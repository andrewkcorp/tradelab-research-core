"""Local, read-only, self-contained live dashboard for ``tradelab.
research_run`` artifacts.

**Read-only by construction.** The HTTP server this module runs
implements only ``do_GET`` -- there is no handler for POST/PUT/DELETE/
PATCH at all, so ``http.server.BaseHTTPRequestHandler``'s own default
behavior (501 Unsupported Method) answers any other verb. Every
endpoint only ever reads files already on disk via ``tradelab.
research_run``'s own reader functions (``read_live_status``, ``read_
run_manifest``, ``read_leaderboard``, ``read_progress_events``, ``list_
run_ids``) -- this module never imports, calls, or influences anything
that writes to a run directory, and can never alter, pause, or steer
the process actually producing those artifacts.

**No external dependencies, no CDN.** The dashboard page is one
self-contained HTML document (inline CSS + vanilla JS, no ``<script
src="https://...">``) served directly from a Python string constant in
this module -- nothing is downloaded at request time, so the dashboard
works fully offline. Charts are hand-drawn on ``<canvas>`` with plain
JS (line/bar/histogram/scatter helpers) -- no charting library.

**Polling, not push.** The page polls ``/api/runs/<run_id>/status``
(and the other JSON endpoints) on a short interval while the run's own
``phase`` is not ``completed``/``failed``, and stops polling once it
is (a manual "Refresh" button remains available). This keeps the
server itself trivial (plain request/response, no websockets) while
still being fluid enough for a human watching a live run.

Run it directly: ``python3 -m tradelab.research_dashboard --runs-dir
runs`` (see ``docs/research-dashboard-v1.md``).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tradelab.research_run import list_run_ids, read_leaderboard, read_live_status, read_progress_events, read_run_manifest

DEFAULT_PORT = 8765
UNIVERSE_PAGE_DEFAULT = 100
UNIVERSE_PAGE_MAX = 250
UNIVERSE_CACHE_SECONDS = 60.0
UNIVERSE_CACHE_MAX_ORDERS = 2
_LOSS_LIMITS_2000_SQL = "__loss_limits_2000__"
_PROFITABLE_MONTHS_SQL = "__profitable_months__"
_WORST_MONTH_PNL_SQL = "__worst_month_pnl__"
_PERF_JSON = "performance_validation_json"
def _perf_sql(field: str) -> str:
    return f"json_extract({_PERF_JSON}, '$.{field}')"

_UNIVERSE_SORT_COLUMNS = {
    "rank": "net_pnl_usd",
    "pnl": "net_pnl_usd",
    "roi": "roi",
    "win_rate": "win_rate",
    "trade_count": "trade_count",
    "max_drawdown": "max_drawdown_usd",
    # The user-facing value is the number of complete $2k risk limits
    # contained in the worst observed drawdown, not the legacy count of
    # threshold-entry edges.  Ordering by the underlying drawdown is exactly
    # monotonic with floor(drawdown / 2000) and avoids an expression index.
    "drawdowns_over_2000_count": _LOSS_LIMITS_2000_SQL,
    "profit_factor": _perf_sql("profit_factor"),
    "average_r": _perf_sql("average_r"),
    "daily_sharpe": _perf_sql("annualized_daily_sharpe"),
    "expectancy": _perf_sql("expectancy_usd"),
    "top_winner_share": _perf_sql("top_winner_share_of_positive_pnl"),
    "profitable_months": _PROFITABLE_MONTHS_SQL,
    "worst_month_pnl": _WORST_MONTH_PNL_SQL,
}
_RANGE_FILTER_COLUMNS = {
    "pnl": "net_pnl_usd", "roi": "roi", "win_rate": "win_rate",
    "trade_count": "trade_count", "max_drawdown": "max_drawdown_usd",
    "drawdowns_over_2000_count": _LOSS_LIMITS_2000_SQL,
    "completed_trades_2024": "completed_trades_2024",
    "completed_trades_2025": "completed_trades_2025",
    "mtm_max_drawdown": "mtm_max_drawdown_usd",
    "profit_factor": _perf_sql("profit_factor"),
    "average_r": _perf_sql("average_r"),
    "daily_sharpe": _perf_sql("annualized_daily_sharpe"),
    "expectancy": _perf_sql("expectancy_usd"),
    "top_winner_share": _perf_sql("top_winner_share_of_positive_pnl"),
    "profitable_months": _PROFITABLE_MONTHS_SQL,
    "worst_month_pnl": _WORST_MONTH_PNL_SQL,
}
_BOOLEAN_FILTER_COLUMNS = {"apex_has_breach": "apex_has_breach"}


def _loss_limits_2000(*, max_drawdown_usd: float | None, mtm_max_drawdown_usd: float | None) -> int:
    """Return complete $2,000 risk limits consumed by worst drawdown.

    New journals prefer conservative mark-to-market drawdown. Historic runs
    fall back to realised max drawdown. This is deliberately derived at read
    time so a live run never needs to be stopped or its durable journal rows
    rewritten merely to correct dashboard semantics.
    """
    realised = float(max_drawdown_usd or 0.0)
    mtm = float(mtm_max_drawdown_usd or 0.0)
    return max(0, int(max(realised, mtm) // 2_000.0))


def _parse_universe_filters(query: dict) -> dict[str, object]:
    """Parse allow-listed numeric/categorical/strategy-token filters."""
    filters: dict[str, object] = {}
    for key in _RANGE_FILTER_COLUMNS:
        for bound in ("min", "max"):
            raw = query.get(f"{key}_{bound}", [""])[0].strip()
            if raw:
                filters[f"{key}_{bound}"] = float(raw)
    for key in ("direction", "instrument", "tp_kind", "sl_kind"):
        values = tuple(v.strip() for v in query.get(key, [""])[0].split(",") if v.strip())
        if values:
            filters[key] = values
    # ``apex_has_breach=0`` is the useful production-research filter: it
    # hides strategies that violate the selected prop-account audit without
    # retroactively changing their fills or P&L.
    raw_apex = query.get("apex_has_breach", [""])[0].strip()
    if raw_apex:
        if raw_apex not in {"0", "1"}:
            raise ValueError("apex_has_breach must be 0 or 1")
        filters["apex_has_breach"] = int(raw_apex)
    # All confluences, session names and future selector labels are represented
    # in strategy_variant_id. Comma-separated terms are ANDed, so this remains
    # forward-compatible without hard-coding a fragile selector vocabulary.
    for key in ("confluence", "session", "contains"):
        values = tuple(v.strip() for v in query.get(key, [""])[0].split(",") if v.strip())
        if values:
            filters[key] = values
    return filters


def _monthly_equity_curve(monthly: dict[str, object], *, starting_equity: float = 50_000.0) -> list[dict[str, object]]:
    """Build a deliberately coarse, monthly realised-equity line for a
    variant that is not one of the persisted P&L leaderboard entries.

    The journal stores monthly P&L for *every* completed variant, but stores
    per-trade detail only for the conventional P&L top list.  This lets a
    user inspect any strategy selected by a global sort without pretending
    that a monthly reconstruction has intratrade precision.
    """
    equity = starting_equity
    points: list[dict[str, object]] = []
    for month, value in sorted(monthly.items()):
        if not isinstance(value, dict):
            continue
        equity += float(value.get("net_pnl_usd", 0.0))
        points.append({"ts": f"{month}-01T00:00:00+00:00", "equity": equity})
    return points


def _monthly_stability_metrics(monthly: dict[str, object]) -> dict[str, float | int | None]:
    """Return calendar-month stability diagnostics from durable journal data.

    These deliberately replace the much coarser year diagnostics in the UI.
    They work for both historic rows and rows written by an active run, without
    mutating the journal or changing any backtest result.
    """
    values = [float(value.get("net_pnl_usd", 0.0)) for value in monthly.values() if isinstance(value, dict)]
    return {
        "profitable_month_count": sum(value > 0 for value in values),
        "evaluated_month_count": len(values),
        "minimum_month_net_pnl_usd": min(values) if values else None,
        "maximum_month_net_pnl_usd": max(values) if values else None,
    }


def _build_universe_order(run_dir: Path, *, sort_key: str, direction: str, min_trades: int, filters: dict[str, object]) -> list[int] | None:
    """Build a compact, read-only order snapshot (SQLite rowids only)."""
    sort_column = _UNIVERSE_SORT_COLUMNS.get(sort_key)
    if sort_column is None:
        raise ValueError(f"unsupported sort key {sort_key!r}")
    if direction not in {"asc", "desc"}:
        raise ValueError(f"unsupported sort direction {direction!r}")
    db_path = run_dir / "backtest_v2_journal.sqlite3"
    if not db_path.is_file():
        return None
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(variant_summaries)")}
        loss_drawdown_column = (
            "MAX(max_drawdown_usd, COALESCE(mtm_max_drawdown_usd, max_drawdown_usd))"
            if "mtm_max_drawdown_usd" in columns else "max_drawdown_usd"
        )
        if sort_column == _LOSS_LIMITS_2000_SQL:
            sort_column = f"CAST(({loss_drawdown_column}) / 2000.0 AS INTEGER)"
        elif sort_column == _PROFITABLE_MONTHS_SQL:
            sort_column = "(SELECT COUNT(*) FROM json_each(monthly_json) AS month WHERE CAST(json_extract(month.value, '$.net_pnl_usd') AS REAL) > 0)"
        elif sort_column == _WORST_MONTH_PNL_SQL:
            sort_column = "(SELECT MIN(CAST(json_extract(month.value, '$.net_pnl_usd') AS REAL)) FROM json_each(monthly_json) AS month)"
        elif _PERF_JSON in sort_column and _PERF_JSON not in columns:
            sort_column = "net_pnl_usd"
        order = "ASC" if direction == "asc" else "DESC"
        clauses = ["trade_count >= ?"]
        values: list[object] = [min_trades]
        # New journals persist the final research rule explicitly: >=100
        # actually completed trades in both complete years. Old journals
        # remain readable and fall back to the historic total-trade filter.
        if "research_eligible" in columns:
            clauses.append("research_eligible = 1")
        for key, filter_column in _RANGE_FILTER_COLUMNS.items():
            if filter_column == _LOSS_LIMITS_2000_SQL:
                filter_column = f"CAST(({loss_drawdown_column}) / 2000.0 AS INTEGER)"
            elif filter_column == _PROFITABLE_MONTHS_SQL:
                filter_column = "(SELECT COUNT(*) FROM json_each(monthly_json) AS month WHERE CAST(json_extract(month.value, '$.net_pnl_usd') AS REAL) > 0)"
            elif filter_column == _WORST_MONTH_PNL_SQL:
                filter_column = "(SELECT MIN(CAST(json_extract(month.value, '$.net_pnl_usd') AS REAL)) FROM json_each(monthly_json) AS month)"
            elif _PERF_JSON in filter_column and _PERF_JSON not in columns:
                continue
            elif _PERF_JSON not in filter_column and filter_column not in columns:
                continue
            if f"{key}_min" in filters:
                clauses.append(f"{filter_column} >= ?"); values.append(filters[f"{key}_min"])
            if f"{key}_max" in filters:
                clauses.append(f"{filter_column} <= ?"); values.append(filters[f"{key}_max"])
        for key in ("direction", "instrument", "tp_kind", "sl_kind"):
            if key in filters:
                selected = filters[key]
                clauses.append(f"{key} IN ({','.join('?' for _ in selected)})"); values.extend(selected)
        for key, column in _BOOLEAN_FILTER_COLUMNS.items():
            if key in filters and column in columns:
                clauses.append(f"{column} = ?"); values.append(filters[key])
        for key in ("confluence", "session", "contains"):
            for term in filters.get(key, ()):
                clauses.append("strategy_variant_id LIKE ?"); values.append(f"%{term}%")
        return [row[0] for row in connection.execute(
            f"""SELECT rowid FROM variant_summaries
                WHERE {' AND '.join(clauses)}
                ORDER BY {sort_column} {order}, strategy_variant_id ASC""",
            values,
        )]
    finally:
        connection.close()


def _journal_supports_live_readers(run_dir: Path) -> bool:
    """Return true only for a WAL journal safe to query during writes.

    WAL readers see a stable snapshot and do not block the writer.  Historic
    rollback-journal runs retain the lightweight top-N fallback, because a
    full filter could otherwise delay a critical durable checkpoint.
    """
    db_path = run_dir / "backtest_v2_journal.sqlite3"
    if not db_path.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.25)
        try:
            row = connection.execute("PRAGMA journal_mode").fetchone()
            return bool(row and str(row[0]).lower() == "wal")
        finally:
            connection.close()
    except sqlite3.Error:
        return False


def _read_universe_leaders(
    run_dir: Path, *, order_rowids: list[int], sort_key: str, direction: str,
    min_trades: int, offset: int, limit: int,
) -> dict:
    """Return a small, globally sorted view of completed journal rows.

    Unlike ``leaderboard.json`` (which intentionally persists only the P&L
    leaders with complete trade detail), this read-only SQL query can rank
    every completed strategy by any dashboard metric.  The expensive order is
    supplied as a cached list of SQLite rowids; this function reads only one
    small page of details.  The browser can therefore append pages during
    scrolling without repeatedly sorting the whole journal.
    """
    db_path = run_dir / "backtest_v2_journal.sqlite3"
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), UNIVERSE_PAGE_MAX))
    requested_rowids = order_rowids[offset:offset + limit]
    if not requested_rowids:
        return {
            "schema_version": "dashboard-universe-v2", "run_id": run_dir.name,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": [], "universe_scope": "all_completed_variants",
            "sort_key": sort_key, "sort_direction": direction,
            "min_trades": min_trades, "total_eligible": len(order_rowids), "offset": offset,
        }
    placeholders = ",".join("?" for _ in requested_rowids)
    # Rowids originate solely from our preceding SELECT. mode=ro is important:
    # the dashboard never creates a journal, index, WAL checkpoint, or other artifact.
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(variant_summaries)")}
        # Old completed journals predate the Apex audit columns. The
        # dashboard must remain readable for them; absent values mean the
        # audit was not recorded, not a passed audit.
        def optional_column(name: str, default: str) -> str:
            return name if name in columns else f"{default} AS {name}"
        rows = connection.execute(
            f"""SELECT rowid, strategy_variant_id, net_pnl_usd, roi, win_rate, trade_count,
                   max_drawdown_usd, drawdowns_over_2000_count,
                   {optional_column('apex_has_breach', '0')},
                   {optional_column('apex_breach_count', '0')},
                   {optional_column('apex_max_daily_loss_usd', '0.0')},
                   {optional_column('apex_largest_eod_drawdown_breach_usd', '0.0')}, monthly_json
                   , {optional_column('research_eligible', '0')}
                   , {optional_column('completed_trades_2024', '0')}
                   , {optional_column('completed_trades_2025', '0')}
                   , {optional_column('mtm_worst_equity_usd', 'NULL')}
                   , {optional_column('mtm_max_drawdown_usd', 'NULL')}
                   , {optional_column('mtm_worst_equity_at', 'NULL')}
                   , {optional_column('ambiguous_same_bar_exit_count', '0')}
                   , {optional_column('performance_validation_json', "'{}'")}
                   , {optional_column('stop_slippage_sensitivity_json', "'{}'")}
            FROM variant_summaries
            WHERE rowid IN ({placeholders})""",
            requested_rowids,
        ).fetchall()
    finally:
        connection.close()

    persisted = read_leaderboard(run_dir)
    detailed_by_id = {entry.strategy_id: entry.to_dict() for entry in (persisted.entries if persisted else ())}
    by_rowid = {row[0]: row[1:] for row in rows}
    entries = []
    for position, rowid in enumerate(requested_rowids, start=offset + 1):
        (strategy_id, pnl, roi, win_rate, trade_count, max_drawdown, dd_count,
         apex_has_breach, apex_breach_count, apex_max_daily_loss,
         apex_largest_eod_breach, monthly_json, research_eligible,
         completed_trades_2024, completed_trades_2025, mtm_worst_equity,
         mtm_max_drawdown, mtm_worst_equity_at, ambiguous_same_bar_exit_count,
         performance_validation_json, stop_slippage_sensitivity_json) = by_rowid[rowid]
        performance_validation = json.loads(performance_validation_json or "{}")
        monthly = json.loads(monthly_json or "{}")
        performance_validation.update(_monthly_stability_metrics(monthly))
        slippage_sensitivity = json.loads(stop_slippage_sensitivity_json or "{}")
        detailed = detailed_by_id.get(strategy_id)
        if detailed is not None:
            # leaderboard.json retains rich trade detail but predates the
            # journal's Apex audit columns. Preserve that detail and augment
            # it from the authoritative per-variant journal row.
            detailed = dict(detailed)
            detailed.update({
                "apex_has_breach": bool(apex_has_breach),
                "apex_breach_count": apex_breach_count,
                "apex_max_daily_loss_usd": apex_max_daily_loss,
                "apex_largest_eod_drawdown_breach_usd": apex_largest_eod_breach,
                "research_eligible": bool(research_eligible),
                "completed_trades_2024": completed_trades_2024,
                "completed_trades_2025": completed_trades_2025,
                "mtm_worst_equity_usd": mtm_worst_equity,
                "mtm_max_drawdown_usd": mtm_max_drawdown,
                "mtm_worst_equity_at": mtm_worst_equity_at,
                "ambiguous_same_bar_exit_count": ambiguous_same_bar_exit_count,
                "performance_validation": performance_validation,
                "stop_slippage_sensitivity_usd": slippage_sensitivity,
            })
            detailed["drawdowns_over_2000_count"] = _loss_limits_2000(
                max_drawdown_usd=max_drawdown, mtm_max_drawdown_usd=mtm_max_drawdown,
            )
            detailed["rank"] = position
            entries.append(detailed)
            continue
        trades_by_month = {month: int(value.get("trade_count", 0)) for month, value in monthly.items() if isinstance(value, dict)}
        entries.append({
            "rank": position,
            "strategy_id": strategy_id,
            "pnl": pnl,
            "roi": roi,
            "win_rate": win_rate or 0.0,
            "trade_count": trade_count,
            "max_drawdown": max_drawdown,
            "drawdowns_over_2000_count": _loss_limits_2000(
                max_drawdown_usd=max_drawdown, mtm_max_drawdown_usd=mtm_max_drawdown,
            ),
            "apex_has_breach": bool(apex_has_breach),
            "apex_breach_count": apex_breach_count,
            "apex_max_daily_loss_usd": apex_max_daily_loss,
            "apex_largest_eod_drawdown_breach_usd": apex_largest_eod_breach,
            "research_eligible": bool(research_eligible),
            "completed_trades_2024": completed_trades_2024,
            "completed_trades_2025": completed_trades_2025,
            "mtm_worst_equity_usd": mtm_worst_equity,
            "mtm_max_drawdown_usd": mtm_max_drawdown,
            "mtm_worst_equity_at": mtm_worst_equity_at,
            "ambiguous_same_bar_exit_count": ambiguous_same_bar_exit_count,
            "performance_validation": performance_validation,
            "stop_slippage_sensitivity_usd": slippage_sensitivity,
            "equity_curve": _monthly_equity_curve(monthly),
            "trade_pnls": [],
            "trades_by_month": trades_by_month,
            "detail_granularity": "monthly",
        })
    return {
        "schema_version": "dashboard-universe-v2",
        "run_id": run_dir.name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": entries,
        "universe_scope": "all_completed_variants",
        "sort_key": sort_key,
        "sort_direction": direction,
        "min_trades": min_trades,
        "total_eligible": len(order_rowids),
        "offset": offset,
    }


def _json_bytes(payload) -> bytes:
    return json.dumps(payload).encode("utf-8")


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    """Implements only GET -- see module docstring's read-only
    guarantee. ``runs_dir`` is bound per-server via a small factory
    (``_make_handler_class``) rather than a constructor argument,
    because ``http.server`` instantiates a fresh handler per request."""

    runs_dir: Path = Path("runs")  # overridden by _make_handler_class
    server_version = "TradeLabResearchDashboard/1.0"
    # Keys map to compact rowid ordering snapshots, not full strategy objects.
    # Keeping only the two most recently requested sorts prevents dashboard
    # exploration from competing materially with the active backtest's RAM.
    universe_cache: dict[tuple[str, str, str, int], tuple[float, list[int] | None]] = {}
    universe_cache_lock = threading.Lock()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 -- matches BaseHTTPRequestHandler's own signature
        pass  # quiet by default; nothing here is a security-relevant access log

    def _send_json(self, payload, status: int = 200) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's own naming
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._send_html(DASHBOARD_HTML)
                return
            if path == "/api/runs":
                self._handle_list_runs()
                return
            if path.startswith("/api/runs/"):
                self._handle_run_endpoint(path, query)
                return
            self._send_json({"error": f"not found: {path}"}, status=404)
        except FileNotFoundError:
            self._send_json({"error": "run artifact not found -- the run may not have written it yet"}, status=404)
        except Exception as exc:  # pragma: no cover -- defensive: never crash the read-only server on a bad artifact
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _handle_list_runs(self) -> None:
        runs = []
        for run_id in list_run_ids(self.runs_dir):
            try:
                manifest = read_run_manifest(self.runs_dir / run_id)
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
            runs.append({"run_id": run_id, "created_at": manifest.created_at, "is_demo": manifest.is_demo})
        self._send_json({"runs": runs})

    def _handle_run_endpoint(self, path: str, query: dict) -> None:
        parts = path.split("/")
        # ["", "api", "runs", "<run_id>", "<endpoint>"]
        if len(parts) < 5:
            self._send_json({"error": "malformed run endpoint path"}, status=400)
            return
        run_id, endpoint = parts[3], parts[4]
        run_dir = self.runs_dir / run_id
        if not run_dir.is_dir():
            self._send_json({"error": f"unknown run_id {run_id!r}"}, status=404)
            return

        if endpoint == "status":
            status = read_live_status(run_dir)
            self._send_json(status.to_dict())
        elif endpoint == "manifest":
            manifest = read_run_manifest(run_dir)
            self._send_json(manifest.to_dict())
        elif endpoint == "leaderboard":
            scope = query.get("scope", [""])[0]
            if scope != "all":
                leaderboard = read_leaderboard(run_dir)
                self._send_json(leaderboard.to_dict() if leaderboard is not None else None)
                return
            # A live full-table ORDER BY can hold a SQLite read transaction
            # long enough for the writer's short checkpoint transaction to
            # fail. While a backtest is actively writing, serve only its
            # atomic JSON P&L snapshot. Full global filtering is enabled once
            # the run is terminal, when no writer can be blocked.
            live_status = read_live_status(run_dir)
            if live_status.phase == "simulating_backtests" and not _journal_supports_live_readers(run_dir):
                leaderboard = read_leaderboard(run_dir)
                payload = leaderboard.to_dict() if leaderboard is not None else {"entries": []}
                sort_key = query.get("sort", ["pnl"])[0]
                direction = query.get("dir", ["desc"])[0]
                column = _UNIVERSE_SORT_COLUMNS.get(sort_key, "net_pnl_usd")
                if sort_key == "drawdowns_over_2000_count":
                    for entry in payload.get("entries", []):
                        entry["drawdowns_over_2000_count"] = _loss_limits_2000(
                            max_drawdown_usd=entry.get("max_drawdown"),
                            mtm_max_drawdown_usd=entry.get("mtm_max_drawdown_usd"),
                        )
                    entry_key = "drawdowns_over_2000_count"
                elif _PERF_JSON in column:
                    field = {
                        "profit_factor": "profit_factor", "average_r": "average_r",
                        "daily_sharpe": "annualized_daily_sharpe", "expectancy": "expectancy_usd",
                        "top_winner_share": "top_winner_share_of_positive_pnl",
                        "profitable_months": "profitable_month_count", "worst_month_pnl": "minimum_month_net_pnl_usd",
                    }[sort_key]
                    for entry in payload.get("entries", []):
                        entry[sort_key] = (entry.get("performance_validation") or {}).get(field)
                    entry_key = sort_key
                else:
                    entry_key = {
                        "net_pnl_usd": "pnl", "roi": "roi", "win_rate": "win_rate",
                        "trade_count": "trade_count", "max_drawdown_usd": "max_drawdown",
                    }[column]
                # Sorting this atomic JSON snapshot is in-memory and tiny
                # (top-N only), so it never holds a SQLite read lock against
                # the active writer. It makes the live dashboard responsive
                # while preserving the safety reason full-universe SQL is
                # paused until completion.
                payload["entries"] = sorted(
                    payload.get("entries", []),
                    key=lambda entry: (entry.get(entry_key) is None, entry.get(entry_key, 0.0), entry.get("strategy_id", "")),
                    reverse=(direction != "asc"),
                )
                for rank, entry in enumerate(payload["entries"], start=1):
                    entry["rank"] = rank
                payload.update({
                    "universe_scope": "live_pnl_snapshot_only",
                    "filtering_available": False,
                    "sort_key": sort_key,
                    "sort_direction": direction,
                    "total_eligible": len(payload.get("entries", [])),
                    "notice": "Global filters are paused while the backtest writes its journal; they become available automatically after completion.",
                })
                self._send_json(payload)
                return
            sort_key = query.get("sort", ["pnl"])[0]
            direction = query.get("dir", ["desc"])[0]
            min_trades = max(1, int(query.get("min_trades", ["100"])[0]))
            offset = max(0, int(query.get("offset", ["0"])[0]))
            limit = max(1, min(int(query.get("limit", [str(UNIVERSE_PAGE_DEFAULT)])[0]), UNIVERSE_PAGE_MAX))
            filters = _parse_universe_filters(query)
            filter_key = tuple(sorted((key, tuple(value) if isinstance(value, tuple) else value) for key, value in filters.items()))
            cache_key = (run_id, sort_key, direction, min_trades, filter_key)
            now = time.monotonic()
            with self.universe_cache_lock:
                cached = self.universe_cache.get(cache_key)
                if cached is not None and now - cached[0] < UNIVERSE_CACHE_SECONDS:
                    order_rowids = cached[1]
                else:
                    order_rowids = _build_universe_order(run_dir, sort_key=sort_key, direction=direction, min_trades=min_trades, filters=filters)
                    self.universe_cache[cache_key] = (now, order_rowids)
                    while len(self.universe_cache) > UNIVERSE_CACHE_MAX_ORDERS:
                        self.universe_cache.pop(next(iter(self.universe_cache)))
            payload = None if order_rowids is None else _read_universe_leaders(
                run_dir, order_rowids=order_rowids, sort_key=sort_key, direction=direction,
                min_trades=min_trades, offset=offset, limit=limit,
            )
            self._send_json(payload)
        elif endpoint == "events":
            limit = int(query.get("limit", ["50"])[0])
            self._send_json({"events": read_progress_events(run_dir, limit=limit)})
        else:
            self._send_json({"error": f"unknown endpoint {endpoint!r}"}, status=404)


def _make_handler_class(runs_dir: Path) -> type:
    return type("_BoundDashboardRequestHandler", (_DashboardRequestHandler,), {"runs_dir": runs_dir})


def serve(runs_dir: Path | str, *, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Build (but do not run the forever-loop of) a
    ``ThreadingHTTPServer`` bound to ``(host, port)``, serving the
    dashboard for runs under ``runs_dir``. Returned so a caller (e.g. a
    test) can call ``.handle_request()`` once, or a CLI can call
    ``.serve_forever()``; either way the server itself does nothing
    beyond respond to HTTP GETs -- see module docstring."""
    handler_class = _make_handler_class(Path(runs_dir))
    return ThreadingHTTPServer((host, port), handler_class)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local, read-only live dashboard for tradelab.research_run artifacts.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run_id subdirectories (default: ./runs)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard in the default browser once the server is up")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    server = serve(runs_dir, port=args.port, host=args.host)
    url = f"http://{args.host}:{args.port}/"
    print(f"TradeLab research dashboard: {url}")
    print(f"Serving runs from: {runs_dir.resolve()}")
    print("Read-only: this server never writes to the runs directory. Press Ctrl+C to stop.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TradeLab Research Dashboard</title>
<style>
  :root {
    --bg: #0b0e14; --bg-panel: #12161f; --bg-card: #161b26; --border: #232938;
    --text: #e6e9f0; --text-dim: #8b93a7; --text-faint: #5b6377;
    --accent: #4f8cff; --good: #34d399; --bad: #f87171; --warn: #fbbf24; --demo: #a855f7;
    --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  header { display: flex; align-items: center; gap: 16px; padding: 14px 20px; background: var(--bg-panel); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: 0.2px; }
  header h1 span { color: var(--accent); }
  select, button { background: var(--bg-card); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  select:hover, button:hover { border-color: var(--accent); }
  .badge { padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }
  .badge-phase { background: rgba(79,140,255,0.15); color: var(--accent); }
  .badge-completed { background: rgba(52,211,153,0.15); color: var(--good); }
  .badge-failed { background: rgba(248,113,113,0.15); color: var(--bad); }
  .badge-demo { background: rgba(168,85,247,0.18); color: var(--demo); }
  .badge-mode-indexing { background: rgba(251,191,36,0.15); color: var(--warn); }
  .badge-mode-resolving { background: rgba(168,85,247,0.15); color: var(--demo); }
  .badge-mode-simulating { background: rgba(52,211,153,0.15); color: var(--good); }
  .spacer { flex: 1; }
  main { padding: 20px; max-width: 1560px; margin: 0 auto; }
  .demo-banner { background: linear-gradient(90deg, rgba(168,85,247,0.18), rgba(168,85,247,0.05)); border: 1px solid rgba(168,85,247,0.4); color: var(--demo); padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 600; font-size: 13px; display: none; }
  .empty-state { color: var(--text-faint); font-size: 13px; text-align: center; padding: 30px 10px; }
  section { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }
  section h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text-dim); margin: 0 0 12px 0; }
  .progress-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
  .progress-bar-track { flex: 1; min-width: 200px; height: 10px; background: var(--bg-card); border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
  .progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), #7db2ff); transition: width 0.3s ease; }
  .stat { font-family: var(--mono); font-size: 13px; color: var(--text-dim); }
  .stat b { color: var(--text); font-weight: 600; }
  .timeline { display: flex; gap: 6px; flex-wrap: wrap; }
  .timeline-step { flex: 1; min-width: 110px; padding: 8px 10px; border-radius: 6px; background: var(--bg-card); border: 1px solid var(--border); font-size: 12px; text-align: center; color: var(--text-faint); }
  .timeline-step.done { color: var(--good); border-color: rgba(52,211,153,0.35); }
  .timeline-step.current { color: var(--accent); border-color: var(--accent); background: rgba(79,140,255,0.08); font-weight: 600; }
  .timeline-step.failed-step { color: var(--bad); border-color: rgba(248,113,113,0.4); }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--text-faint); margin-bottom: 6px; }
  .card .value { font-family: var(--mono); font-size: 22px; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { color: var(--text-faint); font-weight: 600; text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.3px; }
  th .sort-button { appearance: none; border: 0; padding: 0; background: transparent; color: inherit; font: inherit; letter-spacing: inherit; text-transform: inherit; cursor: pointer; }
  th .sort-button:hover, th .sort-button.active { color: var(--accent); }
  .sort-indicator { display: inline-block; width: 12px; color: var(--accent); }
  td.num, th.num { text-align: right; font-family: var(--mono); }
  td.strategy-cell { white-space: normal; min-width: 340px; max-width: 520px; font: 11px/1.35 var(--mono); color: var(--text-dim); overflow-wrap: anywhere; }
  tr:hover td { background: rgba(79,140,255,0.08); }
  tr.selected td { background: rgba(79,140,255,0.16); border-bottom-color: rgba(79,140,255,0.55); }
  tr.selectable { cursor: pointer; }
  .pnl-pos { color: var(--good); } .pnl-neg { color: var(--bad); }
  .strategy-summary { display: grid; grid-template-columns: minmax(260px, 2fr) repeat(5, minmax(105px, 1fr)); gap: 10px; }
  .strategy-summary .card { min-height: 90px; }
  .strategy-summary .strategy-name { font: 12px/1.45 var(--mono); overflow-wrap: anywhere; color: var(--text-dim); }
  .strategy-summary .pnl-value { font-size: 27px; }
  .subtle { color: var(--text-faint); font-size: 11px; }
  .leaders-note { color: var(--text-faint); margin: -4px 0 12px; font-size: 12px; }
  .leaders-scroll { max-height: 620px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; }
  .leaders-scroll table { min-width: 1120px; }
  .leaders-load-more { color: var(--text-faint); text-align: center; padding: 12px; font-size: 12px; }
  .filters { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:8px; margin: 10px 0 12px; }
  .filters input, .filters select { width:100%; }
  .filters .wide { grid-column:span 2; }
  .filters-panel { display:none; border:1px solid var(--border); border-radius:8px; padding:12px; margin:10px 0; background:var(--bg-card); }
  .filters-panel.open { display:block; }
  .filter-toggle { margin:0 0 2px; }
  .check-group { border:1px solid var(--border); border-radius:6px; padding:8px; min-width:170px; }
  .check-group .group-label { display:block; color:var(--text-faint); font-size:10px; text-transform:uppercase; letter-spacing:.4px; margin-bottom:5px; }
  .check-group label { display:inline-flex; align-items:center; gap:5px; margin:2px 10px 2px 0; font-size:12px; color:var(--text-dim); white-space:nowrap; }
  .check-group input { width:auto; }
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .chart-box { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px; min-height: 285px; position: relative; }
  .chart-box.wide { grid-column: 1 / -1; }
  .chart-title-row { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 0 0 4px; }
  .chart-box h3 { font-size: 13px; color: var(--text); margin: 0; font-weight: 650; }
  .chart-subtitle { font-size: 11px; color: var(--text-faint); margin-bottom: 8px; min-height: 15px; }
  canvas { width: 100%; height: 220px; display: block; cursor: crosshair; }
  .chart-tooltip { position: fixed; display: none; pointer-events: none; z-index: 30; background: #080b12; border: 1px solid #3b4762; border-radius: 6px; padding: 7px 9px; color: var(--text); font: 11px/1.45 var(--mono); box-shadow: 0 8px 24px rgba(0,0,0,.35); max-width: 320px; }
  .chart-key { color: var(--text-faint); font-size: 11px; }
  @media (max-width: 900px) { .strategy-summary { grid-template-columns: repeat(2, 1fr); } .charts-grid { grid-template-columns: 1fr; } }
  .warnings-list, .events-list { max-height: 220px; overflow-y: auto; font-family: var(--mono); font-size: 12px; }
  .warnings-list div, .events-list div { padding: 5px 0; border-bottom: 1px solid var(--border); color: var(--text-dim); }
  .events-list .ev-type { color: var(--accent); font-weight: 600; }
  .warnings-list div { color: var(--warn); }
  .errors-list div { color: var(--bad); }
  footer { text-align: center; padding: 20px; color: var(--text-faint); font-size: 11.5px; }
  a { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>TradeLab <span>Research Dashboard</span></h1>
  <select id="run-select"><option>Loading runs...</option></select>
  <span id="phase-badge" class="badge badge-phase">-</span>
  <span id="demo-badge" class="badge badge-demo" style="display:none">DEMO</span>
  <div class="spacer"></div>
  <button id="refresh-btn">Refresh</button>
</header>
<main>
  <div id="demo-banner" class="demo-banner">DEMO RUN &mdash; all numbers on this page are synthetic, not real MNQ research results.</div>

  <section id="section-progress">
    <h2>Run status</h2>
    <div id="progress-empty" class="empty-state">No runs found yet. Start a demo with <code>python3 -m tradelab.research_demo</code> or point --runs-dir at a real run.</div>
    <div id="progress-content" style="display:none">
      <div class="progress-row">
        <span class="stat">Elapsed: <b id="stat-elapsed">-</b></span>
        <span class="stat">Items: <b id="stat-items">-</b></span>
        <span class="stat">Throughput: <b id="stat-throughput">-</b></span>
        <span class="stat">ETA: <b id="stat-eta">-</b></span>
        <span class="stat">RAM: <b id="stat-ram">-</b></span>
        <span class="stat">CPU: <b id="stat-cpu">-</b></span>
      </div>
      <div class="progress-row">
        <div class="progress-bar-track"><div id="progress-bar-fill" class="progress-bar-fill" style="width:0%"></div></div>
      </div>
      <div class="timeline" id="phase-timeline"></div>
    </div>
  </section>

  <section id="section-active-work">
    <h2>Live activity</h2>
    <div id="active-work-empty" class="empty-state">No live activity yet.</div>
    <div id="active-work-content" style="display:none">
      <div class="progress-row">
        <span id="active-work-mode-badge" class="badge badge-phase">-</span>
        <span class="stat">Strategy variant: <b id="stat-variant-id">-</b></span>
        <span class="stat">Updated: <b id="stat-active-work-updated">-</b></span>
      </div>
      <div class="progress-row">
        <span class="stat" id="stat-operation-detail">-</span>
      </div>
      <div class="progress-row" id="active-work-unit-row">
        <span class="stat" style="min-width:170px" id="stat-unit-progress-label">Progress: <b id="stat-unit-progress">-</b></span>
        <div class="progress-bar-track"><div id="unit-progress-bar-fill" class="progress-bar-fill" style="width:0%"></div></div>
      </div>
      <div class="progress-row" id="active-work-bars-row">
        <span class="stat" style="min-width:170px">Bars scanned: <b id="stat-bar-progress">-</b></span>
        <div class="progress-bar-track"><div id="bar-progress-bar-fill" class="progress-bar-fill" style="width:0%"></div></div>
      </div>
    </div>
  </section>

  <section id="section-cards">
    <h2>Discovery / backtest counts</h2>
    <div class="cards" id="cards-grid"></div>
  </section>

  <section id="section-selected-strategy">
    <h2>Selected strategy · execution-quality snapshot</h2>
    <div id="strategy-summary-empty" class="empty-state">Select a strategy from the leaderboard when results are available.</div>
    <div id="strategy-summary" class="strategy-summary" style="display:none"></div>
  </section>

  <section id="section-leaders">
    <h2 id="leaders-title">Completed strategies</h2>
    <div class="leaders-note">Only research-eligible variants are shown: at least 100 actually completed trades in both 2024 and 2025. Click a column to sort the entire completed universe; scroll to load more rows without re-sorting. Click a row to inspect it.</div>
    <button id="toggle-filters" class="filter-toggle">Show filters</button>
    <div class="filters-panel" id="filters-panel">
    <div class="filters" id="leader-filters">
      <input id="f-pnl-min" type="number" placeholder="Min P&amp;L ($)"><input id="f-pnl-max" type="number" placeholder="Max P&amp;L ($)">
      <input id="f-roi-min" type="number" placeholder="Min ROI (9.0 = 900%)"><input id="f-roi-max" type="number" placeholder="Max ROI (9.0 = 900%)">
      <input id="f-win-rate-min" type="number" step="0.01" placeholder="Min win rate (0.30 = 30%)"><input id="f-win-rate-max" type="number" step="0.01" placeholder="Max win rate (0.30 = 30%)">
      <input id="f-trade-count-min" type="number" value="100" placeholder="Min trades"><input id="f-trade-count-max" type="number" placeholder="Max trades">
      <input id="f-max-drawdown-min" type="number" placeholder="Min max DD ($)"><input id="f-max-drawdown-max" type="number" placeholder="Max max DD ($)">
      <input id="f-profit-factor-min" type="number" step="0.01" placeholder="Min profit factor"><input id="f-profit-factor-max" type="number" step="0.01" placeholder="Max profit factor">
      <input id="f-average-r-min" type="number" step="0.01" placeholder="Min average R"><input id="f-average-r-max" type="number" step="0.01" placeholder="Max average R">
      <input id="f-daily-sharpe-min" type="number" step="0.01" placeholder="Min daily Sharpe"><input id="f-daily-sharpe-max" type="number" step="0.01" placeholder="Max daily Sharpe">
      <input id="f-expectancy-min" type="number" placeholder="Min expectancy ($)"><input id="f-expectancy-max" type="number" placeholder="Max expectancy ($)">
      <input id="f-top-winner-share-min" type="number" step="0.01" placeholder="Min top-winner share"><input id="f-top-winner-share-max" type="number" step="0.01" placeholder="Max top-winner share">
      <input id="f-profitable-months-min" type="number" min="0" step="1" placeholder="Min profitable months"><input id="f-profitable-months-max" type="number" min="0" step="1" placeholder="Max profitable months">
      <input id="f-worst-month-pnl-min" type="number" placeholder="Min worst-month P&amp;L ($)"><input id="f-worst-month-pnl-max" type="number" placeholder="Max worst-month P&amp;L ($)">
      <div class="check-group"><span class="group-label">Direction</span><label><input type="checkbox" data-filter="direction" value="long">Long</label><label><input type="checkbox" data-filter="direction" value="short">Short</label><label><input type="checkbox" data-filter="direction" value="both">Both</label></div>
      <div class="check-group"><span class="group-label">Instrument</span><label><input type="checkbox" data-filter="instrument" value="MNQ">MNQ</label></div>
      <div class="check-group"><span class="group-label">Apex EOD audit</span><label><input id="f-apex-pass" type="checkbox">No recorded breach</label><label><input id="f-apex-breach" type="checkbox">Recorded breach</label></div>
      <div class="check-group"><span class="group-label">TP target</span><label><input type="checkbox" data-filter="tp_kind" value="data_wick">Data wick</label><label><input type="checkbox" data-filter="tp_kind" value="liquidity_rank">Liquidity</label></div>
      <div class="check-group"><span class="group-label">SL target</span><label><input type="checkbox" data-filter="sl_kind" value="liquidity_rank">Liquidity</label><label><input type="checkbox" data-filter="sl_kind" value="fvg_boundary">FVG boundary</label></div>
      <div class="check-group wide"><span class="group-label">Confluences — selected boxes must all be present</span><label><input type="checkbox" data-filter="confluence" value="fvg_touch">FVG touch</label><label><input type="checkbox" data-filter="confluence" value="ifvg">iFVG</label><label><input type="checkbox" data-filter="confluence" value="liquidity_sweep">Liquidity sweep</label><label><input type="checkbox" data-filter="confluence" value="fvg-1m-fvg_touch">1m FVG</label><label><input type="checkbox" data-filter="confluence" value="fvg-5m-fvg_touch">5m FVG</label><label><input type="checkbox" data-filter="confluence" value="fvg-15m-fvg_touch">15m FVG</label></div>
      <div class="check-group wide"><span class="group-label">Liquidity sessions — selected boxes must all be present</span><label><input type="checkbox" data-filter="session" value="session_asia">Asia</label><label><input type="checkbox" data-filter="session" value="session_london">London</label><label><input type="checkbox" data-filter="session" value="session_ny_am">NY AM</label><label><input type="checkbox" data-filter="session" value="session_ny_pm">NY PM</label><label><input type="checkbox" data-filter="session" value="session_overnight">Overnight</label></div>
      <input id="f-contains" class="wide" placeholder="Other exact strategy token (advanced filter)">
      <button id="apply-filters">Apply filters</button><button id="clear-filters">Clear filters</button>
    </div>
    </div>
    <div id="leaders-empty" class="empty-state">No leaderboard data yet.</div>
    <div id="leaders-scroll" class="leaders-scroll">
      <table id="leaders-table" style="display:none">
        <thead><tr>
          <th class="num">Position</th><th>Strategy</th><th class="num"><button class="sort-button" data-sort-key="pnl">P&amp;L <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="roi">ROI <span class="sort-indicator"></span></button></th>
          <th class="num"><button class="sort-button" data-sort-key="win_rate">Win rate <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="trade_count">Trades <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="max_drawdown">Max DD <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="profit_factor">PF <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="average_r">Avg R <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="daily_sharpe">Sharpe <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="profitable_months">Positive months <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="worst_month_pnl">Worst month <span class="sort-indicator"></span></button></th><th class="num"><button class="sort-button" data-sort-key="drawdowns_over_2000_count">$2k limits <span class="sort-indicator"></span></button></th>
        </tr></thead>
        <tbody id="leaders-tbody"></tbody>
      </table>
      <div id="leaders-load-more" class="leaders-load-more" style="display:none">Scroll to load more strategies.</div>
    </div>
  </section>

  <section id="section-charts">
    <h2>Research charts</h2>
    <div class="charts-grid">
      <div class="chart-box wide"><div class="chart-title-row"><h3 id="equity-title">Equity curve</h3><span class="chart-key">Y: account equity (USD) · X: trade time</span></div><div id="equity-subtitle" class="chart-subtitle">Select a strategy to inspect it.</div><canvas id="chart-equity"></canvas></div>
      <div class="chart-box"><div class="chart-title-row"><h3>P&amp;L distribution</h3><span class="chart-key">Top 5 only</span></div><div class="chart-subtitle">X: realised P&amp;L in USD · Y: strategies in each range</div><canvas id="chart-pnl-dist"></canvas></div>
      <div class="chart-box"><div class="chart-title-row"><h3>Return versus risk</h3><span class="chart-key">Each dot = one strategy</span></div><div class="chart-subtitle">X: maximum drawdown (USD) · Y: realised P&amp;L (USD)</div><canvas id="chart-pnl-dd"></canvas></div>
      <div class="chart-box"><div class="chart-title-row"><h3>Trade count distribution</h3><span class="chart-key">Top 5 only</span></div><div class="chart-subtitle">X: completed trades · Y: strategies in each range</div><canvas id="chart-trade-count"></canvas></div>
      <div class="chart-box"><div class="chart-title-row"><h3>Win-rate distribution</h3><span class="chart-key">Top 5 only</span></div><div class="chart-subtitle">X: win rate (%) · Y: strategies in each range</div><canvas id="chart-win-rate"></canvas></div>
      <div class="chart-box"><div class="chart-title-row"><h3>Backtest throughput</h3><span class="chart-key">Recent progress events</span></div><div class="chart-subtitle">X: update sequence · Y: variants completed per second</div><canvas id="chart-throughput"></canvas></div>
      <div class="chart-box"><div class="chart-title-row"><h3>Selected trade P&amp;L distribution</h3><span class="chart-key">Net P&amp;L per trade</span></div><div class="chart-subtitle">X: realised net P&amp;L (USD) · Y: number of trades</div><canvas id="chart-trade-pnl"></canvas></div>
      <div class="chart-box"><div class="chart-title-row"><h3>Selected realised drawdown</h3><span class="chart-key">Cash-based</span></div><div class="chart-subtitle">X: trade time · Y: peak-to-trough drawdown (USD)</div><canvas id="chart-drawdown"></canvas></div>
      <div class="chart-box wide"><div class="chart-title-row"><h3 id="month-title">Monthly trade activity</h3><span class="chart-key">Selected strategy</span></div><div class="chart-subtitle">X: month · Y: completed trades</div><canvas id="chart-by-month"></canvas></div>
    </div>
  </section>

  <section id="section-warnings">
    <h2>Warnings &amp; errors</h2>
    <div id="warnings-empty" class="empty-state">No warnings or errors.</div>
    <div class="warnings-list" id="warnings-list"></div>
    <div class="errors-list" id="errors-list"></div>
  </section>

  <section id="section-events">
    <h2>Latest events</h2>
    <div id="events-empty" class="empty-state">No events yet.</div>
    <div class="events-list" id="events-list"></div>
  </section>
</main>
<div id="chart-tooltip" class="chart-tooltip"></div>
<footer>Read-only dashboard &middot; polls run artifacts only &middot; never alters a running process</footer>

<script>
const PHASES = ["preparing_data","discovering_entries","deduplicating_entries","expanding_exit_variants","simulating_backtests","ranking_results"];
const PHASE_LABELS = {preparing_data:"Preparing data", discovering_entries:"Discovering entries", deduplicating_entries:"Deduplicating", expanding_exit_variants:"Exit variants", simulating_backtests:"Backtests", ranking_results:"Ranking", completed:"Completed", failed:"Failed"};

let currentRunId = null;
let pollTimer = null;
let selectedStrategyId = null;
let lastLeaderboard = null;
let lastEventsData = null;
let leaderboardSort = {key: "pnl", dir: "desc"};
let leaderboardOffset = 0;
let leaderboardTotal = 0;
let leaderboardLoading = false;
const POLL_MS = 1000; // task 044: live status can update at most once/second server-side -- poll to match
const throughputHistory = []; // [{t, throughput}]

function fmtNum(n) { if (n === null || n === undefined) return "-"; return Number(n).toLocaleString(undefined, {maximumFractionDigits: 2}); }
function fmtSeconds(s) { if (s === null || s === undefined) return "-"; s = Math.round(s); const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60; return (h>0?h+"h ":"") + (m>0||h>0?m+"m ":"") + sec+"s"; }
function fmtMoney(n) { if (n === null || n === undefined) return "-"; const sign = n < 0 ? "-" : ""; return sign + "$" + Math.abs(n).toLocaleString(undefined, {maximumFractionDigits: 2}); }
function fmtPct(n) { if (n === null || n === undefined) return "-"; return (n*100).toFixed(1) + "%"; }
function fmtBytes(n) { if (n === null || n === undefined) return "-"; const units = ["B","KB","MB","GB","TB"]; let i = 0; let v = n; while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; } return v.toFixed(v >= 10 || i === 0 ? 0 : 1) + " " + units[i]; }
function fmtAgo(iso) { if (!iso) return "-"; const s = (Date.now() - new Date(iso).getTime()) / 1000; if (s < 0) return "just now"; if (s < 2) return "just now"; return fmtSeconds(s) + " ago"; }

async function fetchJson(url) {
  const res = await fetch(url, {cache: "no-store"});
  if (!res.ok) return null;
  return res.json();
}

async function loadRuns() {
  const data = await fetchJson("/api/runs");
  const select = document.getElementById("run-select");
  if (!data || !data.runs || data.runs.length === 0) {
    select.innerHTML = "<option>No runs found</option>";
    showEmptyProgress();
    return;
  }
  const previouslySelected = currentRunId;
  select.innerHTML = data.runs.map(r => `<option value="${r.run_id}">${r.run_id}${r.is_demo ? " (demo)" : ""}</option>`).join("");
  if (previouslySelected && data.runs.some(r => r.run_id === previouslySelected)) {
    select.value = previouslySelected;
  } else {
    currentRunId = data.runs[0].run_id;
    select.value = currentRunId;
  }
}

function showEmptyProgress() {
  document.getElementById("progress-empty").style.display = "block";
  document.getElementById("progress-content").style.display = "none";
}

function renderTimeline(phase) {
  const el = document.getElementById("phase-timeline");
  const failed = phase === "failed";
  const currentIndex = PHASES.indexOf(phase);
  el.innerHTML = PHASES.map((p, i) => {
    let cls = "timeline-step";
    if (failed) cls += "";
    else if (phase === "completed" || i < currentIndex) cls += " done";
    else if (i === currentIndex) cls += " current";
    return `<div class="${cls}">${PHASE_LABELS[p]}</div>`;
  }).join("") + (failed ? `<div class="timeline-step failed-step">Failed</div>` : (phase === "completed" ? `<div class="timeline-step done">Completed</div>` : ""));
}

function renderCards(status) {
  const cards = [
    ["Discovered specs", status.discovered_specs],
    ["Rejected specs", status.rejected_specs],
    ["Eligible specs", status.eligible_specs],
    ["Unique entry classes", status.unique_entry_classes],
    ["Exit variants generated", status.exit_variants_generated],
    ["Backtests completed", status.backtests_completed],
  ];
  document.getElementById("cards-grid").innerHTML = cards.map(([label, value]) => `
    <div class="card"><div class="label">${label}</div><div class="value">${value === null || value === undefined ? "&mdash;" : fmtNum(value)}</div></div>
  `).join("");
}

// task 046: active_work.operation is a free-form string, optionally with
// ":"-separated detail (e.g. "building_range_index:highs:level_3") -- this
// maps the *prefix* to one of the four modes the dashboard must clearly
// distinguish, falling back to the raw operation string (still shown, never
// hidden) for anything unrecognized so a future/unmapped operation degrades
// gracefully instead of disappearing.
const ACTIVE_WORK_MODES = [
  { prefix: "preparing_data", label: "Preparing data", badgeClass: "badge-phase", unitLabel: "Progress" },
  { prefix: "discovering_entries", label: "Discovering entries", badgeClass: "badge-phase", unitLabel: "Primaries processed" },
  { prefix: "building_range_index", label: "Building range index", badgeClass: "badge-mode-indexing", unitLabel: "Cells built" },
  { prefix: "resolving_entry_targets", label: "Resolving entry targets", badgeClass: "badge-mode-resolving", unitLabel: "Entries resolved" },
  { prefix: "simulating_backtests", label: "Simulating trade exits", badgeClass: "badge-mode-simulating", unitLabel: "Variant" },
  { prefix: "sampling_throughput", label: "Simulating trade exits", badgeClass: "badge-mode-simulating", unitLabel: "Sample" },
];
function activeWorkMode(operation) {
  const op = operation || "";
  const found = ACTIVE_WORK_MODES.find(m => op === m.prefix || op.startsWith(m.prefix + ":"));
  return found || { label: op || "-", badgeClass: "badge-phase", unitLabel: "Progress" };
}

// task 044: live_status.active_work is optional -- absent entirely on
// older run folders written before this field existed, and null on any
// run (old or new) that isn't currently between heartbeats. Both cases
// render the same "no live activity yet" empty state, never an error.
function renderActiveWork(status) {
  const empty = document.getElementById("active-work-empty");
  const content = document.getElementById("active-work-content");
  const work = status && status.active_work;
  if (!work) {
    empty.style.display = "block";
    content.style.display = "none";
    return;
  }
  empty.style.display = "none";
  content.style.display = "block";

  const mode = activeWorkMode(work.operation);
  const modeBadge = document.getElementById("active-work-mode-badge");
  modeBadge.textContent = mode.label;
  modeBadge.className = "badge " + mode.badgeClass;
  // the raw operation string carries detail the mode label alone doesn't
  // (e.g. which sparse-table stage/level) -- shown, not discarded.
  document.getElementById("stat-operation-detail").textContent = work.operation && work.operation !== mode.prefix ? work.operation : "";
  document.getElementById("stat-variant-id").textContent = work.variant_id || "-";
  document.getElementById("stat-active-work-updated").textContent = fmtAgo(work.updated_at);

  const unitRow = document.getElementById("active-work-unit-row");
  if (work.unit_index === null || work.unit_index === undefined) {
    unitRow.style.display = "none";
  } else {
    unitRow.style.display = "flex";
    document.getElementById("stat-unit-progress-label").firstChild.textContent = mode.unitLabel + ": ";
    document.getElementById("stat-unit-progress").textContent = fmtNum(work.unit_index) + " / " + (work.unit_total === null || work.unit_total === undefined ? "?" : fmtNum(work.unit_total));
    const unitPct = (work.unit_total && work.unit_total > 0) ? Math.min(100, (work.unit_index / work.unit_total) * 100) : 0;
    document.getElementById("unit-progress-bar-fill").style.width = unitPct + "%";
  }

  // task 046: bars_scanned/bars_total are only ever populated while a bar
  // is genuinely being scanned (simulating_backtests/sampling_throughput)
  // -- every other mode leaves them unset, so this row simply doesn't
  // render rather than showing a stale or fabricated bar count.
  const barsRow = document.getElementById("active-work-bars-row");
  if (work.bars_scanned === null || work.bars_scanned === undefined) {
    barsRow.style.display = "none";
  } else {
    barsRow.style.display = "flex";
    document.getElementById("stat-bar-progress").textContent = fmtNum(work.bars_scanned) + " / " + (work.bars_total === null || work.bars_total === undefined ? "?" : fmtNum(work.bars_total));
    const barsPct = (work.bars_total && work.bars_total > 0) ? Math.min(100, (work.bars_scanned / work.bars_total) * 100) : 0;
    document.getElementById("bar-progress-bar-fill").style.width = barsPct + "%";
  }
}

const SORT_LABELS = {rank:"original rank", pnl:"realised P&L", roi:"ROI", win_rate:"win rate", trade_count:"trade count", max_drawdown:"max drawdown", profit_factor:"profit factor", average_r:"average R", daily_sharpe:"daily Sharpe", expectancy:"expectancy", top_winner_share:"top-winner concentration", profitable_months:"profitable months", worst_month_pnl:"worst-month P&L", drawdowns_over_2000_count:"$2k loss limits consumed"};
function sortedEntries(leaderboard) {
  const sign = leaderboardSort.dir === "asc" ? 1 : -1;
  return ((leaderboard && leaderboard.entries) || []).slice().sort((a,b) => {
    const metric = e => ({profit_factor:e.performance_validation?.profit_factor, average_r:e.performance_validation?.average_r, daily_sharpe:e.performance_validation?.annualized_daily_sharpe, expectancy:e.performance_validation?.expectancy_usd, top_winner_share:e.performance_validation?.top_winner_share_of_positive_pnl, profitable_months:e.performance_validation?.profitable_month_count, worst_month_pnl:e.performance_validation?.minimum_month_net_pnl_usd}[leaderboardSort.key] ?? e[leaderboardSort.key]);
    const av = Number(metric(a)), bv = Number(metric(b));
    if (av === bv) return a.rank - b.rank;
    return (av - bv) * sign;
  });
}
function topEntries(leaderboard) {
  return sortedEntries(leaderboard).slice(0, 5);
}
function renderSortControls() {
  document.getElementById("leaders-title").textContent = `Completed strategies · ${SORT_LABELS[leaderboardSort.key]} (${leaderboardSort.dir === "asc" ? "lowest first" : "highest first"}) · ${fmtNum((lastLeaderboard && lastLeaderboard.entries || []).length)} / ${fmtNum(leaderboardTotal)} loaded`;
  document.querySelectorAll("[data-sort-key]").forEach(button => {
    const active = button.dataset.sortKey === leaderboardSort.key;
    button.classList.toggle("active", active);
    button.querySelector(".sort-indicator").textContent = active ? (leaderboardSort.dir === "asc" ? "▲" : "▼") : "";
  });
}

function selectedEntry(leaderboard) {
  const entries = (leaderboard && leaderboard.entries) || [];
  if (!entries.length) return null;
  return entries.find(e => e.strategy_id === selectedStrategyId) || entries[0];
}

function renderStrategySummary(leaderboard) {
  const entry = selectedEntry(leaderboard);
  const empty = document.getElementById("strategy-summary-empty");
  const box = document.getElementById("strategy-summary");
  if (!entry) { empty.style.display = "block"; box.style.display = "none"; return; }
  empty.style.display = "none"; box.style.display = "grid";
  const pnlClass = entry.pnl >= 0 ? "pnl-pos" : "pnl-neg";
  const pnls = (entry.trade_pnls || []).filter(Number.isFinite);
  const wins = pnls.filter(v => v > 0), losses = pnls.filter(v => v < 0);
  const grossWins = wins.reduce((a,b)=>a+b,0), grossLosses = Math.abs(losses.reduce((a,b)=>a+b,0));
  const profitFactor = grossLosses ? grossWins / grossLosses : null;
  const averageTrade = pnls.length ? pnls.reduce((a,b)=>a+b,0) / pnls.length : null;
  const biggestWin = wins.length ? Math.max(...wins) : null;
  const biggestLoss = losses.length ? Math.min(...losses) : null;
  const outlierShare = biggestWin && entry.pnl > 0 ? biggestWin / entry.pnl : null;
  const validation = entry.performance_validation || {};
  const slippage3 = (entry.stop_slippage_sensitivity_usd || {})["3"];
  box.innerHTML = `
    <div class="card"><div class="label">#${entry.rank} · selected setup</div><div class="strategy-name">${escapeHtml(entry.strategy_id)}</div></div>
    <div class="card"><div class="label">Realised P&amp;L</div><div class="value pnl-value ${pnlClass}">${fmtMoney(entry.pnl)}</div><div class="subtle">after modelled costs</div></div>
    <div class="card"><div class="label">ROI</div><div class="value">${fmtPct(entry.roi)}</div><div class="subtle">relative to $50,000</div></div>
    <div class="card"><div class="label">Max drawdown</div><div class="value pnl-neg">${fmtMoney(entry.max_drawdown)}</div><div class="subtle">largest peak-to-trough fall</div></div>
    <div class="card"><div class="label">Trades</div><div class="value">${fmtNum(entry.trade_count)}</div><div class="subtle">completed positions</div></div>
    <div class="card"><div class="label">Complete-year frequency</div><div class="value">${fmtNum(entry.completed_trades_2024)} / ${fmtNum(entry.completed_trades_2025)}</div><div class="subtle">actual trades · 2024 / 2025</div></div>
    <div class="card"><div class="label">Win rate</div><div class="value">${fmtPct(entry.win_rate)}</div><div class="subtle">not the ranking criterion</div></div>
    <div class="card"><div class="label">Apex EOD audit</div><div class="value ${entry.apex_has_breach ? "pnl-neg" : "pnl-pos"}">${entry.apex_has_breach ? "BREACH" : "No recorded breach"}</div><div class="subtle">${entry.apex_has_breach ? `${fmtNum(entry.apex_breach_count || 0)} event(s)` : "audit only — does not alter trades"}</div></div>
    <div class="card"><div class="label">Mark-to-market drawdown</div><div class="value pnl-neg">${fmtMoney(entry.mtm_max_drawdown_usd)}</div><div class="subtle">includes open-position path</div></div>
    <div class="card"><div class="label">$2k loss limits consumed</div><div class="value pnl-neg">${fmtNum(entry.drawdowns_over_2000_count)}</div><div class="subtle">floor(worst realised/MTM DD ÷ $2,000)</div></div>
    <div class="card"><div class="label">Ambiguous exit bars</div><div class="value">${fmtNum(entry.ambiguous_same_bar_exit_count)}</div><div class="subtle">TP and SL both touched · SL-first policy</div></div>
    <div class="card"><div class="label">Profit factor</div><div class="value">${profitFactor === null ? "–" : fmtNum(profitFactor)}</div><div class="subtle">gross wins ÷ gross losses</div></div>
    <div class="card"><div class="label">Average trade</div><div class="value ${averageTrade >= 0 ? "pnl-pos" : "pnl-neg"}">${fmtMoney(averageTrade)}</div><div class="subtle">net realised P&amp;L</div></div>
    <div class="card"><div class="label">Largest win</div><div class="value pnl-pos">${fmtMoney(biggestWin)}</div><div class="subtle">share of P&amp;L: ${outlierShare === null ? "–" : fmtPct(outlierShare)}</div></div>
    <div class="card"><div class="label">Largest loss</div><div class="value pnl-neg">${fmtMoney(biggestLoss)}</div><div class="subtle">realised single-trade loss</div></div>`;
  box.innerHTML += `
    <div class="card"><div class="label">Average R</div><div class="value">${fmtNum(validation.average_r)}</div><div class="subtle">net P&amp;L ÷ planned stop risk</div></div>
    <div class="card"><div class="label">Daily Sharpe</div><div class="value">${fmtNum(validation.annualized_daily_sharpe)}</div><div class="subtle">annualized · no-trade weekdays = 0</div></div>
    <div class="card"><div class="label">P&amp;L with +3 stop ticks</div><div class="value ${slippage3 >= 0 ? "pnl-pos" : "pnl-neg"}">${fmtMoney(slippage3)}</div><div class="subtle">analytic sensitivity, signals unchanged</div></div>
    <div class="card"><div class="label">Top-winner concentration</div><div class="value">${fmtPct(validation.top_winner_share_of_positive_pnl)}</div><div class="subtle">largest win ÷ all positive P&amp;L</div></div>`;
  box.innerHTML += `
    <div class="card"><div class="label">Profitable months</div><div class="value">${fmtNum(validation.profitable_month_count)} / ${fmtNum(validation.evaluated_month_count)}</div><div class="subtle">calendar-month net P&amp;L above zero</div></div>
    <div class="card"><div class="label">Worst calendar month</div><div class="value ${(validation.minimum_month_net_pnl_usd ?? 0) >= 0 ? "pnl-pos" : "pnl-neg"}">${fmtMoney(validation.minimum_month_net_pnl_usd)}</div><div class="subtle">period-stability diagnostic</div></div>`;
}

function renderLeaders(leaderboard) {
  const empty = document.getElementById("leaders-empty");
  const table = document.getElementById("leaders-table");
  const entries = (leaderboard && leaderboard.entries) || [];
  if (!entries.length) { empty.style.display = "block"; table.style.display = "none"; return; }
  empty.style.display = "none"; table.style.display = "table";
  renderSortControls();
  document.getElementById("leaders-tbody").innerHTML = entries.map(e => `
    <tr class="selectable ${e.strategy_id === selectedStrategyId ? "selected" : ""}" data-strategy-id="${escapeHtml(e.strategy_id)}" title="Click to inspect this strategy">
      <td class="num">${e.rank}</td>
      <td class="strategy-cell">${escapeHtml(e.strategy_id)}</td>
      <td class="num ${e.pnl >= 0 ? "pnl-pos" : "pnl-neg"}">${fmtMoney(e.pnl)}</td>
      <td class="num">${fmtPct(e.roi)}</td>
      <td class="num">${fmtPct(e.win_rate)}</td>
      <td class="num">${fmtNum(e.trade_count)}</td>
      <td class="num">${fmtMoney(e.max_drawdown)}</td>
      <td class="num">${fmtNum(e.performance_validation?.profit_factor)}</td>
      <td class="num">${fmtNum(e.performance_validation?.average_r)}</td>
      <td class="num">${fmtNum(e.performance_validation?.annualized_daily_sharpe)}</td>
      <td class="num">${fmtNum(e.performance_validation?.profitable_month_count)} / ${fmtNum(e.performance_validation?.evaluated_month_count)}</td>
      <td class="num ${(e.performance_validation?.minimum_month_net_pnl_usd ?? 0) >= 0 ? "pnl-pos" : "pnl-neg"}">${fmtMoney(e.performance_validation?.minimum_month_net_pnl_usd)}</td>
      <td class="num">${fmtNum(e.drawdowns_over_2000_count)}</td>
    </tr>`).join("");
  document.querySelectorAll("#leaders-tbody tr").forEach(row => row.addEventListener("click", () => {
    selectedStrategyId = row.dataset.strategyId;
    renderStrategySummary(lastLeaderboard);
    renderLeaders(lastLeaderboard);
    renderCharts(lastLeaderboard, lastEventsData);
  }));
  const more = document.getElementById("leaders-load-more");
  more.style.display = entries.length < leaderboardTotal ? "block" : "none";
  more.textContent = leaderboardLoading ? "Loading more strategies…" : (entries.length < leaderboardTotal ? `Scroll to load more · ${fmtNum(entries.length)} of ${fmtNum(leaderboardTotal)}` : "All eligible strategies loaded.");
}

function renderWarnings(status) {
  const warnings = status.warnings || [];
  const errors = status.errors || [];
  document.getElementById("warnings-empty").style.display = (warnings.length === 0 && errors.length === 0) ? "block" : "none";
  document.getElementById("warnings-list").innerHTML = warnings.map(w => `<div>&#9888; ${escapeHtml(w)}</div>`).join("");
  document.getElementById("errors-list").innerHTML = errors.map(e => `<div>&#10007; ${escapeHtml(e)}</div>`).join("");
}

function renderEvents(eventsData) {
  const events = (eventsData && eventsData.events) || [];
  const empty = document.getElementById("events-empty");
  const list = document.getElementById("events-list");
  if (events.length === 0) { empty.style.display = "block"; list.innerHTML = ""; return; }
  empty.style.display = "none";
  list.innerHTML = events.slice().reverse().map(ev => `<div><span class="ev-type">${ev.event_type}</span> &middot; ${ev.phase || "-"} &middot; ${ev.ts}</div>`).join("");
}

function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

// ---------- canvas chart helpers (no external library) ----------
function clearCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(rect.width, 200), h = Math.max(rect.height, 200);
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return {ctx, w, h};
}
function emptyChart(canvas, message) {
  const {ctx, w, h} = clearCanvas(canvas);
  ctx.fillStyle = "#5b6377"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(message, w/2, h/2);
}
function drawLineChart(canvas, points, {color = "#4f8cff", fill = true} = {}) {
  if (!points || points.length === 0) { emptyChart(canvas, "No data yet"); return; }
  const {ctx, w, h} = clearCanvas(canvas);
  const pad = 24;
  const ys = points.map(p => p.y);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const range = (maxY - minY) || 1;
  const stepX = points.length > 1 ? (w - 2*pad) / (points.length - 1) : 0;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = pad + i * stepX;
    const y = h - pad - ((p.y - minY) / range) * (h - 2*pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  if (fill) {
    ctx.lineTo(pad + (points.length-1)*stepX, h-pad); ctx.lineTo(pad, h-pad); ctx.closePath();
    ctx.fillStyle = color + "22"; ctx.fill();
  }
  ctx.fillStyle = "#8b93a7"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(minY.toFixed(0), 2, h - 4);
  ctx.textAlign = "right"; ctx.fillText(maxY.toFixed(0), w - 2, 12);
}
function drawHistogram(canvas, values, {bins = 12, color = "#4f8cff"} = {}) {
  if (!values || values.length === 0) { emptyChart(canvas, "No data yet"); return; }
  const {ctx, w, h} = clearCanvas(canvas);
  const pad = 20;
  const minV = Math.min(...values), maxV = Math.max(...values);
  const range = (maxV - minV) || 1;
  const counts = new Array(bins).fill(0);
  values.forEach(v => { let idx = Math.floor(((v - minV) / range) * bins); if (idx >= bins) idx = bins - 1; if (idx < 0) idx = 0; counts[idx]++; });
  const maxCount = Math.max(...counts) || 1;
  const barW = (w - 2*pad) / bins;
  counts.forEach((c, i) => {
    const barH = (c / maxCount) * (h - 2*pad);
    ctx.fillStyle = color;
    ctx.fillRect(pad + i*barW + 1, h - pad - barH, barW - 2, barH);
  });
  ctx.fillStyle = "#8b93a7"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(minV.toFixed(0), pad, h - 4);
  ctx.textAlign = "right"; ctx.fillText(maxV.toFixed(0), w - pad, h - 4);
}
function drawScatter(canvas, points, {color = "#4f8cff"} = {}) {
  if (!points || points.length === 0) { emptyChart(canvas, "No data yet"); return; }
  const {ctx, w, h} = clearCanvas(canvas);
  const pad = 24;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), rangeX = (maxX - minX) || 1;
  const minY = Math.min(...ys), maxY = Math.max(...ys), rangeY = (maxY - minY) || 1;
  points.forEach(p => {
    const x = pad + ((p.x - minX) / rangeX) * (w - 2*pad);
    const y = h - pad - ((p.y - minY) / rangeY) * (h - 2*pad);
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 2*Math.PI); ctx.fillStyle = color + "cc"; ctx.fill();
  });
  ctx.fillStyle = "#8b93a7"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText("dd " + minX.toFixed(0) + " - " + maxX.toFixed(0), pad, h - 4);
}
function drawBarChartLabeled(canvas, labels, values, {color = "#4f8cff"} = {}) {
  if (!labels || labels.length === 0) { emptyChart(canvas, "No data yet"); return; }
  const {ctx, w, h} = clearCanvas(canvas);
  const pad = 24;
  const maxV = Math.max(...values) || 1;
  const barW = (w - 2*pad) / labels.length;
  values.forEach((v, i) => {
    const barH = (v / maxV) * (h - 2*pad - 14);
    ctx.fillStyle = color;
    ctx.fillRect(pad + i*barW + 1, h - pad - barH, Math.max(barW - 3, 1), barH);
  });
  ctx.fillStyle = "#8b93a7"; ctx.font = "9px monospace"; ctx.textAlign = "center";
  const step = Math.max(1, Math.ceil(labels.length / 8));
  labels.forEach((l, i) => { if (i % step === 0) ctx.fillText(l, pad + i*barW + barW/2, h - 4); });
}

// The original minimal canvas plots intentionally had no axes. These
// helpers keep the dashboard dependency-free while making every numerical
// chart legible, unit-labelled and inspectable on hover.
const CHART = { grid: "#2a3141", text: "#9aa4bb", faint: "#69738a", panel: "#161b26" };
function chartArea(ctx, w, h, {bottom = 34, left = 62, right = 14, top = 16} = {}) {
  ctx.strokeStyle = CHART.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const y = top + ((h - top - bottom) * i / 4); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w-right, y); ctx.stroke(); }
  ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, h-bottom); ctx.lineTo(w-right, h-bottom); ctx.stroke();
  return {left, right, top, bottom, pw:w-left-right, ph:h-top-bottom};
}
function axisText(ctx, text, x, y, align = "center") { ctx.fillStyle = CHART.text; ctx.font = "10px var(--mono)"; ctx.textAlign = align; ctx.fillText(text, x, y); }
function setTooltip(canvas, hitTest) {
  const tip = document.getElementById("chart-tooltip");
  canvas.onmousemove = ev => { const r = canvas.getBoundingClientRect(); const content = hitTest(ev.clientX-r.left, ev.clientY-r.top); if (!content) { tip.style.display="none"; return; } tip.innerHTML = content; tip.style.left = (ev.clientX + 14) + "px"; tip.style.top = (ev.clientY + 14) + "px"; tip.style.display = "block"; };
  canvas.onmouseleave = () => { tip.style.display = "none"; };
}
function drawAxisLine(canvas, points, {color="#34d399", valueLabel=fmtMoney, xLabel=(p,i)=>String(i), tooltip=(p)=>""}={}) {
  if (!points.length) { emptyChart(canvas, "No data yet"); return; }
  const {ctx,w,h}=clearCanvas(canvas), a=chartArea(ctx,w,h);
  const ys=points.map(p=>p.y), min=Math.min(...ys), max=Math.max(...ys), range=(max-min)||1;
  const pos = points.map((p,i)=>({p, x:a.left+(points.length===1?a.pw/2:i*a.pw/(points.length-1)), y:a.top+a.ph-(p.y-min)/range*a.ph}));
  for(let i=0;i<=4;i++) axisText(ctx,valueLabel(min+range*(4-i)/4), a.left-7, a.top+a.ph*i/4+3, "right");
  axisText(ctx,xLabel(points[0],0),a.left,h-10,"left"); axisText(ctx,xLabel(points[points.length-1],points.length-1),w-a.right,h-10,"right");
  ctx.beginPath(); pos.forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y)); ctx.strokeStyle=color; ctx.lineWidth=2.4; ctx.stroke();
  ctx.lineTo(pos[pos.length-1].x,a.top+a.ph); ctx.lineTo(pos[0].x,a.top+a.ph); ctx.closePath(); ctx.fillStyle=color+"22"; ctx.fill();
  setTooltip(canvas,(x,y)=>{ let best=null, d=Infinity; pos.forEach(q=>{const n=Math.abs(q.x-x); if(n<d){d=n;best=q;}}); return d<18 ? tooltip(best.p) : null; });
}
function drawAxisHistogram(canvas, values, {bins=8,color="#4f8cff",valueLabel=fmtMoney}={}) {
  if (!values.length) { emptyChart(canvas,"No data yet"); return; }
  const {ctx,w,h}=clearCanvas(canvas), a=chartArea(ctx,w,h); const min=Math.min(...values), max=Math.max(...values), range=(max-min)||1, counts=Array(bins).fill(0);
  values.forEach(v=>{let i=Math.floor((v-min)/range*bins); counts[Math.min(bins-1,Math.max(0,i))]++;}); const maxC=Math.max(...counts)||1, bars=[];
  for(let i=0;i<=4;i++) axisText(ctx,String(Math.round(maxC*(4-i)/4)),a.left-7,a.top+a.ph*i/4+3,"right");
  axisText(ctx,valueLabel(min),a.left,h-10,"left"); axisText(ctx,valueLabel(max),w-a.right,h-10,"right");
  counts.forEach((c,i)=>{const x=a.left+i*a.pw/bins+2,bw=a.pw/bins-4,bh=c/maxC*a.ph,y=a.top+a.ph-bh;ctx.fillStyle=color;ctx.fillRect(x,y,bw,bh);bars.push({x,y,bw,bh,c,from:min+range*i/bins,to:min+range*(i+1)/bins});});
  setTooltip(canvas,(x,y)=>{const b=bars.find(q=>x>=q.x&&x<=q.x+q.bw&&y>=q.y&&y<=q.y+q.bh);return b?`${valueLabel(b.from)} – ${valueLabel(b.to)}<br><b>${b.c}</b> strategies`:null;});
}
function drawAxisScatter(canvas, entries) {
  if (!entries.length) { emptyChart(canvas,"No data yet"); return; }
  const {ctx,w,h}=clearCanvas(canvas), a=chartArea(ctx,w,h), xs=entries.map(e=>e.max_drawdown),ys=entries.map(e=>e.pnl),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),rx=(maxX-minX)||1,ry=(maxY-minY)||1;
  for(let i=0;i<=4;i++){axisText(ctx,fmtMoney(minY+ry*(4-i)/4),a.left-7,a.top+a.ph*i/4+3,"right");axisText(ctx,fmtMoney(minX+rx*i/4),a.left+a.pw*i/4,h-10,"center");}
  const points=entries.map(e=>({e,x:a.left+(e.max_drawdown-minX)/rx*a.pw,y:a.top+a.ph-(e.pnl-minY)/ry*a.ph}));
  points.forEach(q=>{ctx.beginPath();ctx.arc(q.x,q.y,q.e.strategy_id===selectedStrategyId?6:4.5,0,Math.PI*2);ctx.fillStyle=q.e.strategy_id===selectedStrategyId?"#34d399":"#fbbf24";ctx.fill();});
  setTooltip(canvas,(x,y)=>{let best=null,d=18;points.forEach(q=>{const n=Math.hypot(q.x-x,q.y-y);if(n<d){d=n;best=q;}});return best?`#${best.e.rank} · ${escapeHtml(best.e.strategy_id)}<br>P&amp;L: <b>${fmtMoney(best.e.pnl)}</b><br>Max DD: ${fmtMoney(best.e.max_drawdown)}`:null;});
}
function drawAxisBars(canvas, labels, values, {color="#34d399", valueLabel=fmtNum}={}) {
  if(!labels.length){emptyChart(canvas,"No data yet");return;} const {ctx,w,h}=clearCanvas(canvas),a=chartArea(ctx,w,h),max=Math.max(...values)||1,bars=[];
  for(let i=0;i<=4;i++) axisText(ctx,valueLabel(max*(4-i)/4),a.left-7,a.top+a.ph*i/4+3,"right");
  const step=Math.max(1,Math.ceil(labels.length/8)); labels.forEach((l,i)=>{if(i%step===0||i===labels.length-1)axisText(ctx,l,a.left+a.pw*(i+.5)/labels.length,h-10,"center");});
  values.forEach((v,i)=>{const x=a.left+i*a.pw/labels.length+1,bw=Math.max(1,a.pw/labels.length-2),bh=v/max*a.ph,y=a.top+a.ph-bh;ctx.fillStyle=color;ctx.fillRect(x,y,bw,bh);bars.push({x,y,bw,bh,l:labels[i],v});});
  setTooltip(canvas,(x,y)=>{const b=bars.find(q=>x>=q.x&&x<=q.x+q.bw&&y>=q.y&&y<=q.y+q.bh);return b?`${b.l}<br><b>${valueLabel(b.v)}</b> completed trades`:null;});
}

function renderCharts(leaderboard, eventsData) {
  const entries = topEntries(leaderboard);
  const selected = selectedEntry(leaderboard);
  document.getElementById("equity-title").textContent = selected ? `Equity curve · #${selected.rank} selected strategy` : "Equity curve";
  document.getElementById("equity-subtitle").textContent = selected ? `${selected.strategy_id} · final realised P&L ${fmtMoney(selected.pnl)} · max drawdown ${fmtMoney(selected.max_drawdown)}${selected.detail_granularity === "monthly" ? " · monthly reconstruction" : ""}` : "Select a strategy to inspect it.";
  document.getElementById("month-title").textContent = selected ? `Monthly trade activity · #${selected.rank} selected strategy` : "Monthly trade activity";
  const equityPoints = ((selected && selected.equity_curve) || []).map((p, i) => ({x:i,y:p.equity,ts:p.ts}));
  drawAxisLine(document.getElementById("chart-equity"), equityPoints, {color:"#34d399", valueLabel:fmtMoney, xLabel:p=>p.ts ? new Date(p.ts).toLocaleDateString(undefined,{year:"numeric",month:"short"}) : "", tooltip:p=>`${new Date(p.ts).toLocaleString()}<br>Account equity: <b>${fmtMoney(p.y)}</b>`});

  drawAxisHistogram(document.getElementById("chart-pnl-dist"), entries.map(e => e.pnl), {color: "#4f8cff", valueLabel:fmtMoney});
  drawAxisScatter(document.getElementById("chart-pnl-dd"), entries);
  drawAxisHistogram(document.getElementById("chart-trade-count"), entries.map(e => e.trade_count), {color: "#a855f7", valueLabel:fmtNum});
  drawAxisHistogram(document.getElementById("chart-win-rate"), entries.map(e => e.win_rate * 100), {color: "#f87171", valueLabel:v=>v.toFixed(1)+"%"});
  drawAxisHistogram(document.getElementById("chart-trade-pnl"), ((selected && selected.trade_pnls) || []), {color:"#34d399", valueLabel:fmtMoney});

  const balances = ((selected && selected.equity_curve) || []).map(p => p.equity);
  let peak = -Infinity;
  const drawdownPoints = balances.map((balance, i) => { peak = Math.max(peak, balance); return {x:i, y:peak-balance, ts:selected.equity_curve[i].ts}; });
  drawAxisLine(document.getElementById("chart-drawdown"), drawdownPoints, {color:"#f87171",valueLabel:fmtMoney,xLabel:p=>p.ts ? new Date(p.ts).toLocaleDateString(undefined,{year:"numeric",month:"short"}) : "",tooltip:p=>`${new Date(p.ts).toLocaleString()}<br>Realised drawdown: <b>${fmtMoney(p.y)}</b>`});

  // throughput over time, from progress events
  const events = (eventsData && eventsData.events) || [];
  const progressEvents = events.filter(e => e.event_type === "progress" && e.detail && e.detail.items_completed !== null && e.detail.items_completed !== undefined);
  const throughputPoints = [];
  for (let i = 1; i < progressEvents.length; i++) {
    const prev = progressEvents[i-1], cur = progressEvents[i];
    const dt = (new Date(cur.ts) - new Date(prev.ts)) / 1000;
    const dc = cur.detail.items_completed - prev.detail.items_completed;
    if (dt > 0 && dc >= 0) throughputPoints.push({x: i, y: dc / dt});
  }
  drawAxisLine(document.getElementById("chart-throughput"), throughputPoints, {color:"#4f8cff",fill:false,valueLabel:v=>fmtNum(v)+"/s",xLabel:(_p,i)=>i===0?"earlier":"latest",tooltip:p=>`Throughput: <b>${fmtNum(p.y)}/s</b>`});

  const byMonth = (selected && selected.trades_by_month) || {};
  const months = Object.keys(byMonth).sort();
  drawAxisBars(document.getElementById("chart-by-month"), months, months.map(m => byMonth[m]), {color:"#34d399", valueLabel:fmtNum});
}

async function loadLeaderboardPage(reset = false) {
  if (!currentRunId || leaderboardLoading) return;
  leaderboardLoading = true;
  if (reset) { leaderboardOffset = 0; leaderboardTotal = 0; lastLeaderboard = null; }
  const offset = reset ? 0 : leaderboardOffset;
  const filters = new URLSearchParams();
  const fieldMap = {"pnl-min":"pnl_min","pnl-max":"pnl_max","roi-min":"roi_min","roi-max":"roi_max","win-rate-min":"win_rate_min","win-rate-max":"win_rate_max","trade-count-min":"min_trades","trade-count-max":"trade_count_max","max-drawdown-min":"max_drawdown_min","max-drawdown-max":"max_drawdown_max","profit-factor-min":"profit_factor_min","profit-factor-max":"profit_factor_max","average-r-min":"average_r_min","average-r-max":"average_r_max","daily-sharpe-min":"daily_sharpe_min","daily-sharpe-max":"daily_sharpe_max","expectancy-min":"expectancy_min","expectancy-max":"expectancy_max","top-winner-share-min":"top_winner_share_min","top-winner-share-max":"top_winner_share_max","profitable-months-min":"profitable_months_min","profitable-months-max":"profitable_months_max","worst-month-pnl-min":"worst_month_pnl_min","worst-month-pnl-max":"worst_month_pnl_max","contains":"contains"};
  Object.entries(fieldMap).forEach(([id,key]) => { const value = document.getElementById("f-"+id).value.trim(); if (value) filters.set(key,value); });
  ["direction","instrument","tp_kind","sl_kind","confluence","session"].forEach(key => { const values = [...document.querySelectorAll(`[data-filter="${key}"]:checked`)].map(el => el.value); if (values.length) filters.set(key, values.join(",")); });
  const apexPass = document.getElementById("f-apex-pass").checked, apexBreach = document.getElementById("f-apex-breach").checked;
  if (apexPass && !apexBreach) filters.set("apex_has_breach", "0");
  if (apexBreach && !apexPass) filters.set("apex_has_breach", "1");
  if (!filters.has("min_trades")) filters.set("min_trades","100");
  filters.set("scope","all"); filters.set("sort",leaderboardSort.key); filters.set("dir",leaderboardSort.dir); filters.set("offset",offset); filters.set("limit",100);
  const url = `/api/runs/${currentRunId}/leaderboard?${filters.toString()}`;
  const page = await fetchJson(url);
  leaderboardLoading = false;
  if (!page) return;
  leaderboardTotal = page.total_eligible || 0;
  const previous = reset || !lastLeaderboard ? [] : lastLeaderboard.entries;
  lastLeaderboard = {...page, entries: previous.concat(page.entries || [])};
  leaderboardOffset = lastLeaderboard.entries.length;
  if (selectedStrategyId && !lastLeaderboard.entries.some(e => e.strategy_id === selectedStrategyId)) selectedStrategyId = null;
  if (!selectedStrategyId && lastLeaderboard.entries.length) selectedStrategyId = lastLeaderboard.entries[0].strategy_id;
  renderStrategySummary(lastLeaderboard);
  renderLeaders(lastLeaderboard);
  renderCharts(lastLeaderboard, lastEventsData);
}

async function refresh() {
  await loadRuns();
  if (!currentRunId) return;

  const [status, eventsData, manifest] = await Promise.all([
    fetchJson(`/api/runs/${currentRunId}/status`),
    fetchJson(`/api/runs/${currentRunId}/events?limit=100`),
    fetchJson(`/api/runs/${currentRunId}/manifest`),
  ]);

  if (!status) { showEmptyProgress(); return; }

  document.getElementById("progress-empty").style.display = "none";
  document.getElementById("progress-content").style.display = "block";

  const phaseBadge = document.getElementById("phase-badge");
  phaseBadge.textContent = PHASE_LABELS[status.phase] || status.phase;
  phaseBadge.className = "badge " + (status.phase === "completed" ? "badge-completed" : status.phase === "failed" ? "badge-failed" : "badge-phase");

  const demoBadge = document.getElementById("demo-badge");
  const demoBanner = document.getElementById("demo-banner");
  const isDemo = !!(status.is_demo || (manifest && manifest.is_demo));
  demoBadge.style.display = isDemo ? "inline-block" : "none";
  demoBanner.style.display = isDemo ? "block" : "none";

  document.getElementById("stat-elapsed").textContent = fmtSeconds(status.elapsed_seconds);
  document.getElementById("stat-items").textContent = (status.items_completed === null ? "-" : fmtNum(status.items_completed)) + " / " + (status.items_total === null || status.items_total === undefined ? "?" : fmtNum(status.items_total));
  document.getElementById("stat-throughput").textContent = status.throughput_per_second === null || status.throughput_per_second === undefined ? "-" : fmtNum(status.throughput_per_second) + "/s";
  document.getElementById("stat-eta").textContent = status.eta_seconds === null || status.eta_seconds === undefined ? "n/a" : fmtSeconds(status.eta_seconds);
  const usage = status.resource_usage || {};
  document.getElementById("stat-ram").textContent = fmtBytes(usage.rss_bytes);
  document.getElementById("stat-cpu").textContent = usage.cpu_seconds === null || usage.cpu_seconds === undefined ? "-" : fmtSeconds(usage.cpu_seconds);

  const pct = (status.items_total && status.items_total > 0 && status.items_completed !== null) ? Math.min(100, (status.items_completed / status.items_total) * 100) : (status.phase === "completed" ? 100 : 0);
  document.getElementById("progress-bar-fill").style.width = pct + "%";

  renderTimeline(status.phase);
  renderActiveWork(status);
  renderCards(status);
  lastEventsData = eventsData;
  renderWarnings(status);
  renderEvents(eventsData);
  if (!lastLeaderboard) await loadLeaderboardPage(true);
  else renderCharts(lastLeaderboard, eventsData);

  const isTerminal = status.phase === "completed" || status.phase === "failed";
  if (isTerminal && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

document.getElementById("run-select").addEventListener("change", (e) => { currentRunId = e.target.value; selectedStrategyId = null; lastLeaderboard = null; leaderboardOffset = 0; leaderboardTotal = 0; refresh(); });
document.getElementById("refresh-btn").addEventListener("click", refresh);
document.getElementById("toggle-filters").addEventListener("click", () => { const panel = document.getElementById("filters-panel"); panel.classList.toggle("open"); document.getElementById("toggle-filters").textContent = panel.classList.contains("open") ? "Hide filters" : "Show filters"; });
document.getElementById("apply-filters").addEventListener("click", () => { selectedStrategyId = null; loadLeaderboardPage(true); });
document.getElementById("clear-filters").addEventListener("click", () => { document.querySelectorAll("#leader-filters input:not([type=checkbox])").forEach(input => input.value = ""); document.querySelectorAll("#leader-filters input[type=checkbox]").forEach(input => input.checked = false); document.getElementById("f-trade-count-min").value = "100"; selectedStrategyId = null; loadLeaderboardPage(true); });
document.querySelectorAll("[data-sort-key]").forEach(button => button.addEventListener("click", () => {
  const key = button.dataset.sortKey;
  leaderboardSort = leaderboardSort.key === key ? {key, dir: leaderboardSort.dir === "asc" ? "desc" : "asc"} : {key, dir: key === "rank" || key === "max_drawdown" || key === "drawdowns_over_2000_count" ? "asc" : "desc"};
  selectedStrategyId = null;
  loadLeaderboardPage(true);
}));
document.getElementById("leaders-scroll").addEventListener("scroll", (event) => {
  const box = event.currentTarget;
  if (box.scrollTop + box.clientHeight >= box.scrollHeight - 120 && leaderboardOffset < leaderboardTotal) loadLeaderboardPage(false);
});

(async () => {
  await refresh();
  pollTimer = setInterval(refresh, POLL_MS);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
