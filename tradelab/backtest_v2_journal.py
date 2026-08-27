"""SQLite-backed, resumable, crash-safe durable journal for
``tradelab.backtest_v2``'s production runner -- the authoritative
resume source for that runner (task 043).

**Authoritative, not "an" export.** ``tradelab.backtest_v2``'s own
``variant_summaries.jsonl`` (task 042) remains a plain append-only
export/audit artifact -- fine to tail, grep, or ship elsewhere -- but
it is never read back to decide what has already been completed. This
module's SQLite database is the sole source of truth resume logic
reads from: one row per completed ``strategy_variant_id``, written in
its own durable transaction *before* the run reports that variant as
completed (see ``commit_summary``), and a manifest row identifying
exactly what configuration produced those rows.

**Mirrors, but does not reuse, ``tradelab.research_journal``'s own
lifecycle.** That module is a real, tested, SQLite-backed
create/verify/resume pattern -- but for the *older*
``StrategySpec``/``EntrySignal`` evaluation stream
(``tradelab.research_runner.iter_research_evaluations``), which
``tradelab.backtest_v2`` deliberately does not use as an entry source
(see that module's own docstring). This module reimplements the same
proven pattern (one manifest row, verified field-by-field on every
open; refuse to resume on any mismatch, naming exactly which fields
differed) against ``backtest_v2``'s own, different schema -- never
imports from or delegates to ``tradelab.research_journal`` itself,
since the two journal manifests protect entirely different pipelines.

**Compatibility fields, exactly** (spec, this task): market artifact
provenance (path + sha256), news provenance (path + sha256), strategy
parameters (tick size, pivot strength, reclaim window, minimum entries
per year), risk configuration, equity configuration (not explicitly
named by the task, but included defensively -- it directly affects ROI
and the drawdown-episode count, so a silent mismatch there would be
just as unsafe as a risk-config mismatch), the exact set of exit
variant ids, the instrument set, and the code commit. On first use for
a given ``run_id`` these are written once; on every subsequent open
they are re-computed from the caller's current arguments and compared
field-by-field against what is already stored -- any mismatch raises
``ValueError`` immediately, naming exactly which field(s) differed,
and refuses to touch the existing data at all.

**Uniqueness.** ``variant_summaries.strategy_variant_id`` is the
table's own primary key -- a resumed run's own skip-already-completed
logic means a duplicate ``INSERT`` should never be attempted in normal
operation, but the constraint itself is never relaxed (no ``INSERT OR
IGNORE``): a duplicate-key attempt raises a loud ``sqlite3.
IntegrityError`` rather than being silently swallowed, since it would
indicate a genuine bug in the skip logic, not a benign race.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

JOURNAL_SCHEMA_VERSION = "1.3"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS run_manifest (
    run_id TEXT PRIMARY KEY,
    journal_schema_version TEXT NOT NULL,
    market_artifact_path TEXT,
    market_artifact_sha256 TEXT,
    news_csv_path TEXT,
    news_csv_sha256 TEXT,
    git_commit TEXT,
    parameters_json TEXT NOT NULL,
    risk_config_json TEXT NOT NULL,
    equity_config_json TEXT NOT NULL,
    exit_variant_ids_json TEXT NOT NULL,
    instruments_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS variant_summaries (
    strategy_variant_id TEXT PRIMARY KEY,
    equivalence_class_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    instrument TEXT NOT NULL,
    exit_variant_id TEXT NOT NULL,
    tp_kind TEXT NOT NULL,
    tp_rank INTEGER,
    sl_kind TEXT NOT NULL,
    sl_rank INTEGER,
    sl_fvg_timeframe TEXT,
    net_pnl_usd REAL NOT NULL,
    roi REAL NOT NULL,
    win_rate REAL,
    trade_count INTEGER NOT NULL,
    max_drawdown_usd REAL NOT NULL,
    drawdowns_over_2000_count INTEGER NOT NULL,
    drawdown_over_2000_episode_count INTEGER NOT NULL DEFAULT 0,
    research_eligible INTEGER NOT NULL DEFAULT 0,
    completed_trades_2024 INTEGER NOT NULL DEFAULT 0,
    completed_trades_2025 INTEGER NOT NULL DEFAULT 0,
    ambiguous_same_bar_exit_count INTEGER NOT NULL DEFAULT 0,
    apex_has_breach INTEGER NOT NULL DEFAULT 0,
    apex_breach_count INTEGER NOT NULL DEFAULT 0,
    apex_breached_trading_day_count INTEGER NOT NULL DEFAULT 0,
    apex_max_daily_loss_usd REAL NOT NULL DEFAULT 0,
    apex_largest_eod_drawdown_breach_usd REAL NOT NULL DEFAULT 0,
    mtm_worst_equity_usd REAL,
    mtm_max_drawdown_usd REAL,
    mtm_worst_equity_at TEXT,
    skipped_entries_by_reason_json TEXT NOT NULL,
    yearly_json TEXT NOT NULL,
    monthly_json TEXT NOT NULL,
    performance_validation_json TEXT NOT NULL DEFAULT '{}',
    stop_slippage_sensitivity_json TEXT NOT NULL DEFAULT '{}',
    committed_at TEXT NOT NULL
);

-- Task 048, backlog item C: trade-level rows for the run's leaderboard
-- variants, so the dashboard can slice an already-finished result by session
-- and hour without re-running anything.
--
-- Deliberately NOT one row per trade for every variant. A full run enumerates
-- on the order of a million variants at hundreds of trades each; persisting
-- all of them is terabytes, which is precisely why variant_summaries is lean.
-- This table is bounded to the variants the dashboard can actually select --
-- the final top-N leaderboard -- and is written ONCE, after the run's own
-- simulation work is finished, never from the hot commit path. That is what
-- keeps it from competing with a live run for the write lock.
CREATE TABLE IF NOT EXISTS variant_trades (
    strategy_variant_id TEXT NOT NULL,
    trade_seq INTEGER NOT NULL,
    direction TEXT NOT NULL,
    entry_at TEXT NOT NULL,          -- ISO-8601 UTC, always
    exit_at TEXT,                    -- NULL only for an unresolved open trade
    entry_price REAL,
    exit_price REAL,
    exit_reason TEXT NOT NULL,
    net_pnl_usd REAL,                -- NULL for an unresolved open trade
    gross_pnl_usd REAL,              -- NULL for an unresolved open trade
    entry_cost_usd REAL NOT NULL,    -- charged even for an unresolved trade
    contracts INTEGER,
    entry_session TEXT NOT NULL,     -- tradelab.trading_sessions label at write time
    entry_hour_ny INTEGER NOT NULL,  -- New York local hour, 0-23
    PRIMARY KEY (strategy_variant_id, trade_seq)
);

-- The two access paths the session view actually uses: "all trades of one
-- variant, in order" and "trades of one variant in these sessions". The
-- primary key already covers the first; this covers the second without a
-- scan, and keeps the ORDER BY free.
CREATE INDEX IF NOT EXISTS idx_variant_trades_session
    ON variant_trades (strategy_variant_id, entry_session, entry_at);
CREATE INDEX IF NOT EXISTS idx_variant_trades_hour
    ON variant_trades (strategy_variant_id, entry_hour_ny, entry_at);

-- The universe browser sorts and filters variant_summaries over the whole
-- population on every page. These cover its two default orderings and its
-- eligibility predicate, which were full scans before.
CREATE INDEX IF NOT EXISTS idx_variant_summaries_pnl
    ON variant_summaries (research_eligible, net_pnl_usd DESC, max_drawdown_usd ASC);
CREATE INDEX IF NOT EXISTS idx_variant_summaries_trades
    ON variant_summaries (research_eligible, trade_count DESC);
"""

_MANIFEST_COLUMNS = (
    "journal_schema_version", "market_artifact_path", "market_artifact_sha256", "news_csv_path", "news_csv_sha256",
    "git_commit", "parameters_json", "risk_config_json", "equity_config_json", "exit_variant_ids_json", "instruments_json",
)

_SUMMARY_COLUMNS = (
    "strategy_variant_id", "equivalence_class_id", "direction", "instrument", "exit_variant_id", "tp_kind", "tp_rank",
    "sl_kind", "sl_rank", "sl_fvg_timeframe", "net_pnl_usd", "roi", "win_rate", "trade_count", "max_drawdown_usd",
    "drawdowns_over_2000_count", "drawdown_over_2000_episode_count",
    "research_eligible", "completed_trades_2024", "completed_trades_2025",
    "ambiguous_same_bar_exit_count",
    "apex_has_breach", "apex_breach_count", "apex_breached_trading_day_count",
    "apex_max_daily_loss_usd", "apex_largest_eod_drawdown_breach_usd",
    "mtm_worst_equity_usd", "mtm_max_drawdown_usd", "mtm_worst_equity_at",
    "skipped_entries_by_reason_json", "yearly_json", "monthly_json",
    "performance_validation_json", "stop_slippage_sensitivity_json",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JournalVariantSummary:
    """One durable, completed variant record read back from
    ``variant_summaries`` -- field-for-field the same shape as
    ``tradelab.backtest_v2.VariantSummaryRow`` (this module has no
    dependency on that module -- it is a plain, JSON-friendly mirror,
    kept independent so the journal never needs to import the
    simulation engine, only the reverse)."""

    strategy_variant_id: str
    equivalence_class_id: str
    direction: str
    instrument: str
    exit_variant_id: str
    tp_kind: str
    tp_rank: int | None
    sl_kind: str
    sl_rank: int | None
    sl_fvg_timeframe: str | None
    net_pnl_usd: float
    roi: float
    win_rate: float | None
    trade_count: int
    max_drawdown_usd: float
    drawdowns_over_2000_count: int
    drawdown_over_2000_episode_count: int
    research_eligible: bool
    completed_trades_2024: int
    completed_trades_2025: int
    ambiguous_same_bar_exit_count: int
    apex_has_breach: bool
    apex_breach_count: int
    apex_breached_trading_day_count: int
    apex_max_daily_loss_usd: float
    apex_largest_eod_drawdown_breach_usd: float
    mtm_worst_equity_usd: float | None
    mtm_max_drawdown_usd: float | None
    mtm_worst_equity_at: str | None
    skipped_entries_by_reason: Mapping[str, int]
    yearly: Mapping[str, Mapping[str, float | int]]
    monthly: Mapping[str, Mapping[str, float | int]]
    performance_validation: Mapping[str, Any]
    stop_slippage_sensitivity_usd: Mapping[str, float]

    @staticmethod
    def from_row(row: tuple) -> "JournalVariantSummary":
        (
            strategy_variant_id, equivalence_class_id, direction, instrument, exit_variant_id, tp_kind, tp_rank, sl_kind,
            sl_rank, sl_fvg_timeframe, net_pnl_usd, roi, win_rate, trade_count, max_drawdown_usd, drawdowns_over_2000_count,
            drawdown_over_2000_episode_count,
            research_eligible, completed_trades_2024, completed_trades_2025, ambiguous_same_bar_exit_count,
            apex_has_breach, apex_breach_count, apex_breached_trading_day_count, apex_max_daily_loss_usd,
            apex_largest_eod_drawdown_breach_usd, mtm_worst_equity_usd, mtm_max_drawdown_usd,
            mtm_worst_equity_at, skipped_json, yearly_json, monthly_json,
            performance_validation_json, stop_slippage_sensitivity_json,
        ) = row
        return JournalVariantSummary(
            strategy_variant_id=strategy_variant_id, equivalence_class_id=equivalence_class_id, direction=direction,
            instrument=instrument, exit_variant_id=exit_variant_id, tp_kind=tp_kind, tp_rank=tp_rank, sl_kind=sl_kind,
            sl_rank=sl_rank, sl_fvg_timeframe=sl_fvg_timeframe, net_pnl_usd=net_pnl_usd, roi=roi, win_rate=win_rate,
            trade_count=trade_count, max_drawdown_usd=max_drawdown_usd, drawdowns_over_2000_count=drawdowns_over_2000_count,
            drawdown_over_2000_episode_count=drawdown_over_2000_episode_count,
            research_eligible=bool(research_eligible), completed_trades_2024=completed_trades_2024,
            completed_trades_2025=completed_trades_2025,
            ambiguous_same_bar_exit_count=ambiguous_same_bar_exit_count,
            apex_has_breach=bool(apex_has_breach), apex_breach_count=apex_breach_count,
            apex_breached_trading_day_count=apex_breached_trading_day_count,
            apex_max_daily_loss_usd=apex_max_daily_loss_usd,
            apex_largest_eod_drawdown_breach_usd=apex_largest_eod_drawdown_breach_usd,
            mtm_worst_equity_usd=mtm_worst_equity_usd, mtm_max_drawdown_usd=mtm_max_drawdown_usd,
            mtm_worst_equity_at=mtm_worst_equity_at,
            skipped_entries_by_reason=json.loads(skipped_json), yearly=json.loads(yearly_json), monthly=json.loads(monthly_json),
            performance_validation=json.loads(performance_validation_json),
            stop_slippage_sensitivity_usd=json.loads(stop_slippage_sensitivity_json),
        )


class BacktestV2Journal:
    """One open SQLite connection to ``<run_dir>/backtest_v2_journal.
    sqlite3``, scoped to exactly one ``run_id``. Construction either
    creates a fresh manifest row (first use) or verifies the current
    call's own configuration against the one already stored -- see
    module docstring."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        run_id: str,
        market_artifact_path: str | None,
        market_artifact_sha256: str | None,
        news_csv_path: str | None,
        news_csv_sha256: str | None,
        git_commit: str | None,
        parameters: Mapping[str, Any],
        risk_config: Mapping[str, Any],
        equity_config: Mapping[str, Any],
        exit_variant_ids: list[str],
        instruments: list[str],
    ) -> None:
        if not run_id:
            raise ValueError("BacktestV2Journal: run_id must be non-empty")
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # WAL lets the dashboard read a consistent snapshot without blocking
        # the backtest's per-variant checkpoint transaction.  The rollback
        # journal default (``DELETE``) cannot provide that guarantee.
        # Keep FULL sync: this journal is the authoritative resume source.
        self._conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA_SQL)
        _migrate_summary_schema(self._conn)
        self._conn.commit()
        self._run_id = run_id

        self._verify_or_write_manifest(
            market_artifact_path=market_artifact_path, market_artifact_sha256=market_artifact_sha256,
            news_csv_path=news_csv_path, news_csv_sha256=news_csv_sha256, git_commit=git_commit,
            parameters=dict(parameters), risk_config=dict(risk_config), equity_config=dict(equity_config),
            exit_variant_ids=sorted(exit_variant_ids), instruments=sorted(instruments),
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _verify_or_write_manifest(
        self,
        *,
        market_artifact_path: str | None,
        market_artifact_sha256: str | None,
        news_csv_path: str | None,
        news_csv_sha256: str | None,
        git_commit: str | None,
        parameters: dict,
        risk_config: dict,
        equity_config: dict,
        exit_variant_ids: list[str],
        instruments: list[str],
    ) -> None:
        fields = {
            "journal_schema_version": JOURNAL_SCHEMA_VERSION,
            "market_artifact_path": market_artifact_path,
            "market_artifact_sha256": market_artifact_sha256,
            "news_csv_path": news_csv_path,
            "news_csv_sha256": news_csv_sha256,
            "git_commit": git_commit,
            "parameters_json": json.dumps(parameters, sort_keys=True),
            "risk_config_json": json.dumps(risk_config, sort_keys=True),
            "equity_config_json": json.dumps(equity_config, sort_keys=True),
            "exit_variant_ids_json": json.dumps(exit_variant_ids, sort_keys=True),
            "instruments_json": json.dumps(instruments, sort_keys=True),
        }

        row = self._conn.execute(
            f"SELECT {', '.join(_MANIFEST_COLUMNS)} FROM run_manifest WHERE run_id = ?", (self._run_id,)
        ).fetchone()

        if row is None:
            with self._conn:
                self._conn.execute(
                    f"INSERT INTO run_manifest (run_id, {', '.join(_MANIFEST_COLUMNS)}, created_at) "
                    f"VALUES (?, {', '.join('?' for _ in _MANIFEST_COLUMNS)}, ?)",
                    (self._run_id, *[fields[c] for c in _MANIFEST_COLUMNS], _utc_now_iso()),
                )
            return

        stored_fields = dict(zip(_MANIFEST_COLUMNS, row))
        # market_artifact_path/news_csv_path are audit-only -- a caller
        # pointing at a different path with identical content (matching
        # sha256) is not an unsafe mismatch; only the hashes (and every
        # other field) are enforced.
        enforced_keys = [c for c in _MANIFEST_COLUMNS if c not in ("market_artifact_path", "news_csv_path")]
        mismatched = sorted(key for key in enforced_keys if fields[key] != stored_fields[key])
        if mismatched:
            raise ValueError(
                f"cannot resume tradelab.backtest_v2 run {self._run_id!r} at {self._db_path}: manifest mismatch in "
                f"field(s) {mismatched} against the existing run -- refusing to resume with a different configuration "
                f"(market artifact / news provenance, strategy parameters, risk config, equity config, exit variants, "
                f"instruments, and code commit must all match exactly)"
            )

    def is_completed(self, strategy_variant_id: str) -> bool:
        """The resumed run's own skip-check -- one bounded, indexed
        lookup by primary key per candidate variant. Deliberately never
        backed by loading every completed id into a Python set first:
        at real-data scale (a full exhaustive run can be millions of
        variants) that set would itself be a significant, unbounded-
        with-run-size memory allocation. Correctness and bounded memory
        matter more here than the per-lookup query overhead."""
        row = self._conn.execute(
            "SELECT 1 FROM variant_summaries WHERE strategy_variant_id = ? LIMIT 1", (strategy_variant_id,)
        ).fetchone()
        return row is not None

    def completed_count(self) -> int:
        """A single ``COUNT(*)`` -- used to initialize a resumed run's
        progress counter without reading any row's own data into Python."""
        return self._conn.execute("SELECT COUNT(*) FROM variant_summaries").fetchone()[0]

    def completed_ids_for(self, candidate_ids: Sequence[str]) -> set[str]:
        """Which of ``candidate_ids`` are already durably completed.

        Asked per shard rather than for the whole run, so a coordinator never
        materialises every completed id of a million-variant run at once. The
        query is chunked because SQLite caps host parameters per statement.
        """
        found: set[str] = set()
        chunk = 500
        for start in range(0, len(candidate_ids), chunk):
            window = list(candidate_ids[start : start + chunk])
            if not window:
                continue
            placeholders = ",".join("?" * len(window))
            rows = self._conn.execute(
                f"SELECT strategy_variant_id FROM variant_summaries WHERE strategy_variant_id IN ({placeholders})",
                window,
            ).fetchall()
            found.update(row[0] for row in rows)
        return found

    def commit_summary(self, row: Mapping[str, Any]) -> None:
        """Durably commit one completed variant's summary -- a single,
        atomic SQLite transaction. Must be called *before* the run
        reports that variant as completed to ``tradelab.research_run``
        (crash safety, spec this task): if the process dies between
        this call and the next reporter call, the completed work is
        still safely durable and will be correctly skipped on resume;
        if it dies *before* this call, the variant is correctly
        re-attempted on resume, never silently lost."""
        with self._conn:
            self._conn.execute(
                f"INSERT INTO variant_summaries ({', '.join(_SUMMARY_COLUMNS)}, committed_at) "
                f"VALUES ({', '.join('?' for _ in _SUMMARY_COLUMNS)}, ?)",
                (
                    row["strategy_variant_id"], row["equivalence_class_id"], row["direction"], row["instrument"],
                    row["exit_variant_id"], row["tp_kind"], row["tp_rank"], row["sl_kind"], row["sl_rank"],
                    row["sl_fvg_timeframe"], row["net_pnl_usd"], row["roi"], row["win_rate"], row["trade_count"],
                    row["max_drawdown_usd"], row["drawdowns_over_2000_count"],
                    int(row.get("drawdown_over_2000_episode_count", 0)),
                    int(bool(row.get("research_eligible", False))),
                    int(row.get("completed_trades_2024", 0)), int(row.get("completed_trades_2025", 0)),
                    int(row.get("ambiguous_same_bar_exit_count", 0)),
                    int(bool((row.get("apex_eod_audit") or {}).get("has_breach", False))),
                    int((row.get("apex_eod_audit") or {}).get("breach_count", 0)),
                    int((row.get("apex_eod_audit") or {}).get("breached_trading_day_count", 0)),
                    float((row.get("apex_eod_audit") or {}).get("max_daily_loss_usd", 0.0)),
                    float((row.get("apex_eod_audit") or {}).get("largest_eod_drawdown_breach_usd", 0.0)),
                    (row.get("mark_to_market_audit") or {}).get("worst_equity_usd"),
                    (row.get("mark_to_market_audit") or {}).get("max_mark_to_market_drawdown_usd"),
                    (row.get("mark_to_market_audit") or {}).get("timestamp_of_worst_equity"),
                    json.dumps(dict(row["skipped_entries_by_reason"]), sort_keys=True),
                    json.dumps(dict(row["yearly"]), sort_keys=True), json.dumps(dict(row["monthly"]), sort_keys=True),
                    json.dumps(dict(row.get("performance_validation") or {}), sort_keys=True),
                    json.dumps(dict(row.get("stop_slippage_sensitivity_usd") or {}), sort_keys=True),
                    _utc_now_iso(),
                ),
            )

    def replace_variant_trades(self, trades_by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> int:
        """Write the trade-level rows for the run's leaderboard variants.

        **Called once, after the run's simulation work is finished** -- never
        from ``commit_summary``'s hot path. One transaction; every variant it
        names is fully replaced, so re-running finalisation (a resumed run
        reaching the end again, say) is idempotent rather than duplicating.
        Variants not named are left alone.

        Returns the number of trade rows written, so the caller can report a
        real figure rather than asserting it worked.

        Each mapping needs ``direction``, ``entry_at`` (tz-aware datetime or
        ISO-8601 UTC string), ``exit_reason``, and optionally ``exit_at``,
        ``entry_price``, ``exit_price``, ``net_pnl_usd`` and ``contracts``.
        The session label and NY hour are derived here, from ``entry_at``, so
        exactly one implementation owns that conversion.
        """
        from tradelab.trading_sessions import ny_hour, session_for

        def _as_utc_iso(value: Any) -> str:
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    raise ValueError("variant trade timestamps must be timezone-aware")
                return value.astimezone(timezone.utc).isoformat()
            parsed = datetime.fromisoformat(str(value))
            if parsed.tzinfo is None:
                raise ValueError("variant trade timestamps must be timezone-aware")
            return parsed.astimezone(timezone.utc).isoformat()

        def _as_utc(value: Any) -> datetime:
            return datetime.fromisoformat(_as_utc_iso(value))

        written = 0
        with self._conn:
            for strategy_variant_id, trades in trades_by_variant.items():
                self._conn.execute("DELETE FROM variant_trades WHERE strategy_variant_id = ?", (strategy_variant_id,))
                payload = []
                for seq, trade in enumerate(trades, start=1):
                    entry_at = _as_utc(trade["entry_at"])
                    payload.append((
                        strategy_variant_id, seq, trade["direction"], entry_at.isoformat(),
                        _as_utc_iso(trade["exit_at"]) if trade.get("exit_at") is not None else None,
                        trade.get("entry_price"), trade.get("exit_price"), trade["exit_reason"],
                        trade.get("net_pnl_usd"), trade.get("gross_pnl_usd"),
                        float(trade.get("entry_cost_usd") or 0.0), trade.get("contracts"),
                        session_for(entry_at), ny_hour(entry_at),
                    ))
                if payload:
                    self._conn.executemany(
                        "INSERT INTO variant_trades (strategy_variant_id, trade_seq, direction, entry_at, exit_at, "
                        "entry_price, exit_price, exit_reason, net_pnl_usd, gross_pnl_usd, entry_cost_usd, contracts, "
                        "entry_session, entry_hour_ny) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        payload,
                    )
                    written += len(payload)
        return written

    def variant_trades(self, strategy_variant_id: str) -> list[dict[str, Any]]:
        """Every persisted trade for one variant, in execution order."""
        rows = self._conn.execute(
            "SELECT trade_seq, direction, entry_at, exit_at, entry_price, exit_price, exit_reason, net_pnl_usd, "
            "gross_pnl_usd, entry_cost_usd, contracts, entry_session, entry_hour_ny "
            "FROM variant_trades WHERE strategy_variant_id = ? ORDER BY trade_seq",
            (strategy_variant_id,),
        ).fetchall()
        keys = ("trade_seq", "direction", "entry_at", "exit_at", "entry_price", "exit_price", "exit_reason",
                "net_pnl_usd", "gross_pnl_usd", "entry_cost_usd", "contracts", "entry_session", "entry_hour_ny")
        return [dict(zip(keys, row)) for row in rows]

    def variants_with_trades(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT strategy_variant_id FROM variant_trades ORDER BY strategy_variant_id"
        ).fetchall()
        return [row[0] for row in rows]

    def top_n_summaries(self, n: int) -> list[JournalVariantSummary]:
        """The current top ``n`` completed summaries, ranked exactly
        like ``tradelab.backtest_v2``'s own leaderboard sort key (net
        P&L descending, ties broken by lower max drawdown, then by
        ``strategy_variant_id`` descending) -- a single bounded SQL
        query, never a full table scan into Python."""
        if n < 1:
            return []
        rows = self._conn.execute(
            f"SELECT {', '.join(_SUMMARY_COLUMNS)} FROM variant_summaries "
            "WHERE research_eligible = 1 "
            "ORDER BY net_pnl_usd DESC, max_drawdown_usd ASC, strategy_variant_id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [JournalVariantSummary.from_row(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BacktestV2Journal":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


_SUMMARY_MIGRATION_COLUMNS = {
    "drawdown_over_2000_episode_count": "INTEGER NOT NULL DEFAULT 0",
    "apex_has_breach": "INTEGER NOT NULL DEFAULT 0",
    "apex_breach_count": "INTEGER NOT NULL DEFAULT 0",
    "apex_breached_trading_day_count": "INTEGER NOT NULL DEFAULT 0",
    "apex_max_daily_loss_usd": "REAL NOT NULL DEFAULT 0",
    "apex_largest_eod_drawdown_breach_usd": "REAL NOT NULL DEFAULT 0",
    "research_eligible": "INTEGER NOT NULL DEFAULT 0",
    "completed_trades_2024": "INTEGER NOT NULL DEFAULT 0",
    "completed_trades_2025": "INTEGER NOT NULL DEFAULT 0",
    "ambiguous_same_bar_exit_count": "INTEGER NOT NULL DEFAULT 0",
    "mtm_worst_equity_usd": "REAL",
    "mtm_max_drawdown_usd": "REAL",
    "mtm_worst_equity_at": "TEXT",
    "performance_validation_json": "TEXT NOT NULL DEFAULT '{}'",
    "stop_slippage_sensitivity_json": "TEXT NOT NULL DEFAULT '{}'",
}


def _migrate_summary_schema(conn: sqlite3.Connection) -> None:
    """Add read-only audit columns to pre-existing local journals safely."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(variant_summaries)")}
    added_eligibility = "research_eligible" not in existing
    added_episode_count = "drawdown_over_2000_episode_count" not in existing
    for name, definition in _SUMMARY_MIGRATION_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE variant_summaries ADD COLUMN {name} {definition}")
    if added_eligibility:
        # Historic journals stored yearly trade counts only in JSON.  Migrate
        # them deterministically using the long-standing 100/year research
        # threshold so the dashboard does not silently hide valid old rows.
        for rowid, yearly_json in conn.execute("SELECT rowid, yearly_json FROM variant_summaries"):
            yearly = json.loads(yearly_json)
            count_2024 = int(yearly.get("2024", {}).get("trade_count", 0))
            count_2025 = int(yearly.get("2025", {}).get("trade_count", 0))
            conn.execute(
                "UPDATE variant_summaries SET research_eligible = ?, completed_trades_2024 = ?, completed_trades_2025 = ? WHERE rowid = ?",
                (int(count_2024 >= 100 and count_2025 >= 100), count_2024, count_2025, rowid),
            )
    if added_episode_count:
        # Preserve the old rising-edge metric under an honest name, then
        # correct the user-facing field to complete $2k loss limits consumed.
        conn.execute(
            "UPDATE variant_summaries "
            "SET drawdown_over_2000_episode_count = drawdowns_over_2000_count"
        )
        conn.execute(
            "UPDATE variant_summaries SET drawdowns_over_2000_count = "
            "CAST(MAX(max_drawdown_usd, COALESCE(mtm_max_drawdown_usd, 0)) / 2000.0 AS INTEGER)"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_variant_summaries_eligible_pnl "
        "ON variant_summaries(research_eligible, net_pnl_usd DESC, max_drawdown_usd ASC)"
    )
