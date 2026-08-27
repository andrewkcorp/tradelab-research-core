"""Post-hoc session analysis (task 048, backlog item C).

Three things are proved here, in increasing strength:

1. the filters select the trades they claim to,
2. the metrics equal a fixture computed by hand, on paper, in the test's own
   comments -- not by calling the same code twice,
3. selecting every trade reproduces ``tradelab.equity``'s own equity curve
   exactly, so the post-hoc view and the engine cannot silently disagree.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.accounting import AccountBacktestResult, AccountedTrade, Instrument, RiskConfig
from tradelab.backtest import TradeExitReason
from tradelab.equity import EquityConfig, build_equity_curve
from tradelab.session_analysis import SessionFilter, analyse, filter_trades
from tradelab.trading_sessions import session_for

UTC = timezone.utc


def trade(seq, entry_at, exit_at, gross, *, cost=20.0, direction="long"):
    net = None if gross is None else gross - cost
    return {
        "trade_id": f"t{seq}",
        "trade_seq": seq,
        "direction": direction,
        "entry_at": entry_at.isoformat(),
        "exit_at": exit_at.isoformat() if exit_at is not None else None,
        "exit_reason": "tp_hit" if (gross or 0) > 0 else ("sl_hit" if gross is not None else "open_unresolved"),
        "gross_pnl_usd": gross,
        "entry_cost_usd": cost,
        "net_pnl_usd": net,
        "contracts": 1,
        "entry_session": session_for(entry_at),
        "entry_hour_ny": entry_at.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).hour,
    }


# The hand-computed fixture. All times UTC; the New York local time and the
# session each one lands in are spelled out, because that mapping is the thing
# under test and must not be produced by calling the code being tested.
#
#   B  entry 2024-05-06 08:00Z = 04:00 NY  london        gross -300  net -320
#   A  entry 2024-05-06 14:00Z = 10:00 NY  new_york_am   gross +200  net +180
#   C  entry 2024-05-06 16:00Z = 12:00 NY  new_york_pm   gross +100  net  +80
#   D  entry 2024-05-07 00:00Z = 20:00 NY  asia          gross -1900 net -1920
#
# Every exit is 30 minutes after its own entry, so the event order is
# B-entry, B-exit, A-entry, A-exit, C-entry, C-exit, D-entry, D-exit.
D0 = datetime(2024, 5, 6, tzinfo=UTC)


def fixture_trades():
    return [
        trade(1, D0 + timedelta(hours=8), D0 + timedelta(hours=8, minutes=30), -300.0, direction="short"),
        trade(2, D0 + timedelta(hours=14), D0 + timedelta(hours=14, minutes=30), 200.0),
        trade(3, D0 + timedelta(hours=16), D0 + timedelta(hours=16, minutes=30), 100.0),
        trade(4, D0 + timedelta(hours=24), D0 + timedelta(hours=24, minutes=30), -1900.0, direction="short"),
    ]


class SessionAssignmentTests(unittest.TestCase):
    def test_each_fixture_trade_lands_in_the_session_written_above(self):
        self.assertEqual(
            [t["entry_session"] for t in fixture_trades()],
            ["london", "new_york_am", "new_york_pm", "asia"],
        )

    def test_ny_hours_are_the_ones_written_above(self):
        self.assertEqual([t["entry_hour_ny"] for t in fixture_trades()], [4, 10, 12, 20])


class FilterTests(unittest.TestCase):
    def test_empty_filter_selects_everything(self):
        trades = fixture_trades()
        self.assertEqual(len(filter_trades(trades, SessionFilter())), 4)
        self.assertTrue(SessionFilter().is_identity)

    def test_single_session(self):
        selected = filter_trades(fixture_trades(), SessionFilter(sessions=frozenset({"new_york_am"})))
        self.assertEqual([t["trade_seq"] for t in selected], [2])

    def test_session_combination(self):
        selected = filter_trades(fixture_trades(), SessionFilter(sessions=frozenset({"london", "asia"})))
        self.assertEqual([t["trade_seq"] for t in selected], [1, 4])

    def test_hour_range_inclusive_on_both_ends(self):
        selected = filter_trades(fixture_trades(), SessionFilter(hour_from=10, hour_to=12))
        self.assertEqual([t["trade_seq"] for t in selected], [2, 3])
        self.assertEqual([t["trade_seq"] for t in filter_trades(fixture_trades(), SessionFilter(hour_from=10, hour_to=10))], [2])

    def test_hour_range_wraps_past_midnight(self):
        """20:00 -> 04:00 NY must include both ends of the wrap. Without this
        the Asian session is simply not expressible."""
        selected = filter_trades(fixture_trades(), SessionFilter(hour_from=20, hour_to=4))
        self.assertEqual([t["trade_seq"] for t in selected], [1, 4])

    def test_session_and_hour_filters_intersect(self):
        selected = filter_trades(
            fixture_trades(), SessionFilter(sessions=frozenset({"london", "asia"}), hour_from=0, hour_to=6)
        )
        self.assertEqual([t["trade_seq"] for t in selected], [1])

    def test_invalid_filters_rejected(self):
        with self.assertRaises(ValueError):
            SessionFilter(sessions=frozenset({"tokyo"}))
        with self.assertRaises(ValueError):
            SessionFilter(hour_from=24, hour_to=1)
        with self.assertRaises(ValueError):
            SessionFilter(hour_from=1)


class HandComputedMetricTests(unittest.TestCase):
    """Every expected number below was computed on paper first.

    Unfiltered, starting from $50,000, booking each $20 entry cost at its entry
    and each gross P&L at its exit, in chronological order:

        50,000 - 20   = 49,980   peak 50,000  drawdown    20
                - 300 = 49,680                drawdown   320
                -  20 = 49,660                drawdown   340   <- worst so far
                + 200 = 49,860                drawdown   140
                -  20 = 49,840                drawdown   160
                + 100 = 49,940                drawdown    60
                -  20 = 49,920                drawdown    80
                -1,900= 48,020                drawdown 1,980   <- maximum

    final balance 48,020, net -1,980, ROI -1,980/50,000 = -3.96%,
    max drawdown 1,980, episodes past $2,000: none (1,980 < 2,000),
    win rate 2 of 4 = 50%.
    """

    def test_unfiltered_metrics(self):
        result = analyse(fixture_trades())
        self.assertEqual(result["trade_count"], 4)
        self.assertAlmostEqual(result["final_balance_usd"], 48_020.0)
        self.assertAlmostEqual(result["net_pnl_usd"], -1_980.0)
        self.assertAlmostEqual(result["roi"], -1_980.0 / 50_000.0)
        self.assertAlmostEqual(result["max_drawdown_usd"], 1_980.0)
        self.assertEqual(result["drawdown_over_2000_episode_count"], 0)
        self.assertAlmostEqual(result["win_rate"], 0.5)
        self.assertEqual(len(result["equity_curve"]), 8)  # two events per trade

    def test_new_york_am_only(self):
        """Only trade A: 50,000 - 20 = 49,980 (drawdown 20), then +200 =
        50,180. Max drawdown 20, net +180, win rate 100%."""
        selected = filter_trades(fixture_trades(), SessionFilter(sessions=frozenset({"new_york_am"})))
        result = analyse(selected)
        self.assertEqual(result["trade_count"], 1)
        self.assertAlmostEqual(result["final_balance_usd"], 50_180.0)
        self.assertAlmostEqual(result["net_pnl_usd"], 180.0)
        self.assertAlmostEqual(result["max_drawdown_usd"], 20.0)
        self.assertAlmostEqual(result["win_rate"], 1.0)

    def test_london_plus_asia_crosses_the_2000_episode_threshold(self):
        """Trades B and D: -20 (dd 20), -300 (dd 320), -20 (dd 340),
        -1,900 = 47,760 (dd 2,240) -- past $2,000, so exactly one episode."""
        selected = filter_trades(fixture_trades(), SessionFilter(sessions=frozenset({"london", "asia"})))
        result = analyse(selected)
        self.assertAlmostEqual(result["final_balance_usd"], 47_760.0)
        self.assertAlmostEqual(result["net_pnl_usd"], -2_240.0)
        self.assertAlmostEqual(result["max_drawdown_usd"], 2_240.0)
        self.assertEqual(result["drawdown_over_2000_episode_count"], 1)
        self.assertAlmostEqual(result["win_rate"], 0.0)

    def test_hour_range_10_to_12(self):
        """Trades A and C: -20 (dd 20), +200 = 50,180 (new peak), -20 = 50,160
        (dd 20), +100 = 50,260. Max drawdown 20, net +260."""
        selected = filter_trades(fixture_trades(), SessionFilter(hour_from=10, hour_to=12))
        result = analyse(selected)
        self.assertAlmostEqual(result["final_balance_usd"], 50_260.0)
        self.assertAlmostEqual(result["net_pnl_usd"], 260.0)
        self.assertAlmostEqual(result["max_drawdown_usd"], 20.0)

    def test_breakdown_by_session(self):
        by_session = analyse(fixture_trades())["by_session"]
        self.assertEqual(sorted(by_session), ["asia", "london", "new_york_am", "new_york_pm"])
        self.assertAlmostEqual(by_session["asia"]["net_pnl_usd"], -1_920.0)
        self.assertAlmostEqual(by_session["new_york_am"]["net_pnl_usd"], 180.0)
        self.assertEqual(by_session["london"]["trade_count"], 1)
        self.assertAlmostEqual(by_session["new_york_pm"]["win_rate"], 1.0)

    def test_unresolved_trade_charges_its_cost_and_no_pnl(self):
        trades = fixture_trades()[:1]
        trades.append(trade(9, D0 + timedelta(hours=20), None, None))
        result = analyse(trades)
        # B's -320 net, plus the open trade's $20 cost and nothing else.
        self.assertAlmostEqual(result["final_balance_usd"], 50_000.0 - 20.0 - 300.0 - 20.0)
        self.assertEqual(result["trade_count"], 2)
        self.assertEqual(result["resolved_trade_count"], 1)


class EngineEquivalenceTests(unittest.TestCase):
    """Selecting everything must reproduce tradelab.equity exactly.

    This is the check that keeps the post-hoc view honest: if the two ever
    diverge, a filtered figure could not be compared to the run's own headline
    numbers at all.
    """

    def _accounted(self):
        accounted = []
        for spec in fixture_trades():
            gross = spec["gross_pnl_usd"]
            accounted.append(
                AccountedTrade(
                    trade_id=spec["trade_id"], strategy_id="s", direction=spec["direction"],
                    instrument=Instrument.MNQ,
                    entry_timestamp=datetime.fromisoformat(spec["entry_at"]),
                    exit_timestamp=datetime.fromisoformat(spec["exit_at"]),
                    entry_price=100.0, sl_price=99.0, tp_price=101.0, quantity=1,
                    risk_per_contract_usd=2.0, planned_risk_usd=2.0, entry_cost_usd=spec["entry_cost_usd"],
                    exit_reason=TradeExitReason.TP_HIT if gross > 0 else TradeExitReason.SL_HIT,
                    gross_pnl_usd=gross, net_pnl_usd=spec["net_pnl_usd"],
                )
            )
        return accounted

    def test_identity_filter_matches_build_equity_curve(self):
        accounted = self._accounted()
        engine = build_equity_curve(
            AccountBacktestResult(
                strategy_id="s", instrument=Instrument.MNQ, risk_config=RiskConfig(),
                accounted_trades=accounted, untradeable_trades=[], trade_count=len(accounted),
                untradeable_count=0, open_unresolved_count=0,
                realized_gross_pnl_usd_total=sum(t.gross_pnl_usd for t in accounted),
                realized_net_pnl_usd_total=sum(t.net_pnl_usd for t in accounted),
                entry_costs_charged_usd_total=sum(t.entry_cost_usd for t in accounted),
            ),
            EquityConfig(),
        )
        posthoc = analyse(fixture_trades())
        self.assertAlmostEqual(posthoc["final_balance_usd"], engine.final_balance_usd)
        self.assertAlmostEqual(posthoc["net_pnl_usd"], engine.net_cash_pnl_usd)
        self.assertAlmostEqual(posthoc["max_drawdown_usd"], engine.max_drawdown_usd)
        self.assertEqual(posthoc["drawdown_over_2000_episode_count"], engine.drawdown_over_threshold_episode_count)
        self.assertEqual(
            [round(p["equity"], 6) for p in posthoc["equity_curve"]],
            [round(p.balance_usd, 6) for p in engine.equity_points],
        )


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Journal round-trip and the dashboard endpoint
# ---------------------------------------------------------------------------

import http.client  # noqa: E402
import json  # noqa: E402
import sqlite3  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import urlencode  # noqa: E402

from tradelab.backtest_v2_journal import BacktestV2Journal  # noqa: E402
from tradelab.research_dashboard import serve  # noqa: E402
from tradelab.research_run import LeaderboardEntry, RunPhase, create_run  # noqa: E402


def open_journal(run_dir: Path) -> BacktestV2Journal:
    return BacktestV2Journal(
        Path(run_dir) / "backtest_v2_journal.sqlite3", run_id="r1",
        market_artifact_path=None, market_artifact_sha256=None, news_csv_path=None, news_csv_sha256=None,
        git_commit=None, parameters={}, risk_config={}, equity_config={}, exit_variant_ids=["v"], instruments=["MNQ"],
    )


class JournalTradePersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_preserves_utc_and_derives_the_session(self):
        with open_journal(self.run_dir) as journal:
            written = journal.replace_variant_trades({"v1": fixture_trades()})
            self.assertEqual(written, 4)
            rows = journal.variant_trades("v1")
        self.assertEqual([r["entry_session"] for r in rows], ["london", "new_york_am", "new_york_pm", "asia"])
        self.assertEqual([r["entry_hour_ny"] for r in rows], [4, 10, 12, 20])
        self.assertTrue(all(r["entry_at"].endswith("+00:00") for r in rows))
        self.assertAlmostEqual(rows[0]["entry_cost_usd"], 20.0)
        self.assertAlmostEqual(rows[3]["gross_pnl_usd"], -1900.0)

    def test_rewriting_a_variant_replaces_rather_than_duplicates(self):
        with open_journal(self.run_dir) as journal:
            journal.replace_variant_trades({"v1": fixture_trades()})
            journal.replace_variant_trades({"v1": fixture_trades()[:2]})
            self.assertEqual(len(journal.variant_trades("v1")), 2)
            self.assertEqual(journal.variants_with_trades(), ["v1"])

    def test_a_naive_timestamp_is_refused_rather_than_assumed_to_be_utc(self):
        bad = fixture_trades()[:1]
        bad[0]["entry_at"] = "2024-05-06T08:00:00"
        with open_journal(self.run_dir) as journal:
            with self.assertRaises(ValueError):
                journal.replace_variant_trades({"v1": bad})

    def test_the_session_and_hour_indexes_exist(self):
        with open_journal(self.run_dir) as journal:
            journal.replace_variant_trades({"v1": fixture_trades()})
        connection = sqlite3.connect(str(self.run_dir / "backtest_v2_journal.sqlite3"))
        try:
            names = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        finally:
            connection.close()
        self.assertIn("idx_variant_trades_session", names)
        self.assertIn("idx_variant_trades_hour", names)
        self.assertIn("idx_variant_summaries_pnl", names)

    def test_the_session_index_is_actually_used_by_the_filtering_query(self):
        with open_journal(self.run_dir) as journal:
            journal.replace_variant_trades({"v1": fixture_trades()})
        connection = sqlite3.connect(str(self.run_dir / "backtest_v2_journal.sqlite3"))
        try:
            plan = connection.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM variant_trades "
                "WHERE strategy_variant_id = ? AND entry_session = ? ORDER BY entry_at",
                ("v1", "london"),
            ).fetchall()
        finally:
            connection.close()
        self.assertIn("idx_variant_trades_session", " ".join(str(row) for row in plan))


class SessionAnalysisEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self._tmp.name)
        reporter = create_run(
            runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None,
            parameters={}, command="test", is_demo=True,
        )
        self.run_id = reporter.run_id
        self.run_dir = Path(reporter.run_dir)
        reporter.completed()
        with open_journal(self.run_dir) as journal:
            journal.replace_variant_trades({"cls::tp=fixed_rr_1|sl=liquidity_rank_1": fixture_trades()})
        self.server = serve(self.runs_dir, port=0, host="127.0.0.1")
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self._tmp.cleanup()

    def _get(self, params: dict):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", f"/api/runs/{self.run_id}/session-analysis?{urlencode(params)}")
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read())
        finally:
            conn.close()

    def test_lists_the_variants_with_persisted_trades_and_the_session_catalog(self):
        status, payload = self._get({})
        self.assertEqual(status, 200)
        self.assertTrue(payload["post_hoc"])
        self.assertIn("holdout", payload["disclaimer"])
        self.assertEqual(payload["available_strategy_ids"], ["cls::tp=fixed_rr_1|sl=liquidity_rank_1"])
        self.assertEqual([s["key"] for s in payload["sessions"]], ["asia", "london", "new_york_am", "new_york_pm", "close"])

    def test_unfiltered_result_matches_the_hand_computed_baseline(self):
        _status, payload = self._get({"strategy_id": "cls::tp=fixed_rr_1|sl=liquidity_rank_1"})
        self.assertAlmostEqual(payload["result"]["net_pnl_usd"], -1_980.0)
        self.assertAlmostEqual(payload["result"]["max_drawdown_usd"], 1_980.0)
        self.assertTrue(payload["filter"]["is_identity"])

    def test_session_filter_changes_only_the_result_never_the_baseline(self):
        _status, payload = self._get({
            "strategy_id": "cls::tp=fixed_rr_1|sl=liquidity_rank_1", "sessions": "new_york_am",
        })
        self.assertAlmostEqual(payload["baseline"]["net_pnl_usd"], -1_980.0)
        self.assertAlmostEqual(payload["result"]["net_pnl_usd"], 180.0)
        self.assertEqual(payload["result"]["trade_count"], 1)

    def test_hour_range_filter(self):
        _status, payload = self._get({
            "strategy_id": "cls::tp=fixed_rr_1|sl=liquidity_rank_1", "hour_from": "10", "hour_to": "12",
        })
        self.assertAlmostEqual(payload["result"]["net_pnl_usd"], 260.0)

    def test_unknown_session_is_reported_not_silently_ignored(self):
        _status, payload = self._get({
            "strategy_id": "cls::tp=fixed_rr_1|sl=liquidity_rank_1", "sessions": "tokyo",
        })
        self.assertIsNone(payload["result"])
        self.assertIn("tokyo", payload["error"])

    def test_the_endpoint_opens_the_journal_read_only(self):
        """The dashboard must never be able to take a write lock on a journal a
        run might still be using."""
        from tradelab.research_dashboard import _open_journal_readonly

        connection = _open_journal_readonly(self.run_dir)
        self.assertIsNotNone(connection)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM variant_trades")
        finally:
            connection.close()


class EndToEndPersistenceTests(unittest.TestCase):
    """A real (if tiny) production run must leave the session view usable.

    Every other test here builds the trade rows by hand. This one runs the
    actual production entry point and checks that the rows arrive, with the
    session labels derived and the summary's own trade count agreeing -- which
    is what makes the panel show real numbers rather than an empty state.
    """

    def test_a_completed_production_run_persists_its_leaderboards_trades(self):
        from unittest.mock import patch

        from tests.test_tradelab_backtest_v2 import _write_smoke_market_artifact
        from tradelab.backtest_v2 import run_production_backtest_v2

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"
            # Pinned clean, like the other production-runner tests: this test is
            # about trade persistence, not about the dirty-working-tree gate,
            # which has its own dedicated coverage.
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", False)):
                reporter = run_production_backtest_v2(
                    market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                    pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                    run_id="session-analysis-e2e", command="test",
                )
            db_path = Path(reporter.run_dir) / "backtest_v2_journal.sqlite3"
            connection = sqlite3.connect(str(db_path))
            try:
                rows = connection.execute(
                    "SELECT strategy_variant_id, entry_at, entry_session, entry_hour_ny, entry_cost_usd "
                    "FROM variant_trades"
                ).fetchall()
                summaries = dict(connection.execute(
                    "SELECT strategy_variant_id, trade_count FROM variant_summaries"
                ).fetchall())
            finally:
                connection.close()

        self.assertGreater(len(rows), 0, "a completed run persisted no trade-level rows at all")
        persisted = {}
        for variant_id, entry_at, session, hour, cost in rows:
            persisted[variant_id] = persisted.get(variant_id, 0) + 1
            self.assertTrue(entry_at.endswith("+00:00"), entry_at)
            self.assertIn(session, {"asia", "london", "new_york_am", "new_york_pm", "close", "maintenance_break"})
            self.assertTrue(0 <= hour <= 23)
            self.assertGreater(cost, 0)
        for variant_id, count in persisted.items():
            # The rows are the variant's own trades, all of them -- not a sample.
            self.assertEqual(count, summaries[variant_id], variant_id)
