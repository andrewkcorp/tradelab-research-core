#!/usr/bin/env python3
"""Research journal tests: a crash mid-run followed by resume never
duplicates a committed (strategy_id, instrument) row and never
re-runs a strategy every one of whose instruments is already
committed; an interruption before a commit actually lands leaves no
false checkpoint behind, and resuming afterward completes correctly;
any manifest field mismatch blocks resume outright with a readable
error; qualified evaluations persist their six summary metrics while
rejected/prefiltered ones persist none of them (but always persist
strategy_spec/window_trade_counts/rejection_reason); prefiltered rows
remain fully auditable and distinguishable from real rejections;
export is deterministic and ordered by realized_net_pnl_usd desc, then
max_drawdown_usd asc, then trade_count desc, then strategy_id asc; and
run() never materializes its strategy_specs input."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradelab.accounting import Instrument, RiskConfig
from tradelab.backtest import BacktestResult, ExecutedTrade, PreparedBacktestContext, TradeExitReason
from tradelab.entry_signal_timeline import EntrySignal, EntrySignalTimeline, SourceKind
from tradelab.equity import EquityConfig
from tradelab.research_journal import (
    EvaluationStatus,
    JournalEntry,
    ResearchJournal,
)
from tradelab.research_runner import ResearchConfig, ResearchWindow
from tradelab.sl_targets import SLSourceType
from tradelab.strategy_spec import LONG, BiasPolicy, SLChoice, StrategySpec, TPChoice
from tradelab.tp_targets import TPSourceType

UTC = timezone.utc
T0 = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)
RC = RiskConfig()
EC = EquityConfig()


def window(window_id: str = "w1", start: datetime = T0, end: datetime = T0 + timedelta(days=1)) -> ResearchWindow:
    return ResearchWindow(window_id=window_id, start_utc=start, end_utc=end)


def config(*windows: ResearchWindow, min_trades: int = 3) -> ResearchConfig:
    return ResearchConfig(windows=windows or (window(),), risk_config=RC, equity_config=EC, min_trades_per_window=min_trades)


def make_signal(selector: str, direction: str, known_at: datetime) -> EntrySignal:
    return EntrySignal(
        candidate_id=f"cand-{selector}-{direction}-{known_at.isoformat()}",
        selector=selector,
        source_kind=SourceKind.LIQUIDITY,
        timeframe="1m",
        direction=direction,
        timestamp=known_at,
        known_at=known_at,
        trading_day_id="2024-03-04",
        source=None,
    )


def make_context(signals: list) -> PreparedBacktestContext:
    by_selector: dict[str, list] = {}
    for s in signals:
        by_selector.setdefault(s.selector, []).append(s)
    timeline = EntrySignalTimeline(signals=list(signals), signals_by_selector=by_selector)
    return PreparedBacktestContext(
        bars_1m=(), news_events=(), tick_size=0.25, pivot_strength=3, reclaim_window_bars=5,
        fvg_results=None, fvg_touch_results=None, fvg_entry_result=None,
        swing_equal_liquidity_results=None, session_liquidity_results=None,
        liquidity_entry_result=None, data_wick_result=None, htf_bias_result=None, timeline=timeline,
    )


def make_strategy(strategy_id: str, primary: str = "fvg-1m-fvg_touch") -> StrategySpec:
    return StrategySpec(
        strategy_id=strategy_id, direction=LONG, primary_entry_selector=primary,
        required_confluence_selectors=(), bias_policy=BiasPolicy.IGNORE,
        tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY, rank=1),
        sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1),
    )


def make_trades(n: int, *, strategy_id: str, start: datetime, entry_price: float = 100.0, tp_price: float = 101.0, sl_price: float = 99.0) -> list[ExecutedTrade]:
    trades = []
    for i in range(n):
        ts = start + timedelta(minutes=i)
        trades.append(
            ExecutedTrade(
                trade_id=f"{strategy_id}-t{i}", strategy_id=strategy_id, direction=LONG,
                signal_known_at=ts, entry_timestamp=ts, entry_price=entry_price, tp_price=tp_price,
                tp_source_type="liquidity", tp_source_id=f"tp-{i}", sl_price=sl_price, sl_source_type="liquidity",
                sl_source_id=f"sl-{i}", exit_timestamp=ts + timedelta(minutes=1), exit_price=tp_price,
                exit_reason=TradeExitReason.TP_HIT, gross_pnl_points=tp_price - entry_price,
            )
        )
    return trades


def make_backtest_result(trades: list[ExecutedTrade], strategy_id: str) -> BacktestResult:
    return BacktestResult(
        strategy_id=strategy_id, trades=trades, skipped_signals=[], trade_count=len(trades),
        tp_hit_count=len(trades), sl_hit_count=0, open_unresolved_count=0,
        gross_pnl_points_total=sum(t.gross_pnl_points for t in trades),
    )


def journal(db_path, *, data_artifact_id="artifact-1", policy_version="policy-1", generator_version="gen-1", research_config=None, instruments=(Instrument.NQ,)) -> ResearchJournal:
    return ResearchJournal(
        db_path, data_artifact_id=data_artifact_id, policy_version=policy_version, generator_version=generator_version,
        research_config=research_config or config(), instruments=instruments,
    )


def patched_backtest(results_by_id: dict):
    def fake(strategy, ctx):
        return results_by_id[strategy.strategy_id]
    return unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", side_effect=fake)


class CrashResumeNoDuplicatesTests(unittest.TestCase):
    def test_resume_after_full_run_produces_no_new_rows_and_skips_backtest_entirely(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "journal.sqlite3"
            signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
            context = make_context(signals)
            strategies = [make_strategy(f"s{i}") for i in range(3)]
            results = {s.strategy_id: make_backtest_result(make_trades(3, strategy_id=s.strategy_id, start=T0), s.strategy_id) for s in strategies}

            with patched_backtest(results):
                j1 = journal(db_path)
                list(j1.run(strategies, context))
                first_status = j1.status()
                j1.close()

            with patched_backtest(results) as mocked:
                mocked.side_effect = AssertionError("backtest must not be re-run for a fully-committed strategy")
                j2 = journal(db_path)
                entries = list(j2.run(strategies, context))
                second_status = j2.status()
                j2.close()

            self.assertEqual(entries, [])
            self.assertEqual(second_status, first_status)
            self.assertEqual(second_status.completed_count, 3)  # 3 strategies x 1 instrument

    def test_partial_instrument_crash_resume_commits_only_the_missing_instrument(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "journal.sqlite3"
            signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
            context = make_context(signals)
            strategy = make_strategy("s0")
            result = make_backtest_result(make_trades(3, strategy_id="s0", start=T0), "s0")
            results = {"s0": result}

            # Both instruments are configured from the start (instruments are
            # part of the manifest -- they can never change between runs).
            # Simulate a crash that lands the NQ commit but not MNQ's, by
            # making the *second* _commit_evaluation call (MNQ, since NQ is
            # first in instruments order) fail.
            real_commit = ResearchJournal._commit_evaluation
            call_count = {"n": 0}

            def flaky_commit(self, evaluation):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("simulated crash before MNQ's commit landed")
                return real_commit(self, evaluation)

            with patched_backtest(results):
                with unittest.mock.patch.object(ResearchJournal, "_commit_evaluation", flaky_commit):
                    j1 = journal(db_path, instruments=(Instrument.NQ, Instrument.MNQ))
                    with self.assertRaises(RuntimeError):
                        list(j1.run([strategy], context))
                    interrupted_status = j1.status()
                    j1.close()

            self.assertEqual(interrupted_status.completed_count, 1)
            self.assertEqual(interrupted_status.last_checkpoint.instrument, "NQ")

            # resume with the same (NQ, MNQ) configuration -- NQ already
            # committed, MNQ still missing.
            with patched_backtest(results) as mocked:
                j2 = journal(db_path, instruments=(Instrument.NQ, Instrument.MNQ))
                entries = list(j2.run([strategy], context))
                status = j2.status()
                j2.close()

            self.assertEqual(mocked.call_count, 1)  # backtest re-run once (strategy wasn't fully done), never duplicated
            self.assertEqual([(e.strategy_id, e.instrument.value) for e in entries], [("s0", "MNQ")])
            self.assertEqual(status.completed_count, 2)

    def test_no_duplicate_rows_in_underlying_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "journal.sqlite3"
            signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
            context = make_context(signals)
            strategies = [make_strategy(f"s{i}") for i in range(2)]
            results = {s.strategy_id: make_backtest_result(make_trades(3, strategy_id=s.strategy_id, start=T0), s.strategy_id) for s in strategies}

            with patched_backtest(results):
                j1 = journal(db_path, instruments=(Instrument.NQ, Instrument.MNQ))
                list(j1.run(strategies, context))
                j1.close()
                j2 = journal(db_path, instruments=(Instrument.NQ, Instrument.MNQ))
                list(j2.run(strategies, context))  # run again on an already-complete journal
                j2.close()

            conn = sqlite3.connect(str(db_path))
            row_count = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
            distinct_count = conn.execute("SELECT COUNT(DISTINCT strategy_id || ':' || instrument) FROM evaluations").fetchone()[0]
            conn.close()
            self.assertEqual(row_count, 4)
            self.assertEqual(row_count, distinct_count)


class InterruptionBeforeCommitTests(unittest.TestCase):
    def test_failure_mid_run_leaves_no_false_checkpoint_and_preserves_prior_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "journal.sqlite3"
            signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
            context = make_context(signals)
            strategies = [make_strategy(f"s{i}") for i in range(4)]
            results = {s.strategy_id: make_backtest_result(make_trades(3, strategy_id=s.strategy_id, start=T0), s.strategy_id) for s in strategies}

            real_commit = ResearchJournal._commit_evaluation
            call_count = {"n": 0}

            def flaky_commit(self, evaluation):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("simulated failure before this pair's transaction committed")
                return real_commit(self, evaluation)

            with patched_backtest(results):
                with unittest.mock.patch.object(ResearchJournal, "_commit_evaluation", flaky_commit):
                    j1 = journal(db_path)
                    with self.assertRaises(RuntimeError):
                        list(j1.run(strategies, context))
                    interrupted_status = j1.status()
                    j1.close()

            self.assertEqual(interrupted_status.completed_count, 1)  # only the first pair committed before the failure

            # resume: the interrupted pair (and everything after it) completes; nothing duplicated
            with patched_backtest(results):
                j2 = journal(db_path)
                entries = list(j2.run(strategies, context))
                final_status = j2.status()
                j2.close()

            self.assertEqual(len(entries), 3)
            self.assertEqual(final_status.completed_count, 4)

    def test_error_message_names_the_run_and_the_pair_in_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "journal.sqlite3"
            signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
            context = make_context(signals)
            strategy = make_strategy("s0")
            results = {"s0": make_backtest_result(make_trades(3, strategy_id="s0", start=T0), "s0")}

            with patched_backtest(results):
                with unittest.mock.patch.object(ResearchJournal, "_commit_evaluation", side_effect=RuntimeError("disk full")):
                    j = journal(db_path)
                    with self.assertRaises(RuntimeError) as ctx:
                        list(j.run([strategy], context))
                    message = str(ctx.exception)
                    self.assertIn(j.run_id, message)
                    self.assertIn("s0", message)
                    j.close()


class ManifestMismatchTests(unittest.TestCase):
    def test_identical_config_resumes_with_the_same_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j1 = journal(db_path)
            run_id = j1.run_id
            j1.close()
            j2 = journal(db_path)
            self.assertEqual(j2.run_id, run_id)
            j2.close()

    def test_different_data_artifact_id_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            journal(db_path, data_artifact_id="artifact-1").close()
            with self.assertRaises(ValueError) as ctx:
                journal(db_path, data_artifact_id="artifact-2")
            self.assertIn("data_artifact_id", str(ctx.exception))

    def test_different_policy_version_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            journal(db_path, policy_version="p1").close()
            with self.assertRaises(ValueError) as ctx:
                journal(db_path, policy_version="p2")
            self.assertIn("policy_version", str(ctx.exception))

    def test_different_generator_version_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            journal(db_path, generator_version="g1").close()
            with self.assertRaises(ValueError) as ctx:
                journal(db_path, generator_version="g2")
            self.assertIn("generator_version", str(ctx.exception))

    def test_different_research_config_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            journal(db_path, research_config=config(min_trades=3)).close()
            with self.assertRaises(ValueError) as ctx:
                journal(db_path, research_config=config(min_trades=5))
            self.assertIn("research_config", str(ctx.exception))

    def test_different_instruments_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            journal(db_path, instruments=(Instrument.NQ,)).close()
            with self.assertRaises(ValueError) as ctx:
                journal(db_path, instruments=(Instrument.NQ, Instrument.MNQ))
            self.assertIn("instruments", str(ctx.exception))

    def test_mismatch_never_touches_existing_checkpoint_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "journal.sqlite3"
            signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
            context = make_context(signals)
            strategy = make_strategy("s0")
            results = {"s0": make_backtest_result(make_trades(3, strategy_id="s0", start=T0), "s0")}
            with patched_backtest(results):
                j1 = journal(db_path)
                list(j1.run([strategy], context))
                before = j1.status()
                j1.close()

            with self.assertRaises(ValueError):
                journal(db_path, policy_version="different")

            conn = sqlite3.connect(str(db_path))
            row_count = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
            conn.close()
            self.assertEqual(row_count, before.completed_count)


class QualifiedRejectedPrefilteredFieldsTests(unittest.TestCase):
    def _run_mixed(self, tmp: Path):
        db_path = tmp / "journal.sqlite3"
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
        context = make_context(signals)
        qualifies = make_strategy("s-qualifies")
        rejects = make_strategy("s-rejects")
        prefiltered = make_strategy("s-prefiltered", primary="fvg-5m-ifvg")  # zero raw signals -> prefiltered
        results = {
            "s-qualifies": make_backtest_result(make_trades(3, strategy_id="s-qualifies", start=T0), "s-qualifies"),
            "s-rejects": make_backtest_result(make_trades(1, strategy_id="s-rejects", start=T0), "s-rejects"),
        }
        with patched_backtest(results):
            j = journal(db_path, instruments=(Instrument.NQ,))
            list(j.run([qualifies, rejects, prefiltered], context))
            j.close()
        return db_path

    def _row(self, db_path: Path, strategy_id: str) -> dict:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM evaluations WHERE strategy_id = ?", (strategy_id,)).fetchone()
        conn.close()
        return dict(row)

    def test_qualified_row_has_all_six_summary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._run_mixed(Path(tmp))
            row = self._row(db_path, "s-qualifies")
            self.assertEqual(row["status"], EvaluationStatus.QUALIFIED.value)
            self.assertIsNone(row["rejection_reason"])
            self.assertIsNotNone(row["realized_net_pnl_usd"])
            self.assertIsNotNone(row["roi"])
            self.assertIsNotNone(row["max_drawdown_usd"])
            self.assertIsNotNone(row["drawdown_episode_count"])
            self.assertIsNotNone(row["trade_count"])
            self.assertIsNotNone(row["entry_costs_usd_total"])
            spec = json.loads(row["strategy_spec_json"])
            self.assertEqual(spec["strategy_id"], "s-qualifies")
            wtc = json.loads(row["window_trade_counts_json"])
            self.assertEqual(len(wtc), 1)
            self.assertTrue(wtc[0]["meets_minimum"])

    def test_rejected_row_has_no_summary_metrics_but_has_audit_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._run_mixed(Path(tmp))
            row = self._row(db_path, "s-rejects")
            self.assertEqual(row["status"], EvaluationStatus.REJECTED.value)
            self.assertEqual(row["rejection_reason"], "insufficient_trade_frequency")
            for field in ("realized_net_pnl_usd", "roi", "max_drawdown_usd", "drawdown_episode_count", "trade_count", "entry_costs_usd_total"):
                self.assertIsNone(row[field])
            spec = json.loads(row["strategy_spec_json"])
            self.assertEqual(spec["strategy_id"], "s-rejects")
            wtc = json.loads(row["window_trade_counts_json"])
            self.assertFalse(wtc[0]["meets_minimum"])
            self.assertEqual(wtc[0]["trade_count"], 1)

    def test_prefiltered_row_is_auditable_and_distinct_from_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._run_mixed(Path(tmp))
            row = self._row(db_path, "s-prefiltered")
            self.assertEqual(row["status"], EvaluationStatus.PREFILTERED.value)
            self.assertNotEqual(row["status"], EvaluationStatus.REJECTED.value)
            self.assertEqual(row["rejection_reason"], "insufficient_trade_frequency")
            for field in ("realized_net_pnl_usd", "roi", "max_drawdown_usd", "drawdown_episode_count", "trade_count", "entry_costs_usd_total"):
                self.assertIsNone(row[field])
            wtc = json.loads(row["window_trade_counts_json"])
            self.assertEqual(wtc[0]["trade_count"], 0)  # zero raw signals for this selector

    def test_status_counts_reflect_all_three_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._run_mixed(Path(tmp))
            j = journal(db_path, instruments=(Instrument.NQ,))
            status = j.status()
            j.close()
            self.assertEqual(status.qualified_count, 1)
            self.assertEqual(status.rejected_count, 1)
            self.assertEqual(status.prefiltered_count, 1)
            self.assertEqual(status.completed_count, 3)


class DeterministicExportAndRankingTests(unittest.TestCase):
    def _journal_with_three_qualified(self, tmp: Path) -> Path:
        db_path = tmp / "journal.sqlite3"
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
        context = make_context(signals)
        # b-mid: highest pnl; a-low: lowest pnl; c-tie1/c-tie2: identical pnl+drawdown+trades, broken by strategy_id.
        strategies = [make_strategy("b-mid"), make_strategy("a-low"), make_strategy("c-tie2"), make_strategy("c-tie1")]
        results = {
            "b-mid": make_backtest_result(make_trades(3, strategy_id="b-mid", start=T0, tp_price=110.0), "b-mid"),
            "a-low": make_backtest_result(make_trades(3, strategy_id="a-low", start=T0, tp_price=100.5), "a-low"),
            "c-tie2": make_backtest_result(make_trades(3, strategy_id="c-tie2", start=T0, tp_price=101.0), "c-tie2"),
            "c-tie1": make_backtest_result(make_trades(3, strategy_id="c-tie1", start=T0, tp_price=101.0), "c-tie1"),
        }
        with patched_backtest(results):
            j = journal(db_path, instruments=(Instrument.NQ,))
            list(j.run(strategies, context))
            j.close()
        return db_path

    def test_export_order_matches_pnl_desc_then_drawdown_asc_then_trades_desc_then_strategy_id_asc(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = self._journal_with_three_qualified(tmp)
            j = journal(db_path, instruments=(Instrument.NQ,))
            out = tmp / "export.jsonl"
            count = j.export_qualifying(out, fmt="jsonl")
            j.close()

            self.assertEqual(count, 4)
            records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            ids_in_order = [r["strategy_id"] for r in records]
            # b-mid has the highest pnl; a-low the lowest; c-tie1/c-tie2 tie on
            # everything but strategy_id, which breaks ascending -> c-tie1 before c-tie2.
            self.assertEqual(ids_in_order, ["b-mid", "c-tie1", "c-tie2", "a-low"])
            pnls = [r["realized_net_pnl_usd"] for r in records]
            self.assertEqual(pnls, sorted(pnls, reverse=True))

    def test_csv_and_jsonl_exports_are_deterministic_across_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = self._journal_with_three_qualified(tmp)
            j = journal(db_path, instruments=(Instrument.NQ,))
            j.export_qualifying(tmp / "a.csv", fmt="csv")
            j.export_qualifying(tmp / "b.csv", fmt="csv")
            j.export_qualifying(tmp / "a.jsonl", fmt="jsonl")
            j.export_qualifying(tmp / "b.jsonl", fmt="jsonl")
            j.close()
            self.assertEqual((tmp / "a.csv").read_bytes(), (tmp / "b.csv").read_bytes())
            self.assertEqual((tmp / "a.jsonl").read_bytes(), (tmp / "b.jsonl").read_bytes())

    def test_export_only_includes_qualified_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "journal.sqlite3"
            signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
            context = make_context(signals)
            qualifies = make_strategy("s-qualifies")
            rejects = make_strategy("s-rejects")
            prefiltered = make_strategy("s-prefiltered", primary="fvg-5m-ifvg")
            results = {
                "s-qualifies": make_backtest_result(make_trades(3, strategy_id="s-qualifies", start=T0), "s-qualifies"),
                "s-rejects": make_backtest_result(make_trades(1, strategy_id="s-rejects", start=T0), "s-rejects"),
            }
            with patched_backtest(results):
                j = journal(db_path, instruments=(Instrument.NQ,))
                list(j.run([qualifies, rejects, prefiltered], context))
                out = tmp / "export.csv"
                count = j.export_qualifying(out, fmt="csv")
                j.close()
            self.assertEqual(count, 1)
            self.assertIn("s-qualifies", out.read_text())
            self.assertNotIn("s-rejects", out.read_text())
            self.assertNotIn("s-prefiltered", out.read_text())

    def test_invalid_export_format_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j = journal(db_path)
            with self.assertRaises(ValueError):
                j.export_qualifying(Path(tmp) / "out.xml", fmt="xml")
            j.close()


class LazyStreamNotMaterializedTests(unittest.TestCase):
    def test_run_does_not_materialize_strategy_specs_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "journal.sqlite3"
            signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
            context = make_context(signals)
            strategies = [make_strategy(f"s{i}") for i in range(5)]
            results = {s.strategy_id: make_backtest_result(make_trades(3, strategy_id=s.strategy_id, start=T0), s.strategy_id) for s in strategies}

            pulled: list[str] = []

            def tracking_iter():
                for s in strategies:
                    pulled.append(s.strategy_id)
                    yield s

            with patched_backtest(results):
                j = journal(db_path)
                gen = j.run(tracking_iter(), context)
                first = next(gen)
                self.assertEqual(pulled, ["s0"])
                self.assertLess(len(pulled), len(strategies))
                self.assertEqual(first.strategy_id, "s0")
                j.close()

    def test_run_returns_a_generator(self):
        import inspect

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j = journal(db_path)
            gen = j.run(iter(()), make_context([]))
            self.assertTrue(inspect.isgenerator(gen))
            j.close()


class ContextManagerAndSchemaTests(unittest.TestCase):
    def test_context_manager_closes_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            with journal(db_path) as j:
                run_id = j.run_id
            self.assertTrue(run_id)
            with self.assertRaises(sqlite3.ProgrammingError):
                j._conn.execute("SELECT 1")

    def test_manifest_is_queryable_and_matches_construction_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j = journal(db_path, data_artifact_id="artifact-x", policy_version="policy-y", generator_version="gen-z")
            manifest = j.manifest
            self.assertEqual(manifest.data_artifact_id, "artifact-x")
            self.assertEqual(manifest.policy_version, "policy-y")
            self.assertEqual(manifest.generator_version, "gen-z")
            self.assertEqual(manifest.run_id, j.run_id)
            j.close()


if __name__ == "__main__":
    unittest.main()
