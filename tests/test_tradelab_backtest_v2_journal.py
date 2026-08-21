#!/usr/bin/env python3
"""tradelab.backtest_v2_journal tests: fresh manifest creation, successful
reopen with identical configuration, refusal to reopen with any mismatched
compatibility field (naming exactly which field(s) differ), commit_summary
durability + the primary-key uniqueness constraint (a duplicate strategy_
variant_id raises rather than being silently ignored), completed_count/
is_completed (the memory-bounded resume-check pair -- there is
deliberately no method that loads every completed id into a Python set),
and top_n_summaries ordering (net P&L desc, then max drawdown asc, then
strategy_variant_id desc -- ties broken deterministically)."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tradelab.backtest_v2_journal import BacktestV2Journal, JournalVariantSummary


def _summary_row(variant_id: str, *, net_pnl_usd: float = 100.0, max_drawdown_usd: float = 50.0) -> dict:
    return {
        "strategy_variant_id": variant_id, "equivalence_class_id": "CLASS_A", "direction": "long", "instrument": "MNQ",
        "exit_variant_id": "tp_first_swing_high_r1__sl_first_swing_low_r1", "tp_kind": "first_swing_high", "tp_rank": 1,
        "sl_kind": "first_swing_low", "sl_rank": 1, "sl_fvg_timeframe": None, "net_pnl_usd": net_pnl_usd, "roi": 0.1,
        "win_rate": 0.5, "trade_count": 10, "max_drawdown_usd": max_drawdown_usd, "drawdowns_over_2000_count": 0,
        "skipped_entries_by_reason": {}, "yearly": {}, "monthly": {},
    }


def _open(db_path: Path, **overrides) -> BacktestV2Journal:
    kwargs = dict(
        run_id="r1", market_artifact_path="/data/artifact.json", market_artifact_sha256="abc123",
        news_csv_path="/data/news.csv", news_csv_sha256="def456", git_commit="deadbeef",
        parameters={"tick_size": 0.25, "pivot_strength": 2}, risk_config={"target_risk_usd": 500.0},
        equity_config={"initial_balance_usd": 100000.0}, exit_variant_ids=["v1", "v2"], instruments=["MNQ", "NQ"],
    )
    kwargs.update(overrides)
    return BacktestV2Journal(db_path, **kwargs)


class ManifestCreateAndVerifyTests(unittest.TestCase):
    def test_fresh_journal_uses_wal_for_nonblocking_dashboard_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            journal = _open(db_path)
            self.assertEqual(journal._conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
            journal.close()

    def test_fresh_creation_writes_manifest_and_reopen_with_identical_config_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j1 = _open(db_path)
            self.assertEqual(j1.run_id, "r1")
            self.assertEqual(j1.completed_count(), 0)
            j1.close()

            j2 = _open(db_path)  # identical config -- must not raise
            self.assertEqual(j2.completed_count(), 0)
            j2.close()

    def test_reopen_with_different_market_artifact_sha256_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            _open(db_path).close()

            with self.assertRaises(ValueError) as ctx:
                _open(db_path, market_artifact_sha256="different-hash")
            self.assertIn("market_artifact_sha256", str(ctx.exception))
            self.assertIn("r1", str(ctx.exception))

    def test_reopen_with_different_risk_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            _open(db_path).close()

            with self.assertRaises(ValueError) as ctx:
                _open(db_path, risk_config={"target_risk_usd": 999.0})
            self.assertIn("risk_config_json", str(ctx.exception))

    def test_reopen_with_different_exit_variants_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            _open(db_path).close()

            with self.assertRaises(ValueError) as ctx:
                _open(db_path, exit_variant_ids=["v1"])
            self.assertIn("exit_variant_ids_json", str(ctx.exception))

    def test_reopen_with_different_git_commit_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            _open(db_path).close()

            with self.assertRaises(ValueError) as ctx:
                _open(db_path, git_commit="other-commit")
            self.assertIn("git_commit", str(ctx.exception))

    def test_reopen_with_multiple_mismatches_names_every_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            _open(db_path).close()

            with self.assertRaises(ValueError) as ctx:
                _open(db_path, git_commit="other-commit", instruments=["MNQ"])
            message = str(ctx.exception)
            self.assertIn("git_commit", message)
            self.assertIn("instruments_json", message)

    def test_different_market_artifact_path_with_same_sha256_is_not_a_mismatch(self):
        # path is audit-only -- only the hash (content) is enforced.
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            _open(db_path).close()

            j2 = _open(db_path, market_artifact_path="/some/other/path/artifact.json")  # must not raise
            j2.close()

    def test_empty_run_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            with self.assertRaises(ValueError):
                _open(db_path, run_id="")


class CommitSummaryAndUniquenessTests(unittest.TestCase):
    def test_commit_summary_is_durable_and_readable_via_is_completed_and_completed_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j = _open(db_path)
            self.assertFalse(j.is_completed("v1"))
            self.assertFalse(j.is_completed("v2"))
            j.commit_summary(_summary_row("v1"))
            self.assertTrue(j.is_completed("v1"))
            self.assertFalse(j.is_completed("v2"))  # per-candidate lookup, never a full-table load
            self.assertEqual(j.completed_count(), 1)
            j.close()

            # durable across a reopen of the same db_path.
            j2 = _open(db_path)
            self.assertTrue(j2.is_completed("v1"))
            self.assertEqual(j2.completed_count(), 1)
            j2.close()

    def test_persists_apex_audit_summary_without_requiring_it_on_legacy_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = _open(Path(tmp) / "journal.sqlite3")
            row = _summary_row("breached")
            row["apex_eod_audit"] = {
                "has_breach": True, "breach_count": 2,
                "breached_trading_day_count": 1, "max_daily_loss_usd": 1200.0,
                "largest_eod_drawdown_breach_usd": 50.0,
            }
            j.commit_summary(row)
            stored = j.top_n_summaries(1)[0]
            self.assertTrue(stored.apex_has_breach)
            self.assertEqual(stored.apex_breach_count, 2)
            self.assertEqual(stored.apex_breached_trading_day_count, 1)
            self.assertEqual(stored.apex_max_daily_loss_usd, 1200.0)
            self.assertEqual(stored.apex_largest_eod_drawdown_breach_usd, 50.0)
            j.close()

    def test_duplicate_strategy_variant_id_raises_rather_than_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j = _open(db_path)
            j.commit_summary(_summary_row("v1"))
            with self.assertRaises(sqlite3.IntegrityError):
                j.commit_summary(_summary_row("v1"))
            self.assertEqual(j.completed_count(), 1)  # the failed duplicate never landed
            j.close()


class TopNSummariesOrderingTests(unittest.TestCase):
    def test_ranked_by_net_pnl_desc_then_drawdown_asc_then_id_desc(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j = _open(db_path)
            j.commit_summary(_summary_row("low_pnl", net_pnl_usd=10.0))
            j.commit_summary(_summary_row("high_pnl", net_pnl_usd=500.0))
            j.commit_summary(_summary_row("mid_pnl", net_pnl_usd=200.0))
            # tie on net_pnl_usd -- broken by lower max_drawdown_usd first.
            j.commit_summary(_summary_row("tie_low_dd", net_pnl_usd=300.0, max_drawdown_usd=10.0))
            j.commit_summary(_summary_row("tie_high_dd", net_pnl_usd=300.0, max_drawdown_usd=90.0))

            top = j.top_n_summaries(10)
            ids = [s.strategy_variant_id for s in top]
            self.assertEqual(ids, ["high_pnl", "tie_low_dd", "tie_high_dd", "mid_pnl", "low_pnl"])
            self.assertIsInstance(top[0], JournalVariantSummary)
            j.close()

    def test_top_n_is_bounded_by_n(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            j = _open(db_path)
            for i in range(10):
                j.commit_summary(_summary_row(f"v{i}", net_pnl_usd=float(i)))
            self.assertEqual(len(j.top_n_summaries(3)), 3)
            self.assertEqual(j.top_n_summaries(3)[0].strategy_variant_id, "v9")
            self.assertEqual(j.top_n_summaries(0), [])
            j.close()


class ContextManagerTests(unittest.TestCase):
    def test_usable_as_a_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "journal.sqlite3"
            with _open(db_path) as j:
                j.commit_summary(_summary_row("v1"))
                self.assertEqual(j.completed_count(), 1)


if __name__ == "__main__":
    unittest.main()
