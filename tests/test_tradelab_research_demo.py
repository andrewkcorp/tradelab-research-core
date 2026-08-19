#!/usr/bin/env python3
"""tradelab.research_demo tests: the demo run is deterministic given a
seed, always labeled is_demo=True, produces every run artifact through
every phase, never touches real detection/discovery/backtest code, and
its leaderboard/checkpoint/event data is internally consistent."""
from __future__ import annotations

import tempfile
import unittest

from tradelab.research_demo import run_demo
from tradelab.research_run import RunPhase, read_leaderboard, read_live_status, read_progress_events, read_run_manifest


class DemoRunTests(unittest.TestCase):
    def test_demo_run_is_labeled_demo_everywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = run_demo(runs_dir=tmp, seed=1, real_time=False)
            manifest = read_run_manifest(reporter.run_dir)
            self.assertTrue(manifest.is_demo)
            self.assertTrue(reporter.run_id.startswith("demo-"))
            status = read_live_status(reporter.run_dir)
            self.assertTrue(status.is_demo)

    def test_demo_run_completes_successfully(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = run_demo(runs_dir=tmp, seed=1, real_time=False)
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.phase, RunPhase.COMPLETED.value)
            self.assertEqual(status.errors, ())

    def test_demo_run_visits_every_work_phase_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = run_demo(runs_dir=tmp, seed=1, real_time=False)
            events = read_progress_events(reporter.run_dir)
            phase_starts = [e["phase"] for e in events if e["event_type"] == "phase_started"]
            self.assertEqual(
                phase_starts,
                ["preparing_data", "discovering_entries", "deduplicating_entries", "expanding_exit_variants", "simulating_backtests", "ranking_results"],
            )

    def test_demo_run_produces_a_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = run_demo(runs_dir=tmp, seed=1, real_time=False)
            leaderboard = read_leaderboard(reporter.run_dir)
            self.assertIsNotNone(leaderboard)
            self.assertGreater(len(leaderboard.entries), 0)
            ranks = [e.rank for e in leaderboard.entries]
            self.assertEqual(ranks, sorted(ranks))
            pnls = [e.pnl for e in leaderboard.entries]
            self.assertEqual(pnls, sorted(pnls, reverse=True))  # ranked by descending pnl

    def test_demo_run_produces_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = run_demo(runs_dir=tmp, seed=1, real_time=False)
            checkpoint_files = list((reporter.run_dir / "checkpoints").glob("*.json"))
            self.assertGreater(len(checkpoint_files), 0)

    def test_demo_run_is_deterministic_given_the_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            r1 = run_demo(runs_dir=tmp1, seed=42, real_time=False, run_id="demo-fixed")
            r2 = run_demo(runs_dir=tmp2, seed=42, real_time=False, run_id="demo-fixed")
            lb1 = read_leaderboard(r1.run_dir)
            lb2 = read_leaderboard(r2.run_dir)
            self.assertEqual([e.to_dict() for e in lb1.entries], [e.to_dict() for e in lb2.entries])
            status1 = read_live_status(r1.run_dir)
            status2 = read_live_status(r2.run_dir)
            self.assertEqual(status1.discovered_specs, status2.discovered_specs)
            self.assertEqual(status1.eligible_specs, status2.eligible_specs)
            self.assertEqual(status1.backtests_completed, status2.backtests_completed)

    def test_different_seeds_produce_different_leaderboards(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            r1 = run_demo(runs_dir=tmp1, seed=1, real_time=False)
            r2 = run_demo(runs_dir=tmp2, seed=2, real_time=False)
            lb1 = read_leaderboard(r1.run_dir)
            lb2 = read_leaderboard(r2.run_dir)
            self.assertNotEqual([e.pnl for e in lb1.entries], [e.pnl for e in lb2.entries])

    def test_never_imports_real_discovery_or_backtest_modules(self):
        import ast

        import tradelab.research_demo as demo_module

        with open(demo_module.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        for forbidden in ("tradelab.entry_strategy_universe", "tradelab.backtest", "tradelab.research_runner"):
            self.assertNotIn(forbidden, imported_modules, f"tradelab.research_demo must never import {forbidden}")


if __name__ == "__main__":
    unittest.main()
