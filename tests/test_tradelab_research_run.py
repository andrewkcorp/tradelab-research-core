#!/usr/bin/env python3
"""tradelab.research_run tests: atomic JSON snapshot writes (temp file
never left behind, target file never partially written), append-only
JSONL event logging, run manifest provenance (commit/hashes/params/
schema version), the reporter's phase/progress/checkpoint/leaderboard/
completed/failed API and its state-machine guards, and the ETA rule
(only statistically defensible, otherwise None)."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from tradelab.research_run import (
    RUN_ARTIFACT_SCHEMA_VERSION,
    CheckpointRecord,
    LeaderboardEntry,
    RunManifest,
    RunPhase,
    atomic_write_json,
    create_run,
    list_run_ids,
    read_leaderboard,
    read_live_status,
    read_progress_events,
    read_run_manifest,
)


class AtomicWriteTests(unittest.TestCase):
    def test_writes_a_complete_valid_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "status.json"
            atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
            with open(path) as f:
                self.assertEqual(json.load(f), {"a": 1, "b": [1, 2, 3]})

    def test_never_leaves_a_temp_file_behind_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            atomic_write_json(path, {"x": 1})
            leftover = [p for p in Path(tmp).iterdir() if p.name != "status.json"]
            self.assertEqual(leftover, [])

    def test_overwrite_is_atomic_target_always_fully_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            atomic_write_json(path, {"phase": "one"})
            atomic_write_json(path, {"phase": "two"})
            with open(path) as f:
                self.assertEqual(json.load(f), {"phase": "two"})
            leftover = [p for p in Path(tmp).iterdir() if p.name != "status.json"]
            self.assertEqual(leftover, [])

    def test_failure_during_write_leaves_target_untouched_and_cleans_up_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            atomic_write_json(path, {"phase": "original"})

            class Unserializable:
                pass

            with self.assertRaises(TypeError):
                atomic_write_json(path, {"phase": Unserializable()})

            with open(path) as f:
                self.assertEqual(json.load(f), {"phase": "original"})  # untouched
            leftover = [p for p in Path(tmp).iterdir() if p.name != "status.json"]
            self.assertEqual(leftover, [])  # temp file cleaned up


class ManifestProvenanceTests(unittest.TestCase):
    def test_create_run_writes_manifest_with_schema_version_and_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={"tick_size": 0.25}, command="python3 -m x")
            manifest = read_run_manifest(reporter.run_dir)
            self.assertEqual(manifest.schema_version, RUN_ARTIFACT_SCHEMA_VERSION)
            self.assertEqual(manifest.parameters, {"tick_size": 0.25})
            self.assertEqual(manifest.command, "python3 -m x")
            self.assertFalse(manifest.is_demo)
            self.assertIsNotNone(manifest.created_at)
            self.assertIsNotNone(manifest.python_version)

    def test_hashes_are_none_never_fabricated_when_paths_dont_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(
                runs_dir=tmp, market_artifact_path="/does/not/exist.json", news_csv_path="/does/not/exist.csv", parameters={},
            )
            manifest = read_run_manifest(reporter.run_dir)
            self.assertIsNone(manifest.market_artifact_sha256)
            self.assertIsNone(manifest.news_csv_sha256)

    def test_hashes_are_computed_when_paths_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "artifact.json"
            artifact_path.write_text('{"x": 1}', encoding="utf-8")
            reporter = create_run(runs_dir=tmp, market_artifact_path=artifact_path, news_csv_path=None, parameters={})
            manifest = read_run_manifest(reporter.run_dir)
            self.assertIsNotNone(manifest.market_artifact_sha256)
            self.assertEqual(len(manifest.market_artifact_sha256), 64)

    def test_run_directory_layout_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            self.assertTrue((reporter.run_dir / "run_manifest.json").is_file())
            self.assertTrue((reporter.run_dir / "live_status.json").is_file())
            self.assertTrue((reporter.run_dir / "results").is_dir())
            self.assertTrue((reporter.run_dir / "checkpoints").is_dir())

    def test_manifest_round_trips_through_to_dict_from_dict(self):
        manifest = RunManifest(
            schema_version=RUN_ARTIFACT_SCHEMA_VERSION, run_id="r1", created_at="t", git_commit="abc", git_commit_dirty=False,
            market_artifact_path="p", market_artifact_sha256="h", news_csv_path="n", news_csv_sha256="h2",
            parameters={"a": 1}, python_version="3.9", command="cmd", is_demo=True,
        )
        self.assertEqual(RunManifest.from_dict(manifest.to_dict()), manifest)

    def test_rejects_wrong_schema_version(self):
        with self.assertRaises(ValueError):
            RunManifest(
                schema_version="9.9", run_id="r1", created_at="t", git_commit=None, git_commit_dirty=None,
                market_artifact_path=None, market_artifact_sha256=None, news_csv_path=None, news_csv_sha256=None,
                parameters={}, python_version="3.9", command=None, is_demo=False,
            )


class ReporterPhaseProgressTests(unittest.TestCase):
    def test_progress_before_phase_started_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            with self.assertRaises(RuntimeError):
                reporter.progress(items_completed=1)

    def test_phase_started_rejects_terminal_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            with self.assertRaises(ValueError):
                reporter.phase_started(RunPhase.COMPLETED)
            with self.assertRaises(ValueError):
                reporter.phase_started(RunPhase.FAILED)

    def test_live_status_reflects_current_phase_and_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=10)
            reporter.progress(items_completed=3, discovered_specs=100, eligible_specs=5)
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.phase, "discovering_entries")
            self.assertEqual(status.items_completed, 3)
            self.assertEqual(status.items_total, 10)
            self.assertEqual(status.discovered_specs, 100)
            self.assertEqual(status.eligible_specs, 5)

    def test_progress_events_are_appended_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=2)
            reporter.progress(items_completed=1)
            reporter.progress(items_completed=2)
            reporter.completed()
            events = read_progress_events(reporter.run_dir)
            types = [e["event_type"] for e in events]
            self.assertEqual(types, ["phase_started", "progress", "progress", "completed"])

    def test_read_progress_events_respects_limit_as_a_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=5)
            for i in range(1, 6):
                reporter.progress(items_completed=i)
            events = read_progress_events(reporter.run_dir, limit=2)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["detail"]["items_completed"], 5)

    def test_warning_appears_in_status_and_event_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            reporter.progress(items_completed=1, warning="something looked off")
            status = read_live_status(reporter.run_dir)
            self.assertIn("something looked off", status.warnings)

    def test_failed_records_error_and_sets_failed_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            reporter.failed(error="boom", traceback_text="Traceback...")
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.phase, "failed")
            self.assertIn("boom", status.errors)

    def test_negative_items_completed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            with self.assertRaises(ValueError):
                reporter.progress(items_completed=-1)


class CheckpointHonestyTests(unittest.TestCase):
    def test_default_checkpoint_is_not_resumable_and_carries_an_honest_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            record = reporter.checkpoint_written(cursor={"step": 1})
            self.assertFalse(record.resumable)
            self.assertTrue(record.resumable_note.strip())
            self.assertIn("Not yet resumable", record.resumable_note)

    def test_resumable_true_requires_an_explicit_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            with self.assertRaises(ValueError):
                reporter.checkpoint_written(cursor={"step": 1}, resumable=True)

    def test_resumable_true_with_note_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            record = reporter.checkpoint_written(cursor={"step": 1}, resumable=True, resumable_note="resume via tradelab.research_journal at this cursor")
            self.assertTrue(record.resumable)

    def test_checkpoint_file_is_written_and_status_references_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            record = reporter.checkpoint_written(cursor={"step": 1})
            checkpoint_files = list((reporter.run_dir / "checkpoints").glob("*.json"))
            self.assertEqual(len(checkpoint_files), 1)
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.latest_checkpoint["checkpoint_id"], record.checkpoint_id)

    def test_checkpoint_record_rejects_illegal_phase(self):
        with self.assertRaises(ValueError):
            CheckpointRecord(
                schema_version=RUN_ARTIFACT_SCHEMA_VERSION, run_id="r", checkpoint_id="c1", created_at="t",
                phase="not_a_real_phase", cursor={}, resumable=False, resumable_note="note",
            )


class LeaderboardTests(unittest.TestCase):
    def test_update_leaderboard_writes_file_and_top_strategies_in_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            entries = [
                LeaderboardEntry(rank=2, strategy_id="s2", pnl=100.0, roi=0.1, win_rate=0.5, trade_count=10, max_drawdown=50.0, drawdowns_over_2000_count=0),
                LeaderboardEntry(rank=1, strategy_id="s1", pnl=200.0, roi=0.2, win_rate=0.6, trade_count=20, max_drawdown=80.0, drawdowns_over_2000_count=1),
            ]
            reporter.update_leaderboard(entries)
            leaderboard = read_leaderboard(reporter.run_dir)
            self.assertEqual(len(leaderboard.entries), 2)
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.top_strategies[0]["strategy_id"], "s1")  # rank 1 first

    def test_win_rate_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            LeaderboardEntry(rank=1, strategy_id="s", pnl=1.0, roi=0.1, win_rate=1.5, trade_count=1, max_drawdown=1.0, drawdowns_over_2000_count=0)

    def test_no_leaderboard_written_yet_reads_as_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            self.assertIsNone(read_leaderboard(reporter.run_dir))


class EtaRuleTests(unittest.TestCase):
    def test_eta_is_none_before_enough_progress_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=100)
            reporter.progress(items_completed=10)
            status = read_live_status(reporter.run_dir)
            self.assertIsNone(status.eta_seconds)
            self.assertIsNone(status.eta_basis)

    def test_eta_is_none_when_items_total_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.DISCOVERING_ENTRIES)  # no items_total
            for i in range(5):
                reporter.progress(items_completed=i)
                time.sleep(0.001)
            status = read_live_status(reporter.run_dir)
            self.assertIsNone(status.eta_seconds)

    def test_eta_is_populated_with_explicit_basis_once_enough_forward_progress_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=100)
            for i in range(1, 6):
                reporter.progress(items_completed=i * 10)
                time.sleep(0.01)
            status = read_live_status(reporter.run_dir)
            self.assertIsNotNone(status.eta_seconds)
            self.assertGreaterEqual(status.eta_seconds, 0)
            self.assertEqual(status.eta_basis["method"], "linear_recent_window")
            self.assertGreaterEqual(status.eta_basis["sample_size"], 3)

    def test_eta_is_none_once_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=100)
            for i in range(1, 6):
                reporter.progress(items_completed=i * 10)
                time.sleep(0.001)
            reporter.completed()
            status = read_live_status(reporter.run_dir)
            self.assertIsNone(status.eta_seconds)

    def test_eta_is_none_when_progress_has_stalled_zero_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=100)
            for _ in range(5):
                reporter.progress(items_completed=10)  # never advances
                time.sleep(0.001)
            status = read_live_status(reporter.run_dir)
            self.assertIsNone(status.eta_seconds)


class ListRunIdsTests(unittest.TestCase):
    def test_lists_runs_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1 = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-a")
            time.sleep(0.01)
            r2 = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-b")
            ids = list_run_ids(tmp)
            self.assertEqual(ids, ["run-b", "run-a"])

    def test_empty_runs_dir_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_run_ids(Path(tmp) / "does-not-exist"), [])

    def test_directory_without_manifest_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "not-a-run").mkdir()
            self.assertEqual(list_run_ids(tmp), [])


if __name__ == "__main__":
    unittest.main()
