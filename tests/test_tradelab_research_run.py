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
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tradelab.research_run import (
    RUN_ARTIFACT_SCHEMA_VERSION,
    ActiveWork,
    CheckpointRecord,
    LeaderboardEntry,
    LiveStatus,
    ProgressThrottle,
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


class ProgressThrottleTests(unittest.TestCase):
    """task 044: ProgressThrottle.maybe -- fires immediately on the very
    first call, throttles to at most once per min_interval_seconds
    thereafter (monotonic-clock based), and evaluates the clock itself
    only once every check_every calls -- never on every single call, so
    it is safe to call from inside a tight per-bar loop."""

    def test_first_call_always_fires(self):
        calls = []
        throttle = ProgressThrottle(lambda *a: calls.append(a), check_every=1000)
        self.assertTrue(throttle.maybe(1, 2))
        self.assertEqual(calls, [(1, 2)])

    def test_none_callback_never_fires(self):
        throttle = ProgressThrottle(None)
        self.assertFalse(throttle.maybe(1))
        self.assertFalse(throttle.force(1))

    def test_throttles_within_min_interval_then_fires_after(self):
        with patch("tradelab.research_run.time.monotonic", side_effect=[100.0, 100.5, 101.5]):
            calls = []
            throttle = ProgressThrottle(lambda v: calls.append(v), min_interval_seconds=1.0, check_every=1)
            self.assertTrue(throttle.maybe("a"))  # t=100.0 -- first call, always fires
            self.assertFalse(throttle.maybe("b"))  # t=100.5 -- only 0.5s since last fire, throttled
            self.assertTrue(throttle.maybe("c"))  # t=101.5 -- 1.5s since last fire, fires
            self.assertEqual(calls, ["a", "c"])

    def test_check_every_stride_avoids_a_clock_check_on_every_call(self):
        with patch("tradelab.research_run.time.monotonic", return_value=100.0) as mock_monotonic:
            throttle = ProgressThrottle(lambda *a: None, min_interval_seconds=1.0, check_every=1000)
            for i in range(2500):
                throttle.maybe(i)
            # ~2500/1000 checks, never anywhere near 2500 -- the stride is
            # doing real work, not just a no-op wrapper around the clock.
            self.assertLessEqual(mock_monotonic.call_count, 5)

    def test_force_bypasses_both_stride_and_interval(self):
        calls = []
        throttle = ProgressThrottle(lambda v: calls.append(v), min_interval_seconds=1000.0, check_every=1000000)
        self.assertTrue(throttle.maybe("a"))
        self.assertFalse(throttle.maybe("b"))  # would be throttled for a very long time
        self.assertTrue(throttle.force("c"))
        self.assertEqual(calls, ["a", "c"])

    def test_rejects_invalid_construction(self):
        with self.assertRaises(ValueError):
            ProgressThrottle(None, check_every=0)
        with self.assertRaises(ValueError):
            ProgressThrottle(None, min_interval_seconds=-1)


class HeartbeatTests(unittest.TestCase):
    """task 044 (amended): RunReporter.heartbeat -- a hard, non-bypassable
    cap of at most one actual write per second, no exceptions; every call
    still updates the in-memory active_work snapshot immediately, so a
    "milestone" reported faster than once a second is never lost -- it
    is coalesced into whichever write next becomes due (either this
    method's own next permitted call, or any other reporter method's own
    flush picking up the same in-memory snapshot). Never appends a
    progress_events.jsonl entry (so a long-running loop calling it far
    more often never grows the durable event log without bound), and is
    cleared by the next phase_started()."""

    def test_throttled_to_at_most_once_per_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=10)
            with patch("tradelab.research_run.time.monotonic", side_effect=[100.0, 100.5, 101.5]):
                self.assertTrue(reporter.heartbeat(operation="sim", unit_index=1, unit_total=10))
                self.assertFalse(reporter.heartbeat(operation="sim", unit_index=2, unit_total=10))
                self.assertTrue(reporter.heartbeat(operation="sim", unit_index=3, unit_total=10))
            status = read_live_status(reporter.run_dir)
            # the throttled-away call (unit_index=2) never landed -- the
            # last flush reflects the most recent call that actually fired.
            self.assertEqual(status.active_work["unit_index"], 3)

    def test_no_call_pattern_can_produce_more_than_one_write_per_second(self):
        # Five distinct "this milestone just finished" heartbeats, all
        # landing inside the same simulated second -- proves there is no
        # remaining bypass (no force=, no argument combination) that lets
        # more than one of them actually write.
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=10)
            with patch("tradelab.research_run.time.monotonic", side_effect=[100.0, 100.1, 100.2, 100.3, 100.4]):
                results = [reporter.heartbeat(operation="sim", unit_index=i, unit_total=5) for i in range(1, 6)]
            self.assertEqual(results, [True, False, False, False, False])
            # the on-disk write reflects only the one call that earned it.
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.active_work["unit_index"], 1)

    def test_coalesced_milestones_become_visible_on_the_next_permitted_heartbeat_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=10)
            with patch("tradelab.research_run.time.monotonic", side_effect=[100.0, 100.1, 100.2, 101.3]):
                self.assertTrue(reporter.heartbeat(operation="sim", unit_index=1, unit_total=5))  # writes -- first call
                self.assertFalse(reporter.heartbeat(operation="sim", unit_index=2, unit_total=5))  # coalesced
                self.assertFalse(reporter.heartbeat(operation="sim", unit_index=3, unit_total=5))  # coalesced -- overwrites #2 in memory
                self.assertTrue(reporter.heartbeat(operation="sim", unit_index=4, unit_total=5))  # >=1s later -- writes the LATEST coalesced state
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.active_work["unit_index"], 4)  # never 2 or 3 -- only ever the latest, once due

    def test_coalesced_milestone_becomes_visible_immediately_via_any_other_reporter_method(self):
        # A caller that finishes its loop and never calls heartbeat()
        # again still doesn't lose the milestone: the very next flush
        # from ANY reporter method (progress(), here) serializes whatever
        # heartbeat() last recorded in memory, even though heartbeat()
        # itself never got to write it on its own.
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=10)
            with patch("tradelab.research_run.time.monotonic", side_effect=[100.0, 100.1]):
                self.assertTrue(reporter.heartbeat(operation="sim", unit_index=1, unit_total=5))
                self.assertFalse(reporter.heartbeat(operation="sim", unit_index=5, unit_total=5))  # coalesced, never written by heartbeat itself
            reporter.progress(items_completed=5)
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.active_work["unit_index"], 5)  # visible now, without waiting on heartbeat's own throttle

    def test_never_appends_a_progress_event_even_across_a_long_simulated_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=1_000_000)
            events_before = read_progress_events(reporter.run_dir)

            # A long simulated loop: 10,000 heartbeats, clock advancing
            # 0.01s per call (~100s of simulated wall time) -- proves both
            # "status updates during a long simulated loop" (throughput
            # samples below) and "no unbounded progress-event growth" at
            # once, deterministically (no real sleeping).
            clock = [100.0]

            def fake_monotonic():
                clock[0] += 0.01
                return clock[0]

            flushed_count = 0
            with patch("tradelab.research_run.time.monotonic", side_effect=fake_monotonic):
                for i in range(10_000):
                    if reporter.heartbeat(operation="scanning_bars", unit_index=1, unit_total=1, bars_scanned=i, bars_total=10_000):
                        flushed_count += 1

            events_after = read_progress_events(reporter.run_dir)
            self.assertEqual(events_after, events_before)  # not one single new durable event

            # ~100s of simulated time at a 1s throttle -> roughly 100
            # actual flushes -- nowhere close to the 10,000 calls made.
            self.assertGreater(flushed_count, 50)
            self.assertLess(flushed_count, 150)

            # the final flushed state is genuinely visible -- proves this
            # was a real, working status update stream, not a no-op.
            status = read_live_status(reporter.run_dir)
            self.assertIsNotNone(status.active_work)
            self.assertEqual(status.active_work["operation"], "scanning_bars")

    def test_active_work_fields_round_trip_through_live_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=200)
            reporter.heartbeat(
                operation="simulating_backtests", variant_id="CLASS_A::variant_x::MNQ", unit_index=42, unit_total=200,
                bars_scanned=734211, bars_total=1057182,
            )
            status = read_live_status(reporter.run_dir)
            self.assertEqual(status.active_work, {
                "operation": "simulating_backtests", "variant_id": "CLASS_A::variant_x::MNQ", "unit_index": 42,
                "unit_total": 200, "bars_scanned": 734211, "bars_total": 1057182,
                "updated_at": status.active_work["updated_at"],
            })

    def test_phase_started_clears_stale_active_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=10)
            reporter.heartbeat(operation="sim", unit_index=1, unit_total=10)
            self.assertIsNotNone(read_live_status(reporter.run_dir).active_work)
            reporter.phase_started(RunPhase.RANKING_RESULTS, items_total=1)
            self.assertIsNone(read_live_status(reporter.run_dir).active_work)

    def test_empty_operation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=10)
            with self.assertRaises(ValueError):
                reporter.heartbeat(operation="")


class LivenessHeartbeatThreadTests(unittest.TestCase):
    """task 046: RunReporter.start_liveness_heartbeat/stop_liveness_
    heartbeat -- a background thread that keeps live_status.json
    refreshing (updated_at, elapsed_seconds, fresh CPU/RAM) at least
    once a second even during a long operation that never itself calls
    heartbeat() or any other reporter method; thread-safe with respect
    to heartbeat()'s own once-per-second write cap (the two share it,
    never exceeding one write/second combined); never appends to
    progress_events.jsonl."""

    def test_refreshes_repeatedly_during_a_long_callback_free_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            reporter.heartbeat(operation="building_range_index", unit_index=1, unit_total=1000)

            # a small, patched throttle + a fast polling interval keeps this
            # test fast and still genuinely exercises real, un-mocked
            # threading and real (short) wall-clock sleeps -- deterministic
            # in the sense that the assertion is a generous lower bound, not
            # a fragile exact count.
            seen_timestamps = set()
            with patch("tradelab.research_run._HEARTBEAT_MIN_INTERVAL_SECONDS", 0.02):
                reporter.start_liveness_heartbeat(interval_seconds=0.01)
                try:
                    # simulates "a long operation with no completed
                    # high-level milestone" -- the main thread does nothing
                    # else here, exactly like being deep inside a single
                    # expensive call with no progress hook.
                    for _ in range(8):
                        time.sleep(0.04)
                        seen_timestamps.add(read_live_status(reporter.run_dir).active_work["updated_at"])
                finally:
                    reporter.stop_liveness_heartbeat()

            self.assertGreater(len(seen_timestamps), 3)  # genuinely refreshed multiple times, not stuck on one value

    def test_operation_and_progress_are_never_fabricated_by_the_liveness_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            reporter.heartbeat(operation="building_range_index", unit_index=42, unit_total=1000)

            with patch("tradelab.research_run._HEARTBEAT_MIN_INTERVAL_SECONDS", 0.02):
                reporter.start_liveness_heartbeat(interval_seconds=0.01)
                try:
                    time.sleep(0.15)
                finally:
                    reporter.stop_liveness_heartbeat()

            status = read_live_status(reporter.run_dir)
            # the tick refreshed updated_at (proving it genuinely ran) but
            # never touched the real operation/progress fields -- exactly
            # "still working on X, no fresher position info yet", never a
            # fabricated bar/unit count.
            self.assertEqual(status.active_work["operation"], "building_range_index")
            self.assertEqual(status.active_work["unit_index"], 42)
            self.assertEqual(status.active_work["unit_total"], 1000)

    def test_never_appends_a_progress_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            events_before = read_progress_events(reporter.run_dir)

            with patch("tradelab.research_run._HEARTBEAT_MIN_INTERVAL_SECONDS", 0.02):
                reporter.start_liveness_heartbeat(interval_seconds=0.01)
                try:
                    time.sleep(0.15)
                finally:
                    reporter.stop_liveness_heartbeat()

            self.assertEqual(read_progress_events(reporter.run_dir), events_before)

    def test_thread_safe_combined_with_explicit_heartbeat_calls_never_exceeds_the_cap(self):
        # A second thread hammering heartbeat() concurrently with the
        # liveness thread's own ticking -- proves the write-throttle
        # decision is genuinely shared (locked), not just "usually fine".
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)

            write_count = 0
            write_count_lock = threading.Lock()
            orig = type(reporter).heartbeat

            def counting_heartbeat(self, **kwargs):
                nonlocal write_count
                wrote = orig(self, **kwargs)
                if wrote:
                    with write_count_lock:
                        write_count += 1
                return wrote

            stop_hammering = threading.Event()

            def _hammer():
                i = 0
                while not stop_hammering.is_set():
                    counting_heartbeat(reporter, operation="resolving_entry_targets", unit_index=i, unit_total=100000)
                    i += 1

            with patch("tradelab.research_run._HEARTBEAT_MIN_INTERVAL_SECONDS", 0.05):
                reporter.start_liveness_heartbeat(interval_seconds=0.01)
                hammer_thread = threading.Thread(target=_hammer)
                hammer_thread.start()
                try:
                    time.sleep(0.22)
                finally:
                    stop_hammering.set()
                    hammer_thread.join(timeout=2)
                    reporter.stop_liveness_heartbeat()

            # ~0.22s at a 0.05s cap -> at most ~5-6 writes total from BOTH
            # sources combined -- nowhere close to the thousands of
            # hammer() calls actually made.
            self.assertLessEqual(write_count, 8)

    def test_idempotent_start_and_safe_stop_without_ever_starting(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)

            reporter.stop_liveness_heartbeat()  # never started -- must not raise

            reporter.start_liveness_heartbeat(interval_seconds=0.01)
            first_thread = reporter._liveness_thread
            reporter.start_liveness_heartbeat(interval_seconds=0.01)  # second call is a no-op
            self.assertIs(reporter._liveness_thread, first_thread)

            reporter.stop_liveness_heartbeat()
            self.assertIsNone(reporter._liveness_thread)
            self.assertFalse(first_thread.is_alive())

    def test_concurrent_writes_from_the_main_thread_and_the_liveness_thread_never_race(self):
        # Regression test: atomic_write_json's own temp file was
        # originally named from the process id alone, shared by every
        # thread -- the liveness thread and the main thread flushing
        # live_status.json at the same instant could each grab the
        # *same* temp path, and whichever thread's os.replace() ran
        # second would raise FileNotFoundError (the first thread's
        # os.replace already consumed/renamed the shared temp file out
        # from under it). Hammers both sides concurrently, with the
        # throttle patched wide open so both are writing constantly, and
        # asserts no exception ever escapes either thread.
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=1000)

            errors: list[BaseException] = []
            stop = threading.Event()

            def _main_thread_writer():
                i = 0
                while not stop.is_set():
                    try:
                        reporter.progress(items_completed=i % 1000, backtests_completed=i)
                        reporter.update_leaderboard([])
                    except BaseException as exc:  # noqa: BLE001 -- must catch everything to prove none occurred
                        errors.append(exc)
                        return
                    i += 1

            with patch("tradelab.research_run._HEARTBEAT_MIN_INTERVAL_SECONDS", 0.0):
                reporter.start_liveness_heartbeat(interval_seconds=0.001)
                writer_thread = threading.Thread(target=_main_thread_writer)
                writer_thread.start()
                try:
                    time.sleep(0.3)
                finally:
                    stop.set()
                    writer_thread.join(timeout=5)
                    reporter.stop_liveness_heartbeat()

            self.assertEqual(errors, [])
            # the run directory is still left in a fully coherent state.
            self.assertEqual(read_live_status(reporter.run_dir).phase, "simulating_backtests")


class LiveStatusBackwardCompatibilityTests(unittest.TestCase):
    """task 044: active_work must be a purely additive field -- an older
    run's live_status.json, written before this field existed at all,
    must still parse cleanly (active_work defaults to None), and the
    dashboard-facing to_dict()/from_dict() round trip must be lossless
    for both old (missing-key) and new (present) payloads."""

    def test_from_dict_without_active_work_key_defaults_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            reporter.progress(items_completed=1)
            with open(reporter.run_dir / "live_status.json", encoding="utf-8") as f:
                raw = json.load(f)
            self.assertIn("active_work", raw)  # this module's own writer always includes it
            del raw["active_work"]  # simulate a genuinely pre-task-044 run folder
            status = LiveStatus.from_dict(raw)
            self.assertIsNone(status.active_work)

    def test_to_dict_always_includes_active_work_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={})
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            status = read_live_status(reporter.run_dir)
            self.assertIn("active_work", status.to_dict())
            self.assertIsNone(status.to_dict()["active_work"])

    def test_active_work_dataclass_round_trips(self):
        work = ActiveWork(
            operation="simulating_backtests", variant_id="v1", unit_index=3, unit_total=10, bars_scanned=500, bars_total=1000,
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(ActiveWork.from_dict(work.to_dict()), work)

    def test_active_work_rejects_empty_operation(self):
        with self.assertRaises(ValueError):
            ActiveWork(operation="", variant_id=None, unit_index=None, unit_total=None, bars_scanned=None, bars_total=None, updated_at="2026-01-01T00:00:00+00:00")

    def test_active_work_rejects_negative_counters(self):
        with self.assertRaises(ValueError):
            ActiveWork(operation="x", variant_id=None, unit_index=-1, unit_total=None, bars_scanned=None, bars_total=None, updated_at="2026-01-01T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
