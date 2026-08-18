#!/usr/bin/env python3
"""tradelab.profile_research_context tests: a checksum mismatch on the
market artifact blocks the profile entirely (nothing written); an
invalid news CSV likewise blocks it; a failure during the JSON write
leaves neither a partial output file nor a leftover temp file; two runs
against identical input produce a byte-identical profile; the selector/
direction counting helper never collapses two different directions (or
two different timeframes, via the selector string itself) into one
count; the two explicit 2024/2025 profiling windows are half-open
(``[start, end)``); and every module-level counting helper (FVG,
liquidity, data wicks, bias) reports what it's given without losing
distinctions. No real market data anywhere -- every fixture is a tiny
synthetic CSV/JSON written to a temporary directory."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tradelab.bias import BEARISH as BIAS_BEARISH
from tradelab.bias import BULLISH as BIAS_BULLISH
from tradelab.bias import ConfluenceSource, ConfluenceType, HTFBiasResult, TimeframeBiasChange
from tradelab.data_wicks import DataWickDetectionResult, DataWickDirection, DataWickEvent
from tradelab.entry_signal_timeline import EntrySignal, EntrySignalTimeline, SourceKind
from tradelab.fvg import BEARISH, BULLISH, FVGDetectionResult, FVGEvent, FVGZone, FVGStatus
from tradelab.liquidity import LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType
from tradelab.profile_research_context import (
    PROFILE_WINDOWS,
    _bias_counts,
    _data_wick_counts,
    _fvg_event_counts,
    _fvg_zone_counts,
    _liquidity_level_counts,
    _selector_direction_counts,
    _selector_direction_counts_in_window,
    profile_research_context,
)

UTC = timezone.utc

CONTINUOUS_CSV = (
    "timestamp,open,high,low,close,volume,contract\n"
    "2024-01-02T14:30:00Z,100,101,99,100.5,10,MNQ_CONTINUOUS\n"
    "2024-01-02T14:31:00Z,100.5,101.5,99.5,101,12,MNQ_CONTINUOUS\n"
    "2024-01-02T14:32:00Z,101,110,100,109,12,MNQ_CONTINUOUS\n"
)

NEWS_HEADER = "event_id,scheduled_at,country,currency,event_type,source_agency,source_url,source_reference,time_is_exact,schema_version\n"
NEWS_ROW = "CPI-2024-01,2024-01-02T14:30:00Z,United States,USD,CPI,BLS,https://bls.gov/cpi,ref-1,true,1.0\n"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(tmp: Path, *, csv_path: Path, sha256: str) -> Path:
    manifest = {
        "schema_version": "1.0",
        "source_name": "test-source",
        "instrument_root": "MNQ",
        "bar_interval": "1m",
        "timestamp_timezone": "UTC",
        "series_kind": "continuous",
        "data_path": csv_path.name,
        "coverage_start": "2024-01-02T14:30:00Z",
        "coverage_end": "2024-01-02T14:33:00Z",
        "sha256": sha256,
        "roll_manifest_path": None,
        "normalization_method": "volume_front_month_v1+additive_back_adjustment",
    }
    manifest_path = tmp / "continuous_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def write_valid_fixture(tmp: Path) -> tuple[Path, Path]:
    """Writes a small, internally-consistent continuous CSV + manifest +
    valid news CSV to ``tmp``; returns ``(manifest_path, news_path)``."""
    csv_path = tmp / "mnq_continuous_1m.csv"
    csv_path.write_text(CONTINUOUS_CSV, encoding="utf-8")
    manifest_path = write_manifest(tmp, csv_path=csv_path, sha256=sha256_of(csv_path))

    news_path = tmp / "news.csv"
    news_path.write_text(NEWS_HEADER + NEWS_ROW, encoding="utf-8")
    return manifest_path, news_path


class ChecksumMismatchBlocksProfileTests(unittest.TestCase):
    def test_wrong_sha256_raises_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            csv_path = tmp / "mnq_continuous_1m.csv"
            csv_path.write_text(CONTINUOUS_CSV, encoding="utf-8")
            manifest_path = write_manifest(tmp, csv_path=csv_path, sha256="0" * 64)
            news_path = tmp / "news.csv"
            news_path.write_text(NEWS_HEADER + NEWS_ROW, encoding="utf-8")

            output_json = tmp / "context_profile.json"
            with self.assertRaises(ValueError):
                profile_research_context(manifest_path, news_path, output_json)
            self.assertFalse(output_json.exists())


class InvalidNewsBlocksProfileTests(unittest.TestCase):
    def test_duplicate_event_id_raises_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            duplicate_row = NEWS_ROW.replace("2024-01-02T14:30:00Z", "2024-02-02T14:30:00Z")
            news_path.write_text(NEWS_HEADER + NEWS_ROW + duplicate_row, encoding="utf-8")

            output_json = tmp / "context_profile.json"
            with self.assertRaises(ValueError):
                profile_research_context(manifest_path, news_path, output_json)
            self.assertFalse(output_json.exists())

    def test_unknown_event_type_raises_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            news_path.write_text(NEWS_HEADER + NEWS_ROW.replace(",CPI,", ",NOT_A_REAL_EVENT,"), encoding="utf-8")

            output_json = tmp / "context_profile.json"
            with self.assertRaises(ValueError):
                profile_research_context(manifest_path, news_path, output_json)
            self.assertFalse(output_json.exists())


class AtomicWriteTests(unittest.TestCase):
    def test_failure_during_write_leaves_no_partial_output_and_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_json = tmp / "context_profile.json"

            with mock.patch("tradelab.profile_research_context.json.dumps", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    profile_research_context(manifest_path, news_path, output_json)

            self.assertFalse(output_json.exists())
            leftover = list(tmp.glob(".context_profile.json.tmp"))
            self.assertEqual(leftover, [])

    def test_successful_run_creates_exactly_the_output_file(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_json = tmp / "context_profile.json"

            profile_research_context(manifest_path, news_path, output_json)
            self.assertTrue(output_json.exists())
            self.assertEqual(list(tmp.glob(".*.tmp")), [])


class DeterminismTests(unittest.TestCase):
    def test_two_runs_produce_byte_identical_profiles(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)

            output_a = tmp / "profile_a.json"
            output_b = tmp / "profile_b.json"
            profile_research_context(manifest_path, news_path, output_a)
            profile_research_context(manifest_path, news_path, output_b)

            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())

    def test_profile_result_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_json = tmp / "context_profile.json"

            result_a = profile_research_context(manifest_path, news_path, output_json)
            result_b = profile_research_context(manifest_path, news_path, output_json)
            self.assertEqual(result_a, result_b)


def make_signal(selector: str, direction: str, known_at: datetime, timeframe: str = "1m") -> EntrySignal:
    return EntrySignal(
        candidate_id=f"{selector}-{direction}-{known_at.isoformat()}",
        selector=selector,
        source_kind=SourceKind.FVG,
        timeframe=timeframe,
        direction=direction,
        timestamp=known_at,
        known_at=known_at,
        trading_day_id="2024-01-01",
        valid_until=known_at + timedelta(days=1),
        source=None,
    )


def make_timeline(signals: list[EntrySignal]) -> EntrySignalTimeline:
    by_selector: dict[str, list[EntrySignal]] = {}
    for signal in signals:
        by_selector.setdefault(signal.selector, []).append(signal)
    return EntrySignalTimeline(signals=list(signals), signals_by_selector=by_selector)


class SelectorStatsPreserveDirectionAndTimeframeTests(unittest.TestCase):
    def test_two_directions_on_the_same_selector_are_not_collapsed(self):
        t0 = datetime(2024, 6, 1, tzinfo=UTC)
        timeline = make_timeline(
            [
                make_signal("fvg-1m-fvg_touch", "long", t0),
                make_signal("fvg-1m-fvg_touch", "long", t0 + timedelta(minutes=1)),
                make_signal("fvg-1m-fvg_touch", "short", t0 + timedelta(minutes=2)),
            ]
        )
        counts = _selector_direction_counts(timeline)
        by_key = {(row["selector"], row["direction"]): row["count"] for row in counts}
        self.assertEqual(by_key[("fvg-1m-fvg_touch", "long")], 2)
        self.assertEqual(by_key[("fvg-1m-fvg_touch", "short")], 1)

    def test_two_timeframes_on_the_same_source_type_are_not_collapsed(self):
        t0 = datetime(2024, 6, 1, tzinfo=UTC)
        timeline = make_timeline(
            [
                make_signal("fvg-1m-fvg_touch", "long", t0, timeframe="1m"),
                make_signal("fvg-5m-fvg_touch", "long", t0, timeframe="5m"),
            ]
        )
        counts = _selector_direction_counts(timeline)
        selectors = {row["selector"] for row in counts}
        self.assertEqual(selectors, {"fvg-1m-fvg_touch", "fvg-5m-fvg_touch"})
        by_key = {(row["selector"], row["direction"]): row["count"] for row in counts}
        self.assertEqual(by_key[("fvg-1m-fvg_touch", "long")], 1)
        self.assertEqual(by_key[("fvg-5m-fvg_touch", "long")], 1)

    def test_result_is_sorted_deterministically(self):
        t0 = datetime(2024, 6, 1, tzinfo=UTC)
        timeline = make_timeline(
            [
                make_signal("fvg-5m-fvg_touch", "short", t0),
                make_signal("fvg-1m-fvg_touch", "long", t0),
                make_signal("fvg-1m-fvg_touch", "short", t0),
            ]
        )
        counts = _selector_direction_counts(timeline)
        keys = [(row["selector"], row["direction"]) for row in counts]
        self.assertEqual(keys, sorted(keys))


class ExplicitWindowsHalfOpenTests(unittest.TestCase):
    def setUp(self):
        self.window_2024 = next(w for w in PROFILE_WINDOWS if w.window_id == "2024")
        self.window_2025 = next(w for w in PROFILE_WINDOWS if w.window_id == "2025")

    def test_window_boundaries_match_the_literal_spec(self):
        self.assertEqual(self.window_2024.start_utc, datetime(2024, 1, 1, tzinfo=UTC))
        self.assertEqual(self.window_2024.end_utc, datetime(2025, 1, 1, tzinfo=UTC))
        self.assertEqual(self.window_2025.start_utc, datetime(2025, 1, 1, tzinfo=UTC))
        self.assertEqual(self.window_2025.end_utc, datetime(2026, 1, 1, tzinfo=UTC))

    def test_instant_at_2025_boundary_counts_in_2025_not_2024(self):
        boundary = datetime(2025, 1, 1, tzinfo=UTC)
        timeline = make_timeline([make_signal("fvg-1m-fvg_touch", "long", boundary)])

        counts_2024 = _selector_direction_counts_in_window(timeline, self.window_2024)
        counts_2025 = _selector_direction_counts_in_window(timeline, self.window_2025)
        self.assertEqual(counts_2024, [])
        self.assertEqual(counts_2025, [{"selector": "fvg-1m-fvg_touch", "direction": "long", "count": 1}])

    def test_instant_one_minute_before_2025_boundary_counts_in_2024(self):
        just_before = datetime(2025, 1, 1, tzinfo=UTC) - timedelta(minutes=1)
        timeline = make_timeline([make_signal("fvg-1m-fvg_touch", "long", just_before)])

        counts_2024 = _selector_direction_counts_in_window(timeline, self.window_2024)
        counts_2025 = _selector_direction_counts_in_window(timeline, self.window_2025)
        self.assertEqual(counts_2024, [{"selector": "fvg-1m-fvg_touch", "direction": "long", "count": 1}])
        self.assertEqual(counts_2025, [])

    def test_instant_at_2026_boundary_counts_in_neither_window(self):
        boundary_2026 = datetime(2026, 1, 1, tzinfo=UTC)
        timeline = make_timeline([make_signal("fvg-1m-fvg_touch", "long", boundary_2026)])

        counts_2024 = _selector_direction_counts_in_window(timeline, self.window_2024)
        counts_2025 = _selector_direction_counts_in_window(timeline, self.window_2025)
        self.assertEqual(counts_2024, [])
        self.assertEqual(counts_2025, [])

    def test_instant_at_2024_start_counts_in_2024(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        timeline = make_timeline([make_signal("fvg-1m-fvg_touch", "short", start)])
        counts_2024 = _selector_direction_counts_in_window(timeline, self.window_2024)
        self.assertEqual(counts_2024, [{"selector": "fvg-1m-fvg_touch", "direction": "short", "count": 1}])


def zone(zone_id: str, timeframe: str, direction: str, status: FVGStatus = FVGStatus.ACTIVE) -> FVGZone:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return FVGZone(
        zone_id=zone_id,
        timeframe=timeframe,
        direction=direction,
        lower=100.0,
        upper=101.5,
        candle_1_timestamp=t0,
        candle_2_timestamp=t0 + timedelta(minutes=1),
        candle_3_timestamp=t0 + timedelta(minutes=2),
        formed_at=t0 + timedelta(minutes=3),
        status=status,
        resolved_at=None,
        produced_ifvg=False,
    )


def ifvg_event(zone_id: str, timeframe: str, direction: str) -> FVGEvent:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return FVGEvent(
        zone_id=zone_id,
        timeframe=timeframe,
        direction=direction,
        original_zone_direction=BEARISH if direction == BULLISH else BULLISH,
        lower=100.0,
        upper=101.5,
        timestamp=t0 + timedelta(minutes=4),
        known_at=t0 + timedelta(minutes=4),
    )


class FVGCountingHelperTests(unittest.TestCase):
    def test_zone_counts_split_by_direction_and_timeframe(self):
        fvg_results = {
            "1m": FVGDetectionResult(zones=[zone("1m-FVG-1", "1m", BULLISH), zone("1m-FVG-2", "1m", BULLISH), zone("1m-FVG-3", "1m", BEARISH)], events=[]),
            "5m": FVGDetectionResult(zones=[zone("5m-FVG-1", "5m", BEARISH)], events=[]),
        }
        counts = _fvg_zone_counts(fvg_results)
        self.assertEqual(counts["1m"], {"bullish": 2, "bearish": 1})
        self.assertEqual(counts["5m"], {"bullish": 0, "bearish": 1})

    def test_event_counts_track_ifvg_direction_independently_of_zones(self):
        fvg_results = {
            "1m": FVGDetectionResult(zones=[], events=[ifvg_event("1m-FVG-1", "1m", BEARISH), ifvg_event("1m-FVG-2", "1m", BEARISH), ifvg_event("1m-FVG-3", "1m", BULLISH)]),
        }
        counts = _fvg_event_counts(fvg_results)
        self.assertEqual(counts["1m"], {"bullish": 1, "bearish": 2})


def liquidity_level(level_id: str, timeframe: str, liquidity_type: LiquidityType) -> LiquidityLevel:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return LiquidityLevel(
        level_id=level_id,
        timeframe=timeframe,
        liquidity_type=liquidity_type,
        side="high" if "high" in liquidity_type.value else "low",
        lower=100.0,
        upper=100.0,
        member_count=1,
        member_timestamps=(t0,),
        formed_at=t0,
        status=LiquidityStatus.ACTIVE,
        taken_at=None,
        absorbed_by_level_id=None,
    )


class LiquidityCountingHelperTests(unittest.TestCase):
    def test_swing_and_session_levels_are_kept_separate_by_family_timeframe_and_type(self):
        swing_equal = {
            "1m": LiquidityDetectionResult(
                levels=[
                    liquidity_level("l1", "1m", LiquidityType.SWING_HIGH),
                    liquidity_level("l2", "1m", LiquidityType.SWING_HIGH),
                    liquidity_level("l3", "1m", LiquidityType.SWING_LOW),
                ],
                takes=[],
                reclaims=[],
            )
        }
        session = {
            "1m": LiquidityDetectionResult(levels=[liquidity_level("l4", "1m", LiquidityType.SESSION_ASIA_HIGH)], takes=[], reclaims=[])
        }
        counts = _liquidity_level_counts(swing_equal, session)
        self.assertEqual(counts["swing_equal"]["1m"], {"swing_high": 2, "swing_low": 1})
        self.assertEqual(counts["session"]["1m"], {"session_asia_high": 1})


class DataWickCountingHelperTests(unittest.TestCase):
    def test_upper_and_lower_counted_independently(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)

        def wick(direction: DataWickDirection) -> DataWickEvent:
            return DataWickEvent(
                event_ids=("e1",),
                direction=direction,
                timestamp=t0,
                known_at=t0,
                level=100.0,
                candle_range=5.0,
                baseline_median=1.0,
                wick_length=3.0,
            )

        result = DataWickDetectionResult(events=[wick(DataWickDirection.UPPER), wick(DataWickDirection.UPPER), wick(DataWickDirection.LOWER)])
        counts = _data_wick_counts(result)
        self.assertEqual(counts, {"upper": 2, "lower": 1, "total": 3})


class BiasCountingHelperTests(unittest.TestCase):
    def test_state_distribution_and_change_count_per_timeframe(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)

        def change(timeframe: str, state: str, known_at: datetime) -> TimeframeBiasChange:
            source = ConfluenceSource(ConfluenceType.FVG_FORMATION, BIAS_BULLISH if state == BIAS_BULLISH else BIAS_BEARISH, "z1", known_at)
            return TimeframeBiasChange(timeframe=timeframe, known_at=known_at, state=state, sources=(source,))

        result = HTFBiasResult(
            timeframe_changes={
                "1H": [change("1H", BIAS_BULLISH, t0), change("1H", "neutral", t0 + timedelta(hours=1)), change("1H", BIAS_BEARISH, t0 + timedelta(hours=2))],
                "4H": [],
                "1D": [],
            }
        )
        counts = _bias_counts(result)
        self.assertEqual(counts["1H"]["change_count"], 3)
        self.assertEqual(counts["1H"]["state_distribution"], {"bullish": 1, "bearish": 1, "neutral": 1})
        self.assertEqual(counts["4H"], {"change_count": 0, "state_distribution": {"bullish": 0, "bearish": 0, "neutral": 0}})


class NoRealDataInRepoTests(unittest.TestCase):
    def test_gitignore_covers_generated_profile_and_market_artifact_files(self):
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        content = gitignore.read_text(encoding="utf-8")
        for pattern in ("context_profile.json", "mnq_continuous_1m.csv", "continuous_artifact_manifest.json"):
            self.assertIn(pattern, content, f"{pattern!r} should be gitignored so a locally-run profile can never be committed by accident")

    def test_no_generated_artifact_files_are_tracked_by_git(self):
        import subprocess

        repo_root = Path(__file__).resolve().parent.parent
        tracked = subprocess.run(["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.splitlines()
        offenders = [f for f in tracked if f.endswith("context_profile.json") or f.endswith("mnq_continuous_1m.csv")]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
