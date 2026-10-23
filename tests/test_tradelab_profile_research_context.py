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
from tradelab.entry_signal_timeline import (
    EntrySignal,
    EntrySignalTimeline,
    SourceKind,
    build_entry_signal_timeline,
)
from tradelab.entry_signal_timeline import EntryEventKind
from tradelab.fvg import BEARISH, BULLISH, FVGDetectionResult, FVGEvent, FVGMitigationEvent, FVGZone, FVGStatus
from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, FVGEntryCandidateResult
from tradelab.fvg_entry import SourceType as FVGCandidateSourceType
from tradelab.liquidity import LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType
from tradelab.liquidity_entry import CandidatePhase, EntryMode, LiquidityEntryCandidate, LiquidityEntryCandidateResult, SourceFamily
from tradelab.profile_research_context import (
    PROFILE_WINDOWS,
    _active_pool_stats,
    _bias_counts,
    _comparison_section,
    _compare_raw_signal_counts,
    _data_wick_counts,
    _entry_event_model_section,
    _entry_signal_timeline_stats,
    _fvg_event_counts,
    _fvg_ifvg_candidate_counts,
    _fvg_mitigation_counts,
    _fvg_touch_candidate_counts,
    _fvg_zone_counts,
    _git_provenance,
    _is_immediate_resolution_restoration,
    _liquidity_level_counts,
    _percentile,
    _provenance_section,
    _selector_direction_counts,
    _selector_direction_counts_in_window,
    _stat_summary,
    _sweep_fvg_setup_model_section,
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


def zone(
    zone_id: str, timeframe: str, direction: str, status: FVGStatus = FVGStatus.ACTIVE, formed_at: datetime | None = None
) -> FVGZone:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    resolved_formed_at = formed_at if formed_at is not None else t0 + timedelta(minutes=3)
    return FVGZone(
        zone_id=zone_id,
        timeframe=timeframe,
        direction=direction,
        lower=100.0,
        upper=101.5,
        candle_1_timestamp=resolved_formed_at - timedelta(minutes=3),
        candle_2_timestamp=resolved_formed_at - timedelta(minutes=2),
        candle_3_timestamp=resolved_formed_at - timedelta(minutes=1),
        formed_at=resolved_formed_at,
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


def fvg_touch_candidate(
    zone_id: str, timeframe: str, direction: str, contact_type: FVGContactType, known_at: datetime
) -> FVGEntryCandidate:
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{FVGCandidateSourceType.FVG_TOUCH.value}-{zone_id}",
        source_type=FVGCandidateSourceType.FVG_TOUCH,
        direction=direction,
        timeframe=timeframe,
        zone_id=zone_id,
        lower=100.0,
        upper=101.5,
        contact_type=contact_type,
        timestamp=known_at,
        known_at=known_at,
    )


def fvg_ifvg_candidate(zone_id: str, timeframe: str, direction: str, known_at: datetime) -> FVGEntryCandidate:
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{FVGCandidateSourceType.IFVG.value}-{zone_id}",
        source_type=FVGCandidateSourceType.IFVG,
        direction=direction,
        timeframe=timeframe,
        zone_id=zone_id,
        lower=100.0,
        upper=101.5,
        contact_type=None,
        timestamp=known_at,
        known_at=known_at,
    )


def fvg_mitigation(zone_id: str, timeframe: str, direction: str, known_at: datetime) -> FVGMitigationEvent:
    return FVGMitigationEvent(
        zone_id=zone_id, timeframe=timeframe, direction=direction, lower=100.0, upper=101.5, timestamp=known_at, known_at=known_at
    )


class FVGEntryCandidateCountingHelperTests(unittest.TestCase):
    def test_touch_counts_split_by_timeframe_direction_and_contact_type(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        result = FVGEntryCandidateResult(
            candidates=[
                fvg_touch_candidate("z1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0),
                fvg_touch_candidate("z2", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0),
                fvg_touch_candidate("z3", "1m", "short", FVGContactType.FULL_MITIGATION_NO_INVERSION, t0),
                fvg_ifvg_candidate("z4", "1m", "long", t0),
            ],
        )
        counts = _fvg_touch_candidate_counts(result)
        self.assertEqual(counts["1m"]["long"], {"partial_touch": 2})
        self.assertEqual(counts["1m"]["short"], {"full_mitigation_no_inversion": 1})

    def test_ifvg_counts_split_by_timeframe_and_direction(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        result = FVGEntryCandidateResult(
            candidates=[
                fvg_ifvg_candidate("z1", "1m", "long", t0),
                fvg_ifvg_candidate("z2", "1m", "long", t0),
                fvg_ifvg_candidate("z3", "5m", "short", t0),
                fvg_touch_candidate("z4", "1m", "short", FVGContactType.PARTIAL_TOUCH, t0),
            ],
        )
        counts = _fvg_ifvg_candidate_counts(result)
        self.assertEqual(counts["1m"], {"long": 2})
        self.assertEqual(counts["5m"], {"short": 1})

    def test_mitigation_counts_split_by_timeframe_and_direction(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        result = FVGEntryCandidateResult(
            candidates=[],
            zone_mitigations=(
                fvg_mitigation("z1", "1m", "long", t0),
                fvg_mitigation("z2", "1m", "long", t0),
                fvg_mitigation("z3", "1m", "short", t0),
            ),
        )
        counts = _fvg_mitigation_counts(result)
        self.assertEqual(counts["1m"], {"long": 2, "short": 1})


class PercentileHelperTests(unittest.TestCase):
    def test_single_value_returns_itself_for_any_fraction(self):
        self.assertEqual(_percentile([42.0], 0.5), 42.0)
        self.assertEqual(_percentile([42.0], 0.9), 42.0)

    def test_median_of_odd_count_is_the_middle_value(self):
        self.assertEqual(_percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_interpolates_linearly_between_closest_ranks(self):
        self.assertEqual(_percentile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(_percentile([0.0, 10.0], 0.9), 9.0)

    def test_endpoints_return_min_and_max(self):
        values = [1.0, 5.0, 9.0, 20.0]
        self.assertEqual(_percentile(values, 0.0), 1.0)
        self.assertEqual(_percentile(values, 1.0), 20.0)


def build_fvg_only_timeline(candidates: list[FVGEntryCandidate], mitigations: tuple[FVGMitigationEvent, ...] = ()) -> EntrySignalTimeline:
    fvg_result = FVGEntryCandidateResult(candidates=candidates, zone_mitigations=mitigations)
    liquidity_result = LiquidityEntryCandidateResult(candidates=[])
    return build_entry_signal_timeline(fvg_result, liquidity_result)


class EntrySignalTimelineStatsTests(unittest.TestCase):
    def test_no_fvg_signals_reports_all_zero_and_no_duration_stats(self):
        timeline = build_fvg_only_timeline([])
        stats = _entry_signal_timeline_stats(timeline)
        self.assertEqual(stats["suspension_count_total"], 0)
        self.assertEqual(stats["restoration_count_total"], 0)
        self.assertEqual(stats["still_suspended_at_end_count"], 0)
        duration_stats = stats["active_fvg_confluence_duration_stats_minutes"]
        self.assertEqual(duration_stats["completed_count"], 0)
        self.assertEqual(duration_stats["never_suspended_count"], 0)
        self.assertIsNone(duration_stats["min"])
        self.assertIsNone(duration_stats["max"])
        self.assertIsNone(duration_stats["mean"])

    def test_never_suspended_touch_counts_as_never_suspended_not_completed(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        timeline = build_fvg_only_timeline([fvg_touch_candidate("z1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)])
        stats = _entry_signal_timeline_stats(timeline)
        self.assertEqual(stats["suspension_count_total"], 0)
        duration_stats = stats["active_fvg_confluence_duration_stats_minutes"]
        self.assertEqual(duration_stats["completed_count"], 0)
        self.assertEqual(duration_stats["never_suspended_count"], 1)

    def test_suspension_and_restoration_are_counted_and_duration_measures_time_to_first_suspension(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        bull_touch = fvg_touch_candidate("bull-1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        bear_touch = fvg_touch_candidate("bear-1", "1m", "short", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=10))
        bear_ifvg = fvg_ifvg_candidate("bear-1", "1m", "long", t0 + timedelta(minutes=25))
        timeline = build_fvg_only_timeline([bull_touch, bear_touch, bear_ifvg])

        stats = _entry_signal_timeline_stats(timeline)
        self.assertEqual(stats["suspension_count_total"], 1)
        self.assertEqual(stats["restoration_count_total"], 1)
        self.assertEqual(stats["still_suspended_at_end_count"], 0)

        duration_stats = stats["active_fvg_confluence_duration_stats_minutes"]
        self.assertEqual(duration_stats["completed_count"], 1)
        self.assertEqual(duration_stats["never_suspended_count"], 2)  # bear_touch and bear_ifvg were never themselves suspended
        self.assertEqual(duration_stats["min"], 10.0)
        self.assertEqual(duration_stats["max"], 10.0)
        self.assertEqual(duration_stats["median"], 10.0)

    def test_still_suspended_at_end_is_never_restored_and_has_no_duration(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        bull_touch = fvg_touch_candidate("bull-1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        bear_touch = fvg_touch_candidate("bear-1", "1m", "short", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=5))
        timeline = build_fvg_only_timeline([bull_touch, bear_touch])

        stats = _entry_signal_timeline_stats(timeline)
        self.assertEqual(stats["suspension_count_total"], 1)
        self.assertEqual(stats["restoration_count_total"], 0)
        self.assertEqual(stats["still_suspended_at_end_count"], 1)
        duration_stats = stats["active_fvg_confluence_duration_stats_minutes"]
        self.assertEqual(duration_stats["completed_count"], 1)
        self.assertEqual(duration_stats["min"], 5.0)

    def test_mitigation_event_also_restores_and_is_reflected_in_counts(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        bull_touch = fvg_touch_candidate("bull-1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        bear_touch = fvg_touch_candidate("bear-1", "1m", "short", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=1))
        bear_mitigation = fvg_mitigation("bear-1", "1m", "short", t0 + timedelta(minutes=8))
        timeline = build_fvg_only_timeline([bull_touch, bear_touch], mitigations=(bear_mitigation,))

        stats = _entry_signal_timeline_stats(timeline)
        self.assertEqual(stats["suspension_count_total"], 1)
        self.assertEqual(stats["restoration_count_total"], 1)
        self.assertEqual(stats["still_suspended_at_end_count"], 0)

    def test_immediate_full_mitigation_challenge_never_suspends_and_is_excluded_from_still_suspended_math(self):
        """A FULL_MITIGATION_NO_INVERSION-typed touch challenges the
        opposite pool but, per tradelab.entry_signal_timeline._apply_
        fvg_batch step 2, its own zone is already resolved the instant
        it is known -- the target is never actually removed from
        active, yet a SignalRestoration is still recorded for audit
        transparency. suspension_count_total must stay 0 here (nothing
        was ever suspended), restoration_count_total must be 1 (the
        audit-only record), and still_suspended_at_end_count must use
        only the non-immediate-resolution subset -- never go negative
        or otherwise misrepresent a real suspend/restore pairing that
        never happened."""
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        bull_touch = fvg_touch_candidate("bull-1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        bear_full_mitigation_touch = fvg_touch_candidate(
            "bear-1", "1m", "short", FVGContactType.FULL_MITIGATION_NO_INVERSION, t0 + timedelta(minutes=5)
        )
        timeline = build_fvg_only_timeline([bull_touch, bear_full_mitigation_touch])

        stats = _entry_signal_timeline_stats(timeline)
        self.assertEqual(stats["suspension_count_total"], 0)
        self.assertEqual(stats["restoration_count_total"], 1)
        self.assertEqual(stats["restoration_of_suspended_signal_count_total"], 0)
        self.assertEqual(stats["immediate_resolution_restoration_count_total"], 1)
        self.assertEqual(stats["still_suspended_at_end_count"], 0)
        duration_stats = stats["active_fvg_confluence_duration_stats_minutes"]
        self.assertEqual(duration_stats["completed_count"], 0)
        self.assertEqual(duration_stats["never_suspended_count"], 2)

    def test_real_suspension_and_immediate_resolution_restoration_do_not_cross_contaminate_counts(self):
        """A genuine suspend/restore pair (via a later IFVG) coexists
        with an unrelated immediate-resolution restoration on a second,
        independent zone -- the still_suspended_at_end_count math must
        only ever consult the first pair, never be thrown off by the
        second, audit-only restoration."""
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        bull_touch = fvg_touch_candidate("bull-1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        bear_touch = fvg_touch_candidate("bear-1", "1m", "short", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=1))
        bear_ifvg = fvg_ifvg_candidate("bear-1", "1m", "long", t0 + timedelta(minutes=2))
        # Independent zone on a different timeframe -- never interacts with the pair above.
        long5m_touch = fvg_touch_candidate("bull5-1", "5m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        short5m_full_mitigation = fvg_touch_candidate(
            "bear5-1", "5m", "short", FVGContactType.FULL_MITIGATION_NO_INVERSION, t0 + timedelta(minutes=3)
        )
        timeline = build_fvg_only_timeline([bull_touch, bear_touch, bear_ifvg, long5m_touch, short5m_full_mitigation])

        stats = _entry_signal_timeline_stats(timeline)
        self.assertEqual(stats["suspension_count_total"], 1)
        self.assertEqual(stats["restoration_count_total"], 2)
        self.assertEqual(stats["restoration_of_suspended_signal_count_total"], 1)
        self.assertEqual(stats["immediate_resolution_restoration_count_total"], 1)
        self.assertEqual(stats["still_suspended_at_end_count"], 0)


class StatSummaryTests(unittest.TestCase):
    def test_empty_values_report_zero_count_and_none_stats(self):
        self.assertEqual(_stat_summary([]), {"count": 0, "min": None, "mean": None, "median": None, "p90": None, "max": None})

    def test_known_values_produce_exact_deterministic_percentiles(self):
        summary = _stat_summary([1, 2, 3, 4, 5])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["min"], 1)
        self.assertEqual(summary["max"], 5)
        self.assertEqual(summary["mean"], 3.0)
        self.assertEqual(summary["median"], 3)
        # rank = 0.9 * 4 = 3.6 -> interpolate between sorted[3]=4 and sorted[4]=5
        self.assertAlmostEqual(summary["p90"], 4.6)

    def test_input_order_never_affects_output(self):
        self.assertEqual(_stat_summary([5, 1, 4, 2, 3]), _stat_summary([1, 2, 3, 4, 5]))


class ActivePoolStatsTests(unittest.TestCase):
    def test_empty_timeline_reports_zero_snapshots_everywhere(self):
        stats = _active_pool_stats(build_fvg_only_timeline([]))
        self.assertEqual(stats["global"]["snapshot_count"], 0)
        self.assertEqual(stats["global"]["active_selector_count_including_primary"]["count"], 0)
        self.assertEqual(stats["by_direction"], {})
        self.assertEqual(stats["by_primary_selector_direction"], [])

    def test_overlapping_same_selector_signals_dedupe_to_one_unique_selector(self):
        """Two FVG_TOUCH candidates on the exact same timeframe/direction
        (hence the exact same selector) that are both simultaneously
        active at the second one's own snapshot must count as ONE
        active selector, never two -- while active_signal_instance_count
        still reflects both individual instances, so the two metrics can
        never be confused."""
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        first = fvg_touch_candidate("z1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        second = fvg_touch_candidate("z2", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=1))
        timeline = build_fvg_only_timeline([first, second])

        stats = _active_pool_stats(timeline)
        selector_entry = next(
            row for row in stats["by_primary_selector_direction"] if row["selector"] == "fvg-1m-fvg_touch" and row["direction"] == "long"
        )
        self.assertEqual(selector_entry["snapshot_count"], 2)
        # first's own snapshot: only itself active -> 1 selector, 1 instance.
        # second's own snapshot: both first and second active, same selector -> still 1 selector, 2 instances.
        self.assertEqual(selector_entry["active_selector_count_including_primary"], {"count": 2, "min": 1, "mean": 1.0, "median": 1.0, "p90": 1.0, "max": 1})
        self.assertEqual(selector_entry["active_signal_instance_count"]["min"], 1)
        self.assertEqual(selector_entry["active_signal_instance_count"]["max"], 2)
        self.assertEqual(selector_entry["required_confluence_selector_count_excluding_primary"]["max"], 0)

    def test_required_confluence_excludes_only_the_primarys_own_selector(self):
        """A second, different selector active alongside the primary
        must be counted as one required-confluence selector; the
        primary's own selector must never count against itself."""
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        long_1m = fvg_touch_candidate("z1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        long_5m = fvg_touch_candidate("z2", "5m", "long", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=1))
        timeline = build_fvg_only_timeline([long_1m, long_5m])

        stats = _active_pool_stats(timeline)
        primary_entry = next(
            row for row in stats["by_primary_selector_direction"] if row["selector"] == "fvg-5m-fvg_touch" and row["direction"] == "long"
        )
        # At long_5m's own snapshot: active selectors = {fvg-1m-fvg_touch, fvg-5m-fvg_touch} = 2,
        # required confluence excludes fvg-5m-fvg_touch itself -> only fvg-1m-fvg_touch = 1.
        self.assertEqual(primary_entry["active_selector_count_including_primary"]["max"], 2)
        self.assertEqual(primary_entry["required_confluence_selector_count_excluding_primary"]["max"], 1)

        first_entry = next(
            row for row in stats["by_primary_selector_direction"] if row["selector"] == "fvg-1m-fvg_touch" and row["direction"] == "long"
        )
        # long_1m is the very first signal known -- nothing else active yet.
        self.assertEqual(first_entry["active_selector_count_including_primary"]["max"], 1)
        self.assertEqual(first_entry["required_confluence_selector_count_excluding_primary"]["max"], 0)

    def test_directions_are_never_mixed_in_any_grouping(self):
        """A short-direction primary's own snapshot must never include a
        long-direction selector, even one active on a different
        timeframe at an earlier known_at -- and by_direction/by_primary_
        selector_direction must reflect that isolation, never a
        cross-direction pool."""
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        long_1m = fvg_touch_candidate("z1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        long_5m = fvg_touch_candidate("z2", "5m", "long", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=1))
        short_15m = fvg_touch_candidate("z3", "15m", "short", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=2))
        timeline = build_fvg_only_timeline([long_1m, long_5m, short_15m])

        stats = _active_pool_stats(timeline)
        self.assertEqual(set(stats["by_direction"].keys()), {"long", "short"})
        self.assertEqual(stats["by_direction"]["long"]["snapshot_count"], 2)
        self.assertEqual(stats["by_direction"]["short"]["snapshot_count"], 1)
        # The short primary sees only its own selector -- never the two long ones active before it.
        short_entry = next(row for row in stats["by_primary_selector_direction"] if row["direction"] == "short")
        self.assertEqual(short_entry["active_selector_count_including_primary"]["max"], 1)
        self.assertEqual(short_entry["required_confluence_selector_count_excluding_primary"]["max"], 0)
        self.assertEqual(short_entry["active_signal_instance_count"]["max"], 1)
        # The long side, unaffected by the later short signal, still reaches 2 unique selectors.
        self.assertEqual(stats["by_direction"]["long"]["active_selector_count_including_primary"]["max"], 2)

    def test_deterministic_percentile_output_across_five_snapshots_of_the_same_selector(self):
        """Five distinct selectors, each activated one at a time on the
        same direction, so the Nth snapshot always sees exactly N unique
        active selectors -- an exact, hand-computable global percentile
        curve: [1, 2, 3, 4, 5]."""
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        timeframes = ["1m", "5m", "15m", "1H", "4H"]
        candidates = [
            fvg_touch_candidate(f"z{i}", tf, "long", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=i))
            for i, tf in enumerate(timeframes)
        ]
        timeline = build_fvg_only_timeline(candidates)

        stats = _active_pool_stats(timeline)
        summary = stats["global"]["active_selector_count_including_primary"]
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["min"], 1)
        self.assertEqual(summary["max"], 5)
        self.assertEqual(summary["mean"], 3.0)
        self.assertEqual(summary["median"], 3)
        self.assertAlmostEqual(summary["p90"], 4.6)

    def test_note_states_selector_count_not_instance_count_is_tractability_relevant(self):
        stats = _active_pool_stats(build_fvg_only_timeline([]))
        note = stats["note"]
        self.assertIn("Selector count", note)
        self.assertIn("not raw signal-instance count", note)
        self.assertIn("tradelab.observed_strategy_universe", note)


class IsImmediateResolutionRestorationTests(unittest.TestCase):
    def test_restored_by_fvg_touch_candidate_is_immediate_resolution(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        signal = EntrySignal(
            candidate_id="c1",
            selector="fvg-1m-fvg_touch",
            source_kind=SourceKind.FVG,
            timeframe="1m",
            direction="short",
            timestamp=t0,
            known_at=t0,
            trading_day_id="2024-01-01",
            source=fvg_touch_candidate("z1", "1m", "short", FVGContactType.FULL_MITIGATION_NO_INVERSION, t0),
        )
        restoration = mock.Mock(restored_by=signal)
        self.assertTrue(_is_immediate_resolution_restoration(restoration))

    def test_restored_by_ifvg_candidate_is_not_immediate_resolution(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        signal = EntrySignal(
            candidate_id="c1",
            selector="fvg-1m-ifvg",
            source_kind=SourceKind.FVG,
            timeframe="1m",
            direction="long",
            timestamp=t0,
            known_at=t0,
            trading_day_id="2024-01-01",
            source=fvg_ifvg_candidate("z1", "1m", "long", t0),
        )
        restoration = mock.Mock(restored_by=signal)
        self.assertFalse(_is_immediate_resolution_restoration(restoration))

    def test_restored_by_mitigation_event_is_not_immediate_resolution(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        restoration = mock.Mock(restored_by=fvg_mitigation("z1", "1m", "short", t0))
        self.assertFalse(_is_immediate_resolution_restoration(restoration))


class ProvenanceSectionTests(unittest.TestCase):
    def test_git_provenance_returns_none_pair_when_subprocess_fails(self):
        with mock.patch("tradelab.profile_research_context.subprocess.run", side_effect=OSError("git not found")):
            info = _git_provenance()
        self.assertEqual(info, {"git_commit": None, "git_commit_dirty": None})

    def test_git_provenance_reports_commit_and_dirty_flag_from_subprocess(self):
        def fake_run(args, **kwargs):
            result = mock.Mock()
            if args[1] == "rev-parse":
                result.stdout = "deadbeef\n"
            else:
                result.stdout = " M some_file.py\n"
            return result

        with mock.patch("tradelab.profile_research_context.subprocess.run", side_effect=fake_run):
            info = _git_provenance()
        self.assertEqual(info, {"git_commit": "deadbeef", "git_commit_dirty": True})

    def test_git_provenance_reports_clean_when_status_is_empty(self):
        def fake_run(args, **kwargs):
            result = mock.Mock()
            result.stdout = "deadbeef\n" if args[1] == "rev-parse" else ""
            return result

        with mock.patch("tradelab.profile_research_context.subprocess.run", side_effect=fake_run):
            info = _git_provenance()
        self.assertEqual(info, {"git_commit": "deadbeef", "git_commit_dirty": False})

    def test_provenance_section_carries_checksums_and_command_verbatim(self):
        with mock.patch(
            "tradelab.profile_research_context._git_provenance",
            return_value={"git_commit": "abc123", "git_commit_dirty": False},
        ):
            section = _provenance_section(market_artifact_sha256="a" * 64, news_csv_sha256="b" * 64, command="python3 -m tradelab.profile_research_context --market-artifact x")
        self.assertEqual(section["market_artifact_sha256"], "a" * 64)
        self.assertEqual(section["news_csv_sha256"], "b" * 64)
        self.assertEqual(section["command"], "python3 -m tradelab.profile_research_context --market-artifact x")
        self.assertEqual(section["git_commit"], "abc123")
        self.assertFalse(section["git_commit_dirty"])
        self.assertIn("python_version", section)

    def test_provenance_section_command_is_none_when_not_supplied(self):
        with mock.patch(
            "tradelab.profile_research_context._git_provenance",
            return_value={"git_commit": None, "git_commit_dirty": None},
        ):
            section = _provenance_section(market_artifact_sha256="a" * 64, news_csv_sha256="b" * 64, command=None)
        self.assertIsNone(section["command"])


class CompareRawSignalCountsTests(unittest.TestCase):
    def test_identical_lists_report_identical_true_and_no_diffs(self):
        counts = [{"selector": "fvg-1m-fvg_touch", "direction": "long", "count": 3}]
        diff = _compare_raw_signal_counts(counts, counts)
        self.assertEqual(diff, {"identical": True, "only_in_previous": [], "only_in_current": [], "changed": []})

    def test_added_removed_and_changed_counts_are_all_reported(self):
        previous = [
            {"selector": "fvg-1m-fvg_touch", "direction": "long", "count": 3},
            {"selector": "fvg-1m-fvg_touch", "direction": "short", "count": 5},
        ]
        current = [
            {"selector": "fvg-1m-fvg_touch", "direction": "long", "count": 4},
            {"selector": "fvg-5m-fvg_touch", "direction": "long", "count": 1},
        ]
        diff = _compare_raw_signal_counts(previous, current)
        self.assertFalse(diff["identical"])
        self.assertEqual(diff["only_in_previous"], ["fvg-1m-fvg_touch/short"])
        self.assertEqual(diff["only_in_current"], ["fvg-5m-fvg_touch/long"])
        self.assertEqual(diff["changed"], ["fvg-1m-fvg_touch/long: 3 -> 4"])


class ComparisonSectionTests(unittest.TestCase):
    def test_no_previous_profile_reports_all_none(self):
        section = _comparison_section(
            previous_profile_json_path=None,
            previous_profile=None,
            current_entry_signals_by_selector_direction=[{"selector": "fvg-1m-fvg_touch", "direction": "long", "count": 1}],
        )
        self.assertEqual(
            section,
            {
                "previous_profile_path": None,
                "previous_profile_schema_version": None,
                "previous_profile_obsolete_validity_model_warning": None,
                "raw_signal_count_comparison": None,
                "active_pool_comparison_possible": None,
                "active_pool_comparison_note": None,
            },
        )

    def test_previous_profile_given_reports_warning_and_comparison_but_never_active_pool(self):
        previous_profile = {
            "profile_schema_version": "1.0",
            "entry_signals": {"by_selector_direction": [{"selector": "fvg-1m-fvg_touch", "direction": "long", "count": 1}]},
        }
        section = _comparison_section(
            previous_profile_json_path=Path("/tmp/old_profile.json"),
            previous_profile=previous_profile,
            current_entry_signals_by_selector_direction=[{"selector": "fvg-1m-fvg_touch", "direction": "long", "count": 1}],
        )
        self.assertEqual(section["previous_profile_path"], "/tmp/old_profile.json")
        self.assertEqual(section["previous_profile_schema_version"], "1.0")
        self.assertIn("OBSOLETE", section["previous_profile_obsolete_validity_model_warning"])
        self.assertIn("end-of-NY-trading-day", section["previous_profile_obsolete_validity_model_warning"])
        self.assertEqual(section["raw_signal_count_comparison"], {"identical": True, "only_in_previous": [], "only_in_current": [], "changed": []})
        self.assertFalse(section["active_pool_comparison_possible"])
        self.assertIn("NOT possible", section["active_pool_comparison_note"])


class ProfileSchemaVersionAndProvenanceEndToEndTests(unittest.TestCase):
    def test_written_profile_has_v2_schema_new_sections_and_recorded_command(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_json = tmp / "context_profile.json"

            profile_research_context(manifest_path, news_path, output_json, command="python3 -m tradelab.profile_research_context --test")
            profile = json.loads(output_json.read_text(encoding="utf-8"))

            self.assertEqual(profile["profile_schema_version"], "2.0")
            self.assertIn("fvg_entry_candidates", profile)
            self.assertIn("touch_counts_by_timeframe_direction_contact_type", profile["fvg_entry_candidates"])
            self.assertIn("ifvg_counts_by_timeframe_direction", profile["fvg_entry_candidates"])
            self.assertIn("mitigation_counts_by_timeframe_direction", profile["fvg_entry_candidates"])
            self.assertIn("entry_signal_timeline", profile)
            self.assertIn("suspension_count_total", profile["entry_signal_timeline"])
            self.assertIn("restoration_count_total", profile["entry_signal_timeline"])
            self.assertIn("active_fvg_confluence_duration_stats_minutes", profile["entry_signal_timeline"])
            self.assertIn("active_pool_stats", profile)
            self.assertIn("global", profile["active_pool_stats"])
            self.assertIn("by_direction", profile["active_pool_stats"])
            self.assertIn("by_primary_selector_direction", profile["active_pool_stats"])
            self.assertIn("active_selector_count_including_primary", profile["active_pool_stats"]["global"])
            self.assertIn("required_confluence_selector_count_excluding_primary", profile["active_pool_stats"]["global"])
            self.assertIn("active_signal_instance_count", profile["active_pool_stats"]["global"])
            self.assertIn("entry_event_model", profile)
            self.assertIn("persistent_state_fact_counts", profile["entry_event_model"])
            self.assertIn("fvg_zone_count_total", profile["entry_event_model"]["persistent_state_fact_counts"])
            self.assertIn("htf_bias_change_count_total", profile["entry_event_model"]["persistent_state_fact_counts"])
            self.assertIn("entry_event_count_by_kind", profile["entry_event_model"])
            self.assertEqual(
                set(profile["entry_event_model"]["entry_event_count_by_kind"]), {"fvg_touch", "ifvg", "liquidity_sweep", "total"}
            )
            self.assertIn("candidate_entry_snapshot_count", profile["entry_event_model"])
            self.assertEqual(
                profile["entry_event_model"]["candidate_entry_snapshot_count"],
                profile["entry_event_model"]["entry_event_count_by_kind"]["total"],
            )
            self.assertIn("active_persistent_context_at_candidate_timestamps", profile["entry_event_model"])
            self.assertIn("no_accumulation_evidence", profile["entry_event_model"])
            self.assertIn("entry_event_count_equals_candidate_snapshot_count", profile["entry_event_model"]["no_accumulation_evidence"])
            self.assertLessEqual(profile["entry_event_model"]["no_accumulation_evidence"]["max_active_fvg_selectors_observed"], 6)
            self.assertEqual(profile["entry_event_model"]["no_accumulation_evidence"]["max_possible_fvg_selectors_by_catalog"], 6)
            self.assertIn("sweep_fvg_setup_model", profile)
            self.assertIn("sweep_count_total", profile["sweep_fvg_setup_model"])
            self.assertIn("pending_context_transition_counts", profile["sweep_fvg_setup_model"])
            self.assertEqual(
                set(profile["sweep_fvg_setup_model"]["pending_context_transition_counts"]),
                {"created", "replaced", "invalidated", "consumed"},
            )
            self.assertIn("setup_event_count_total", profile["sweep_fvg_setup_model"])
            self.assertIn("setup_event_counts_by_sweep_timeframe", profile["sweep_fvg_setup_model"])
            self.assertIn("setup_event_counts_by_fvg_timeframe", profile["sweep_fvg_setup_model"])
            self.assertIn("setup_event_counts_by_fvg_kind", profile["sweep_fvg_setup_model"])
            self.assertIn("setup_event_counts_by_direction", profile["sweep_fvg_setup_model"])
            self.assertIn("age_minutes_sweep_known_at_to_setup_known_at", profile["sweep_fvg_setup_model"])
            self.assertIn("one_time_consumption_evidence", profile["sweep_fvg_setup_model"])
            self.assertTrue(profile["sweep_fvg_setup_model"]["one_time_consumption_evidence"]["every_multi_use_is_a_same_instant_tie"])
            self.assertTrue(profile["sweep_fvg_setup_model"]["one_time_consumption_evidence"]["created_count_equals_sweep_count"])
            self.assertEqual(profile["provenance"]["command"], "python3 -m tradelab.profile_research_context --test")
            self.assertEqual(profile["provenance"]["market_artifact_sha256"], profile["market_artifact"]["sha256"])
            self.assertEqual(profile["provenance"]["news_csv_sha256"], profile["news"]["sha256"])
            self.assertEqual(
                profile["comparison"],
                {
                    "previous_profile_path": None,
                    "previous_profile_schema_version": None,
                    "previous_profile_obsolete_validity_model_warning": None,
                    "raw_signal_count_comparison": None,
                    "active_pool_comparison_possible": None,
                    "active_pool_comparison_note": None,
                },
            )

    def test_previous_profile_json_path_populates_comparison_section(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            previous_output = tmp / "previous_profile.json"
            profile_research_context(manifest_path, news_path, previous_output)

            current_output = tmp / "current_profile.json"
            profile_research_context(manifest_path, news_path, current_output, previous_profile_json_path=previous_output)
            profile = json.loads(current_output.read_text(encoding="utf-8"))

            self.assertEqual(profile["comparison"]["previous_profile_path"], str(previous_output))
            self.assertEqual(profile["comparison"]["previous_profile_schema_version"], "2.0")
            self.assertFalse(profile["comparison"]["active_pool_comparison_possible"])
            self.assertTrue(profile["comparison"]["raw_signal_count_comparison"]["identical"])

    def test_two_runs_with_same_command_and_no_previous_profile_stay_byte_identical(self):
        """Guards the pre-existing determinism contract (see
        ``DeterminismTests``) against a regression from any of the new
        v2.0 fields -- ``provenance`` carries no wall-clock timestamp,
        and ``_git_provenance`` reads the checkout's actual state, which
        cannot change between these two calls."""
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_a = tmp / "profile_a.json"
            output_b = tmp / "profile_b.json"

            profile_research_context(manifest_path, news_path, output_a, command="fixed-command")
            profile_research_context(manifest_path, news_path, output_b, command="fixed-command")

            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())


class PhaseCallbackTests(unittest.TestCase):
    """``phase_callback`` exposes per-phase wall-clock timing for CLI
    progress visibility on a long real run -- purely a side-effecting
    hook. Must never appear in, or affect, the deterministic JSON
    output, and must never be required (omitting it changes nothing)."""

    def test_callback_receives_all_seven_expected_phases_in_order(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_json = tmp / "context_profile.json"

            observed_phases: list[str] = []
            profile_research_context(manifest_path, news_path, output_json, phase_callback=lambda phase, duration: observed_phases.append(phase))

            self.assertEqual(
                observed_phases,
                [
                    "preparation",
                    "lifecycle_profiling",
                    "pool_statistics",
                    "entry_event_model",
                    "sweep_fvg_setup_model",
                    "serialization",
                    "total",
                ],
            )

    def test_callback_receives_non_negative_float_durations(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_json = tmp / "context_profile.json"

            observed_durations: dict[str, float] = {}
            profile_research_context(
                manifest_path, news_path, output_json, phase_callback=lambda phase, duration: observed_durations.__setitem__(phase, duration)
            )

            for phase, duration in observed_durations.items():
                self.assertIsInstance(duration, float, f"{phase} duration should be a float")
                self.assertGreaterEqual(duration, 0.0, f"{phase} duration should never be negative")
            # total must be at least as large as the sum of the four named sub-phases.
            sub_phase_total = sum(d for p, d in observed_durations.items() if p != "total")
            self.assertGreaterEqual(observed_durations["total"], sub_phase_total)

    def test_omitting_the_callback_changes_nothing_about_the_written_profile(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_with_callback = tmp / "with_callback.json"
            output_without_callback = tmp / "without_callback.json"

            profile_research_context(manifest_path, news_path, output_with_callback, command="fixed", phase_callback=lambda phase, duration: None)
            profile_research_context(manifest_path, news_path, output_without_callback, command="fixed")

            self.assertEqual(output_with_callback.read_bytes(), output_without_callback.read_bytes())

    def test_no_phase_timing_key_ever_appears_in_the_written_json(self):
        def _collect_keys(value) -> set:
            keys: set = set()
            if isinstance(value, dict):
                for key, sub_value in value.items():
                    keys.add(key)
                    keys |= _collect_keys(sub_value)
            elif isinstance(value, list):
                for item in value:
                    keys |= _collect_keys(item)
            return keys

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_valid_fixture(tmp)
            output_json = tmp / "context_profile.json"

            profile_research_context(manifest_path, news_path, output_json, phase_callback=lambda phase, duration: None)
            profile = json.loads(output_json.read_text(encoding="utf-8"))
            all_keys = _collect_keys(profile)

            # Exact key names, never substring-matched, and excluding phase names that
            # collide with real, legitimate, unrelated existing keys ("total" is
            # tradelab.data_wicks's own upper/lower/total breakdown; "preparation" is a
            # substring of "context_preparation_params") -- those are covered instead by
            # DeterminismTests/test_two_runs_with_same_command_and_no_previous_profile_stay_
            # byte_identical, which would fail the instant any wall-clock duration leaked in.
            for phase_key in ("lifecycle_profiling", "pool_statistics", "serialization", "phase_timings", "phase_timings_seconds"):
                self.assertNotIn(phase_key, all_keys)


def liquidity_candidate_for_model(cid: str, tf: str, direction: str, known_at: datetime) -> LiquidityEntryCandidate:
    return LiquidityEntryCandidate(
        candidate_id=cid,
        source_family=SourceFamily.SWING_EQUAL,
        timeframe=tf,
        level_id="l1",
        liquidity_type=LiquidityType.SWING_HIGH,
        side="high" if direction == "short" else "low",
        lower=100.0,
        upper=100.0,
        trigger_phase=CandidatePhase.TAKE,
        entry_mode=EntryMode.REVERSAL,
        direction=direction,
        timestamp=known_at,
        known_at=known_at,
    )


class EntryEventModelSectionTests(unittest.TestCase):
    """``_entry_event_model_section``: persistent-state fact counts,
    atomic EntryEvent counts by kind, candidate-entry snapshot count,
    active-persistent-context distribution, and machine-checkable
    no-accumulation evidence -- see task 036's own requirement to
    report these separately from the older, EntrySignal-based
    ``active_pool_stats`` section."""

    def test_persistent_state_fact_counts_and_event_counts_by_kind(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        touch = fvg_touch_candidate("z1", "1m", "long", FVGContactType.PARTIAL_TOUCH, t0)
        ifvg = fvg_ifvg_candidate("z2", "1m", "long", t0 + timedelta(minutes=1))
        sweep = liquidity_candidate_for_model("sw1", "1m", "long", t0 + timedelta(minutes=2))

        fvg_results = {
            "1m": FVGDetectionResult(
                zones=[
                    zone("z1", "1m", BULLISH, formed_at=t0 - timedelta(minutes=1)),
                    zone("z2", "1m", BEARISH, formed_at=t0 - timedelta(minutes=1)),
                ],
                events=[],
            )
        }
        fvg_entry_result = FVGEntryCandidateResult(candidates=[touch, ifvg])
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=[sweep])
        htf_bias_result = HTFBiasResult(timeframe_changes={"1H": [], "4H": [], "1D": []})

        section = _entry_event_model_section(fvg_results, fvg_entry_result, liquidity_entry_result, htf_bias_result)

        self.assertEqual(section["persistent_state_fact_counts"], {"fvg_zone_count_total": 2, "htf_bias_change_count_total": 0})
        self.assertEqual(section["entry_event_count_by_kind"], {"fvg_touch": 1, "ifvg": 1, "liquidity_sweep": 1, "total": 3})
        self.assertEqual(section["candidate_entry_snapshot_count"], 3)

    def test_excluding_primary_never_goes_negative_when_the_primarys_own_zone_is_suspended_in_the_same_batch(self):
        """Under the zone-keyed model, a primary event's own zone can
        itself be suspended in this exact same batch by a same-instant,
        same-timeframe opposing challenger -- the challenger targets the
        pre-batch pool, which includes zones active since formation, not
        just zones this same batch happens to touch. Found via a real-
        MNQ smoke run: required_confluence_selector_count_excluding_
        primary's min was -1 before this was fixed to check actual
        selector presence instead of blindly subtracting one whenever
        the event was FVG-family."""
        t_form = datetime(2024, 1, 1, tzinfo=UTC)
        t_touch = t_form + timedelta(hours=1)
        zone_a = zone("A", "1m", BULLISH, formed_at=t_form)
        zone_b = zone("B", "1m", BEARISH, formed_at=t_form)
        touch_a = fvg_touch_candidate("A", "1m", "long", FVGContactType.PARTIAL_TOUCH, t_touch)
        touch_b = fvg_touch_candidate("B", "1m", "short", FVGContactType.PARTIAL_TOUCH, t_touch)

        fvg_results = {"1m": FVGDetectionResult(zones=[zone_a, zone_b], events=[])}
        fvg_entry_result = FVGEntryCandidateResult(candidates=[touch_a, touch_b])
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=[])
        htf_bias_result = HTFBiasResult(timeframe_changes={"1H": [], "4H": [], "1D": []})

        section = _entry_event_model_section(fvg_results, fvg_entry_result, liquidity_entry_result, htf_bias_result)

        excluding_stats = section["active_persistent_context_at_candidate_timestamps"][
            "required_confluence_selector_count_excluding_primary"
        ]
        self.assertGreaterEqual(excluding_stats["min"], 0)

    def test_no_accumulation_evidence_holds_on_a_realistic_fixture(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        candidates = [
            fvg_touch_candidate(f"z{i}", tf, direction, FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=i))
            for i, (tf, direction) in enumerate([("1m", "long"), ("5m", "short"), ("15m", "long"), ("1m", "short")])
        ]
        fvg_entry_result = FVGEntryCandidateResult(candidates=candidates)
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=[])
        htf_bias_result = HTFBiasResult(timeframe_changes={"1H": [], "4H": [], "1D": []})

        section = _entry_event_model_section({}, fvg_entry_result, liquidity_entry_result, htf_bias_result)

        self.assertTrue(section["no_accumulation_evidence"]["entry_event_count_equals_candidate_snapshot_count"])
        self.assertLessEqual(section["no_accumulation_evidence"]["max_active_fvg_selectors_observed"], 6)
        self.assertEqual(section["no_accumulation_evidence"]["max_possible_fvg_selectors_by_catalog"], 6)

    def test_empty_inputs_report_all_zero_and_still_hold_the_evidence(self):
        fvg_entry_result = FVGEntryCandidateResult(candidates=[])
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=[])
        htf_bias_result = HTFBiasResult(timeframe_changes={"1H": [], "4H": [], "1D": []})

        section = _entry_event_model_section({}, fvg_entry_result, liquidity_entry_result, htf_bias_result)

        self.assertEqual(section["entry_event_count_by_kind"]["total"], 0)
        self.assertEqual(section["candidate_entry_snapshot_count"], 0)
        self.assertTrue(section["no_accumulation_evidence"]["entry_event_count_equals_candidate_snapshot_count"])
        self.assertEqual(section["no_accumulation_evidence"]["max_active_fvg_selectors_observed"], 0)


class SweepFVGSetupModelSectionTests(unittest.TestCase):
    """``_sweep_fvg_setup_model_section`` (task 038): sweep counts,
    pending-context transition counts, setup-event breakdowns, age
    distribution, and the one-time-consumption proof."""

    def test_one_sweep_and_one_compatible_touch_produce_one_setup_and_the_expected_transitions(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        sweep = liquidity_candidate_for_model("sw1", "1m", "long", t0)
        touch = fvg_touch_candidate("z1", "5m", "long", FVGContactType.PARTIAL_TOUCH, t0 + timedelta(minutes=5))

        fvg_entry_result = FVGEntryCandidateResult(candidates=[touch])
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=[sweep])

        section = _sweep_fvg_setup_model_section(fvg_entry_result, liquidity_entry_result)

        self.assertEqual(section["sweep_count_total"], 1)
        self.assertEqual(
            section["pending_context_transition_counts"], {"created": 1, "replaced": 0, "invalidated": 0, "consumed": 1}
        )
        self.assertEqual(section["setup_event_count_total"], 1)
        self.assertEqual(section["setup_event_counts_by_sweep_timeframe"], {"1m": 1})
        self.assertEqual(section["setup_event_counts_by_fvg_timeframe"], {"5m": 1})
        self.assertEqual(section["setup_event_counts_by_fvg_kind"], {"fvg_touch": 1})
        self.assertEqual(section["setup_event_counts_by_direction"], {"long": 1})
        self.assertEqual(section["age_minutes_sweep_known_at_to_setup_known_at"]["min"], 5.0)
        self.assertEqual(section["age_minutes_sweep_known_at_to_setup_known_at"]["max"], 5.0)

    def test_one_time_consumption_evidence_holds_and_flags_a_genuine_tie(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = t0 + timedelta(minutes=5)
        sweep = liquidity_candidate_for_model("sw1", "1m", "long", t0)
        touch_a = fvg_touch_candidate("za", "1m", "long", FVGContactType.PARTIAL_TOUCH, t1)
        touch_b = fvg_touch_candidate("zb", "5m", "long", FVGContactType.PARTIAL_TOUCH, t1)  # exact same known_at -- a tie

        fvg_entry_result = FVGEntryCandidateResult(candidates=[touch_a, touch_b])
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=[sweep])

        section = _sweep_fvg_setup_model_section(fvg_entry_result, liquidity_entry_result)

        self.assertEqual(section["setup_event_count_total"], 2)
        evidence = section["one_time_consumption_evidence"]
        self.assertEqual(evidence["distinct_sweeps_consumed"], 1)
        self.assertEqual(evidence["sweeps_consumed_by_more_than_one_tied_event"], 1)
        self.assertTrue(evidence["every_multi_use_is_a_same_instant_tie"])
        self.assertTrue(evidence["created_count_equals_sweep_count"])
        self.assertEqual(section["pending_context_transition_counts"]["consumed"], 1)  # one context, consumed once

    def test_empty_inputs_report_all_zero(self):
        fvg_entry_result = FVGEntryCandidateResult(candidates=[])
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=[])

        section = _sweep_fvg_setup_model_section(fvg_entry_result, liquidity_entry_result)

        self.assertEqual(section["sweep_count_total"], 0)
        self.assertEqual(section["setup_event_count_total"], 0)
        self.assertEqual(section["pending_context_transition_counts"], {"created": 0, "replaced": 0, "invalidated": 0, "consumed": 0})
        self.assertTrue(section["one_time_consumption_evidence"]["every_multi_use_is_a_same_instant_tie"])
        self.assertTrue(section["one_time_consumption_evidence"]["created_count_equals_sweep_count"])


class NoRealDataInRepoTests(unittest.TestCase):
    def test_gitignore_covers_generated_profile_and_market_artifact_files(self):
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        content = gitignore.read_text(encoding="utf-8")
        for pattern in ("context_profile.json", "mnq_continuous_1m.csv", "continuous_artifact_manifest.json"):
            self.assertIn(pattern, content, f"{pattern!r} should be gitignored so a locally-run profile can never be committed by accident")

    def test_no_generated_artifact_files_are_tracked_by_git(self):
        import subprocess

        repo_root = Path(__file__).resolve().parent.parent
        if not (repo_root / ".git").exists():
            manifest = json.loads((repo_root / "SOURCE_MANIFEST.json").read_text())
            tracked = list(manifest["files"])
        else:
            tracked = subprocess.run(["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.splitlines()
        offenders = [f for f in tracked if f.endswith("context_profile.json") or f.endswith("mnq_continuous_1m.csv")]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
