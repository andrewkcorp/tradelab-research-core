#!/usr/bin/env python3
"""FVG/iFVG entry candidate tests: FVG touch entry triggers only
(partial_touch/full_mitigation_no_inversion, never inversion_ifvg),
iFVG candidates with direction taken directly from the event, disjoint
source types (no duplication when a zone produces both), bullish/
bearish symmetry, 1m/5m/15m-only restriction, known_at/no-lookahead,
consistency validation, no duplicate candidate_ids, and full audit
retention."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.fvg import (
    BEARISH,
    BULLISH,
    Candle,
    FVGDetectionResult,
    FVGEvent,
    FVGStatus,
    FVGZone,
    detect_fvgs,
)
from tradelab.fvg_entry import (
    FVGContactType,
    SourceType,
    detect_fvg_entry_candidates,
)
from tradelab.fvg_touch import FVGTouchDetectionResult, FVGTouchEvent, detect_fvg_touch_events

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)


def candle(offset_minutes: float, o: float, h: float, l: float, c: float, known_offset_minutes: float | None = None) -> Candle:
    ts = T0 + timedelta(minutes=offset_minutes)
    known_at = T0 + timedelta(minutes=known_offset_minutes if known_offset_minutes is not None else offset_minutes + 1)
    return Candle(ts, o, h, l, c, known_at)


def bullish_zone_setup() -> list[Candle]:
    """3 candles forming a bullish FVG zone = [100.0, 102.0]."""
    return [
        candle(0, 100.0, 100.0, 99.0, 99.5),
        candle(1, 100.5, 101.0, 100.2, 100.8),
        candle(2, 102.3, 102.5, 102.0, 102.35),
    ]


def bearish_zone_setup() -> list[Candle]:
    """3 candles forming a bearish FVG zone = [98.0, 100.0]."""
    return [
        candle(0, 100.5, 101.0, 100.0, 100.8),
        candle(1, 99.5, 99.8, 99.0, 99.3),
        candle(2, 97.8, 98.0, 97.5, 97.9),
    ]


def run(candles: list[Candle], timeframe: str = "1m"):
    fvg_result = detect_fvgs(candles, timeframe)
    touch_result = detect_fvg_touch_events(candles, fvg_result, timeframe)
    entry_result = detect_fvg_entry_candidates({timeframe: fvg_result}, {timeframe: touch_result})
    return fvg_result, touch_result, entry_result


def zone(
    zone_id: str = "1m-FVG-1",
    timeframe: str = "1m",
    direction: str = BULLISH,
    lower: float = 100.0,
    upper: float = 102.0,
    status: FVGStatus = FVGStatus.ACTIVE,
    resolved_at: datetime | None = None,
    produced_ifvg: bool = False,
) -> FVGZone:
    return FVGZone(
        zone_id=zone_id,
        timeframe=timeframe,
        direction=direction,
        lower=lower,
        upper=upper,
        candle_1_timestamp=T0,
        candle_2_timestamp=T0 + timedelta(minutes=1),
        candle_3_timestamp=T0 + timedelta(minutes=2),
        formed_at=T0 + timedelta(minutes=3),
        status=status,
        resolved_at=resolved_at,
        produced_ifvg=produced_ifvg,
    )


def ifvg_event(
    zone_id: str = "1m-FVG-1",
    timeframe: str = "1m",
    original_direction: str = BULLISH,
    lower: float = 100.0,
    upper: float = 102.0,
    known_at: datetime = T0 + timedelta(minutes=5),
    timestamp: datetime | None = None,
) -> FVGEvent:
    opposite = BEARISH if original_direction == BULLISH else BULLISH
    return FVGEvent(
        zone_id=zone_id,
        timeframe=timeframe,
        direction=opposite,
        original_zone_direction=original_direction,
        lower=lower,
        upper=upper,
        timestamp=known_at if timestamp is None else timestamp,
        known_at=known_at,
    )


def touch_event(
    zone_id: str = "1m-FVG-1",
    timeframe: str = "1m",
    direction: str = BULLISH,
    lower: float = 100.0,
    upper: float = 102.0,
    contact_type: FVGContactType = FVGContactType.PARTIAL_TOUCH,
    is_entry_trigger: bool | None = None,
    timestamp: datetime = T0 + timedelta(minutes=4),
    known_at: datetime = T0 + timedelta(minutes=5),
) -> FVGTouchEvent:
    if is_entry_trigger is None:
        is_entry_trigger = contact_type != FVGContactType.INVERSION_IFVG
    return FVGTouchEvent(
        zone_id=zone_id,
        timeframe=timeframe,
        direction=direction,
        lower=lower,
        upper=upper,
        timestamp=timestamp,
        known_at=known_at,
        contact_type=contact_type,
        is_entry_trigger=is_entry_trigger,
    )


def fvg_result_with(zones: list[FVGZone], events: list[FVGEvent]) -> FVGDetectionResult:
    return FVGDetectionResult(zones=zones, events=events)


def touch_result_with(events: list[FVGTouchEvent]) -> FVGTouchDetectionResult:
    return FVGTouchDetectionResult(events=events)


class FVGTouchEntryTests(unittest.TestCase):
    def test_partial_touch_produces_a_touch_entry_candidate(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        _, _, entry_result = run(bullish_zone_setup() + [touch_candle])
        self.assertEqual(len(entry_result.candidates), 1)
        c = entry_result.candidates[0]
        self.assertEqual(c.source_type, SourceType.FVG_TOUCH)
        self.assertEqual(c.contact_type, FVGContactType.PARTIAL_TOUCH)

    def test_full_mitigation_no_inversion_produces_a_touch_entry_candidate(self):
        touch_candle = candle(3, 100.5, 100.6, 99.5, 100.5)  # full pass-through, close inside
        _, _, entry_result = run(bullish_zone_setup() + [touch_candle])
        self.assertEqual(len(entry_result.candidates), 1)
        c = entry_result.candidates[0]
        self.assertEqual(c.source_type, SourceType.FVG_TOUCH)
        self.assertEqual(c.contact_type, FVGContactType.FULL_MITIGATION_NO_INVERSION)

    def test_inversion_touch_never_produces_a_touch_entry_candidate(self):
        touch_candle = candle(3, 100.0, 100.1, 99.0, 99.5)  # full pass-through + close beyond -> inversion
        _, _, entry_result = run(bullish_zone_setup() + [touch_candle])
        touch_candidates = [c for c in entry_result.candidates if c.source_type == SourceType.FVG_TOUCH]
        self.assertEqual(touch_candidates, [])


class IFVGEntryTests(unittest.TestCase):
    def test_bullish_zone_inversion_gives_bearish_ifvg_short_candidate(self):
        touch_candle = candle(3, 100.0, 100.1, 99.0, 99.5)
        fvg_result, _, entry_result = run(bullish_zone_setup() + [touch_candle])
        ifvg_candidates = [c for c in entry_result.candidates if c.source_type == SourceType.IFVG]
        self.assertEqual(len(ifvg_candidates), 1)
        c = ifvg_candidates[0]
        self.assertEqual(c.direction, "short")
        self.assertEqual(c.contact_type, None)
        self.assertEqual(fvg_result.events[0].direction, "bearish")

    def test_bearish_zone_inversion_gives_bullish_ifvg_long_candidate(self):
        touch_candle = candle(3, 100.0, 100.5, 99.9, 100.5)  # bearish zone, full pass through + close beyond
        fvg_result, _, entry_result = run(bearish_zone_setup() + [touch_candle])
        ifvg_candidates = [c for c in entry_result.candidates if c.source_type == SourceType.IFVG]
        self.assertEqual(len(ifvg_candidates), 1)
        c = ifvg_candidates[0]
        self.assertEqual(c.direction, "long")
        self.assertEqual(fvg_result.events[0].direction, "bullish")


class DisjointSourceTests(unittest.TestCase):
    def test_inversion_candle_produces_only_ifvg_never_a_duplicate_touch_candidate(self):
        touch_candle = candle(3, 100.0, 100.1, 99.0, 99.5)  # first touch IS the inversion
        _, _, entry_result = run(bullish_zone_setup() + [touch_candle])
        self.assertEqual(len(entry_result.candidates), 1)
        self.assertEqual(entry_result.candidates[0].source_type, SourceType.IFVG)

    def test_partial_touch_then_later_inversion_produces_both_as_distinct_candidates(self):
        partial = candle(3, 101.5, 101.8, 101.0, 101.5)  # first touch: partial
        later_invert = candle(4, 100.0, 100.1, 99.0, 99.5)  # later, genuine inversion
        fvg_result, touch_result, entry_result = run(bullish_zone_setup() + [partial, later_invert])
        self.assertEqual(len(entry_result.candidates), 2)
        source_types = {c.source_type for c in entry_result.candidates}
        self.assertEqual(source_types, {SourceType.FVG_TOUCH, SourceType.IFVG})
        # distinct candidate_ids, distinct known_at
        ids = {c.candidate_id for c in entry_result.candidates}
        self.assertEqual(len(ids), 2)
        known_ats = sorted(c.known_at for c in entry_result.candidates)
        self.assertEqual(known_ats, [partial.known_at, later_invert.known_at])


class DirectionSymmetryTests(unittest.TestCase):
    def test_bullish_fvg_touch_is_long(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        _, _, entry_result = run(bullish_zone_setup() + [touch_candle])
        self.assertEqual(entry_result.candidates[0].direction, "long")

    def test_bearish_fvg_touch_is_short(self):
        touch_candle = candle(3, 98.5, 99.0, 98.2, 98.7)
        _, _, entry_result = run(bearish_zone_setup() + [touch_candle])
        self.assertEqual(entry_result.candidates[0].direction, "short")


class TimeframeRestrictionTests(unittest.TestCase):
    def test_1m_5m_15m_are_accepted(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        for timeframe in ("1m", "5m", "15m"):
            with self.subTest(timeframe=timeframe):
                _, _, entry_result = run(bullish_zone_setup() + [touch_candle], timeframe)
                self.assertEqual(len(entry_result.candidates), 1)
                self.assertEqual(entry_result.candidates[0].timeframe, timeframe)

    def test_1h_4h_1d_are_rejected_even_with_real_detect_fvgs_output(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        candles = bullish_zone_setup() + [touch_candle]
        for timeframe in ("1H", "4H", "1D"):
            with self.subTest(timeframe=timeframe):
                fvg_result = detect_fvgs(candles, timeframe)
                touch_result = detect_fvg_touch_events(candles, fvg_result, timeframe)
                with self.assertRaises(ValueError):
                    detect_fvg_entry_candidates({timeframe: fvg_result}, {timeframe: touch_result})

    def test_touch_results_timeframe_without_matching_fvg_results_is_rejected(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        candles = bullish_zone_setup() + [touch_candle]
        fvg_result = detect_fvgs(candles, "1m")
        touch_result = detect_fvg_touch_events(candles, fvg_result, "1m")
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({}, {"1m": touch_result})

    def test_fvg_only_timeframe_without_touch_results_still_yields_ifvg_candidates(self):
        touch_candle = candle(3, 100.0, 100.1, 99.0, 99.5)  # inversion
        fvg_result = detect_fvgs(bullish_zone_setup() + [touch_candle], "1m")
        entry_result = detect_fvg_entry_candidates({"1m": fvg_result}, {})
        self.assertEqual(len(entry_result.candidates), 1)
        self.assertEqual(entry_result.candidates[0].source_type, SourceType.IFVG)


class KnownAtNoLookaheadTests(unittest.TestCase):
    def test_touch_candidate_known_at_matches_the_touch_event_exactly(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5, known_offset_minutes=50)
        _, touch_result, entry_result = run(bullish_zone_setup() + [touch_candle])
        c = entry_result.candidates[0]
        self.assertEqual(c.known_at, touch_result.events[0].known_at)
        self.assertEqual(c.timestamp, touch_result.events[0].timestamp)

    def test_ifvg_candidate_known_at_matches_the_fvg_event_exactly(self):
        touch_candle = candle(3, 100.0, 100.1, 99.0, 99.5, known_offset_minutes=50)
        fvg_result, _, entry_result = run(bullish_zone_setup() + [touch_candle])
        c = entry_result.candidates[0]
        self.assertEqual(c.known_at, fvg_result.events[0].known_at)
        self.assertEqual(c.timestamp, fvg_result.events[0].timestamp)

    def test_a_future_candle_appended_after_the_touch_does_not_change_it(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        base = bullish_zone_setup() + [touch_candle]
        extra_future = [candle(4, 200.0, 201.0, 199.0, 200.5)]
        _, _, short_result = run(base)
        _, _, long_result = run(base + extra_future)
        self.assertEqual(short_result.candidates, long_result.candidates)


class ValidationTests(unittest.TestCase):
    def test_zone_timeframe_mismatch_with_dict_key_is_rejected(self):
        fvg_result = fvg_result_with([zone(timeframe="5m")], [])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {})

    def test_unsupported_timeframe_key_is_rejected(self):
        fvg_result = fvg_result_with([zone(timeframe="3m", zone_id="3m-FVG-1")], [])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"3m": fvg_result}, {})

    def test_ifvg_referencing_nonexistent_zone_is_rejected(self):
        fvg_result = fvg_result_with([], [ifvg_event()])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {})

    def test_ifvg_with_wrong_original_zone_direction_is_rejected(self):
        fvg_result = fvg_result_with([zone(direction=BULLISH)], [ifvg_event(original_direction=BEARISH)])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {})

    def test_ifvg_with_direction_not_opposite_of_zone_is_rejected(self):
        bad_event = FVGEvent(
            zone_id="1m-FVG-1",
            timeframe="1m",
            direction=BULLISH,  # should be BEARISH (opposite of a bullish zone)
            original_zone_direction=BULLISH,
            lower=100.0,
            upper=102.0,
            timestamp=T0 + timedelta(minutes=5),
            known_at=T0 + timedelta(minutes=5),
        )
        fvg_result = fvg_result_with([zone(direction=BULLISH)], [bad_event])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {})

    def test_ifvg_with_wrong_bounds_is_rejected(self):
        fvg_result = fvg_result_with([zone(lower=100.0, upper=102.0)], [ifvg_event(lower=99.0, upper=101.0)])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {})

    def test_touch_referencing_nonexistent_zone_is_rejected(self):
        fvg_result = fvg_result_with([], [])
        touch_result = touch_result_with([touch_event()])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {"1m": touch_result})

    def test_touch_with_wrong_direction_is_rejected(self):
        fvg_result = fvg_result_with([zone(direction=BULLISH)], [])
        touch_result = touch_result_with([touch_event(direction=BEARISH)])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {"1m": touch_result})

    def test_touch_with_wrong_bounds_is_rejected(self):
        fvg_result = fvg_result_with([zone(lower=100.0, upper=102.0)], [])
        touch_result = touch_result_with([touch_event(lower=99.0, upper=101.0)])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {"1m": touch_result})

    def test_touch_with_wrong_timeframe_is_rejected(self):
        fvg_result = fvg_result_with([zone()], [])
        touch_result = touch_result_with([touch_event(timeframe="5m")])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {"1m": touch_result})

    def test_inversion_contact_type_wrongly_marked_as_entry_trigger_is_rejected(self):
        fvg_result = fvg_result_with([zone(direction=BULLISH)], [])
        bad_touch = touch_event(contact_type=FVGContactType.INVERSION_IFVG, is_entry_trigger=True)
        touch_result = touch_result_with([bad_touch])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {"1m": touch_result})

    def test_partial_touch_wrongly_marked_as_not_an_entry_trigger_is_rejected(self):
        fvg_result = fvg_result_with([zone(direction=BULLISH)], [])
        bad_touch = touch_event(contact_type=FVGContactType.PARTIAL_TOUCH, is_entry_trigger=False)
        touch_result = touch_result_with([bad_touch])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {"1m": touch_result})

    def test_duplicate_ifvg_event_for_same_zone_is_rejected(self):
        fvg_result = fvg_result_with([zone(direction=BULLISH)], [ifvg_event(), ifvg_event()])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {})

    def test_duplicate_touch_event_for_same_zone_is_rejected(self):
        fvg_result = fvg_result_with([zone(direction=BULLISH)], [])
        touch_result = touch_result_with([touch_event(), touch_event()])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {"1m": touch_result})

    def test_duplicate_zone_id_is_rejected(self):
        fvg_result = fvg_result_with([zone(), zone()], [])
        with self.assertRaises(ValueError):
            detect_fvg_entry_candidates({"1m": fvg_result}, {})


class AuditTests(unittest.TestCase):
    def test_touch_candidate_retains_full_audit_fields(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        fvg_result, touch_result, entry_result = run(bullish_zone_setup() + [touch_candle])
        c = entry_result.candidates[0]
        t = touch_result.events[0]
        self.assertEqual(c.zone_id, t.zone_id)
        self.assertEqual(c.timeframe, t.timeframe)
        self.assertEqual(c.lower, t.lower)
        self.assertEqual(c.upper, t.upper)
        self.assertEqual(c.contact_type, t.contact_type)
        self.assertTrue(c.candidate_id)

    def test_ifvg_candidate_retains_full_audit_fields(self):
        touch_candle = candle(3, 100.0, 100.1, 99.0, 99.5)
        fvg_result, _, entry_result = run(bullish_zone_setup() + [touch_candle])
        c = entry_result.candidates[0]
        e = fvg_result.events[0]
        self.assertEqual(c.zone_id, e.zone_id)
        self.assertEqual(c.timeframe, e.timeframe)
        self.assertEqual(c.lower, e.lower)
        self.assertEqual(c.upper, e.upper)
        self.assertIsNone(c.contact_type)

    def test_candidate_id_is_stable_and_distinct_between_source_types(self):
        partial = candle(3, 101.5, 101.8, 101.0, 101.5)
        later_invert = candle(4, 100.0, 100.1, 99.0, 99.5)
        _, _, entry_result_1 = run(bullish_zone_setup() + [partial, later_invert])
        _, _, entry_result_2 = run(bullish_zone_setup() + [partial, later_invert])
        self.assertEqual(
            sorted(c.candidate_id for c in entry_result_1.candidates),
            sorted(c.candidate_id for c in entry_result_2.candidates),
        )
        ids = [c.candidate_id for c in entry_result_1.candidates]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
