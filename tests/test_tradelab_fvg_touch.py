#!/usr/bin/env python3
"""FVG touch event tests: partial touch, full mitigation without
inversion, inversion (no entry trigger, existing iFVG unaffected),
boundary-without-entry, at-most-one-touch-per-zone, bullish/bearish
symmetry, native timeframes, known_at/no-lookahead, and regression-free
tradelab.fvg semantics."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.fvg import SUPPORTED_TIMEFRAMES, Candle, FVGStatus, detect_fvgs
from tradelab.fvg_touch import FVGContactType, detect_fvg_touch_events

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
    return fvg_result, touch_result


class PartialTouchTests(unittest.TestCase):
    def test_bullish_partial_touch_is_exactly_one_entry_trigger_and_stays_active(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)  # low=101.0, inside (100, 102)
        fvg_result, touch_result = run(bullish_zone_setup() + [touch_candle])

        self.assertEqual(len(touch_result.events), 1)
        event = touch_result.events[0]
        self.assertEqual(event.contact_type, FVGContactType.PARTIAL_TOUCH)
        self.assertTrue(event.is_entry_trigger)
        self.assertEqual(fvg_result.zones[0].status, FVGStatus.ACTIVE)

    def test_bearish_partial_touch_is_exactly_one_entry_trigger_and_stays_active(self):
        touch_candle = candle(3, 98.5, 99.0, 98.2, 98.7)  # high=99.0, inside (98, 100)
        fvg_result, touch_result = run(bearish_zone_setup() + [touch_candle])

        self.assertEqual(len(touch_result.events), 1)
        event = touch_result.events[0]
        self.assertEqual(event.contact_type, FVGContactType.PARTIAL_TOUCH)
        self.assertTrue(event.is_entry_trigger)
        self.assertEqual(fvg_result.zones[0].status, FVGStatus.ACTIVE)


class FullMitigationNoInversionTests(unittest.TestCase):
    def test_bullish_full_mitigation_without_inversion_is_one_entry_trigger_and_makes_zone_inactive(self):
        touch_candle = candle(3, 100.5, 100.6, 99.5, 100.5)  # low=99.5 (<=100), close=100.5 (not <100)
        fvg_result, touch_result = run(bullish_zone_setup() + [touch_candle])

        self.assertEqual(len(touch_result.events), 1)
        event = touch_result.events[0]
        self.assertEqual(event.contact_type, FVGContactType.FULL_MITIGATION_NO_INVERSION)
        self.assertTrue(event.is_entry_trigger)
        self.assertEqual(fvg_result.zones[0].status, FVGStatus.MITIGATED)
        self.assertFalse(fvg_result.zones[0].produced_ifvg)

    def test_bearish_full_mitigation_without_inversion_is_one_entry_trigger_and_makes_zone_inactive(self):
        touch_candle = candle(3, 99.5, 100.5, 99.4, 99.5)  # high=100.5 (>=100), close=99.5 (not >100)
        fvg_result, touch_result = run(bearish_zone_setup() + [touch_candle])

        self.assertEqual(len(touch_result.events), 1)
        event = touch_result.events[0]
        self.assertEqual(event.contact_type, FVGContactType.FULL_MITIGATION_NO_INVERSION)
        self.assertTrue(event.is_entry_trigger)
        self.assertEqual(fvg_result.zones[0].status, FVGStatus.MITIGATED)
        self.assertFalse(fvg_result.zones[0].produced_ifvg)


class InversionTests(unittest.TestCase):
    def test_bullish_inversion_produces_no_entry_trigger_but_keeps_the_ifvg_event(self):
        touch_candle = candle(3, 100.0, 100.1, 99.0, 99.5)  # low=99.0 (<=100), close=99.5 (<100)
        fvg_result, touch_result = run(bullish_zone_setup() + [touch_candle])

        # A touch event is still recorded (at most one per zone, for the
        # first contact of *any* kind) -- it is simply not an entry trigger.
        self.assertEqual(len(touch_result.events), 1)
        self.assertEqual(touch_result.events[0].contact_type, FVGContactType.INVERSION_IFVG)
        self.assertFalse(touch_result.events[0].is_entry_trigger)
        self.assertEqual(len(fvg_result.events), 1)  # the existing iFVG event is unaffected
        self.assertEqual(fvg_result.events[0].direction, "bearish")
        self.assertTrue(fvg_result.zones[0].produced_ifvg)
        self.assertEqual(fvg_result.zones[0].status, FVGStatus.MITIGATED)

    def test_bearish_inversion_produces_no_entry_trigger_but_keeps_the_ifvg_event(self):
        touch_candle = candle(3, 100.0, 100.5, 99.9, 100.5)  # high=100.5 (>=100), close=100.5 (>100)
        fvg_result, touch_result = run(bearish_zone_setup() + [touch_candle])

        self.assertEqual(len(touch_result.events), 1)
        self.assertEqual(touch_result.events[0].contact_type, FVGContactType.INVERSION_IFVG)
        self.assertFalse(touch_result.events[0].is_entry_trigger)
        self.assertEqual(len(fvg_result.events), 1)
        self.assertEqual(fvg_result.events[0].direction, "bullish")
        self.assertTrue(fvg_result.zones[0].produced_ifvg)


class BoundaryWithoutEntryTests(unittest.TestCase):
    def test_bullish_exact_near_boundary_touch_is_not_a_touch(self):
        touch_candle = candle(3, 102.0, 102.2, 102.0, 102.1)  # low == upper (102.0) exactly
        _, touch_result = run(bullish_zone_setup() + [touch_candle])
        self.assertEqual(touch_result.events, [])

    def test_bearish_exact_near_boundary_touch_is_not_a_touch(self):
        touch_candle = candle(3, 97.8, 98.0, 97.7, 97.9)  # high == lower (98.0) exactly
        _, touch_result = run(bearish_zone_setup() + [touch_candle])
        self.assertEqual(touch_result.events, [])

    def test_bullish_one_tick_past_the_boundary_is_a_touch(self):
        touch_candle = candle(3, 101.9, 102.0, 101.9, 101.95)  # low=101.9, strictly < 102.0
        _, touch_result = run(bullish_zone_setup() + [touch_candle])
        self.assertEqual(len(touch_result.events), 1)
        self.assertEqual(touch_result.events[0].contact_type, FVGContactType.PARTIAL_TOUCH)


class AtMostOneTouchPerZoneTests(unittest.TestCase):
    def test_a_later_touch_after_a_partial_touch_never_creates_a_second_event(self):
        partial = candle(3, 101.5, 101.8, 101.0, 101.5)  # first touch: partial
        later_invert = candle(4, 100.0, 100.1, 99.0, 99.5)  # would be inversion if it were the first touch
        fvg_result, touch_result = run(bullish_zone_setup() + [partial, later_invert])

        zone_1_events = [e for e in touch_result.events if e.zone_id == fvg_result.zones[0].zone_id]
        self.assertEqual(len(zone_1_events), 1)
        self.assertEqual(zone_1_events[0].contact_type, FVGContactType.PARTIAL_TOUCH)
        self.assertEqual(zone_1_events[0].known_at, partial.known_at)
        # the zone genuinely does resolve (with an iFVG) on the later candle --
        # tradelab.fvg's own behavior is unaffected -- but that is not a
        # second FVGTouchEvent.
        zone_1_fvg_events = [e for e in fvg_result.events if e.zone_id == fvg_result.zones[0].zone_id]
        self.assertEqual(len(zone_1_fvg_events), 1)
        self.assertEqual(zone_1_fvg_events[0].known_at, later_invert.known_at)

    def test_two_partial_touches_in_a_row_still_produce_only_one_event(self):
        first_partial = candle(3, 101.5, 101.8, 101.2, 101.5)
        second_partial = candle(4, 101.3, 101.6, 101.1, 101.4)
        _, touch_result = run(bullish_zone_setup() + [first_partial, second_partial])
        self.assertEqual(len(touch_result.events), 1)
        self.assertEqual(touch_result.events[0].known_at, first_partial.known_at)


class TimeframeTests(unittest.TestCase):
    def test_every_supported_timeframe_produces_a_touch_event(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        for timeframe in sorted(SUPPORTED_TIMEFRAMES):
            with self.subTest(timeframe=timeframe):
                fvg_result, touch_result = run(bullish_zone_setup() + [touch_candle], timeframe)
                self.assertEqual(len(touch_result.events), 1)
                self.assertEqual(touch_result.events[0].timeframe, timeframe)
                self.assertEqual(touch_result.events[0].zone_id, fvg_result.zones[0].zone_id)

    def test_unsupported_timeframe_raises(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        candles = bullish_zone_setup() + [touch_candle]
        fvg_result = detect_fvgs(candles, "1m")
        with self.assertRaises(ValueError):
            detect_fvg_touch_events(candles, fvg_result, "3m")

    def test_zone_timeframe_mismatch_is_rejected(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        candles = bullish_zone_setup() + [touch_candle]
        fvg_result = detect_fvgs(candles, "1H")  # zones say "1H"
        with self.assertRaises(ValueError):
            detect_fvg_touch_events(candles, fvg_result, "1m")  # but caller claims "1m"


class NoLookaheadTests(unittest.TestCase):
    def test_no_touch_recorded_when_no_candle_follows_the_forming_one(self):
        _, touch_result = run(bullish_zone_setup())
        self.assertEqual(touch_result.events, [])

    def test_the_very_next_candle_after_formation_is_already_eligible(self):
        # Confirms eligibility starts exactly one candle after formation --
        # neither delayed (a bug would miss this touch) nor early (which is
        # structurally impossible here anyway: a bullish zone's near
        # boundary, zone.upper, is literally candle_3.low, so the forming
        # candle itself can never register as "inside" the zone it just
        # defined).
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        _, touch_result = run(bullish_zone_setup() + [touch_candle])
        self.assertEqual(len(touch_result.events), 1)
        self.assertEqual(touch_result.events[0].known_at, touch_candle.known_at)

    def test_touch_known_at_is_the_touching_candles_own_known_at(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5, known_offset_minutes=50)
        _, touch_result = run(bullish_zone_setup() + [touch_candle])
        self.assertEqual(touch_result.events[0].known_at, touch_candle.known_at)
        self.assertEqual(touch_result.events[0].timestamp, touch_candle.timestamp)
        self.assertNotEqual(touch_result.events[0].known_at, touch_result.events[0].timestamp)

    def test_a_future_candle_appended_after_the_first_touch_does_not_change_it(self):
        touch_candle = candle(3, 101.5, 101.8, 101.0, 101.5)
        base = bullish_zone_setup() + [touch_candle]
        extra_future = [candle(4, 200.0, 201.0, 199.0, 200.5), candle(5, 300.0, 301.0, 299.0, 300.5)]

        _, short_result = run(base)
        _, long_result = run(base + extra_future)

        self.assertEqual(len(short_result.events), 1)
        self.assertEqual(short_result.events[0], long_result.events[0])

    def test_candles_must_be_strictly_ordered(self):
        candles = bullish_zone_setup()
        fvg_result = detect_fvgs(candles, "1m")
        out_of_order = candle(1, 50.0, 50.5, 49.5, 50.0)
        with self.assertRaises(ValueError):
            detect_fvg_touch_events(candles + [out_of_order], fvg_result, "1m")

    def test_fvg_result_from_different_candles_is_rejected(self):
        fvg_result = detect_fvgs(bullish_zone_setup(), "1m")
        # A disjoint candle series: none of these timestamps match the
        # zone's forming candle at all, so the zone can never be tracked.
        disjoint_candles = [
            candle(100, 100.0, 100.5, 99.5, 100.0),
            candle(101, 100.0, 100.5, 99.5, 100.0),
        ]
        with self.assertRaises(ValueError):
            detect_fvg_touch_events(disjoint_candles, fvg_result, "1m")


if __name__ == "__main__":
    unittest.main()
