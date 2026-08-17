#!/usr/bin/env python3
"""FVG/iFVG detector tests: formation, tick-size threshold, lifecycle,
one-shot inversion, overlapping zones, no-lookahead, and known_at."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.fvg import (
    DEFAULT_TICK_SIZE,
    SUPPORTED_TIMEFRAMES,
    Candle,
    FVGStatus,
    detect_fvgs,
)

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
CONTRACT_TICK = 0.25  # NQ/MNQ default, passed explicitly -- never assumed by the module


def candle(offset_minutes: float, o: float, h: float, l: float, c: float, known_offset_minutes: float | None = None) -> Candle:
    ts = T0 + timedelta(minutes=offset_minutes)
    known_at = T0 + timedelta(minutes=known_offset_minutes if known_offset_minutes is not None else offset_minutes + 1)
    return Candle(ts, o, h, l, c, known_at)


def bullish_setup(gap_ticks: float, tick_size: float = CONTRACT_TICK) -> list[Candle]:
    """candle_1.high = 100; candle_3.low = 100 + gap_ticks * tick_size."""
    gap = gap_ticks * tick_size
    c1 = candle(0, 100.0, 100.0, 99.0, 99.5)
    c2 = candle(1, 100.5, 101.0, 100.2, 100.8)
    low3 = 100.0 + gap
    c3 = candle(2, low3 + 0.3, low3 + 0.5, low3, low3 + 0.35)
    return [c1, c2, c3]


def bearish_setup(gap_ticks: float, tick_size: float = CONTRACT_TICK) -> list[Candle]:
    """candle_1.low = 100; candle_3.high = 100 - gap_ticks * tick_size."""
    gap = gap_ticks * tick_size
    c1 = candle(0, 100.5, 101.0, 100.0, 100.8)
    c2 = candle(1, 99.8, 100.2, 99.5, 99.9)
    high3 = 100.0 - gap
    c3 = candle(2, high3 - 0.35, high3, high3 - 0.5, high3 - 0.3)
    return [c1, c2, c3]


class FormationTests(unittest.TestCase):
    def test_bullish_fvg_is_detected(self):
        result = detect_fvgs(bullish_setup(5), "1m")
        self.assertEqual(len(result.zones), 1)
        zone = result.zones[0]
        self.assertEqual(zone.direction, "bullish")
        self.assertEqual((zone.lower, zone.upper), (100.0, 101.25))
        self.assertEqual(zone.status, FVGStatus.ACTIVE)

    def test_bearish_fvg_is_detected(self):
        result = detect_fvgs(bearish_setup(5), "1m")
        self.assertEqual(len(result.zones), 1)
        zone = result.zones[0]
        self.assertEqual(zone.direction, "bearish")
        self.assertEqual((zone.lower, zone.upper), (98.75, 100.0))
        self.assertEqual(zone.status, FVGStatus.ACTIVE)

    def test_no_gap_produces_no_zone(self):
        # candle_1.high >= candle_3.low and candle_1.low <= candle_3.high: no bullish or bearish condition.
        c1 = candle(0, 100.0, 100.5, 99.5, 100.0)
        c2 = candle(1, 100.2, 100.6, 99.8, 100.1)
        c3 = candle(2, 100.1, 100.4, 99.9, 100.0)
        result = detect_fvgs([c1, c2, c3], "1m")
        self.assertEqual(result.zones, [])


class TickThresholdTests(unittest.TestCase):
    def test_exactly_5_ticks_qualifies(self):
        result = detect_fvgs(bullish_setup(5), "1m", tick_size=CONTRACT_TICK)
        self.assertEqual(len(result.zones), 1)

    def test_exactly_4_ticks_is_rejected(self):
        result = detect_fvgs(bullish_setup(4), "1m", tick_size=CONTRACT_TICK)
        self.assertEqual(result.zones, [])

    def test_bearish_exactly_5_vs_4_ticks(self):
        self.assertEqual(len(detect_fvgs(bearish_setup(5), "1m", tick_size=CONTRACT_TICK).zones), 1)
        self.assertEqual(detect_fvgs(bearish_setup(4), "1m", tick_size=CONTRACT_TICK).zones, [])

    def test_tick_size_is_a_parameter_not_hardcoded(self):
        # A gap of 5 * 0.01 only qualifies when tick_size=0.01 is actually passed through,
        # not if the module silently assumed the NQ default of 0.25.
        candles = bullish_setup(5, tick_size=0.01)
        self.assertEqual(detect_fvgs(candles, "1m", tick_size=0.01).zones.__len__(), 1)
        self.assertEqual(detect_fvgs(candles, "1m", tick_size=DEFAULT_TICK_SIZE).zones, [])

    def test_default_tick_size_is_nq_quarter_point(self):
        self.assertEqual(DEFAULT_TICK_SIZE, 0.25)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.bullish = bullish_setup(5)  # zone [100.0, 101.25]

    def test_partial_wick_touch_stays_active(self):
        touch = candle(3, 100.9, 101.0, 100.5, 100.8)  # low=100.5: strictly inside (100, 101.25)
        result = detect_fvgs(self.bullish + [touch], "1m")
        self.assertEqual(result.zones[0].status, FVGStatus.ACTIVE)
        self.assertEqual(result.events, [])

    def test_touching_only_the_near_boundary_is_not_real_entry(self):
        # low == upper exactly: grazing the boundary, not entering the interior.
        touch = candle(3, 101.3, 101.4, 101.25, 101.3)
        result = detect_fvgs(self.bullish + [touch], "1m")
        self.assertEqual(result.zones[0].status, FVGStatus.ACTIVE)

    def test_full_pass_through_close_inside_is_mitigated_without_ifvg(self):
        full = candle(3, 101.0, 101.2, 99.9, 100.5)  # low<=100 (full), close 100.5 inside zone
        result = detect_fvgs(self.bullish + [full], "1m")
        self.assertEqual(result.zones[0].status, FVGStatus.MITIGATED)
        self.assertFalse(result.zones[0].produced_ifvg)
        self.assertEqual(result.events, [])

    def test_full_pass_through_close_beyond_produces_exactly_one_inverted_ifvg(self):
        invert = candle(3, 100.8, 100.9, 99.9, 99.5)  # low<=100 (full), close 99.5 < lower(100)
        result = detect_fvgs(self.bullish + [invert], "1m")
        self.assertEqual(result.zones[0].status, FVGStatus.MITIGATED)
        self.assertTrue(result.zones[0].produced_ifvg)
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.direction, "bearish")
        self.assertEqual(event.original_zone_direction, "bullish")
        self.assertEqual((event.lower, event.upper), (100.0, 101.25))

    def test_bearish_mitigation_and_inversion_mirror_bullish(self):
        bearish = bearish_setup(5)  # zone [98.75, 100.0]
        inside = candle(3, 99.0, 100.1, 98.9, 99.5)  # high>=100 (full), close inside -> mitigated
        result = detect_fvgs(bearish + [inside], "1m")
        self.assertEqual(result.zones[0].status, FVGStatus.MITIGATED)
        self.assertEqual(result.events, [])

        beyond = candle(3, 99.9, 100.2, 99.8, 100.5)  # high>=100 (full), close 100.5 > upper(100)
        result2 = detect_fvgs(bearish + [beyond], "1m")
        self.assertTrue(result2.zones[0].produced_ifvg)
        self.assertEqual(result2.events[0].direction, "bullish")
        self.assertEqual(result2.events[0].original_zone_direction, "bearish")

    def test_no_duplicate_ifvg_event_after_inversion(self):
        invert = candle(3, 100.8, 100.9, 99.9, 99.5)
        further = candle(4, 99.4, 99.6, 98.0, 98.5)  # keeps trading through the same zone
        result = detect_fvgs(self.bullish + [invert, further], "1m")
        self.assertEqual(len(result.events), 1)

    def test_mitigated_zone_never_returns_to_active(self):
        full = candle(3, 101.0, 101.2, 99.9, 100.5)  # mitigates, closes inside
        later_partial_looking = candle(4, 100.8, 100.9, 100.6, 100.7)  # would look like a "partial touch" if still active
        result = detect_fvgs(self.bullish + [full, later_partial_looking], "1m")
        self.assertEqual(result.zones[0].status, FVGStatus.MITIGATED)


class OverlappingZonesTests(unittest.TestCase):
    def test_multiple_overlapping_fvgs_tracked_independently(self):
        # Zone A: bullish [100.0, 101.25] from candles 0-2.
        c0, c1, c2 = bullish_setup(5)
        c3 = candle(3, 101.5, 101.6, 101.4, 101.5)  # neutral filler
        # Zone B: bullish [100.9, 102.15] from candles 4-6, genuinely overlapping zone A
        # (100.9 < zone A's upper of 101.25).
        c4 = candle(4, 100.9, 100.9, 100.8, 100.85)
        c5 = candle(5, 101.0, 101.1, 100.9, 101.05)
        c6 = candle(6, 102.3, 102.5, 102.15, 102.4)
        result = detect_fvgs([c0, c1, c2, c3, c4, c5, c6], "1m")
        self.assertEqual(len(result.zones), 2)
        zone_a, zone_b = result.zones
        self.assertEqual((zone_a.lower, zone_a.upper), (100.0, 101.25))
        self.assertEqual((zone_b.lower, zone_b.upper), (100.9, 102.15))
        self.assertTrue(zone_a.upper > zone_b.lower)  # genuinely overlapping ranges

        # One candle, two different independent outcomes: low=100.5 is a real (strict)
        # interior entry into zone A (100 < 100.5 < 101.25, partial -> stays ACTIVE),
        # but a full pass-through of zone B (100.5 <= 100.9) with close 101.5 back
        # inside zone B's range -> zone B resolves to MITIGATED, not inverted.
        differentiator = candle(7, 101.3, 101.4, 100.5, 101.5)
        result2 = detect_fvgs([c0, c1, c2, c3, c4, c5, c6, differentiator], "1m")
        zone_a2, zone_b2 = result2.zones
        self.assertEqual(zone_a2.status, FVGStatus.ACTIVE)
        self.assertEqual(zone_b2.status, FVGStatus.MITIGATED)
        self.assertFalse(zone_b2.produced_ifvg)

    def test_a_shared_candle_evaluates_older_zone_and_forms_a_new_one_correctly(self):
        # Zone A forms on candles 0-2 and stays active (candles 3-4 stay well above it).
        c0, c1, c2 = bullish_setup(5)  # zone A: [100.0, 101.25]
        c3 = candle(3, 102.3, 102.6, 102.15, 102.4)  # candle_1 for zone B; stays above zone A
        c4 = candle(4, 102.0, 102.3, 101.8, 102.1)  # candle_2 for zone B; stays above zone A
        # c5 is candle_3 for zone B (bearish: candle_1(c3).low=102.15 > candle_3(c5).high=100.9)
        # AND simultaneously fully mitigates zone A (low=99.9<=100, close=100.5 inside zone A).
        c5 = candle(5, 100.8, 100.9, 99.9, 100.5)

        result = detect_fvgs([c0, c1, c2, c3, c4, c5], "1m")
        self.assertEqual(len(result.zones), 2)
        zone_a, zone_b = result.zones

        # Zone A: the shared candle c5 correctly resolved it (proves evaluation of a
        # pre-existing zone is not skipped just because the same candle also forms a new one).
        self.assertEqual(zone_a.status, FVGStatus.MITIGATED)
        self.assertEqual(zone_a.resolved_at, c5.known_at)
        self.assertEqual(result.events, [])  # closed inside zone A: mitigation, not inversion

        # Zone B: formed exactly on c5, must still be ACTIVE (not evaluated against itself).
        self.assertEqual(zone_b.direction, "bearish")
        self.assertEqual(zone_b.status, FVGStatus.ACTIVE)
        self.assertIsNone(zone_b.resolved_at)
        self.assertEqual(zone_b.formed_at, c5.known_at)


class FormingCandleNotEvaluatedTests(unittest.TestCase):
    def test_zone_is_active_immediately_after_its_own_forming_candle(self):
        # With only the 3 forming candles present, the zone must be ACTIVE: nothing
        # after it has been processed yet, and the forming candle itself must not
        # have been (impossibly) checked against the zone it just created.
        result = detect_fvgs(bullish_setup(5), "1m")
        self.assertEqual(result.zones[0].status, FVGStatus.ACTIVE)
        self.assertIsNone(result.zones[0].resolved_at)


class TimeframeTests(unittest.TestCase):
    def test_every_supported_timeframe_detects_correctly(self):
        for timeframe in sorted(SUPPORTED_TIMEFRAMES):
            with self.subTest(timeframe=timeframe):
                result = detect_fvgs(bullish_setup(5), timeframe)
                self.assertEqual(len(result.zones), 1)
                self.assertEqual(result.zones[0].timeframe, timeframe)

    def test_all_six_native_timeframes_are_supported(self):
        self.assertEqual(SUPPORTED_TIMEFRAMES, frozenset({"1m", "5m", "15m", "1H", "4H", "1D"}))

    def test_unsupported_timeframe_raises(self):
        with self.assertRaises(ValueError):
            detect_fvgs(bullish_setup(5), "3m")

    def test_fvgs_on_different_timeframes_are_independent_calls(self):
        # Not a real cross-timeframe mixing test (the module never accepts mixed
        # input) -- confirms two separate calls for two timeframes don't share state.
        result_1h = detect_fvgs(bullish_setup(5), "1H")
        result_4h = detect_fvgs(bullish_setup(5), "4H")
        self.assertEqual(result_1h.zones[0].zone_id, result_4h.zones[0].zone_id.replace("4H", "1H"))
        self.assertNotEqual(result_1h.zones[0].timeframe, result_4h.zones[0].timeframe)


class NoLookaheadAndKnownAtTests(unittest.TestCase):
    def test_formed_at_is_the_third_candles_known_at_not_its_open(self):
        c0, c1, c2 = bullish_setup(5)
        result = detect_fvgs([c0, c1, c2], "1m")
        self.assertEqual(result.zones[0].formed_at, c2.known_at)
        self.assertNotEqual(result.zones[0].formed_at, c2.timestamp)

    def test_event_timestamp_and_known_at_equal_the_confirming_candles_known_at(self):
        candles = bullish_setup(5)
        invert = candle(3, 100.8, 100.9, 99.9, 99.5)
        result = detect_fvgs(candles + [invert], "1m")
        event = result.events[0]
        self.assertEqual(event.timestamp, invert.known_at)
        self.assertEqual(event.known_at, invert.known_at)
        self.assertEqual(event.timestamp, event.known_at)

    def test_appending_future_candles_does_not_change_an_already_resolved_zone(self):
        candles = bullish_setup(5)
        invert = candle(3, 100.8, 100.9, 99.9, 99.5)
        base = candles + [invert]
        extra_future = [candle(4, 200.0, 201.0, 199.0, 200.5), candle(5, 300.0, 301.0, 299.0, 300.5)]

        result_short = detect_fvgs(base, "1m")
        result_long = detect_fvgs(base + extra_future, "1m")

        zone_short = result_short.zones[0]
        zone_long = result_long.zones[0]
        self.assertEqual(zone_short.status, zone_long.status)
        self.assertEqual(zone_short.resolved_at, zone_long.resolved_at)
        self.assertEqual(zone_short.produced_ifvg, zone_long.produced_ifvg)
        self.assertEqual(len(result_short.events), len(result_long.events))
        self.assertEqual(result_short.events[0], result_long.events[0])

    def test_zone_with_no_resolving_candle_yet_stays_active_with_no_resolved_at(self):
        result = detect_fvgs(bullish_setup(5), "1m")
        self.assertIsNone(result.zones[0].resolved_at)

    def test_candles_must_be_strictly_ordered(self):
        c0, c1, c2 = bullish_setup(5)
        out_of_order = candle(1, 50.0, 50.5, 49.5, 50.0)  # earlier timestamp than c2
        with self.assertRaises(ValueError):
            detect_fvgs([c0, c1, c2, out_of_order], "1m")

    def test_increasing_timestamps_with_decreasing_known_at_is_rejected(self):
        # Each candle individually has known_at > its own timestamp, but the
        # second candle's known_at is earlier than the first's -- timestamps
        # alone increasing is not enough to rule out lookahead.
        c1 = candle(0, 100.0, 100.0, 99.0, 99.5, known_offset_minutes=10)
        c2 = candle(1, 100.5, 101.0, 100.2, 100.8, known_offset_minutes=2)
        with self.assertRaises(ValueError):
            detect_fvgs([c1, c2], "1m")

    def test_third_candle_known_far_in_future_then_earlier_known_at_is_rejected(self):
        # Regression: the third candle forms an FVG but is (malformed-input)
        # known far in the future; the next candle has an earlier known_at.
        # The detector must reject this outright rather than evaluate the
        # zone's lifecycle against a candle "known" before the zone itself.
        c1 = candle(0, 100.0, 100.0, 99.0, 99.5, known_offset_minutes=1)
        c2 = candle(1, 100.5, 101.0, 100.2, 100.8, known_offset_minutes=2)
        c3 = candle(2, 101.55, 101.75, 101.25, 101.6, known_offset_minutes=100)  # forms zone, known far ahead
        c4 = candle(3, 100.8, 100.9, 99.9, 99.5, known_offset_minutes=4)  # would invert the zone, if evaluated
        with self.assertRaises(ValueError):
            detect_fvgs([c1, c2, c3, c4], "1m")


class CandleKnownAtValidationTests(unittest.TestCase):
    def test_known_at_equal_to_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            Candle(T0, 100.0, 100.5, 99.5, 100.0, T0)

    def test_known_at_before_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            Candle(T0, 100.0, 100.5, 99.5, 100.0, T0 - timedelta(minutes=1))

    def test_known_at_after_timestamp_is_accepted(self):
        c = Candle(T0, 100.0, 100.5, 99.5, 100.0, T0 + timedelta(minutes=1))
        self.assertGreater(c.known_at, c.timestamp)


if __name__ == "__main__":
    unittest.main()
