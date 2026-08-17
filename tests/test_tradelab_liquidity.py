#!/usr/bin/env python3
"""Liquidity levels tests: confirmed swings, equal highs/lows clustering,
take (wick-only), and multi-candle reclaim -- no lookahead, no entry logic."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.fvg import Candle
from tradelab.liquidity import (
    SUPPORTED_TIMEFRAMES,
    LiquidityStatus,
    LiquidityType,
    ReclaimOutcome,
    detect_liquidity,
)

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
TICK = 0.25


def candle(offset_minutes: float, o: float, h: float, l: float, c: float, known_offset_minutes: float | None = None) -> Candle:
    ts = T0 + timedelta(minutes=offset_minutes)
    known_at = T0 + timedelta(minutes=known_offset_minutes if known_offset_minutes is not None else offset_minutes + 1)
    return Candle(ts, o, h, l, c, known_at)


def swing_high_series(pivot_strength: int, pivot_high: float = 100.0, base: float = 99.0) -> list[Candle]:
    """pivot_strength candles below pivot_high on each side, pivot in the middle."""
    left = [candle(i, base, base + 0.2, base - 0.2, base) for i in range(pivot_strength)]
    pivot = candle(pivot_strength, base + 0.3, pivot_high, base, base + 0.1)
    right = [candle(pivot_strength + 1 + i, base, base + 0.2, base - 0.2, base) for i in range(pivot_strength)]
    return left + [pivot] + right


def swing_low_series(pivot_strength: int, pivot_low: float = 99.0, base: float = 100.0) -> list[Candle]:
    left = [candle(i, base, base + 0.2, base - 0.2, base) for i in range(pivot_strength)]
    pivot = candle(pivot_strength, base - 0.3, base, pivot_low, base - 0.1)
    right = [candle(pivot_strength + 1 + i, base, base + 0.2, base - 0.2, base) for i in range(pivot_strength)]
    return left + [pivot] + right


class SwingConfirmationTests(unittest.TestCase):
    def test_swing_high_is_confirmed_after_right_side_candles_close(self):
        candles = swing_high_series(pivot_strength=1)
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        highs = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.SWING_HIGH]
        self.assertEqual(len(highs), 1)
        self.assertEqual(highs[0].lower, 100.0)
        self.assertEqual(highs[0].upper, 100.0)
        self.assertEqual(highs[0].status, LiquidityStatus.ACTIVE)
        # formed_at must be the known_at of the LAST confirming (right-side) candle.
        self.assertEqual(highs[0].formed_at, candles[-1].known_at)

    def test_swing_low_is_confirmed_after_right_side_candles_close(self):
        candles = swing_low_series(pivot_strength=1)
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        lows = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.SWING_LOW]
        self.assertEqual(len(lows), 1)
        self.assertEqual(lows[0].lower, 99.0)
        self.assertEqual(lows[0].formed_at, candles[-1].known_at)

    def test_swing_not_confirmed_without_enough_right_side_candles(self):
        # Only 1 of the 2 required right-side candles is present.
        pivot_strength = 2
        left = [candle(i, 99, 99.2, 98.8, 99) for i in range(pivot_strength)]
        pivot = candle(pivot_strength, 99.3, 100.0, 99.0, 99.1)
        right = [candle(pivot_strength + 1, 99, 99.2, 98.8, 99)]  # only 1, need 2
        result = detect_liquidity(left + [pivot] + right, "1m", pivot_strength=pivot_strength, reclaim_window_bars=3)
        self.assertEqual(result.levels, [])

    def test_candle_forming_the_swing_is_not_itself_a_take_candidate(self):
        # The right-side confirming candles structurally cannot exceed the
        # pivot's own high/low (that's exactly why the swing qualifies), so
        # the level must still be ACTIVE right after confirmation.
        candles = swing_high_series(pivot_strength=2)
        result = detect_liquidity(candles, "1m", pivot_strength=2, reclaim_window_bars=3)
        self.assertEqual(result.levels[0].status, LiquidityStatus.ACTIVE)
        self.assertEqual(result.takes, [])


class PivotStrengthTests(unittest.TestCase):
    def test_various_pivot_strengths_all_confirm_correctly(self):
        for pivot_strength in (1, 2, 3, 5, 8):
            with self.subTest(pivot_strength=pivot_strength):
                candles = swing_high_series(pivot_strength=pivot_strength)
                result = detect_liquidity(candles, "1m", pivot_strength=pivot_strength, reclaim_window_bars=3)
                highs = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.SWING_HIGH]
                self.assertEqual(len(highs), 1, f"pivot_strength={pivot_strength}")
                self.assertEqual(highs[0].formed_at, candles[-1].known_at)

    def test_non_positive_pivot_strength_rejected(self):
        candles = swing_high_series(pivot_strength=1)
        with self.assertRaises(ValueError):
            detect_liquidity(candles, "1m", pivot_strength=0, reclaim_window_bars=3)
        with self.assertRaises(ValueError):
            detect_liquidity(candles, "1m", pivot_strength=-1, reclaim_window_bars=3)


class EqualHighsLowsTests(unittest.TestCase):
    def _two_swing_highs(self, gap: float) -> list[Candle]:
        return [
            candle(0, 99, 99.2, 98.8, 99),
            candle(1, 99.3, 100.0, 98.9, 99.5),  # swing high = 100.0
            candle(2, 99.5, 99.6, 99.0, 99.2),
            candle(3, 99.6, 99.7, 99.1, 99.3),
            candle(4, 99.8, 100.0 + gap, 99.4, 100.0),  # swing high = 100.0 + gap
            candle(5, 99.9, 100.0, 99.5, 99.7),
        ]

    def test_equal_highs_cluster_forms_from_two_swings_within_2_ticks(self):
        result = detect_liquidity(self._two_swing_highs(0.2), "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        clusters = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_HIGHS]
        self.assertEqual(len(clusters), 1)
        self.assertEqual((clusters[0].lower, clusters[0].upper), (100.0, 100.2))
        self.assertEqual(clusters[0].member_count, 2)

    def test_exactly_2_ticks_apart_qualifies(self):
        result = detect_liquidity(self._two_swing_highs(2 * TICK), "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        clusters = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_HIGHS]
        self.assertEqual(len(clusters), 1)

    def test_more_than_2_ticks_apart_is_rejected(self):
        result = detect_liquidity(self._two_swing_highs(2 * TICK + 0.1), "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        clusters = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_HIGHS]
        self.assertEqual(clusters, [])

    def test_equal_lows_cluster_forms_symmetrically(self):
        candles = [
            candle(0, 99.5, 99.8, 99.3, 99.5),
            candle(1, 99.2, 99.5, 99.0, 99.3),  # swing low = 99.0
            candle(2, 99.4, 99.6, 99.2, 99.5),
            candle(3, 99.3, 99.5, 99.1, 99.4),
            candle(4, 99.1, 99.3, 98.8, 99.0),  # swing low = 98.8
            candle(5, 99.2, 99.4, 99.0, 99.3),
        ]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        clusters = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_LOWS]
        self.assertEqual(len(clusters), 1)
        self.assertEqual((clusters[0].lower, clusters[0].upper), (98.8, 99.0))

    def test_third_matching_swing_grows_member_count_not_a_new_level(self):
        candles = self._two_swing_highs(0.2) + [
            candle(6, 99.7, 99.8, 99.3, 99.5),
            candle(7, 99.9, 100.15, 99.4, 99.95),  # 3rd swing high = 100.15, within band of [100.0, 100.2]
            candle(8, 99.8, 99.9, 99.4, 99.6),
        ]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        clusters = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_HIGHS]
        self.assertEqual(len(clusters), 1)  # still exactly one level, not a second one
        self.assertEqual(clusters[0].member_count, 3)

    def test_cluster_known_at_is_the_second_swings_confirmation_not_the_first(self):
        candles = self._two_swing_highs(0.2)
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        clusters = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_HIGHS]
        self.assertEqual(clusters[0].formed_at, candles[5].known_at)  # 2nd swing's right-side confirm

    def test_swings_from_different_days_still_cluster(self):
        far_future_offset = 60 * 24 * 30  # 30 days later, in minutes
        candles = [
            candle(0, 99, 99.2, 98.8, 99),
            candle(1, 99.3, 100.0, 98.9, 99.5),  # swing high day 1
            candle(2, 99.5, 99.6, 99.0, 99.2),
            candle(far_future_offset, 99.6, 99.7, 99.1, 99.3),
            candle(far_future_offset + 1, 99.8, 100.15, 99.4, 100.0),  # swing high 30 days later
            candle(far_future_offset + 2, 99.9, 100.0, 99.5, 99.7),
        ]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        clusters = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_HIGHS]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].member_count, 2)

    def test_cluster_band_growth_does_not_retroactively_affect_earlier_candles(self):
        # A candle right after the 2-member cluster forms breaches the
        # *narrow* band; this must count as a take using the band as it
        # stood then, never a wider band from a member that hasn't
        # confirmed yet.
        candles = self._two_swing_highs(0.2) + [
            candle(6, 100.1, 100.3, 99.9, 99.95),  # high=100.3 >= narrow upper 100.2 -> TAKE
        ]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        clusters = [lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_HIGHS]
        self.assertEqual(clusters[0].status, LiquidityStatus.TAKEN)

    def test_taken_cluster_stops_absorbing_new_members(self):
        # With pivot_strength=1, almost any local extremum becomes its own
        # swing, so later filler candles legitimately spawn *other*,
        # unrelated levels -- that's expected and fine. What must hold is
        # that the *original* [100.0, 100.2] cluster specifically never
        # grows past its 2 members once taken, even though a later,
        # separately-confirmed swing (100.1) falls right back inside its
        # old band.
        candles = self._two_swing_highs(0.2) + [
            candle(6, 100.0, 100.2, 99.9, 100.0),  # takes the [100.0, 100.2] cluster (high >= 100.2)
            candle(7, 99.8, 99.9, 99.5, 99.7),
            candle(8, 99.7, 99.8, 99.4, 99.6),
            candle(9, 99.6, 99.7, 99.3, 99.5),
            candle(10, 99.8, 100.1, 99.4, 99.95),  # a new, independent swing at 100.1 -- inside the old band
            candle(11, 99.5, 99.6, 99.2, 99.4),
        ]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3, tick_size=TICK)
        original_cluster = next(
            lv for lv in result.levels if lv.liquidity_type == LiquidityType.EQUAL_HIGHS and lv.lower == 100.0
        )
        self.assertEqual(original_cluster.member_count, 2)  # did not grow after being taken
        self.assertEqual(original_cluster.status, LiquidityStatus.TAKEN)
        self.assertEqual(original_cluster.upper, 100.2)  # band did not widen to absorb 100.1 either


class TakeTests(unittest.TestCase):
    def test_take_requires_only_a_wick_no_close_beyond_needed(self):
        candles = swing_high_series(pivot_strength=1) + [
            candle(4, 99.5, 100.5, 99.4, 99.6),  # wick to 100.5, closes back at 99.6 -- no close beyond
        ]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        self.assertEqual(len(result.takes), 1)
        self.assertEqual(result.levels[0].status, LiquidityStatus.TAKEN)

    def test_first_take_consumes_the_level(self):
        candles = swing_high_series(pivot_strength=1) + [
            candle(4, 99.5, 100.5, 99.4, 99.6),  # first take
            candle(5, 99.5, 100.8, 99.4, 99.6),  # a second, bigger breach -- must not double-count
        ]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        self.assertEqual(len(result.takes), 1)

    def test_swing_low_take_symmetric(self):
        candles = swing_low_series(pivot_strength=1) + [
            candle(4, 99.5, 99.6, 98.5, 99.4),  # wick below 99.0
        ]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        self.assertEqual(len(result.takes), 1)


class ReclaimTests(unittest.TestCase):
    def _taken_swing_high(self, tail: list[Candle]) -> list[Candle]:
        base = [
            candle(0, 99, 99.2, 98.8, 99),
            candle(1, 99.3, 100.0, 98.9, 99.5),  # swing high = 100.0, confirmed at idx2 close
            candle(2, 99.5, 99.6, 99.0, 99.2),
        ]
        return base + tail

    def test_reclaim_wick(self):
        candles = self._taken_swing_high(
            [
                candle(3, 100.0, 100.1, 99.8, 99.9),  # take (high >= 100.0)
                candle(4, 100.05, 100.1, 99.8, 100.05),  # wick below 100.0, closes back above -> wick reclaim
            ]
        )
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.RECLAIM_WICK)
        self.assertEqual(result.reclaims[0].bars_after_take, 1)

    def test_reclaim_close(self):
        candles = self._taken_swing_high(
            [
                candle(3, 100.0, 100.1, 99.8, 99.9),  # take
                candle(4, 100.0, 100.05, 99.7, 99.8),  # closes below 100.0 -> close reclaim
            ]
        )
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.RECLAIM_CLOSE)
        self.assertEqual(result.reclaims[0].bars_after_take, 1)

    def test_reclaim_appearing_several_bars_after_take(self):
        candles = self._taken_swing_high(
            [
                candle(3, 100.0, 100.1, 99.8, 99.9),  # take
                candle(4, 100.05, 100.2, 100.0, 100.1),  # no reclaim
                candle(5, 100.05, 100.2, 100.0, 100.1),  # no reclaim
                candle(6, 100.0, 100.1, 99.7, 99.8),  # close reclaim, 3rd bar after take
            ]
        )
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.RECLAIM_CLOSE)
        self.assertEqual(result.reclaims[0].bars_after_take, 3)

    def test_no_reclaim_in_window_is_explicit(self):
        candles = self._taken_swing_high(
            [
                candle(3, 100.0, 100.1, 99.8, 99.9),  # take
                candle(4, 100.05, 100.2, 100.0, 100.1),  # no reclaim
                candle(5, 100.05, 100.2, 100.0, 100.1),  # no reclaim, window (2) now exhausted
            ]
        )
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=2)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.NO_RECLAIM_IN_WINDOW)
        self.assertEqual(result.reclaims[0].bars_after_take, 2)
        self.assertEqual(result.reclaims[0].resolved_at, candles[-1].known_at)

    def test_pending_when_window_not_yet_fully_observed(self):
        candles = self._taken_swing_high(
            [
                candle(3, 100.0, 100.1, 99.8, 99.9),  # take
                candle(4, 100.05, 100.2, 100.0, 100.1),  # only 1 of 5 requested window bars available
            ]
        )
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=5)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.PENDING)
        self.assertIsNone(result.reclaims[0].resolved_at)
        self.assertIsNone(result.reclaims[0].bars_after_take)

    def test_reclaim_is_never_evaluated_on_the_take_candle_itself(self):
        # The take candle (100.0, 100.1, 99.8, 99.9) itself closes below 100.0
        # (99.9 < 100.0): if it were wrongly counted as its own first reclaim
        # candle, this would immediately resolve as RECLAIM_CLOSE at
        # bars_after_take=1 with resolved_at == the take's own known_at.
        candles = self._taken_swing_high(
            [
                candle(3, 100.0, 100.1, 99.8, 99.9),  # take; close 99.9 is already < 100.0
                candle(4, 100.05, 100.2, 100.0, 100.1),  # genuinely no reclaim
                candle(5, 100.05, 100.2, 100.0, 100.1),  # genuinely no reclaim; window (2) exhausted
            ]
        )
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=2)
        take = result.takes[0]
        reclaim = result.reclaims[0]
        self.assertNotEqual(reclaim.resolved_at, take.known_at)
        self.assertEqual(reclaim.outcome, ReclaimOutcome.NO_RECLAIM_IN_WINDOW)


class IndependentLevelsTests(unittest.TestCase):
    def test_multiple_levels_tracked_and_resolved_independently(self):
        # A swing high at ~110 and a swing low at ~90, placed far apart in
        # time (the low series starts well after the high series ends) so
        # each level's take only reflects candles genuinely relevant to it.
        high_series = swing_high_series(pivot_strength=1, pivot_high=110.0, base=109.0)
        take_high = candle(len(high_series) + 1, 109.5, 110.5, 109.4, 109.6)  # takes the high only

        shift = 100
        low_series = [
            candle(shift + i, c.open, c.high, c.low, c.close, known_offset_minutes=shift + i + 1)
            for i, c in enumerate(swing_low_series(pivot_strength=1, pivot_low=90.0, base=91.0))
        ]
        take_low = candle(shift + len(low_series) + 1, 91.0, 91.1, 89.0, 90.8)  # takes the low only

        candles = high_series + [take_high] + low_series + [take_low]
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)

        # With pivot_strength=1 the flat filler candles in these helper series
        # legitimately spawn a couple of extra, unrelated swings of their own
        # (e.g. a minor low right next to the intended high) -- expected, and
        # irrelevant here. What matters is that the two *intended* levels
        # (110.0 high, 90.0 low) each resolve correctly and independently.
        intended_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SWING_HIGH and lv.lower == 110.0)
        intended_low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SWING_LOW and lv.lower == 90.0)
        self.assertEqual(intended_high.status, LiquidityStatus.TAKEN)
        self.assertEqual(intended_low.status, LiquidityStatus.TAKEN)
        self.assertIn(intended_high.level_id, [t.level_id for t in result.takes])
        self.assertIn(intended_low.level_id, [t.level_id for t in result.takes])
        self.assertIn(intended_high.level_id, [r.level_id for r in result.reclaims])
        self.assertIn(intended_low.level_id, [r.level_id for r in result.reclaims])


class TimeframeTests(unittest.TestCase):
    def test_every_supported_timeframe_detects_correctly(self):
        for timeframe in sorted(SUPPORTED_TIMEFRAMES):
            with self.subTest(timeframe=timeframe):
                candles = swing_high_series(pivot_strength=1)
                result = detect_liquidity(candles, timeframe, pivot_strength=1, reclaim_window_bars=3)
                self.assertEqual(len(result.levels), 1)
                self.assertEqual(result.levels[0].timeframe, timeframe)

    def test_all_six_native_timeframes_supported(self):
        self.assertEqual(SUPPORTED_TIMEFRAMES, frozenset({"1m", "5m", "15m", "1H", "4H", "1D"}))

    def test_unsupported_timeframe_raises(self):
        candles = swing_high_series(pivot_strength=1)
        with self.assertRaises(ValueError):
            detect_liquidity(candles, "3m", pivot_strength=1, reclaim_window_bars=3)


class KnownAtChronologyTests(unittest.TestCase):
    def test_candles_must_be_strictly_ordered_by_timestamp(self):
        candles = swing_high_series(pivot_strength=1)
        out_of_order = candle(0, 50.0, 50.5, 49.5, 50.0)
        with self.assertRaises(ValueError):
            detect_liquidity(candles + [out_of_order], "1m", pivot_strength=1, reclaim_window_bars=3)

    def test_candles_must_have_strictly_increasing_known_at(self):
        c1 = candle(0, 100.0, 100.0, 99.0, 99.5, known_offset_minutes=10)
        c2 = candle(1, 100.5, 101.0, 100.2, 100.8, known_offset_minutes=2)
        with self.assertRaises(ValueError):
            detect_liquidity([c1, c2], "1m", pivot_strength=1, reclaim_window_bars=3)

    def test_all_recorded_timestamps_are_actual_candle_known_at_values(self):
        candles = swing_high_series(pivot_strength=1) + [
            candle(4, 99.5, 100.5, 99.4, 99.6),  # take
        ]
        known_ats = {c.known_at for c in candles}
        result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        for level in result.levels:
            self.assertIn(level.formed_at, known_ats)
            if level.taken_at is not None:
                self.assertIn(level.taken_at, known_ats)
        for take in result.takes:
            self.assertIn(take.timestamp, known_ats)
            self.assertEqual(take.timestamp, take.known_at)
        for reclaim in result.reclaims:
            if reclaim.resolved_at is not None:
                self.assertIn(reclaim.resolved_at, known_ats)


if __name__ == "__main__":
    unittest.main()
