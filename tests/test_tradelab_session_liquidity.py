#!/usr/bin/env python3
"""Session liquidity tests: high/low of fixed America/New_York windows,
activation only after the session actually closes, strict take/reclaim
(reused from tradelab.liquidity), overlapping windows, DST correctness,
empty/incomplete sessions, and no lookahead."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.fvg import Candle
from tradelab.session_liquidity import (
    LiquidityStatus,
    LiquidityType,
    ReclaimOutcome,
    detect_session_liquidity,
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def ny_candle(ny_dt: datetime, o: float, h: float, l: float, c: float, minutes: int = 1) -> Candle:
    ts = ny_dt.astimezone(UTC)
    return Candle(ts, o, h, l, c, ts + timedelta(minutes=minutes))


def utc_candle(ts: datetime, o: float, h: float, l: float, c: float, minutes: int = 1) -> Candle:
    return Candle(ts, o, h, l, c, ts + timedelta(minutes=minutes))


# A plain Tuesday, far from any DST transition, used as the base date for
# tests that aren't specifically about DST.
DAY = datetime(2026, 1, 6, tzinfo=NY)

NON_WRAPPING_SESSIONS = [
    ("london", 2, 0, 5, 0, LiquidityType.SESSION_LONDON_HIGH, LiquidityType.SESSION_LONDON_LOW),
    ("pre_ny", 8, 30, 9, 30, LiquidityType.SESSION_PRE_NY_HIGH, LiquidityType.SESSION_PRE_NY_LOW),
    ("ny_am", 9, 30, 12, 0, LiquidityType.SESSION_NY_AM_HIGH, LiquidityType.SESSION_NY_AM_LOW),
    ("ny_lunch", 12, 0, 13, 0, LiquidityType.SESSION_NY_LUNCH_HIGH, LiquidityType.SESSION_NY_LUNCH_LOW),
    ("ny_pm", 13, 0, 16, 0, LiquidityType.SESSION_NY_PM_HIGH, LiquidityType.SESSION_NY_PM_LOW),
]


class SessionBoundaryTests(unittest.TestCase):
    def test_each_non_wrapping_session_uses_only_its_own_window(self):
        for key, sh, sm, eh, em, high_type, low_type in NON_WRAPPING_SESSIONS:
            with self.subTest(session=key):
                start = DAY.replace(hour=sh, minute=sm)
                end = DAY.replace(hour=eh, minute=em)
                candles = [
                    ny_candle(start - timedelta(minutes=1), 200, 200.1, 199.9, 200),  # before start -- excluded
                    ny_candle(start, 100, 101, 99, 100.5),  # start, inclusive
                    ny_candle(start + timedelta(minutes=1), 100.5, 105, 99.5, 101),  # inside -- session high
                    ny_candle(end - timedelta(minutes=1), 101, 102, 98, 101.5),  # last minute, inclusive -- session low
                    ny_candle(end, 300, 300.1, 299.9, 300),  # end, exclusive -- confirms close only
                ]
                result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
                high = next(lv for lv in result.levels if lv.liquidity_type == high_type)
                low = next(lv for lv in result.levels if lv.liquidity_type == low_type)
                self.assertEqual(high.upper, 105)
                self.assertEqual(low.lower, 98)
                self.assertEqual(high.member_count, 3)  # only the 3 genuinely in-window candles
                self.assertEqual(high.status, LiquidityStatus.ACTIVE)

    def test_asia_session_spans_midnight_and_excludes_out_of_window_candles(self):
        candles = [
            ny_candle(DAY.replace(hour=19, minute=59), 200, 200.1, 199.9, 200),  # before start -- excluded
            ny_candle(DAY.replace(hour=20, minute=0), 100, 101, 99, 100.5),  # start, inclusive
            ny_candle(DAY.replace(hour=23, minute=59), 100.5, 105, 98, 101),  # last minute before midnight, inclusive
            ny_candle(DAY + timedelta(days=1), 300, 300.1, 299.9, 300),  # next day 00:00 -- exclusive
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_ASIA_HIGH)
        low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_ASIA_LOW)
        self.assertEqual(high.upper, 105)
        self.assertEqual(low.lower, 98)
        self.assertEqual(high.member_count, 2)

    def test_overnight_session_spans_midnight_into_next_morning(self):
        candles = [
            ny_candle(DAY.replace(hour=17, minute=59), 200, 200.1, 199.9, 200),  # before start -- excluded
            ny_candle(DAY.replace(hour=18, minute=0), 100, 101, 99, 100.5),  # start, inclusive
            ny_candle(DAY + timedelta(days=1, hours=9, minutes=29), 100.5, 105, 98, 101),  # 09:29 next day, inclusive
            ny_candle(DAY + timedelta(days=1, hours=9, minutes=30), 300, 300.1, 299.9, 300),  # 09:30 next day -- exclusive
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_HIGH)
        low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_LOW)
        self.assertEqual(high.upper, 105)
        self.assertEqual(low.lower, 98)

    def test_pre_ny_end_and_ny_am_start_hand_off_cleanly_at_09_30(self):
        # 09:30 exactly belongs to NY AM (start inclusive), never to Pre-NY
        # (end exclusive) -- no double counting at the shared boundary.
        candle_at_boundary = ny_candle(DAY.replace(hour=9, minute=30), 150, 150.5, 149.5, 150)
        candles = [
            ny_candle(DAY.replace(hour=8, minute=30), 100, 101, 99, 100.5),
            candle_at_boundary,
            ny_candle(DAY.replace(hour=9, minute=31), 150, 150.1, 149.9, 150),
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        pre_ny_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_PRE_NY_HIGH)
        self.assertNotIn(candle_at_boundary.timestamp, pre_ny_high.member_timestamps)
        self.assertEqual(pre_ny_high.upper, 101)  # the 09:30 spike never entered Pre-NY's high


class ActivationTimingTests(unittest.TestCase):
    def test_no_level_and_no_spurious_take_while_session_is_still_forming(self):
        start = DAY.replace(hour=2, minute=0)  # London
        candles = [
            ny_candle(start, 100, 101, 99, 100.5),
            ny_candle(start + timedelta(minutes=1), 100.5, 103, 99.5, 101),  # a new high within the still-open session
            ny_candle(start + timedelta(minutes=2), 101, 102, 98, 100),  # a new low within the still-open session
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual(result.takes, [])
        self.assertEqual([lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH], [])

    def test_level_becomes_known_only_at_the_sessions_own_last_candle_not_the_confirming_one(self):
        start = DAY.replace(hour=2, minute=0)
        end = DAY.replace(hour=5, minute=0)
        in_session = [
            ny_candle(start, 100, 101, 99, 100.5),
            ny_candle(end - timedelta(minutes=1), 100.5, 102, 99.5, 101),
        ]
        result_incomplete = detect_session_liquidity(in_session, "1m", reclaim_window_bars=3)
        self.assertEqual([lv for lv in result_incomplete.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH], [])

        confirming = ny_candle(end, 300, 300.1, 299.9, 300)
        result = detect_session_liquidity(in_session + [confirming], "1m", reclaim_window_bars=3)
        high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH)
        self.assertEqual(high.status, LiquidityStatus.ACTIVE)
        self.assertEqual(high.formed_at, in_session[-1].known_at)  # last real member, not the confirming candle
        self.assertNotEqual(high.formed_at, confirming.known_at)


class TakeAndReclaimTests(unittest.TestCase):
    def _confirmed_london_session(self) -> list[Candle]:
        start = DAY.replace(hour=2, minute=0)
        end = DAY.replace(hour=5, minute=0)
        return [
            ny_candle(start, 100, 101, 99, 100.5),  # session high=101, low=99
            ny_candle(end, 300, 300.1, 299.9, 300),  # confirms close
        ]

    def test_touch_is_not_a_take(self):
        candles = self._confirmed_london_session() + [
            ny_candle(DAY.replace(hour=5, minute=1), 101, 101, 100.9, 100.95),  # high == 101 exactly
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual(result.takes, [])
        high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH)
        self.assertEqual(high.status, LiquidityStatus.ACTIVE)

    def test_strict_take_above_session_high(self):
        candles = self._confirmed_london_session() + [
            ny_candle(DAY.replace(hour=5, minute=1), 101, 101.25, 100.9, 100.95),  # exceeds by 1 tick
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual(len(result.takes), 1)
        high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH)
        self.assertEqual(high.status, LiquidityStatus.TAKEN)

    def test_strict_take_below_session_low_symmetric(self):
        candles = self._confirmed_london_session() + [
            ny_candle(DAY.replace(hour=5, minute=1), 99, 99.1, 98.75, 99),  # low exceeds (goes below) 99 by 1 tick
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_LOW)
        self.assertEqual(low.status, LiquidityStatus.TAKEN)
        self.assertEqual(len(result.takes), 1)

    def test_reclaim_close_after_take(self):
        candles = self._confirmed_london_session() + [
            ny_candle(DAY.replace(hour=5, minute=1), 101, 101.5, 100.9, 101.2),  # take; closes above 101
            ny_candle(DAY.replace(hour=5, minute=2), 101.2, 101.3, 100.5, 100.8),  # closes back below 101 -> reclaim close
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.RECLAIM_CLOSE)
        self.assertEqual(result.reclaims[0].bars_after_take, 1)

    def test_reclaim_wick(self):
        candles = self._confirmed_london_session() + [
            ny_candle(DAY.replace(hour=5, minute=1), 101, 101.5, 100.9, 101.2),  # take
            ny_candle(DAY.replace(hour=5, minute=2), 101.1, 101.3, 100.9, 101.1),  # wick below 101, closes back above
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.RECLAIM_WICK)

    def test_no_reclaim_in_window_is_explicit(self):
        candles = self._confirmed_london_session() + [
            ny_candle(DAY.replace(hour=5, minute=1), 101, 101.5, 100.9, 101.2),  # take
            ny_candle(DAY.replace(hour=5, minute=2), 101.2, 101.6, 101.1, 101.3),  # no reclaim
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=1)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.NO_RECLAIM_IN_WINDOW)

    def test_pending_when_window_not_yet_fully_observed(self):
        candles = self._confirmed_london_session() + [
            ny_candle(DAY.replace(hour=5, minute=1), 101, 101.5, 100.9, 101.2),  # take
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=5)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.PENDING)
        self.assertIsNone(result.reclaims[0].resolved_at)


class OverlapTests(unittest.TestCase):
    def test_asia_and_overnight_overlap_produce_independent_levels_with_shared_extreme(self):
        candles = [
            ny_candle(DAY.replace(hour=18, minute=0), 100, 100.2, 99.8, 100),  # overnight-only, before Asia opens
            ny_candle(DAY.replace(hour=20, minute=0), 100, 105, 95, 100),  # asia opens -- sets the extreme for BOTH
            ny_candle(DAY.replace(hour=23, minute=59), 100, 100.1, 99.9, 100),  # tame, inside both
            ny_candle(DAY + timedelta(days=1), 100, 100.1, 99.9, 100),  # 00:00 -- confirms Asia close; tame, still inside Overnight
            ny_candle(DAY + timedelta(days=1, hours=9, minutes=29), 100, 100.1, 99.9, 100),  # tame, still inside overnight
            ny_candle(DAY + timedelta(days=1, hours=9, minutes=30), 300, 300.1, 299.9, 300),  # confirms Overnight close
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        asia_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_ASIA_HIGH)
        asia_low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_ASIA_LOW)
        overnight_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_HIGH)
        overnight_low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_LOW)

        self.assertEqual((asia_high.upper, asia_low.lower), (105, 95))
        self.assertEqual((overnight_high.upper, overnight_low.lower), (105, 95))
        self.assertNotEqual(asia_high.level_id, overnight_high.level_id)
        self.assertNotEqual(asia_low.level_id, overnight_low.level_id)
        self.assertNotEqual(asia_high.liquidity_type, overnight_high.liquidity_type)

    def test_asia_take_and_overnight_take_are_reported_independently(self):
        # A wick that only breaks Asia's (narrower) window is not automatically
        # a break of Overnight's own high, even though Overnight contains Asia.
        candles = [
            ny_candle(DAY.replace(hour=18, minute=0), 100, 100.2, 99.8, 100),  # overnight-only
            ny_candle(DAY.replace(hour=20, minute=0), 100, 102, 98, 100),  # asia high=102, low=98
            ny_candle(DAY + timedelta(days=1), 100, 100.1, 99.9, 100),  # confirms Asia close
            ny_candle(DAY + timedelta(days=1, hours=9, minutes=29), 100, 105, 99.9, 100),  # overnight-only new high=105
            ny_candle(DAY + timedelta(days=1, hours=9, minutes=30), 300, 300.1, 299.9, 300),  # confirms Overnight close
            ny_candle(DAY + timedelta(days=1, hours=9, minutes=31), 300, 102.5, 299.9, 300),  # breaks Asia's 102 high only
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        asia_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_ASIA_HIGH)
        overnight_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_HIGH)
        self.assertEqual(asia_high.status, LiquidityStatus.TAKEN)
        self.assertEqual(overnight_high.status, LiquidityStatus.ACTIVE)
        self.assertEqual(len(result.takes), 1)
        self.assertEqual(result.takes[0].level_id, asia_high.level_id)


class DSTTests(unittest.TestCase):
    def test_spring_forward_shrinks_the_missing_clock_hour_not_the_algorithm(self):
        # 2026-03-08: NY wall clocks jump 02:00 -> 03:00, so 02:00-02:59 NY
        # never happens that day. London (02:00-05:00 NY) therefore only
        # has ~2 real clock-hours of candles -- no DST-specific code is
        # needed for this to come out right; astimezone(NY) already
        # reflects the missing hour, and the window check just follows it.
        start = datetime(2026, 3, 8, 5, 0, tzinfo=UTC)  # NY 00:00 EST, pre-transition
        candles = [utc_candle(start + timedelta(minutes=i), 100, 100.5, 99.5, 100) for i in range(300)]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        london = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH)
        self.assertEqual(london.member_count, 120)  # 2 real hours, not 3

    def test_fall_back_counts_the_repeated_clock_hour_exactly_once_per_real_candle(self):
        # 2026-11-01: NY wall clocks fall back 02:00 -> 01:00, so 01:00-01:59
        # NY happens twice in real elapsed time. Overnight (18:00-09:30 NY)
        # spans that hour; every real UTC candle is still only ever counted
        # once (there is no wall-clock deduplication), so the session ends
        # up with one extra real hour of members that day.
        start = datetime(2026, 10, 31, 22, 0, tzinfo=UTC)  # NY 18:00 EDT, pre-transition
        candles = [utc_candle(start + timedelta(minutes=i), 100, 100.5, 99.5, 100) for i in range(1200)]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        overnight = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_HIGH)
        self.assertEqual(overnight.member_count, 990)  # 15.5h + 1 repeated hour = 16.5h = 990 minutes


class EmptyAndIncompleteSessionTests(unittest.TestCase):
    def test_session_with_zero_candles_produces_no_level(self):
        # Candles only during NY lunch -- London never has a single candle.
        candles = [
            ny_candle(DAY.replace(hour=12, minute=0), 100, 101, 99, 100),
            ny_candle(DAY.replace(hour=13, minute=0), 100, 100.1, 99.9, 100),
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual([lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH], [])

    def test_session_still_open_when_data_ends_produces_no_level(self):
        # Data ends mid-Pre-NY: no later candle ever confirmed the window closed.
        candles = [
            ny_candle(DAY.replace(hour=8, minute=30), 100, 101, 99, 100),
            ny_candle(DAY.replace(hour=9, minute=0), 100, 102, 98, 100),
        ]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual(
            [lv for lv in result.levels if lv.liquidity_type in (LiquidityType.SESSION_PRE_NY_HIGH, LiquidityType.SESSION_PRE_NY_LOW)],
            [],
        )

    def test_completely_empty_candle_list_produces_no_levels(self):
        result = detect_session_liquidity([], "1m", reclaim_window_bars=3)
        self.assertEqual(result.levels, [])
        self.assertEqual(result.takes, [])
        self.assertEqual(result.reclaims, [])


class NoLookaheadTests(unittest.TestCase):
    def test_candles_must_be_strictly_ordered_by_timestamp(self):
        candles = [
            ny_candle(DAY.replace(hour=2, minute=0), 100, 101, 99, 100),
            ny_candle(DAY.replace(hour=1, minute=59), 100, 101, 99, 100),  # goes backwards
        ]
        with self.assertRaises(ValueError):
            detect_session_liquidity(candles, "1m", reclaim_window_bars=3)

    def test_candles_must_have_strictly_increasing_known_at(self):
        c1 = ny_candle(DAY.replace(hour=2, minute=0), 100, 101, 99, 100, minutes=10)
        c2 = ny_candle(DAY.replace(hour=2, minute=1), 100, 101, 99, 100, minutes=1)
        with self.assertRaises(ValueError):
            detect_session_liquidity([c1, c2], "1m", reclaim_window_bars=3)

    def test_all_recorded_timestamps_are_actual_candle_known_at_values(self):
        start = DAY.replace(hour=2, minute=0)
        end = DAY.replace(hour=5, minute=0)
        candles = [
            ny_candle(start, 100, 101, 99, 100.5),
            ny_candle(end, 300, 300.1, 299.9, 300),  # confirms close
            ny_candle(end + timedelta(minutes=1), 101, 101.5, 100.9, 101.2),  # take
            ny_candle(end + timedelta(minutes=2), 101.2, 101.3, 100.5, 100.8),  # reclaim close
        ]
        known_ats = {c.known_at for c in candles}
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
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

    def test_unsupported_timeframe_raises(self):
        with self.assertRaises(ValueError):
            detect_session_liquidity([], "3m", reclaim_window_bars=3)

    def test_non_positive_reclaim_window_bars_rejected(self):
        with self.assertRaises(ValueError):
            detect_session_liquidity([], "1m", reclaim_window_bars=0)


if __name__ == "__main__":
    unittest.main()
