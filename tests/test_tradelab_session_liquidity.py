#!/usr/bin/env python3
"""Session liquidity tests: America/New_York window boundaries, strict
completeness (no level from a partial/gapped session), availability
timing (formed_at / first-eligible-take-candle), strict take/reclaim
(reused from tradelab.liquidity), overlapping windows, DST correctness,
empty/incomplete sessions, and no lookahead."""
from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.fvg import Candle
from tradelab.session_liquidity import (
    _SESSION_LIQUIDITY_TYPES,
    _SESSION_WINDOWS,
    LiquidityStatus,
    LiquidityType,
    ReclaimOutcome,
    _expected_utc_timestamps,
    _session_anchor,
    detect_session_liquidity,
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

_DELTAS = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "15m": timedelta(minutes=15)}

# A plain Tuesday, far from any DST transition, used as the base date for
# tests that aren't specifically about DST.
DAY = date(2026, 1, 6)


def ny_candle(ny_dt: datetime, o: float, h: float, l: float, c: float, minutes: int = 1) -> Candle:
    ts = ny_dt.astimezone(UTC)
    return Candle(ts, o, h, l, c, ts + timedelta(minutes=minutes))


def utc_candle(ts: datetime, o: float, h: float, l: float, c: float, minutes: int = 1) -> Candle:
    return Candle(ts, o, h, l, c, ts + timedelta(minutes=minutes))


def complete_session_candles(
    key: str,
    anchor: date,
    timeframe: str = "1m",
    base: float = 100.0,
    highs: dict[int, float] | None = None,
    lows: dict[int, float] | None = None,
) -> list[Candle]:
    """Every expected candle for one full occurrence of session ``key``
    anchored on ``anchor``, generated from the module's own grid so the
    set is exact and contiguous by construction. ``highs``/``lows`` place
    custom extremes at specific 0-based indices; everything else is a flat
    +-0.1 range around ``base``."""
    delta = _DELTAS[timeframe]
    expected = _expected_utc_timestamps(anchor, _SESSION_WINDOWS[key], delta)
    highs = highs or {}
    lows = lows or {}
    out = []
    for i, ts in enumerate(expected):
        h = highs.get(i, base + 0.1)
        l = lows.get(i, base - 0.1)
        out.append(Candle(ts, base, h, l, base, ts + delta))
    return out


SESSION_KEYS = list(_SESSION_WINDOWS)


class SessionAnchorBoundaryTests(unittest.TestCase):
    """Direct tests of the pure start-inclusive/end-exclusive window
    membership function, independent of completeness/detection."""

    def test_each_session_boundary_is_start_inclusive_end_exclusive(self):
        cases = [
            ("asia", "19:59", None), ("asia", "20:00", DAY), ("asia", "23:59", DAY),
            ("london", "01:59", None), ("london", "02:00", DAY), ("london", "04:59", DAY), ("london", "05:00", None),
            ("pre_ny", "08:29", None), ("pre_ny", "08:30", DAY), ("pre_ny", "09:29", DAY), ("pre_ny", "09:30", None),
            ("ny_am", "09:29", None), ("ny_am", "09:30", DAY), ("ny_am", "11:59", DAY), ("ny_am", "12:00", None),
            ("ny_lunch", "11:59", None), ("ny_lunch", "12:00", DAY), ("ny_lunch", "12:59", DAY), ("ny_lunch", "13:00", None),
            ("ny_pm", "12:59", None), ("ny_pm", "13:00", DAY), ("ny_pm", "15:59", DAY), ("ny_pm", "16:00", None),
            ("overnight", "17:59", None), ("overnight", "18:00", DAY),
        ]
        for key, hhmm, expected in cases:
            with self.subTest(key=key, hhmm=hhmm):
                hh, mm = (int(x) for x in hhmm.split(":"))
                local_dt = datetime(DAY.year, DAY.month, DAY.day, hh, mm, tzinfo=NY)
                self.assertEqual(_session_anchor(local_dt, _SESSION_WINDOWS[key]), expected)

    def test_asia_excludes_midnight_and_overnight_wraps_into_next_morning(self):
        next_day = DAY + timedelta(days=1)
        self.assertIsNone(_session_anchor(datetime(next_day.year, next_day.month, next_day.day, 0, 0, tzinfo=NY), _SESSION_WINDOWS["asia"]))
        self.assertEqual(
            _session_anchor(datetime(next_day.year, next_day.month, next_day.day, 9, 29, tzinfo=NY), _SESSION_WINDOWS["overnight"]),
            DAY,
        )
        self.assertIsNone(
            _session_anchor(datetime(next_day.year, next_day.month, next_day.day, 9, 30, tzinfo=NY), _SESSION_WINDOWS["overnight"])
        )


class SessionCompletionTests(unittest.TestCase):
    def test_each_session_produces_correct_high_low_and_member_count_when_complete(self):
        for key in SESSION_KEYS:
            with self.subTest(session=key):
                last_index = len(_expected_utc_timestamps(DAY, _SESSION_WINDOWS[key], _DELTAS["1m"])) - 1
                candles = complete_session_candles(key, DAY, highs={0: 105.0}, lows={last_index: 95.0})
                confirming = Candle(
                    candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=1)
                )
                result = detect_session_liquidity(candles + [confirming], "1m", reclaim_window_bars=3)
                high_type, low_type = _session_types(key)
                high = next(lv for lv in result.levels if lv.liquidity_type == high_type)
                low = next(lv for lv in result.levels if lv.liquidity_type == low_type)
                self.assertEqual(high.upper, 105.0)
                self.assertEqual(low.lower, 95.0)
                self.assertEqual(high.member_count, len(candles))
                self.assertEqual(high.status, LiquidityStatus.ACTIVE)
                self.assertEqual(high.formed_at, candles[-1].known_at)

    def test_level_revisions_preserve_pre_take_state(self):
        candles = complete_session_candles("ny_lunch", DAY, highs={0: 105.0})
        confirming = Candle(
            candles[-1].known_at, 100, 100.1, 99.9, 100,
            candles[-1].known_at + timedelta(minutes=1),
        )
        taking = Candle(
            confirming.known_at, 105.0, 105.25, 104.9, 105.1,
            confirming.known_at + timedelta(minutes=1),
        )
        result = detect_session_liquidity(candles + [confirming, taking], "1m", reclaim_window_bars=3)
        high_type, _ = _session_types("ny_lunch")
        revisions = [r for r in result.level_revisions if r.level.liquidity_type == high_type]
        self.assertEqual([r.level.status for r in revisions], [LiquidityStatus.ACTIVE, LiquidityStatus.TAKEN])
        self.assertIsNone(revisions[0].level.taken_at)
        self.assertEqual(revisions[1].effective_at, taking.known_at)


def _session_types(key: str):
    return _SESSION_LIQUIDITY_TYPES[key]


class TimeframeRestrictionTests(unittest.TestCase):
    def test_1m_5m_15m_are_supported(self):
        for timeframe in ("1m", "5m", "15m"):
            with self.subTest(timeframe=timeframe):
                result = detect_session_liquidity([], timeframe, reclaim_window_bars=3)
                self.assertEqual(result.levels, [])

    def test_higher_timeframes_are_rejected(self):
        for timeframe in ("1H", "4H", "1D"):
            with self.subTest(timeframe=timeframe):
                with self.assertRaises(ValueError):
                    detect_session_liquidity([], timeframe, reclaim_window_bars=3)

    def test_unsupported_timeframe_raises(self):
        with self.assertRaises(ValueError):
            detect_session_liquidity([], "3m", reclaim_window_bars=3)


class ActivationTimingTests(unittest.TestCase):
    def test_no_level_and_no_spurious_take_while_session_still_incomplete(self):
        # Data ends partway through London -- no later candle ever
        # confirmed the window closed, and an extreme wick inside the
        # still-open session must not register as a take of anything.
        window = _SESSION_WINDOWS["london"]
        expected = _expected_utc_timestamps(DAY, window, _DELTAS["1m"])[:5]
        candles = [Candle(ts, 100, 103 if i == 2 else 100.1, 98 if i == 3 else 99.9, 100, ts + timedelta(minutes=1)) for i, ts in enumerate(expected)]
        result = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual(result.takes, [])
        self.assertEqual([lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH], [])

    def test_level_available_exactly_at_last_member_known_at_not_before_and_not_delayed(self):
        candles = complete_session_candles("london", DAY, highs={1: 102.0})
        result_incomplete = detect_session_liquidity(candles, "1m", reclaim_window_bars=3)
        self.assertEqual([lv for lv in result_incomplete.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH], [])

        # The very next candle after the window both confirms completion
        # AND is itself an eligible take candidate -- no separate
        # "confirming-only" candle is needed, and none is skipped.
        next_ts = candles[-1].timestamp + timedelta(minutes=1)
        take_candle = Candle(next_ts, 102, 102.25, 101.9, 102, next_ts + timedelta(minutes=1))  # exceeds 102 by 1 tick
        result = detect_session_liquidity(candles + [take_candle], "1m", reclaim_window_bars=3)
        high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH)
        self.assertEqual(high.formed_at, candles[-1].known_at)  # last real member, not the take candle
        self.assertEqual(high.status, LiquidityStatus.TAKEN)
        self.assertEqual(len(result.takes), 1)
        self.assertEqual(result.takes[0].timestamp, take_candle.known_at)  # first eligible candle, not skipped

    def test_no_lookahead_and_no_skipped_first_bar_together(self):
        # Combined check the review asked for: (a) a wick *inside* the
        # still-forming session cannot be a take of anything (no
        # lookahead), and (b) once the session closes, the very first
        # candle after it *is* eligible to take -- not exempt, not
        # requiring a second candle to "arrive" first.
        candles = complete_session_candles("london", DAY, highs={1: 102.0})
        mid_session_spike = candles[2]
        self.assertLess(mid_session_spike.high, 102.0 + 100)  # sanity: not itself the eventual high

        # (a) truncate right after the spike -- still mid-session, incomplete.
        truncated = candles[:3]
        result_mid = detect_session_liquidity(truncated, "1m", reclaim_window_bars=3)
        self.assertEqual(result_mid.takes, [])

        # (b) full session + immediate next candle breaks the high.
        next_ts = candles[-1].timestamp + timedelta(minutes=1)
        take_candle = Candle(next_ts, 102, 102.5, 101.9, 102, next_ts + timedelta(minutes=1))
        result = detect_session_liquidity(candles + [take_candle], "1m", reclaim_window_bars=3)
        self.assertEqual(len(result.takes), 1)
        self.assertEqual(result.takes[0].known_at, take_candle.known_at)


class TakeAndReclaimTests(unittest.TestCase):
    def _confirmed_london_session(self) -> list[Candle]:
        return complete_session_candles("london", DAY, highs={0: 101.0}, lows={0: 99.0})

    def test_touch_is_not_a_take(self):
        candles = self._confirmed_london_session()
        next_ts = candles[-1].timestamp + timedelta(minutes=1)
        touch = Candle(next_ts, 101, 101, 100.9, 100.95, next_ts + timedelta(minutes=1))  # high == 101 exactly
        result = detect_session_liquidity(candles + [touch], "1m", reclaim_window_bars=3)
        self.assertEqual(result.takes, [])
        high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH)
        self.assertEqual(high.status, LiquidityStatus.ACTIVE)

    def test_strict_take_above_session_high(self):
        candles = self._confirmed_london_session()
        next_ts = candles[-1].timestamp + timedelta(minutes=1)
        take = Candle(next_ts, 101, 101.25, 100.9, 100.95, next_ts + timedelta(minutes=1))
        result = detect_session_liquidity(candles + [take], "1m", reclaim_window_bars=3)
        self.assertEqual(len(result.takes), 1)
        high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH)
        self.assertEqual(high.status, LiquidityStatus.TAKEN)

    def test_strict_take_below_session_low_symmetric(self):
        candles = self._confirmed_london_session()
        next_ts = candles[-1].timestamp + timedelta(minutes=1)
        take = Candle(next_ts, 99, 99.1, 98.75, 99, next_ts + timedelta(minutes=1))  # exceeds (below) 99 by 1 tick
        result = detect_session_liquidity(candles + [take], "1m", reclaim_window_bars=3)
        low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_LOW)
        self.assertEqual(low.status, LiquidityStatus.TAKEN)
        self.assertEqual(len(result.takes), 1)

    def test_reclaim_close_after_take(self):
        candles = self._confirmed_london_session()
        t1 = candles[-1].timestamp + timedelta(minutes=1)
        t2 = t1 + timedelta(minutes=1)
        take = Candle(t1, 101, 101.5, 100.9, 101.2, t1 + timedelta(minutes=1))
        reclaim = Candle(t2, 101.2, 101.3, 100.5, 100.8, t2 + timedelta(minutes=1))
        result = detect_session_liquidity(candles + [take, reclaim], "1m", reclaim_window_bars=3)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.RECLAIM_CLOSE)
        self.assertEqual(result.reclaims[0].bars_after_take, 1)

    def test_reclaim_wick(self):
        candles = self._confirmed_london_session()
        t1 = candles[-1].timestamp + timedelta(minutes=1)
        t2 = t1 + timedelta(minutes=1)
        take = Candle(t1, 101, 101.5, 100.9, 101.2, t1 + timedelta(minutes=1))
        reclaim = Candle(t2, 101.1, 101.3, 100.9, 101.1, t2 + timedelta(minutes=1))
        result = detect_session_liquidity(candles + [take, reclaim], "1m", reclaim_window_bars=3)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.RECLAIM_WICK)

    def test_no_reclaim_in_window_is_explicit(self):
        candles = self._confirmed_london_session()
        t1 = candles[-1].timestamp + timedelta(minutes=1)
        t2 = t1 + timedelta(minutes=1)
        take = Candle(t1, 101, 101.5, 100.9, 101.2, t1 + timedelta(minutes=1))
        no_reclaim = Candle(t2, 101.2, 101.6, 101.1, 101.3, t2 + timedelta(minutes=1))
        result = detect_session_liquidity(candles + [take, no_reclaim], "1m", reclaim_window_bars=1)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.NO_RECLAIM_IN_WINDOW)

    def test_pending_when_window_not_yet_fully_observed(self):
        candles = self._confirmed_london_session()
        t1 = candles[-1].timestamp + timedelta(minutes=1)
        take = Candle(t1, 101, 101.5, 100.9, 101.2, t1 + timedelta(minutes=1))
        result = detect_session_liquidity(candles + [take], "1m", reclaim_window_bars=5)
        self.assertEqual(result.reclaims[0].outcome, ReclaimOutcome.PENDING)
        self.assertIsNone(result.reclaims[0].resolved_at)


class OverlapTests(unittest.TestCase):
    def test_asia_and_overnight_overlap_produce_independent_levels_with_shared_extreme(self):
        window = _SESSION_WINDOWS["overnight"]
        expected = _expected_utc_timestamps(DAY, window, _DELTAS["1m"])
        asia_start_ts = datetime(DAY.year, DAY.month, DAY.day, 20, 0, tzinfo=NY).astimezone(UTC)
        asia_start_idx = expected.index(asia_start_ts)
        candles = [
            Candle(ts, 100, 105.0 if i == asia_start_idx else 100.1, 95.0 if i == asia_start_idx else 99.9, 100, ts + timedelta(minutes=1))
            for i, ts in enumerate(expected)
        ]
        confirming = Candle(candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=1))
        result = detect_session_liquidity(candles + [confirming], "1m", reclaim_window_bars=3)

        asia_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_ASIA_HIGH)
        asia_low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_ASIA_LOW)
        overnight_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_HIGH)
        overnight_low = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_LOW)

        self.assertEqual((asia_high.upper, asia_low.lower), (105.0, 95.0))
        self.assertEqual((overnight_high.upper, overnight_low.lower), (105.0, 95.0))
        self.assertNotEqual(asia_high.level_id, overnight_high.level_id)
        self.assertNotEqual(asia_low.level_id, overnight_low.level_id)
        self.assertNotEqual(asia_high.liquidity_type, overnight_high.liquidity_type)

    def test_asia_take_and_overnight_take_are_reported_independently(self):
        window = _SESSION_WINDOWS["overnight"]
        expected = _expected_utc_timestamps(DAY, window, _DELTAS["1m"])
        asia_start_ts = datetime(DAY.year, DAY.month, DAY.day, 20, 0, tzinfo=NY).astimezone(UTC)
        overnight_only_idx = 0  # 18:00, before Asia opens
        asia_high_idx = expected.index(asia_start_ts)
        candles = []
        for i, ts in enumerate(expected):
            if i == overnight_only_idx:
                candles.append(Candle(ts, 100, 105.0, 99.9, 100, ts + timedelta(minutes=1)))  # overnight-only high=105
            elif i == asia_high_idx:
                candles.append(Candle(ts, 100, 102.0, 98.0, 100, ts + timedelta(minutes=1)))  # asia high=102, low=98
            else:
                candles.append(Candle(ts, 100, 100.1, 99.9, 100, ts + timedelta(minutes=1)))
        confirming = Candle(candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=1))
        breaks_asia_only = Candle(
            confirming.known_at, 100, 102.5, 99.9, 100, confirming.known_at + timedelta(minutes=1)
        )  # > asia high (102), < overnight high (105)
        result = detect_session_liquidity(candles + [confirming, breaks_asia_only], "1m", reclaim_window_bars=3)

        asia_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_ASIA_HIGH)
        overnight_high = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_HIGH)
        # Overnight's full window also happens to contain complete London
        # and Pre-NY sub-ranges, which the same breakout candle can
        # incidentally also take -- irrelevant here. What matters is Asia
        # vs. Overnight specifically: the narrower window's sweep must not
        # register as a sweep of the wider one that contains it.
        take_level_ids = {t.level_id for t in result.takes}
        self.assertEqual(asia_high.status, LiquidityStatus.TAKEN)
        self.assertEqual(overnight_high.status, LiquidityStatus.ACTIVE)
        self.assertIn(asia_high.level_id, take_level_ids)
        self.assertNotIn(overnight_high.level_id, take_level_ids)


class DSTTests(unittest.TestCase):
    def test_spring_forward_shrinks_the_missing_clock_hour_not_the_algorithm(self):
        # 2026-03-08: NY wall clocks jump 02:00 -> 03:00, so 02:00-02:59 NY
        # never happens that day. London (02:00-05:00 NY) therefore only
        # has ~2 real clock-hours of expected candles -- no DST-specific
        # code is needed for this to come out right; astimezone(NY) and
        # the UTC-stepped grid already reflect the missing hour.
        anchor = date(2026, 3, 8)
        candles = complete_session_candles("london", anchor)
        self.assertEqual(len(candles), 120)  # 2 real hours, not 3
        result = detect_session_liquidity(
            candles + [Candle(candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=1))],
            "1m",
            reclaim_window_bars=3,
        )
        london = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH)
        self.assertEqual(london.member_count, 120)

    def test_fall_back_counts_the_repeated_clock_hour_exactly_once_per_real_candle(self):
        # 2026-11-01: NY wall clocks fall back 02:00 -> 01:00, so 01:00-01:59
        # NY happens twice in real elapsed time. Overnight (18:00-09:30 NY)
        # spans that hour; every real UTC candle is still only ever counted
        # once, so the session ends up with one extra real hour of members.
        anchor = date(2026, 10, 31)
        candles = complete_session_candles("overnight", anchor)
        self.assertEqual(len(candles), 990)  # 15.5h + 1 repeated hour = 16.5h = 990 minutes
        result = detect_session_liquidity(
            candles + [Candle(candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=1))],
            "1m",
            reclaim_window_bars=3,
        )
        overnight = next(lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_OVERNIGHT_HIGH)
        self.assertEqual(overnight.member_count, 990)


class EmptyAndIncompleteSessionTests(unittest.TestCase):
    def test_session_with_zero_candles_produces_no_level(self):
        candles = complete_session_candles("ny_lunch", DAY)
        confirming = Candle(candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=1))
        result = detect_session_liquidity(candles + [confirming], "1m", reclaim_window_bars=3)
        self.assertEqual([lv for lv in result.levels if lv.liquidity_type == LiquidityType.SESSION_LONDON_HIGH], [])

    def test_session_still_open_when_data_ends_produces_no_level(self):
        candles = complete_session_candles("pre_ny", DAY)[:10]  # data ends mid-session, no confirming candle
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

    def test_gap_in_the_middle_of_the_session_produces_no_level_1m(self):
        candles = complete_session_candles("pre_ny", DAY, timeframe="1m")
        gapped = candles[:30] + candles[31:]  # drop one expected 1m slot
        confirming = Candle(candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=1))
        result = detect_session_liquidity(gapped + [confirming], "1m", reclaim_window_bars=3)
        self.assertEqual(
            [lv for lv in result.levels if lv.liquidity_type in (LiquidityType.SESSION_PRE_NY_HIGH, LiquidityType.SESSION_PRE_NY_LOW)],
            [],
        )

    def test_gap_in_the_middle_of_the_session_produces_no_level_5m(self):
        candles = complete_session_candles("ny_pm", DAY, timeframe="5m")
        gapped = candles[:10] + candles[11:]  # drop one expected 5m slot
        confirming = Candle(candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=5))
        result = detect_session_liquidity(gapped + [confirming], "5m", reclaim_window_bars=3)
        self.assertEqual(
            [lv for lv in result.levels if lv.liquidity_type in (LiquidityType.SESSION_NY_PM_HIGH, LiquidityType.SESSION_NY_PM_LOW)],
            [],
        )

    def test_gap_in_the_middle_of_the_session_produces_no_level_15m(self):
        candles = complete_session_candles("ny_am", DAY, timeframe="15m")
        gapped = candles[:3] + candles[4:]  # drop one expected 15m slot
        confirming = Candle(candles[-1].known_at, 100, 100.1, 99.9, 100, candles[-1].known_at + timedelta(minutes=15))
        result = detect_session_liquidity(gapped + [confirming], "15m", reclaim_window_bars=3)
        self.assertEqual(
            [lv for lv in result.levels if lv.liquidity_type in (LiquidityType.SESSION_NY_AM_HIGH, LiquidityType.SESSION_NY_AM_LOW)],
            [],
        )

    def test_gap_still_lets_other_sessions_resolve_normally(self):
        # A gap breaking Pre-NY must not affect an unrelated, complete
        # session elsewhere in the same candle list.
        pre_ny = complete_session_candles("pre_ny", DAY)
        gapped_pre_ny = pre_ny[:5] + pre_ny[6:]
        ny_lunch = complete_session_candles("ny_lunch", DAY)
        confirming = Candle(ny_lunch[-1].known_at, 100, 100.1, 99.9, 100, ny_lunch[-1].known_at + timedelta(minutes=1))
        result = detect_session_liquidity(gapped_pre_ny + ny_lunch + [confirming], "1m", reclaim_window_bars=3)
        self.assertEqual(
            [lv for lv in result.levels if lv.liquidity_type in (LiquidityType.SESSION_PRE_NY_HIGH, LiquidityType.SESSION_PRE_NY_LOW)],
            [],
        )
        self.assertTrue(any(lv.liquidity_type == LiquidityType.SESSION_NY_LUNCH_HIGH for lv in result.levels))


class NoLookaheadTests(unittest.TestCase):
    def test_candles_must_be_strictly_ordered_by_timestamp(self):
        candles = [
            ny_candle(datetime(DAY.year, DAY.month, DAY.day, 2, 0, tzinfo=NY), 100, 101, 99, 100),
            ny_candle(datetime(DAY.year, DAY.month, DAY.day, 1, 59, tzinfo=NY), 100, 101, 99, 100),  # goes backwards
        ]
        with self.assertRaises(ValueError):
            detect_session_liquidity(candles, "1m", reclaim_window_bars=3)

    def test_candles_must_have_strictly_increasing_known_at(self):
        c1 = ny_candle(datetime(DAY.year, DAY.month, DAY.day, 2, 0, tzinfo=NY), 100, 101, 99, 100, minutes=10)
        c2 = ny_candle(datetime(DAY.year, DAY.month, DAY.day, 2, 1, tzinfo=NY), 100, 101, 99, 100, minutes=1)
        with self.assertRaises(ValueError):
            detect_session_liquidity([c1, c2], "1m", reclaim_window_bars=3)

    def test_all_recorded_timestamps_are_actual_candle_known_at_values(self):
        candles = complete_session_candles("london", DAY, highs={0: 101.0}, lows={0: 99.0})
        t1 = candles[-1].timestamp + timedelta(minutes=1)
        t2 = t1 + timedelta(minutes=1)
        take = Candle(t1, 101, 101.5, 100.9, 101.2, t1 + timedelta(minutes=1))
        reclaim = Candle(t2, 101.2, 101.3, 100.5, 100.8, t2 + timedelta(minutes=1))
        full = candles + [take, reclaim]
        known_ats = {c.known_at for c in full}
        result = detect_session_liquidity(full, "1m", reclaim_window_bars=3)
        for level in result.levels:
            self.assertIn(level.formed_at, known_ats)
            if level.taken_at is not None:
                self.assertIn(level.taken_at, known_ats)
        for t in result.takes:
            self.assertIn(t.timestamp, known_ats)
            self.assertEqual(t.timestamp, t.known_at)
        for reclaim_result in result.reclaims:
            if reclaim_result.resolved_at is not None:
                self.assertIn(reclaim_result.resolved_at, known_ats)

    def test_non_positive_reclaim_window_bars_rejected(self):
        with self.assertRaises(ValueError):
            detect_session_liquidity([], "1m", reclaim_window_bars=0)


if __name__ == "__main__":
    unittest.main()
