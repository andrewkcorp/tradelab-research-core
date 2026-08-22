#!/usr/bin/env python3
"""Aggregation tests: OHLCV rollup, timeframe boundaries, completeness, trading-day and DST rules."""
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.aggregate import _bucket_start, _four_hour_bucket_start, _trading_day_start, aggregate_bars
from tradelab.bar import Bar1m
from tradelab.session_calendar import expected_open_minutes

NY = ZoneInfo("America/New_York")
CONTRACT = "NQZ25"


def bar_at_ny(ny_dt: datetime, o: float = 1, h: float = 1, l: float = 1, c: float = 1, v: float = 1) -> Bar1m:
    return Bar1m(ny_dt.astimezone(timezone.utc), o, h, l, c, v, CONTRACT)


def bar_at_utc(utc_dt: datetime, o: float = 1, h: float = 1, l: float = 1, c: float = 1, v: float = 1) -> Bar1m:
    return Bar1m(utc_dt, o, h, l, c, v, CONTRACT)


def minute_series_ny(start_ny: datetime, n: int, price: float = 100.0) -> list[Bar1m]:
    """n consecutive 1m bars anchored to an NY wall-clock start, each with a distinct
    high/low/volume so rollup math is checkable. Only safe when [start_ny, start_ny+n
    minutes) does not cross a DST gap/fold or the 17:00-18:00 NY daily break; use
    minute_series_utc / session_bars for those."""
    bars = []
    for i in range(n):
        ts = start_ny + timedelta(minutes=i)
        bars.append(bar_at_ny(ts, price + i, price + i + 0.5, price + i - 0.5, price + i + 0.2, 10 + i))
    return bars


def minute_series_utc(start_utc: datetime, n: int) -> list[Bar1m]:
    """n consecutive 1m bars anchored to a UTC start; safe across any DST transition.
    Does not skip the daily break -- only use for windows that don't touch it."""
    return [bar_at_utc(start_utc + timedelta(minutes=i)) for i in range(n)]


def session_bars(start_utc: datetime, end_utc: datetime) -> list[Bar1m]:
    """One 1m bar per expected-open minute in [start_utc, end_utc), i.e. a
    fully complete window per tradelab.session_calendar (break excluded)."""
    return [bar_at_utc(t) for t in expected_open_minutes(start_utc, end_utc)]


class OhlcvRollupTests(unittest.TestCase):
    def test_open_high_low_close_volume_are_correctly_rolled_up(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        bars = minute_series_ny(start, 5)  # exactly one complete 5m bucket
        [agg] = aggregate_bars(bars, "5m")
        self.assertEqual(agg.open, bars[0].open)
        self.assertEqual(agg.close, bars[-1].close)
        self.assertEqual(agg.high, max(b.high for b in bars))
        self.assertEqual(agg.low, min(b.low for b in bars))
        self.assertEqual(agg.volume, sum(b.volume for b in bars))
        self.assertEqual(agg.contract, CONTRACT)
        self.assertEqual(agg.timeframe, "5m")


class BoundaryLabelTests(unittest.TestCase):
    """Direct checks of the boundary-computation functions, independent of the
    completeness policy exercised in CompletenessTests below."""

    def test_5m_boundary_labels(self):
        self.assertEqual(_bucket_start(datetime(2026, 1, 5, 9, 34, tzinfo=NY), "5m"), datetime(2026, 1, 5, 9, 30, tzinfo=NY))
        self.assertEqual(_bucket_start(datetime(2026, 1, 5, 9, 35, tzinfo=NY), "5m"), datetime(2026, 1, 5, 9, 35, tzinfo=NY))

    def test_15m_boundary_labels(self):
        self.assertEqual(_bucket_start(datetime(2026, 1, 5, 9, 44, tzinfo=NY), "15m"), datetime(2026, 1, 5, 9, 30, tzinfo=NY))
        self.assertEqual(_bucket_start(datetime(2026, 1, 5, 9, 45, tzinfo=NY), "15m"), datetime(2026, 1, 5, 9, 45, tzinfo=NY))

    def test_1h_boundary_labels(self):
        self.assertEqual(_bucket_start(datetime(2026, 1, 5, 9, 59, tzinfo=NY), "1H"), datetime(2026, 1, 5, 9, 0, tzinfo=NY))
        self.assertEqual(_bucket_start(datetime(2026, 1, 5, 10, 0, tzinfo=NY), "1H"), datetime(2026, 1, 5, 10, 0, tzinfo=NY))

    def test_4h_boundary_labels_use_the_18_22_02_06_10_14_anchors(self):
        cases = {
            18: 18, 19: 18, 21: 18,
            22: 22, 23: 22,
            0: 22, 1: 22,
            2: 2, 3: 2, 5: 2,
            6: 6, 9: 6,
            10: 10, 13: 10,
            14: 14, 17: 14,
        }
        for hour, expected_anchor_hour in cases.items():
            with self.subTest(hour=hour):
                ts = datetime(2026, 1, 5, hour, 30, tzinfo=NY)
                got = _four_hour_bucket_start(ts)
                self.assertEqual(got.hour, expected_anchor_hour)

    def test_trading_day_boundary_labels(self):
        self.assertEqual(_trading_day_start(datetime(2026, 1, 5, 17, 59, tzinfo=NY)), datetime(2026, 1, 4, 18, 0, tzinfo=NY))
        self.assertEqual(_trading_day_start(datetime(2026, 1, 5, 18, 0, tzinfo=NY)), datetime(2026, 1, 5, 18, 0, tzinfo=NY))


class CompletenessTests(unittest.TestCase):
    """Per PR #2 review: no bucket -- first, interior, or trailing -- may be
    aggregated unless its bars exactly match the session calendar's expected-open
    minutes for that window (see tradelab.session_calendar)."""

    def test_full_bucket_is_aggregated(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        bars = minute_series_ny(start, 5)
        agg = aggregate_bars(bars, "5m")
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].timestamp, start.astimezone(timezone.utc))

    def test_partial_first_bucket_is_dropped(self):
        # Input starts mid-bucket (09:30-09:35 window), so only 2 of 5 minutes exist.
        start = datetime(2026, 1, 5, 9, 33, tzinfo=NY)
        bars = minute_series_ny(start, 2)  # 09:33, 09:34 only
        self.assertEqual(aggregate_bars(bars, "5m"), [])

    def test_gap_inside_bucket_is_dropped(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        # bucket 09:30-09:35 has only :30 and :31 (gap at :32-:34); next bucket is full.
        gappy = [bar_at_ny(start, 1, 1, 1, 1, 1), bar_at_ny(start + timedelta(minutes=1), 2, 2, 2, 2, 2)]
        full_next = minute_series_ny(start + timedelta(minutes=5), 5, price=200.0)
        agg = aggregate_bars(gappy + full_next, "5m")
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].timestamp, (start + timedelta(minutes=5)).astimezone(timezone.utc))
        self.assertEqual(agg[0].open, full_next[0].open)
        self.assertEqual(agg[0].close, full_next[-1].close)

    def test_boundary_split_does_not_borrow_bars_from_the_next_bucket(self):
        # A full 09:30-09:35 bucket plus one bar already in 09:35-09:40: only the
        # first bucket is complete and returned; the lone 09:35 bar is dropped.
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        bucket_a = minute_series_ny(start, 5)
        stray = [bar_at_ny(start + timedelta(minutes=5), 9, 9, 9, 9, 9)]
        agg = aggregate_bars(bucket_a + stray, "5m")
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].close, bucket_a[-1].close)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(aggregate_bars([], "5m"), [])

    def test_unsupported_timeframe_raises(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        with self.assertRaises(ValueError):
            aggregate_bars(minute_series_ny(start, 5), "3m")


class DailyBreakCompletenessTests(unittest.TestCase):
    """Per PR #2 review: the 17:00-18:00 NY daily maintenance break must not be
    expected to have bars, but every *regular-session* minute still must."""

    def test_full_1d_bucket_with_1380_active_minutes_and_no_break_bars_is_returned(self):
        start_utc = datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        end_utc = datetime(2026, 1, 6, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        bars = session_bars(start_utc, end_utc)
        self.assertEqual(len(bars), 1380)  # sanity check on the fixture itself

        [agg] = aggregate_bars(bars, "1D")
        self.assertEqual(agg.timestamp, start_utc)
        self.assertEqual(agg.known_at, bars[-1].timestamp + timedelta(minutes=1))
        self.assertEqual(agg.open, bars[0].open)
        self.assertEqual(agg.close, bars[-1].close)

    def test_4h_bucket_14_to_18_with_180_active_minutes_is_returned(self):
        start_utc = datetime(2026, 1, 6, 14, 0, tzinfo=NY).astimezone(timezone.utc)
        end_utc = datetime(2026, 1, 6, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        bars = session_bars(start_utc, end_utc)
        self.assertEqual(len(bars), 180)  # sanity check on the fixture itself

        [agg] = aggregate_bars(bars, "4H")
        self.assertEqual(agg.timestamp, start_utc)
        self.assertEqual(agg.known_at, bars[-1].timestamp + timedelta(minutes=1))

    def test_one_missing_regular_minute_still_rejects_the_4h_bucket(self):
        start_utc = datetime(2026, 1, 6, 14, 0, tzinfo=NY).astimezone(timezone.utc)
        end_utc = datetime(2026, 1, 6, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        expected = expected_open_minutes(start_utc, end_utc)
        missing_15_00 = datetime(2026, 1, 6, 15, 0, tzinfo=NY).astimezone(timezone.utc)
        self.assertIn(missing_15_00, expected)  # sanity check on the fixture itself

        bars = [bar_at_utc(t) for t in expected if t != missing_15_00]
        self.assertEqual(aggregate_bars(bars, "4H"), [])

    def test_a_bar_landing_inside_the_break_rejects_the_bucket(self):
        # Anomalous data: a 1m bar timestamped inside the 17:00-18:00 break. This
        # is "data inconsistent with the regular schedule" and must be rejected,
        # not silently folded into the bucket.
        start_utc = datetime(2026, 1, 6, 14, 0, tzinfo=NY).astimezone(timezone.utc)
        end_utc = datetime(2026, 1, 6, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        bars = session_bars(start_utc, end_utc)
        stray_in_break = bar_at_ny(datetime(2026, 1, 6, 17, 30, tzinfo=NY))
        # Insert in timestamp order to satisfy the strict-ordering requirement.
        combined = sorted(bars + [stray_in_break], key=lambda b: b.timestamp)
        self.assertEqual(aggregate_bars(combined, "4H"), [])


class TradingDayTests(unittest.TestCase):
    def test_full_trading_days_roll_over_at_18_00_ny(self):
        day1_start = datetime(2026, 1, 4, 18, 0, tzinfo=NY).astimezone(timezone.utc)  # non-DST day
        day1_end = datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        day2_end = datetime(2026, 1, 6, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        day1 = session_bars(day1_start, day1_end)
        day2 = session_bars(day1_end, day2_end)

        agg = aggregate_bars(day1 + day2, "1D")
        self.assertEqual(len(agg), 2)
        self.assertEqual(agg[0].timestamp, day1_start)
        self.assertEqual(agg[0].open, day1[0].open)
        self.assertEqual(agg[0].close, day1[-1].close)
        self.assertEqual(agg[0].known_at, day1[-1].timestamp + timedelta(minutes=1))
        self.assertEqual(agg[1].open, day2[0].open)


class FourHourAnchorTests(unittest.TestCase):
    def test_all_six_anchors_produce_complete_buckets(self):
        day_start = datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc)  # non-DST day
        day_end = datetime(2026, 1, 6, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        bars = session_bars(day_start, day_end)  # 1380 bars, break excluded

        agg = aggregate_bars(bars, "4H")
        self.assertEqual(len(agg), 6)
        expected_hours = [18, 22, 2, 6, 10, 14]
        self.assertEqual([a.timestamp.astimezone(NY).hour for a in agg], expected_hours)
        # known_at is the close of the final expected member candle. The last
        # bucket is complete at 17:00 because 17:00-18:00 is scheduled closed.
        raw_durations = [(a.known_at - a.timestamp).total_seconds() / 60 for a in agg]
        self.assertEqual(raw_durations, [240, 240, 240, 240, 240, 180])
        # Each bar_at_utc fixture has volume=1, so summed volume reveals how
        # many 1m bars actually fed each bucket: 240 for every segment except
        # the last (14:00-18:00), which loses its final hour to the daily
        # break and is fed by only 180.
        actual_bar_counts = [a.volume for a in agg]
        self.assertEqual(actual_bar_counts, [240, 240, 240, 240, 240, 180])


class DstTransitionTests(unittest.TestCase):
    def test_spring_forward_4h_bucket_needs_only_180_minutes(self):
        # The transition is Sunday while Globex is closed. The old daily-only
        # calendar incorrectly fabricated an open bucket here.
        start_utc = datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc)
        closer = [bar_at_utc(datetime(2026, 3, 8, 10, 5, tzinfo=timezone.utc))]

        complete = minute_series_utc(start_utc, 180)
        self.assertEqual(aggregate_bars(complete + closer, "4H"), [])

    def test_fall_back_4h_bucket_needs_300_minutes(self):
        # Also Sunday/closed: the repeated civil hour cannot create a valid
        # market-data bucket when the market itself is not open.
        start_utc = datetime(2026, 11, 1, 2, 0, tzinfo=timezone.utc)
        closer = [bar_at_utc(datetime(2026, 11, 1, 7, 5, tzinfo=timezone.utc))]

        complete = minute_series_utc(start_utc, 300)
        self.assertEqual(aggregate_bars(complete + closer, "4H"), [])

    def test_fall_back_trading_day_needs_1440_minutes(self):
        # Saturday 18:00 through Sunday 18:00 is the weekly closure, not a
        # trading day. The old calendar's 1440-minute expectation was false.
        day_start = datetime(2026, 10, 31, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        day_end = datetime(2026, 11, 1, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        expected = expected_open_minutes(day_start, day_end)
        self.assertEqual(expected, [])


if __name__ == "__main__":
    unittest.main()
