#!/usr/bin/env python3
"""Aggregation tests: OHLCV rollup, timeframe boundaries, trading-day and DST rules."""
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.aggregate import aggregate_bars
from tradelab.bar import Bar1m

NY = ZoneInfo("America/New_York")
CONTRACT = "NQZ25"


def bar_at_ny(ny_dt: datetime, o: float, h: float, l: float, c: float, v: float) -> Bar1m:
    return Bar1m(ny_dt.astimezone(timezone.utc), o, h, l, c, v, CONTRACT)


def minute_series(start_ny: datetime, n: int, price: float = 100.0) -> list[Bar1m]:
    """n consecutive 1m bars, each with a distinct high/low/volume so rollup math is checkable."""
    bars = []
    for i in range(n):
        ts = start_ny + timedelta(minutes=i)
        bars.append(bar_at_ny(ts, price + i, price + i + 0.5, price + i - 0.5, price + i + 0.2, 10 + i))
    return bars


class OhlcvRollupTests(unittest.TestCase):
    def test_open_high_low_close_volume_are_correctly_rolled_up(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        bars = minute_series(start, 5)  # exactly one closed 5m bucket
        [agg] = aggregate_bars(bars, "5m")
        self.assertEqual(agg.open, bars[0].open)
        self.assertEqual(agg.close, bars[-1].close)
        self.assertEqual(agg.high, max(b.high for b in bars))
        self.assertEqual(agg.low, min(b.low for b in bars))
        self.assertEqual(agg.volume, sum(b.volume for b in bars))
        self.assertEqual(agg.contract, CONTRACT)
        self.assertEqual(agg.timeframe, "5m")


class BoundaryTests(unittest.TestCase):
    def test_5m_boundary_splits_buckets(self):
        start = datetime(2026, 1, 5, 9, 33, tzinfo=NY)  # mid-bucket start (09:30-09:35)
        bars = minute_series(start, 4)  # 09:33, 09:34 (bucket A) ; 09:35, 09:36 (bucket B)
        agg = aggregate_bars(bars, "5m")
        self.assertEqual(len(agg), 1)  # only bucket A is closed; trailing bucket B is dropped
        self.assertEqual(agg[0].timestamp, datetime(2026, 1, 5, 9, 30, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(agg[0].open, bars[0].open)
        self.assertEqual(agg[0].close, bars[1].close)

    def test_15m_boundary_splits_buckets(self):
        start = datetime(2026, 1, 5, 9, 44, tzinfo=NY)
        bars = minute_series(start, 3)  # 09:44 (bucket 09:30-09:45); 09:45, 09:46 (next bucket)
        agg = aggregate_bars(bars, "15m")
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].timestamp, datetime(2026, 1, 5, 9, 30, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(agg[0].open, bars[0].open)
        self.assertEqual(agg[0].close, bars[0].close)

    def test_1h_boundary_splits_buckets(self):
        start = datetime(2026, 1, 5, 9, 59, tzinfo=NY)
        bars = minute_series(start, 3)  # 09:59 (bucket 09:00-10:00); 10:00, 10:01 (next bucket)
        agg = aggregate_bars(bars, "1H")
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].timestamp, datetime(2026, 1, 5, 9, 0, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(agg[0].close, bars[0].close)

    def test_4h_boundary_splits_buckets(self):
        start = datetime(2026, 1, 5, 11, 59, tzinfo=NY)
        bars = minute_series(start, 3)  # 11:59 (bucket 08:00-12:00); 12:00, 12:01 (next bucket)
        agg = aggregate_bars(bars, "4H")
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].timestamp, datetime(2026, 1, 5, 8, 0, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(agg[0].close, bars[0].close)


class TradingDayTests(unittest.TestCase):
    def test_trading_day_rolls_over_at_18_00_ny(self):
        pre_rollover = bar_at_ny(datetime(2026, 1, 5, 17, 59, tzinfo=NY), 1, 1, 1, 1, 1)
        post_rollover = bar_at_ny(datetime(2026, 1, 5, 18, 0, tzinfo=NY), 2, 2, 2, 2, 2)
        next_day_close = bar_at_ny(datetime(2026, 1, 6, 18, 0, tzinfo=NY), 3, 3, 3, 3, 3)
        agg = aggregate_bars([pre_rollover, post_rollover, next_day_close], "1D")

        self.assertEqual(len(agg), 2)
        # 17:59 NY on Jan 5 still belongs to the trading day that opened 18:00 NY on Jan 4.
        self.assertEqual(agg[0].timestamp, datetime(2026, 1, 4, 18, 0, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(agg[0].close, pre_rollover.close)
        # 18:00 NY on Jan 5 opens the next trading day.
        self.assertEqual(agg[1].timestamp, datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(agg[1].close, post_rollover.close)
        self.assertEqual(agg[1].known_at, datetime(2026, 1, 6, 18, 0, tzinfo=NY).astimezone(timezone.utc))

    def test_trading_day_uses_ny_calendar_not_utc_midnight(self):
        # 20:00 UTC on Jan 5 is 15:00 NY (winter, UTC-5): still inside the trading
        # day that opened 18:00 NY on Jan 4, well past UTC midnight.
        bar = bar_at_ny(datetime(2026, 1, 5, 15, 0, tzinfo=NY), 1, 1, 1, 1, 1)
        closer = bar_at_ny(datetime(2026, 1, 5, 18, 0, tzinfo=NY), 2, 2, 2, 2, 2)
        [agg] = aggregate_bars([bar, closer], "1D")
        self.assertEqual(agg.timestamp, datetime(2026, 1, 4, 18, 0, tzinfo=NY).astimezone(timezone.utc))


class DstTransitionTests(unittest.TestCase):
    def test_spring_forward_4h_bucket_has_correct_wall_clock_boundaries(self):
        # 2026-03-08: US clocks jump 02:00 -> 03:00 NY. A 00:00-04:00 bucket
        # spans only 3 real hours that day, but the boundary is still 04:00 wall clock.
        start = datetime(2026, 3, 8, 0, 0, tzinfo=NY)
        bars = minute_series(start, 2)
        closer = bar_at_ny(datetime(2026, 3, 8, 4, 5, tzinfo=NY), 9, 9, 9, 9, 9)
        agg = aggregate_bars(bars + [closer], "4H")
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0].known_at.astimezone(NY), datetime(2026, 3, 8, 4, 0, tzinfo=NY))
        # Real elapsed time for this bucket is 3 hours, not 4, because of the DST jump.
        # (Subtract the fixed-offset UTC values, not NY-zoneinfo ones: CPython's aware
        # datetime subtraction takes an identity-based shortcut when both operands
        # share the same ZoneInfo object, which would silently give 4h here instead.)
        elapsed_hours = (agg[0].known_at - agg[0].timestamp).total_seconds() / 3600
        self.assertEqual(elapsed_hours, 3.0)

    def test_fall_back_trading_day_spans_25_real_hours(self):
        # 2026-11-01: US clocks fall back 02:00 -> 01:00 NY (extra hour).
        # The CME trading day opening 18:00 NY on Oct 31 still ends 18:00 NY on Nov 1,
        # but that wall-clock day is 25 real hours long.
        opener = bar_at_ny(datetime(2026, 10, 31, 18, 0, tzinfo=NY), 1, 1, 1, 1, 1)
        closer = bar_at_ny(datetime(2026, 11, 1, 18, 0, tzinfo=NY), 2, 2, 2, 2, 2)
        [agg] = aggregate_bars([opener, closer], "1D")
        elapsed_hours = (agg.known_at - agg.timestamp).total_seconds() / 3600
        self.assertEqual(elapsed_hours, 25.0)
        self.assertEqual(agg.known_at.astimezone(NY), datetime(2026, 11, 1, 18, 0, tzinfo=NY))


class KnownAtSemanticsTests(unittest.TestCase):
    def test_trailing_incomplete_bucket_is_not_returned(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        bars = minute_series(start, 4)  # only 4 of 5 minutes: bucket never closes
        self.assertEqual(aggregate_bars(bars, "5m"), [])

    def test_interior_bucket_with_missing_minutes_still_closes(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        # bucket 09:30-09:35 only has bars for :30 and :31 (gap at :32-:34), then
        # bucket 09:35-09:40 is fully present: bucket A must still close, because a
        # later bar proves wall-clock time has moved past its boundary.
        gappy = [bar_at_ny(start, 1, 1, 1, 1, 1), bar_at_ny(start + timedelta(minutes=1), 2, 2, 2, 2, 2)]
        full_next = minute_series(start + timedelta(minutes=5), 5, price=200.0)
        agg = aggregate_bars(gappy + full_next, "5m")
        self.assertEqual(len(agg), 2)
        self.assertEqual(agg[0].open, gappy[0].open)
        self.assertEqual(agg[0].close, gappy[-1].close)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(aggregate_bars([], "5m"), [])

    def test_unsupported_timeframe_raises(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY)
        with self.assertRaises(ValueError):
            aggregate_bars(minute_series(start, 5), "3m")


if __name__ == "__main__":
    unittest.main()
