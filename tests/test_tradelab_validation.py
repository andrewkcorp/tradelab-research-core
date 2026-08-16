#!/usr/bin/env python3
"""Validation tests: ordering, uniqueness, OHLC range, volume, missing-minutes registry."""
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.bar import Bar1m
from tradelab.validation import validate_bars

NY = ZoneInfo("America/New_York")
CONTRACT = "NQZ25"
START = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)  # 09:30 NY: well inside regular hours


def bar(offset_minutes: float, o=100.0, h=100.5, l=99.5, c=100.2, v=10.0) -> Bar1m:
    return Bar1m(START + timedelta(minutes=offset_minutes), o, h, l, c, v, CONTRACT)


def bar_at_ny(ny_dt: datetime, o=100.0, h=100.5, l=99.5, c=100.2, v=10.0) -> Bar1m:
    return Bar1m(ny_dt.astimezone(timezone.utc), o, h, l, c, v, CONTRACT)


class HardInvariantTests(unittest.TestCase):
    def test_well_formed_series_passes_with_no_gaps(self):
        bars = [bar(i) for i in range(5)]
        result = validate_bars(bars)
        self.assertEqual(result.gaps, ())

    def test_out_of_order_timestamps_raise(self):
        bars = [bar(1), bar(0)]
        with self.assertRaises(ValueError):
            validate_bars(bars)

    def test_duplicate_timestamps_raise(self):
        bars = [bar(0), bar(0)]
        with self.assertRaises(ValueError):
            validate_bars(bars)

    def test_invalid_ohlc_range_raises(self):
        bad = Bar1m(START, open=100.0, high=99.0, low=99.5, close=100.0, volume=1.0, contract=CONTRACT)
        with self.assertRaises(ValueError):
            validate_bars([bad])

    def test_negative_volume_raises(self):
        bad = Bar1m(START, open=100.0, high=100.5, low=99.5, close=100.0, volume=-1.0, contract=CONTRACT)
        with self.assertRaises(ValueError):
            validate_bars([bad])

    def test_zero_volume_is_allowed(self):
        ok = Bar1m(START, open=100.0, high=100.5, low=99.5, close=100.0, volume=0.0, contract=CONTRACT)
        result = validate_bars([ok])
        self.assertEqual(result.gaps, ())

    def test_empty_series_is_valid(self):
        result = validate_bars([])
        self.assertEqual(result.gaps, ())


class MissingMinutesRegistryTests(unittest.TestCase):
    def test_single_missing_minute_is_recorded(self):
        bars = [bar(0), bar(2)]  # minute 1 is missing
        result = validate_bars(bars)
        self.assertEqual(len(result.gaps), 1)
        gap = result.gaps[0]
        self.assertEqual(gap.first_missing, bars[0].timestamp + timedelta(minutes=1))
        self.assertEqual(gap.last_missing, bars[0].timestamp + timedelta(minutes=1))
        self.assertEqual(gap.missing_count, 1)

    def test_multi_minute_gap_counts_all_missing_minutes(self):
        bars = [bar(0), bar(61)]  # an hour-long gap: 60 missing minutes
        result = validate_bars(bars)
        self.assertEqual(len(result.gaps), 1)
        self.assertEqual(result.gaps[0].missing_count, 60)

    def test_multiple_gaps_are_all_recorded_independently(self):
        bars = [bar(0), bar(2), bar(3), bar(10)]  # gap of 1, then gap of 6
        result = validate_bars(bars)
        self.assertEqual(len(result.gaps), 2)
        self.assertEqual(result.gaps[0].missing_count, 1)
        self.assertEqual(result.gaps[1].missing_count, 6)


class DailyBreakIsNotAGapTests(unittest.TestCase):
    """Per PR #2 review: the 17:00-18:00 NY daily maintenance break must never
    show up in the missing-minutes registry, and a raw gap straddling it must
    be split into separate real-gap runs rather than one lump covering the break."""

    def test_16_59_then_18_00_reports_zero_gaps(self):
        bars = [bar_at_ny(datetime(2026, 1, 5, 16, 59, tzinfo=NY)), bar_at_ny(datetime(2026, 1, 5, 18, 0, tzinfo=NY))]
        result = validate_bars(bars)
        self.assertEqual(result.gaps, ())

    def test_missing_16_58_between_16_57_and_16_59_is_one_real_gap(self):
        bars = [bar_at_ny(datetime(2026, 1, 5, 16, 57, tzinfo=NY)), bar_at_ny(datetime(2026, 1, 5, 16, 59, tzinfo=NY))]
        result = validate_bars(bars)
        self.assertEqual(len(result.gaps), 1)
        gap = result.gaps[0]
        self.assertEqual(gap.first_missing, datetime(2026, 1, 5, 16, 58, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(gap.last_missing, datetime(2026, 1, 5, 16, 58, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(gap.missing_count, 1)

    def test_missing_18_01_between_18_00_and_18_02_is_one_real_gap(self):
        bars = [bar_at_ny(datetime(2026, 1, 5, 18, 0, tzinfo=NY)), bar_at_ny(datetime(2026, 1, 5, 18, 2, tzinfo=NY))]
        result = validate_bars(bars)
        self.assertEqual(len(result.gaps), 1)
        gap = result.gaps[0]
        self.assertEqual(gap.first_missing, datetime(2026, 1, 5, 18, 1, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(gap.last_missing, datetime(2026, 1, 5, 18, 1, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(gap.missing_count, 1)

    def test_gap_straddling_the_break_reports_only_active_session_minutes(self):
        # 16:00 to 19:00 NY with nothing in between: only 16:01-16:59 (59 min)
        # and 18:00-18:59 (59 min) are real gaps; the 17:00-18:00 break itself
        # contributes nothing, and it is never lumped into one 179-minute gap.
        bars = [bar_at_ny(datetime(2026, 1, 5, 16, 0, tzinfo=NY)), bar_at_ny(datetime(2026, 1, 5, 19, 0, tzinfo=NY))]
        result = validate_bars(bars)
        self.assertEqual(len(result.gaps), 2)

        before_break, after_break = result.gaps
        self.assertEqual(before_break.first_missing, datetime(2026, 1, 5, 16, 1, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(before_break.last_missing, datetime(2026, 1, 5, 16, 59, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(before_break.missing_count, 59)

        self.assertEqual(after_break.first_missing, datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(after_break.last_missing, datetime(2026, 1, 5, 18, 59, tzinfo=NY).astimezone(timezone.utc))
        self.assertEqual(after_break.missing_count, 60)

        total_reported = before_break.missing_count + after_break.missing_count
        self.assertEqual(total_reported, 119)  # never the raw 179-minute span


if __name__ == "__main__":
    unittest.main()
