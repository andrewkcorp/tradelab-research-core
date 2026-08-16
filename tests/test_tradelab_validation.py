#!/usr/bin/env python3
"""Validation tests: ordering, uniqueness, OHLC range, volume, missing-minutes registry."""
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.bar import Bar1m
from tradelab.validation import validate_bars

CONTRACT = "NQZ25"
START = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)


def bar(offset_minutes: float, o=100.0, h=100.5, l=99.5, c=100.2, v=10.0) -> Bar1m:
    return Bar1m(START + timedelta(minutes=offset_minutes), o, h, l, c, v, CONTRACT)


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
        self.assertEqual(gap.after, bars[0].timestamp)
        self.assertEqual(gap.before, bars[1].timestamp)
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


if __name__ == "__main__":
    unittest.main()
