#!/usr/bin/env python3
"""Bar1m model tests: timezone normalization and strict minute alignment."""
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.bar import Bar1m

CONTRACT = "NQZ25"


def make(ts: datetime) -> Bar1m:
    return Bar1m(ts, open=100.0, high=100.5, low=99.5, close=100.2, volume=10.0, contract=CONTRACT)


class TimezoneNormalizationTests(unittest.TestCase):
    def test_naive_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            make(datetime(2026, 1, 5, 9, 30))

    def test_utc_timestamp_is_kept_as_utc(self):
        bar = make(datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc))
        self.assertEqual(bar.timestamp, datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc))
        self.assertEqual(bar.timestamp.utcoffset(), timedelta(0))

    def test_offset_timestamp_is_normalized_to_utc(self):
        minus_five = timezone(timedelta(hours=-5))
        bar = make(datetime(2026, 1, 5, 9, 30, tzinfo=minus_five))
        self.assertEqual(bar.timestamp, datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc))
        self.assertEqual(bar.timestamp.utcoffset(), timedelta(0))

    def test_positive_offset_timestamp_is_normalized_to_utc(self):
        plus_nine = timezone(timedelta(hours=9))
        bar = make(datetime(2026, 1, 5, 9, 30, tzinfo=plus_nine))
        self.assertEqual(bar.timestamp, datetime(2026, 1, 5, 0, 30, tzinfo=timezone.utc))


class MinuteAlignmentTests(unittest.TestCase):
    def test_exact_minute_boundary_is_accepted(self):
        bar = make(datetime(2026, 1, 5, 9, 30, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(bar.timestamp.second, 0)
        self.assertEqual(bar.timestamp.microsecond, 0)

    def test_nonzero_seconds_is_rejected(self):
        with self.assertRaises(ValueError):
            make(datetime(2026, 1, 5, 9, 30, 1, tzinfo=timezone.utc))

    def test_nonzero_microseconds_is_rejected(self):
        with self.assertRaises(ValueError):
            make(datetime(2026, 1, 5, 9, 30, 0, 1, tzinfo=timezone.utc))

    def test_misalignment_is_rejected_even_after_utc_normalization(self):
        # 09:30:30 in UTC-5 is exactly 14:30:30 UTC: still off the minute grid,
        # so this must be rejected even though the *local* second looks fixed.
        minus_five = timezone(timedelta(hours=-5))
        with self.assertRaises(ValueError):
            make(datetime(2026, 1, 5, 9, 30, 30, tzinfo=minus_five))


class OtherValidationTests(unittest.TestCase):
    def test_empty_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            Bar1m(datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc), 100.0, 100.5, 99.5, 100.2, 10.0, "")


if __name__ == "__main__":
    unittest.main()
