#!/usr/bin/env python3
"""Regular and exceptional NQ/MNQ session calendar tests."""
import unittest
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from tradelab.session_calendar import expected_open_minutes, is_regular_session_open, load_special_session_schedule

NY = ZoneInfo("America/New_York")


class DailyBreakTests(unittest.TestCase):
    def test_break_minutes_are_closed(self):
        for minute in (0, 1, 30, 59):
            with self.subTest(minute=minute):
                ts = datetime(2026, 1, 5, 17, minute, tzinfo=NY).astimezone(timezone.utc)
                self.assertFalse(is_regular_session_open(ts))

    def test_minutes_outside_the_break_are_open(self):
        for hour in (16, 18, 0, 9, 23):
            with self.subTest(hour=hour):
                ts = datetime(2026, 1, 5, hour, 0, tzinfo=NY).astimezone(timezone.utc)
                self.assertTrue(is_regular_session_open(ts))

    def test_break_boundary_is_exclusive_at_18_00(self):
        just_before = datetime(2026, 1, 5, 16, 59, tzinfo=NY).astimezone(timezone.utc)
        break_start = datetime(2026, 1, 5, 17, 0, tzinfo=NY).astimezone(timezone.utc)
        just_after = datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        self.assertTrue(is_regular_session_open(just_before))
        self.assertFalse(is_regular_session_open(break_start))
        self.assertTrue(is_regular_session_open(just_after))


class ExpectedOpenMinutesTests(unittest.TestCase):
    def test_full_trading_day_has_1380_open_minutes(self):
        start = datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        end = datetime(2026, 1, 6, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        minutes = expected_open_minutes(start, end)
        self.assertEqual(len(minutes), 1380)

    def test_break_window_contributes_no_expected_minutes(self):
        # Directly verifies the scheduled break is never reported as "missing":
        # it simply never appears in the expected set at all.
        start = datetime(2026, 1, 5, 17, 0, tzinfo=NY).astimezone(timezone.utc)
        end = datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        self.assertEqual(expected_open_minutes(start, end), [])

    def test_window_spanning_the_break_excludes_exactly_those_60_minutes(self):
        start = datetime(2026, 1, 5, 14, 0, tzinfo=NY).astimezone(timezone.utc)  # 14:00-18:00
        end = datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        minutes = expected_open_minutes(start, end)
        self.assertEqual(len(minutes), 180)
        break_start = datetime(2026, 1, 5, 17, 0, tzinfo=NY).astimezone(timezone.utc)
        break_end = datetime(2026, 1, 5, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        self.assertTrue(all(m < break_start or m >= break_end for m in minutes))

    def test_window_fully_within_regular_hours_has_no_exclusions(self):
        start = datetime(2026, 1, 5, 9, 30, tzinfo=NY).astimezone(timezone.utc)
        end = datetime(2026, 1, 5, 9, 35, tzinfo=NY).astimezone(timezone.utc)
        self.assertEqual(len(expected_open_minutes(start, end)), 5)

    def test_empty_window_returns_empty_list(self):
        ts = datetime(2026, 1, 5, 9, 30, tzinfo=NY).astimezone(timezone.utc)
        self.assertEqual(expected_open_minutes(ts, ts), [])

    def test_weekend_is_closed_until_sunday_18_ny(self):
        saturday = datetime(2026, 1, 3, 12, 0, tzinfo=NY).astimezone(timezone.utc)
        sunday_before = datetime(2026, 1, 4, 17, 59, tzinfo=NY).astimezone(timezone.utc)
        sunday_open = datetime(2026, 1, 4, 18, 0, tzinfo=NY).astimezone(timezone.utc)
        self.assertFalse(is_regular_session_open(saturday))
        self.assertFalse(is_regular_session_open(sunday_before))
        self.assertTrue(is_regular_session_open(sunday_open))

    def test_versioned_exception_subtracts_an_early_close(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.json"
            path.write_text(json.dumps({
                "schema_version": "cme-special-session-schedule-v1",
                "source_url": "https://www.cmegroup.com/trading-hours.html",
                "retrieved_at": "2026-08-22T00:00:00Z",
                "closed_intervals": [{
                    "start_utc": "2026-01-05T17:00:00Z",
                    "end_utc": "2026-01-05T22:00:00Z",
                    "reason": "test early close",
                }],
            }), encoding="utf-8")
            schedule = load_special_session_schedule(path)
            self.assertEqual(expected_open_minutes(
                datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc),
                special_schedule=schedule,
            ), [])


if __name__ == "__main__":
    unittest.main()
