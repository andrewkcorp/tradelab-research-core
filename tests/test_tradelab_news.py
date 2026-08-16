#!/usr/bin/env python3
"""NewsEvent model, batch validation, and red-folder blackout window tests.

No network access anywhere in this file: every event is built in-process.
"""
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.news import NewsEvent, is_entry_blocked, validate_news_events

DEFAULT_TS = datetime(2026, 1, 13, 13, 30, tzinfo=timezone.utc)


def make_event(
    event_id: str = "CPI-2026-01",
    scheduled_at: datetime = DEFAULT_TS,
    event_type: str = "CPI",
    country: str = "United States",
    currency: str = "USD",
    source_agency: str = "BLS",
    source_url: str = "https://www.bls.gov/cpi/",
    source_reference: str = "CPI-2026-01-REL",
    time_is_exact: bool = True,
    schema_version: str = "1.0",
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        scheduled_at=scheduled_at,
        country=country,
        currency=currency,
        event_type=event_type,
        source_agency=source_agency,
        source_url=source_url,
        source_reference=source_reference,
        time_is_exact=time_is_exact,
        schema_version=schema_version,
    )


class NewsEventModelTests(unittest.TestCase):
    def test_well_formed_event_is_accepted(self):
        event = make_event()
        self.assertEqual(event.event_type, "CPI")
        self.assertEqual(event.scheduled_at, DEFAULT_TS)

    def test_non_us_country_is_rejected(self):
        with self.assertRaises(ValueError):
            make_event(country="Canada")

    def test_non_usd_currency_is_rejected(self):
        with self.assertRaises(ValueError):
            make_event(currency="CAD")

    def test_unknown_event_type_is_rejected(self):
        with self.assertRaises(ValueError):
            make_event(event_type="RANDOM_INDICATOR")

    def test_all_v1_event_types_are_individually_accepted(self):
        from tradelab.news import EVENT_TYPES

        for event_type in EVENT_TYPES:
            with self.subTest(event_type=event_type):
                event = make_event(event_id=f"evt-{event_type}", event_type=event_type)
                self.assertEqual(event.event_type, event_type)

    def test_naive_scheduled_at_is_rejected(self):
        with self.assertRaises(ValueError):
            make_event(scheduled_at=datetime(2026, 1, 13, 13, 30))

    def test_offset_scheduled_at_is_normalized_to_utc(self):
        minus_five = timezone(timedelta(hours=-5))
        event = make_event(scheduled_at=datetime(2026, 1, 13, 8, 30, tzinfo=minus_five))
        self.assertEqual(event.scheduled_at, DEFAULT_TS)

    def test_time_is_exact_false_is_rejected(self):
        with self.assertRaises(ValueError):
            make_event(time_is_exact=False)

    def test_empty_event_id_is_rejected(self):
        with self.assertRaises(ValueError):
            make_event(event_id="")

    def test_empty_source_fields_are_rejected(self):
        for field in ("source_agency", "source_url", "source_reference"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    make_event(**{field: ""})

    def test_model_has_no_actual_or_forecast_fields(self):
        # The blackout decision must be computable before publication: there
        # is nothing on the model that could tempt a caller into using a
        # published number to decide blocking.
        field_names = {f.name for f in NewsEvent.__dataclass_fields__.values()}
        self.assertNotIn("actual", field_names)
        self.assertNotIn("forecast", field_names)

    def test_correct_schema_version_is_accepted(self):
        event = make_event(schema_version="1.0")
        self.assertEqual(event.schema_version, "1.0")

    def test_other_schema_versions_are_rejected(self):
        for bad_version in ("2.0", "1.1", "1", "", "1.0 ", " 1.0"):
            with self.subTest(schema_version=bad_version):
                with self.assertRaises(ValueError):
                    make_event(schema_version=bad_version)

    def test_exact_minute_boundary_scheduled_at_is_accepted(self):
        event = make_event(scheduled_at=datetime(2026, 1, 13, 13, 30, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(event.scheduled_at.second, 0)
        self.assertEqual(event.scheduled_at.microsecond, 0)

    def test_nonzero_seconds_in_scheduled_at_is_rejected(self):
        with self.assertRaises(ValueError):
            make_event(scheduled_at=datetime(2026, 1, 13, 13, 30, 1, tzinfo=timezone.utc))

    def test_nonzero_microseconds_in_scheduled_at_is_rejected(self):
        with self.assertRaises(ValueError):
            make_event(scheduled_at=datetime(2026, 1, 13, 13, 30, 0, 1, tzinfo=timezone.utc))

    def test_minute_misalignment_is_rejected_even_after_utc_normalization(self):
        # 08:30:30 in UTC-5 is exactly 13:30:30 UTC: still off the minute
        # grid, so this must be rejected even though the *local* second
        # looks like part of the offset conversion rather than real data.
        minus_five = timezone(timedelta(hours=-5))
        with self.assertRaises(ValueError):
            make_event(scheduled_at=datetime(2026, 1, 13, 8, 30, 30, tzinfo=minus_five))


class ValidateNewsEventsTests(unittest.TestCase):
    def test_distinct_events_pass(self):
        events = [make_event("a", DEFAULT_TS, "CPI"), make_event("b", DEFAULT_TS, "NFP")]
        validate_news_events(events)  # must not raise

    def test_duplicate_event_id_raises(self):
        events = [make_event("a", DEFAULT_TS, "CPI"), make_event("a", DEFAULT_TS + timedelta(days=30), "PPI")]
        with self.assertRaises(ValueError):
            validate_news_events(events)

    def test_duplicate_event_type_and_scheduled_at_raises(self):
        events = [make_event("a", DEFAULT_TS, "CPI"), make_event("b", DEFAULT_TS, "CPI")]
        with self.assertRaises(ValueError):
            validate_news_events(events)

    def test_same_scheduled_at_different_event_type_is_allowed(self):
        # NFP + unemployment rate + average hourly earnings are all released
        # together as part of the Employment Situation report.
        events = [
            make_event("a", DEFAULT_TS, "NFP"),
            make_event("b", DEFAULT_TS, "UNEMPLOYMENT_RATE"),
            make_event("c", DEFAULT_TS, "AVERAGE_HOURLY_EARNINGS"),
        ]
        validate_news_events(events)  # must not raise

    def test_empty_list_is_valid(self):
        validate_news_events([])  # must not raise


class RedFolderWindowBoundaryTests(unittest.TestCase):
    """Per spec: blackout is [T-15min, T+5min], inclusive on both ends."""

    def setUp(self):
        self.event = make_event(scheduled_at=DEFAULT_TS)

    def test_t_minus_15_is_blocked(self):
        blocked, events = is_entry_blocked(DEFAULT_TS - timedelta(minutes=15), [self.event])
        self.assertTrue(blocked)
        self.assertEqual(events, [self.event])

    def test_t_minus_16_is_not_blocked(self):
        blocked, events = is_entry_blocked(DEFAULT_TS - timedelta(minutes=16), [self.event])
        self.assertFalse(blocked)
        self.assertEqual(events, [])

    def test_t_is_blocked(self):
        blocked, _ = is_entry_blocked(DEFAULT_TS, [self.event])
        self.assertTrue(blocked)

    def test_t_plus_5_is_blocked(self):
        blocked, _ = is_entry_blocked(DEFAULT_TS + timedelta(minutes=5), [self.event])
        self.assertTrue(blocked)

    def test_t_plus_6_is_not_blocked(self):
        blocked, events = is_entry_blocked(DEFAULT_TS + timedelta(minutes=6), [self.event])
        self.assertFalse(blocked)
        self.assertEqual(events, [])

    def test_no_events_never_blocks(self):
        blocked, events = is_entry_blocked(DEFAULT_TS, [])
        self.assertFalse(blocked)
        self.assertEqual(events, [])


class OverlappingWindowsTests(unittest.TestCase):
    def test_overlapping_windows_report_both_blocking_events(self):
        first = make_event("a", DEFAULT_TS, "CPI")
        second = make_event("b", DEFAULT_TS + timedelta(minutes=10), "PPI")  # windows overlap: T+10-15=T-5 <= T+5
        ts_in_both_windows = DEFAULT_TS + timedelta(minutes=5)  # in first's [-15,+5] and second's [-5,+15]
        blocked, events = is_entry_blocked(ts_in_both_windows, [first, second])
        self.assertTrue(blocked)
        self.assertEqual({e.event_id for e in events}, {"a", "b"})

    def test_non_overlapping_events_only_report_the_relevant_one(self):
        first = make_event("a", DEFAULT_TS, "CPI")
        second = make_event("b", DEFAULT_TS + timedelta(hours=2), "PPI")
        blocked, events = is_entry_blocked(DEFAULT_TS, [first, second])
        self.assertTrue(blocked)
        self.assertEqual([e.event_id for e in events], ["a"])

    def test_no_trades_blocked_after_t_plus_5_even_with_multiple_events(self):
        first = make_event("a", DEFAULT_TS, "CPI")
        second = make_event("b", DEFAULT_TS - timedelta(hours=1), "PPI")
        blocked, events = is_entry_blocked(DEFAULT_TS + timedelta(minutes=6), [first, second])
        self.assertFalse(blocked)
        self.assertEqual(events, [])


class DstInvarianceTests(unittest.TestCase):
    """UTC storage must not change blackout behavior across a DST transition
    -- the window function does no timezone conversion at all, so results
    must be identical for equivalent UTC offsets regardless of calendar date."""

    def test_boundary_behavior_is_identical_on_both_sides_of_a_dst_change(self):
        # 2026-03-08 is the US spring-forward date; pick one event well before
        # and one well after, both anchored purely in UTC.
        before_dst = make_event("a", datetime(2026, 3, 1, 13, 30, tzinfo=timezone.utc))
        after_dst = make_event("b", datetime(2026, 3, 15, 13, 30, tzinfo=timezone.utc))

        for event in (before_dst, after_dst):
            with self.subTest(event_id=event.event_id):
                t = event.scheduled_at
                self.assertTrue(is_entry_blocked(t - timedelta(minutes=15), [event])[0])
                self.assertFalse(is_entry_blocked(t - timedelta(minutes=16), [event])[0])
                self.assertTrue(is_entry_blocked(t + timedelta(minutes=5), [event])[0])
                self.assertFalse(is_entry_blocked(t + timedelta(minutes=6), [event])[0])

    def test_events_stored_with_different_utc_offsets_behave_identically_once_normalized(self):
        # Same real instant, supplied with two different (but correct) UTC
        # offsets: after NewsEvent normalizes both to UTC, blackout behavior
        # must be identical.
        utc_event = make_event("a", datetime(2026, 1, 13, 13, 30, tzinfo=timezone.utc))
        offset_event = make_event("b", datetime(2026, 1, 13, 8, 30, tzinfo=timezone(timedelta(hours=-5))))
        self.assertEqual(utc_event.scheduled_at, offset_event.scheduled_at)

        probe = datetime(2026, 1, 13, 13, 35, tzinfo=timezone.utc)  # T+5 for both
        self.assertEqual(
            is_entry_blocked(probe, [utc_event])[0],
            is_entry_blocked(probe, [offset_event])[0],
        )


if __name__ == "__main__":
    unittest.main()
