#!/usr/bin/env python3
"""Data wick tests: fixed-threshold upper/lower detection, candidate
window boundaries (T..T+5), baseline correctness (60 candles, no
lookahead, gap/shortage rejection), overlapping news events, timestamp/
known_at semantics, and Bar1m/Candle adapter reuse."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.bar import Bar1m
from tradelab.data_wicks import (
    BASELINE_LENGTH,
    DataWickDirection,
    detect_data_wicks,
    from_bar1m,
)
from tradelab.news import NewsEvent

TICK = 0.25
T0 = datetime(2026, 1, 5, 13, 0, tzinfo=timezone.utc)
T = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)  # a convenient event scheduled_at


def bar(ts: datetime, o: float, h: float, l: float, c: float) -> Bar1m:
    return Bar1m(ts, o, h, l, c, 1000.0, "NQ")


def flat_baseline(start_ts: datetime, count: int, base: float = 100.0, half_range: float = 0.5) -> list[Bar1m]:
    """`count` contiguous 1m bars, each with range == 2 * half_range."""
    return [bar(start_ts + timedelta(minutes=i), base, base + half_range, base - half_range, base) for i in range(count)]


def wick_bar(ts: datetime, low: float, range_val: float, upper_wick: float, lower_wick: float) -> Bar1m:
    """A single bar with an exact range and exact upper/lower wick sizes."""
    high = low + range_val
    body_top = high - upper_wick
    body_bottom = low + lower_wick
    return bar(ts, body_bottom, high, low, body_top)


def candles(bars: list[Bar1m]):
    return [from_bar1m(b) for b in bars]


def make_event(event_id: str = "CPI-1", scheduled_at: datetime = T, event_type: str = "CPI") -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        scheduled_at=scheduled_at,
        country="United States",
        currency="USD",
        event_type=event_type,
        source_agency="BLS",
        source_url="https://www.bls.gov/x/",
        source_reference="REF-1",
        time_is_exact=True,
        schema_version="1.0",
    )


def baseline_start(event_scheduled_at: datetime) -> datetime:
    return event_scheduled_at - timedelta(minutes=BASELINE_LENGTH)


class UpperLowerDetectionTests(unittest.TestCase):
    def test_upper_wick_detected_as_bearish_confluence_at_the_high(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)  # range=1.0 each, median=1.0
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)  # 6.0 >= 2*1.0; 4.0 >= 0.5*6.0; 4.0 >= 10*0.25
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.direction, DataWickDirection.UPPER)
        self.assertEqual(event.level, 105.0)  # candidate high
        self.assertEqual(event.candle_range, 6.0)
        self.assertEqual(event.baseline_median, 1.0)
        self.assertEqual(event.wick_length, 4.0)

    def test_lower_wick_detected_as_bullish_confluence_at_the_low(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=0.0, lower_wick=4.0)
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.direction, DataWickDirection.LOWER)
        self.assertEqual(event.level, 99.0)  # candidate low
        self.assertEqual(event.wick_length, 4.0)

    def test_both_directions_can_qualify_on_the_same_candle(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=3.0, lower_wick=3.0)  # both exactly 50%
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        directions = {e.direction for e in result.events}
        self.assertEqual(directions, {DataWickDirection.UPPER, DataWickDirection.LOWER})

    def test_ordinary_candle_produces_no_event(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = bar(T, 100.0, 100.5, 99.5, 100.0)  # range == median, no multiplier
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(result.events, [])


class ThresholdBoundaryTests(unittest.TestCase):
    def test_range_exactly_at_2x_median_qualifies(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH, half_range=5.0)  # range=10 each, median=10
        candidate = wick_bar(T, low=90.0, range_val=20.0, upper_wick=15.0, lower_wick=0.0)  # range==2*10 exactly
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(len(result.events), 1)

    def test_range_just_below_2x_median_does_not_qualify(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH, half_range=5.0)  # median=10
        candidate = wick_bar(T, low=90.0, range_val=19.9, upper_wick=15.0, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(result.events, [])

    def test_wick_share_exactly_50_percent_qualifies(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH, half_range=5.0)  # median=10, gate=20
        candidate = wick_bar(T, low=90.0, range_val=30.0, upper_wick=15.0, lower_wick=0.0)  # 30>=20; 15==50% of 30; 15>=2.5 ticks
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(len(result.events), 1)

    def test_wick_share_just_below_50_percent_does_not_qualify(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH, half_range=5.0)
        candidate = wick_bar(T, low=90.0, range_val=30.0, upper_wick=14.9, lower_wick=0.0)  # 49.67%
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(result.events, [])

    def test_wick_exactly_10_ticks_qualifies(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH, half_range=0.5)  # median=1.0, gate=2.0
        candidate = wick_bar(T, low=90.0, range_val=3.0, upper_wick=2.5, lower_wick=0.0)  # 3>=2.0; 2.5==83% of 3 (>=50%); 2.5==10*0.25
        result = detect_data_wicks(candles(base + [candidate]), [make_event()], tick_size=TICK)
        self.assertEqual(len(result.events), 1)

    def test_wick_just_below_10_ticks_does_not_qualify(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH, half_range=0.5)
        candidate = wick_bar(T, low=90.0, range_val=3.0, upper_wick=2.49, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [make_event()], tick_size=TICK)
        self.assertEqual(result.events, [])

    def test_tick_size_parameter_changes_the_absolute_tick_threshold(self):
        # Same 2.5-unit wick: qualifies at tick_size=0.25 (10 ticks) but
        # not at tick_size=0.5 (would need 5.0 to reach 10 ticks).
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH, half_range=0.5)
        candidate = wick_bar(T, low=90.0, range_val=3.0, upper_wick=2.5, lower_wick=0.0)
        result_default = detect_data_wicks(candles(base + [candidate]), [make_event()], tick_size=0.25)
        result_wider = detect_data_wicks(candles(base + [candidate]), [make_event()], tick_size=0.5)
        self.assertEqual(len(result_default.events), 1)
        self.assertEqual(result_wider.events, [])

    def test_non_positive_tick_size_rejected(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = bar(T, 100.0, 100.5, 99.5, 100.0)
        with self.assertRaises(ValueError):
            detect_data_wicks(candles(base + [candidate]), [make_event()], tick_size=0.0)


class CandidateWindowBoundaryTests(unittest.TestCase):
    def _big_wick_at(self, ts: datetime) -> list[Bar1m]:
        base = flat_baseline(baseline_start(ts), BASELINE_LENGTH)
        candidate = wick_bar(ts, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        return base + [candidate]

    def test_t_minus_1_is_excluded(self):
        ts = T - timedelta(minutes=1)
        result = detect_data_wicks(candles(self._big_wick_at(ts)), [make_event()])
        self.assertEqual(result.events, [])

    def test_t_is_included(self):
        result = detect_data_wicks(candles(self._big_wick_at(T)), [make_event()])
        self.assertEqual(len(result.events), 1)

    def test_t_plus_5_is_included(self):
        ts = T + timedelta(minutes=5)
        result = detect_data_wicks(candles(self._big_wick_at(ts)), [make_event()])
        self.assertEqual(len(result.events), 1)

    def test_t_plus_6_is_excluded(self):
        ts = T + timedelta(minutes=6)
        result = detect_data_wicks(candles(self._big_wick_at(ts)), [make_event()])
        self.assertEqual(result.events, [])


class BaselineNoLookaheadTests(unittest.TestCase):
    def test_baseline_excludes_the_candidate_candle_itself(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)  # median=1.0
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        # if the candidate's own huge range had leaked into its baseline,
        # the median would be far higher than 1.0.
        self.assertEqual(result.events[0].baseline_median, 1.0)

    def test_baseline_is_unaffected_by_any_later_candle(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        # Placed well outside [T, T+5] so it is not itself a candidate --
        # this isolates whether it can corrupt the *earlier* candidate's
        # already-computed baseline, which it must not be able to.
        future_giant = bar(T + timedelta(minutes=10), 100.0, 500.0, 1.0, 100.0)
        without_future = detect_data_wicks(candles(base + [candidate]), [make_event()])
        with_future = detect_data_wicks(candles(base + [candidate, future_giant]), [make_event()])
        self.assertEqual(len(with_future.events), 1)
        self.assertEqual(without_future.events[0].baseline_median, with_future.events[0].baseline_median)


class MissingOrGappedBaselineTests(unittest.TestCase):
    def test_fewer_than_60_preceding_candles_skips_the_candidate(self):
        # only 59 candles, run still ends exactly 1 minute before T
        base = flat_baseline(T - timedelta(minutes=BASELINE_LENGTH - 1), BASELINE_LENGTH - 1)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(result.events, [])

    def test_gap_inside_the_60_preceding_candles_skips_the_candidate(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        # Push the first 30 candles one minute earlier each (their mutual
        # spacing is preserved), leaving the last 30 exactly where they
        # were -- this opens a 2-minute gap right at the 29/30 boundary
        # without disturbing the run's end (still exactly 1 minute before
        # the candidate).
        gapped = list(base)
        for i in range(30):
            c = gapped[i]
            gapped[i] = bar(c.timestamp - timedelta(minutes=1), c.open, c.high, c.low, c.close)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        result = detect_data_wicks(candles(gapped + [candidate]), [make_event()])
        self.assertEqual(result.events, [])

    def test_gap_between_last_baseline_candle_and_candidate_skips_it(self):
        base = flat_baseline(T - timedelta(minutes=BASELINE_LENGTH + 1), BASELINE_LENGTH)  # ends 2 min before T
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(result.events, [])

    def test_complete_contiguous_baseline_does_qualify(self):
        # sanity companion to the two gap tests above.
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        self.assertEqual(len(result.events), 1)


class OverlappingNewsTests(unittest.TestCase):
    def test_overlapping_windows_produce_one_event_with_both_event_ids(self):
        event_a = make_event(event_id="CPI-A", scheduled_at=T, event_type="CPI")
        event_b = make_event(event_id="CORE_CPI-B", scheduled_at=T + timedelta(minutes=3), event_type="CORE_CPI")
        shared_minute = T + timedelta(minutes=3)  # inside both [T, T+5] and [T+3, T+8]
        base = flat_baseline(baseline_start(shared_minute), BASELINE_LENGTH)
        candidate = wick_bar(shared_minute, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [event_a, event_b])
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].event_ids, ("CPI-A", "CORE_CPI-B"))

    def test_non_overlapping_minute_only_reports_the_matching_event(self):
        event_a = make_event(event_id="CPI-A", scheduled_at=T, event_type="CPI")
        event_b = make_event(event_id="CORE_CPI-B", scheduled_at=T + timedelta(minutes=20), event_type="CORE_CPI")
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [event_a, event_b])
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].event_ids, ("CPI-A",))


class TimestampKnownAtTests(unittest.TestCase):
    def test_timestamp_is_candle_open_and_known_at_is_candle_close(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        result = detect_data_wicks(candles(base + [candidate]), [make_event()])
        event = result.events[0]
        self.assertEqual(event.timestamp, T)
        self.assertEqual(event.known_at, T + timedelta(minutes=1))
        self.assertNotEqual(event.timestamp, event.known_at)


class AdapterReuseTests(unittest.TestCase):
    def test_detects_via_bar1m_and_from_bar1m_end_to_end(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = wick_bar(T, low=99.0, range_val=6.0, upper_wick=4.0, lower_wick=0.0)
        bar1m_series = base + [candidate]
        self.assertTrue(all(isinstance(b, Bar1m) for b in bar1m_series))
        converted = [from_bar1m(b) for b in bar1m_series]
        result = detect_data_wicks(converted, [make_event()])
        self.assertEqual(len(result.events), 1)

    def test_duplicate_event_id_in_input_is_rejected_via_validate_news_events(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = bar(T, 100.0, 100.5, 99.5, 100.0)
        dup = make_event(event_id="same-id")
        also_dup = make_event(event_id="same-id", scheduled_at=T + timedelta(minutes=30))
        with self.assertRaises(ValueError):
            detect_data_wicks(candles(base + [candidate]), [dup, also_dup])

    def test_out_of_order_candles_are_rejected_via_shared_chronology_check(self):
        base = flat_baseline(baseline_start(T), BASELINE_LENGTH)
        candidate = bar(T, 100.0, 100.5, 99.5, 100.0)
        out_of_order = bar(T - timedelta(minutes=200), 100.0, 100.5, 99.5, 100.0)
        with self.assertRaises(ValueError):
            detect_data_wicks(candles(base + [candidate, out_of_order]), [make_event()])


if __name__ == "__main__":
    unittest.main()
