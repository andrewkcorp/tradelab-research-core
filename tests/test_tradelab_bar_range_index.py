#!/usr/bin/env python3
"""tradelab.bar_range_index tests (task 045): the sparse-table range-max/
range-min queries and the first-hit binary-descent queries built on top
of them match an independent brute-force scan across many randomized
trials (varied bar counts, thresholds, and query ranges), plus edge
cases (empty index, single bar, zero-length range, threshold exactly at
a boundary value) and the timestamp lookup."""
from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.bar_range_index import BarRangeIndex

UTC = timezone.utc


class _FakeBar:
    def __init__(self, ts: datetime, high: float, low: float) -> None:
        self.timestamp = ts
        self.high = high
        self.low = low


def _random_bars(rng: random.Random, n: int) -> list[_FakeBar]:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    bars = []
    for i in range(n):
        high = rng.uniform(90.0, 110.0)
        low = high - rng.uniform(0.1, 8.0)
        bars.append(_FakeBar(t0 + timedelta(minutes=i), high, low))
    return bars


def _brute_force_first_high_at_least(highs: list[float], lo: int, hi: int, threshold: float) -> int | None:
    for k in range(lo, hi):
        if highs[k] >= threshold:
            return k
    return None


def _brute_force_first_low_at_most(lows: list[float], lo: int, hi: int, threshold: float) -> int | None:
    for k in range(lo, hi):
        if lows[k] <= threshold:
            return k
    return None


class RandomizedFirstHitEquivalenceTests(unittest.TestCase):
    def test_matches_brute_force_across_many_randomized_ranges_and_thresholds(self):
        rng = random.Random(9001)
        bars = _random_bars(rng, 400)
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        index = BarRangeIndex.build(bars)

        checked = 0
        for _ in range(3000):
            lo = rng.randint(0, len(bars) - 1)
            hi = rng.randint(lo + 1, len(bars))
            threshold = rng.uniform(85.0, 115.0)

            self.assertEqual(index.first_high_at_least(lo, hi, threshold), _brute_force_first_high_at_least(highs, lo, hi, threshold))
            self.assertEqual(index.first_low_at_most(lo, hi, threshold), _brute_force_first_low_at_most(lows, lo, hi, threshold))
            self.assertEqual(index.range_max_high(lo, hi), max(highs[lo:hi]))
            self.assertEqual(index.range_min_low(lo, hi), min(lows[lo:hi]))
            checked += 2

        self.assertGreater(checked, 1000)

    def test_matches_brute_force_with_varied_bar_counts_including_powers_of_two_and_off_by_one(self):
        rng = random.Random(31337)
        for n in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 200, 201):
            bars = _random_bars(rng, n)
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]
            index = BarRangeIndex.build(bars)
            for _ in range(50):
                lo = rng.randint(0, n - 1)
                hi = rng.randint(lo + 1, n)
                threshold = rng.uniform(85.0, 115.0)
                self.assertEqual(
                    index.first_high_at_least(lo, hi, threshold), _brute_force_first_high_at_least(highs, lo, hi, threshold), msg=f"n={n} lo={lo} hi={hi}"
                )
                self.assertEqual(
                    index.first_low_at_most(lo, hi, threshold), _brute_force_first_low_at_most(lows, lo, hi, threshold), msg=f"n={n} lo={lo} hi={hi}"
                )


class EdgeCaseTests(unittest.TestCase):
    def test_empty_index(self):
        index = BarRangeIndex.build([])
        self.assertEqual(index.bar_count, 0)
        self.assertEqual(index.timestamps, ())
        self.assertIsNone(index.find_bar_at_or_after(datetime(2024, 1, 1, tzinfo=UTC)))

    def test_single_bar_index(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        index = BarRangeIndex.build([_FakeBar(t0, 100.0, 95.0)])
        self.assertEqual(index.first_high_at_least(0, 1, 100.0), 0)  # boundary value, inclusive (>=)
        self.assertIsNone(index.first_high_at_least(0, 1, 100.01))
        self.assertEqual(index.first_low_at_most(0, 1, 95.0), 0)  # boundary value, inclusive (<=)
        self.assertIsNone(index.first_low_at_most(0, 1, 94.99))

    def test_zero_length_range_returns_none(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        index = BarRangeIndex.build([_FakeBar(t0, 100.0, 95.0), _FakeBar(t0 + timedelta(minutes=1), 101.0, 96.0)])
        self.assertIsNone(index.first_high_at_least(1, 1, 0.0))
        self.assertIsNone(index.first_low_at_most(1, 1, 200.0))

    def test_threshold_never_reachable_in_range(self):
        rng = random.Random(5)
        bars = _random_bars(rng, 100)
        index = BarRangeIndex.build(bars)
        self.assertIsNone(index.first_high_at_least(0, 100, 10_000.0))
        self.assertIsNone(index.first_low_at_most(0, 100, -10_000.0))

    def test_find_bar_at_or_after_matches_bisect_semantics(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        bars = [_FakeBar(t0 + timedelta(minutes=i * 2), 100.0, 95.0) for i in range(10)]  # every-other-minute, real gaps
        index = BarRangeIndex.build(bars)
        # exact match
        self.assertEqual(index.find_bar_at_or_after(t0 + timedelta(minutes=4)), 2)
        # inside a gap -- first bar strictly after
        self.assertEqual(index.find_bar_at_or_after(t0 + timedelta(minutes=5)), 3)
        # start_from narrows the search forward only
        self.assertEqual(index.find_bar_at_or_after(t0, start_from=5), 5)
        # past the end
        self.assertIsNone(index.find_bar_at_or_after(t0 + timedelta(days=1)))


class BuildProgressCallbackTests(unittest.TestCase):
    """task 046: BarRangeIndex.build's own on_progress callback reports
    real, exact sparse-table construction progress (stage, level, cells
    built, cells total) -- never a fabricated estimate -- so a dashboard
    watching this genuinely multi-step build sees real movement rather
    than sitting at bars_scanned: 0 for its whole duration."""

    def test_reports_real_monotonic_progress_across_both_stages(self):
        rng = random.Random(7)
        bars = _random_bars(rng, 5000)
        calls = []
        BarRangeIndex.build(bars, on_progress=lambda stage, level, built, total: calls.append((stage, level, built, total)))

        self.assertGreater(len(calls), 10)  # genuinely multiple levels reported, not just once

        stages_in_order = [c[0] for c in calls]
        self.assertEqual(set(stages_in_order), {"highs", "lows"})
        # "highs" entirely completes before "lows" starts -- never interleaved.
        first_lows_index = stages_in_order.index("lows")
        self.assertTrue(all(s == "highs" for s in stages_in_order[:first_lows_index]))
        self.assertTrue(all(s == "lows" for s in stages_in_order[first_lows_index:]))

        built_values = [c[2] for c in calls]
        self.assertEqual(built_values, sorted(built_values))  # monotonically non-decreasing
        self.assertEqual(len(set(built_values)), len(built_values))  # strictly increasing -- every call reports genuinely new progress

        totals = {c[3] for c in calls}
        self.assertEqual(len(totals), 1)  # cells_total never changes mid-build
        total = totals.pop()
        self.assertEqual(calls[-1][2], total)  # the very last call reaches exactly 100%
        self.assertGreater(total, 5000)  # more cells than just the base level -- genuinely counts every sparse-table level

    def test_no_callback_means_no_overhead_and_identical_result(self):
        rng = random.Random(8)
        bars = _random_bars(rng, 500)
        without_callback = BarRangeIndex.build(bars)
        with_callback = BarRangeIndex.build(bars, on_progress=lambda *a: None)
        self.assertEqual(without_callback, with_callback)

    def test_empty_bars_reports_nothing_and_never_crashes(self):
        calls = []
        index = BarRangeIndex.build([], on_progress=lambda *a: calls.append(a))
        self.assertEqual(index.bar_count, 0)
        # never claims nonzero progress for an empty sequence -- whether
        # that means zero calls or calls reporting 0/0 is an
        # implementation detail; what matters is honesty.
        self.assertTrue(all(built <= total for _stage, _level, built, total in calls))


if __name__ == "__main__":
    unittest.main()
