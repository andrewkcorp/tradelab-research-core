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


if __name__ == "__main__":
    unittest.main()
