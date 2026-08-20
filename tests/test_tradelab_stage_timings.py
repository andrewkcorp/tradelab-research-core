#!/usr/bin/env python3
"""tradelab.stage_timings tests (task 047): totals/counts accumulate
correctly across repeated record() calls for the same and different
stages, and to_dict()'s share values are a normalized (sum-to-1)
breakdown of the totals -- or all-zero, never a division error, when
nothing has been recorded yet."""
from __future__ import annotations

import unittest

from tradelab.stage_timings import StageTimings


class StageTimingsTests(unittest.TestCase):
    def test_record_accumulates_totals_and_counts_per_stage(self):
        timings = StageTimings()
        timings.record("exit_resolution", 0.5)
        timings.record("exit_resolution", 0.25)
        timings.record("persistence", 0.1)

        self.assertAlmostEqual(timings.totals_seconds["exit_resolution"], 0.75)
        self.assertEqual(timings.counts["exit_resolution"], 2)
        self.assertAlmostEqual(timings.totals_seconds["persistence"], 0.1)
        self.assertEqual(timings.counts["persistence"], 1)

    def test_to_dict_share_sums_to_one(self):
        timings = StageTimings()
        timings.record("a", 3.0)
        timings.record("b", 1.0)

        d = timings.to_dict()
        self.assertAlmostEqual(sum(d["share"].values()), 1.0)
        self.assertAlmostEqual(d["share"]["a"], 0.75)
        self.assertAlmostEqual(d["share"]["b"], 0.25)

    def test_to_dict_empty_never_divides_by_zero(self):
        timings = StageTimings()
        d = timings.to_dict()
        self.assertEqual(d["totals_seconds"], {})
        self.assertEqual(d["share"], {})


if __name__ == "__main__":
    unittest.main()
