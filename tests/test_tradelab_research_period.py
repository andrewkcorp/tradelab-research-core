from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.bar import Bar1m
from tradelab.research_period import load_research_period_policy, require_discovery_only, require_mature_holdout

UTC = timezone.utc


def bar(ts):
    return Bar1m(timestamp=ts, open=1, high=1, low=1, close=1, volume=1, contract="MNQ")


class ResearchPeriodTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_research_period_policy()

    def test_current_three_year_sample_is_discovery_not_holdout(self):
        require_discovery_only([bar(datetime(2023, 8, 18, tzinfo=UTC)), bar(datetime(2026, 8, 17, tzinfo=UTC))], self.policy)

    def test_future_bar_is_rejected_from_discovery(self):
        with self.assertRaisesRegex(ValueError, "holdout begins"):
            require_discovery_only([bar(self.policy.holdout_start_utc)], self.policy)

    def test_holdout_requires_unlock_and_maturity(self):
        future = [bar(self.policy.holdout_start_utc), bar(self.policy.holdout_start_utc + timedelta(days=366))]
        with self.assertRaisesRegex(ValueError, "explicitly_unlocked"):
            require_mature_holdout(future, self.policy, explicitly_unlocked=False)
        require_mature_holdout(future, self.policy, explicitly_unlocked=True)


if __name__ == "__main__":
    unittest.main()
