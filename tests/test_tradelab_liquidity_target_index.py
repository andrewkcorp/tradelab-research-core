from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.liquidity import HIGH, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType
from tradelab.liquidity_target_index import LiquidityTargetIndex


UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _level(level_id: str, formed_at: datetime) -> LiquidityLevel:
    return LiquidityLevel(
        level_id=level_id, timeframe="1m", liquidity_type=LiquidityType.SWING_HIGH, side=HIGH,
        lower=101.0, upper=101.0, member_count=1, member_timestamps=(formed_at,), formed_at=formed_at,
        status=LiquidityStatus.ACTIVE, taken_at=None, absorbed_by_level_id=None,
    )


class LiquidityTargetIndexTests(unittest.TestCase):
    def test_nearest_keeps_only_the_exact_top_three_from_a_large_equal_price_group(self):
        levels = [_level(f"late-{index:05d}", T0 + timedelta(minutes=index + 10)) for index in range(100)]
        levels += [_level(f"early-{index:05d}", T0 + timedelta(minutes=index)) for index in range(5)]
        index = LiquidityTargetIndex.build({"1m": LiquidityDetectionResult(levels=levels, takes=[], reclaims=[])}, {})
        selected = index.nearest(side=HIGH, entry_price=100.0, entry_known_at=T0 + timedelta(days=1))
        self.assertEqual([level.level_id for level in selected], ["early-00000", "early-00001", "early-00002"])

