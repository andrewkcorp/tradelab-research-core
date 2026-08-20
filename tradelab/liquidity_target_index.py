"""Exact price index for TP/SL liquidity target resolution.

The detectors expose a flat audit list of every level.  Re-scanning that
list for each previously unseen entry is O(levels).  This immutable index
orders levels by the price used by a target (upper for highs, lower for
lows), then expands away from an entry price only until the three nearest
eligible levels are known.  Eligibility itself remains the original
no-look-ahead rule: formed no later than entry and not taken on/before it.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from tradelab.liquidity import HIGH, LOW, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus


@dataclass(frozen=True)
class LiquidityTargetIndex:
    _high_prices: tuple[float, ...]
    _high_levels: tuple[LiquidityLevel, ...]
    _low_prices: tuple[float, ...]
    _low_levels: tuple[LiquidityLevel, ...]

    @classmethod
    def build(
        cls,
        swing_equal_results: Mapping[str, LiquidityDetectionResult],
        session_results: Mapping[str, LiquidityDetectionResult],
    ) -> "LiquidityTargetIndex":
        by_side: dict[str, list[tuple[float, LiquidityLevel]]] = {HIGH: [], LOW: []}
        seen: set[str] = set()
        for results in (swing_equal_results, session_results):
            for timeframe, result in results.items():
                for level in result.levels:
                    if level.timeframe != timeframe:
                        raise ValueError(f"LiquidityLevel {level.level_id!r} has timeframe {level.timeframe!r}, expected {timeframe!r}")
                    if level.level_id in seen:
                        raise ValueError(f"Duplicate liquidity level_id {level.level_id!r}")
                    seen.add(level.level_id)
                    if level.status == LiquidityStatus.TAKEN and (level.taken_at is None or level.taken_at <= level.formed_at):
                        raise ValueError(f"LiquidityLevel {level.level_id!r} has invalid taken_at")
                    price = level.upper if level.side == HIGH else level.lower
                    if price <= 0:
                        raise ValueError(f"LiquidityLevel {level.level_id!r} has non-positive target price {price!r}")
                    by_side[level.side].append((price, level))
        high = sorted(by_side[HIGH], key=lambda item: (item[0], item[1].formed_at, item[1].level_id))
        low = sorted(by_side[LOW], key=lambda item: (item[0], item[1].formed_at, item[1].level_id))
        return cls(tuple(p for p, _ in high), tuple(level for _, level in high), tuple(p for p, _ in low), tuple(level for _, level in low))

    @staticmethod
    def _eligible(level: LiquidityLevel, known_at: datetime) -> bool:
        return (
            level.status != LiquidityStatus.ABSORBED_IN_EQUAL_CLUSTER
            and level.formed_at <= known_at
            and not (level.status == LiquidityStatus.TAKEN and level.taken_at <= known_at)
        )

    def nearest(self, *, side: str, entry_price: float, entry_known_at: datetime, count: int = 3) -> tuple[LiquidityLevel, ...]:
        if side not in {HIGH, LOW}:
            raise ValueError(f"Unsupported liquidity side {side!r}")
        if count <= 0:
            return ()
        prices, levels = (self._high_prices, self._high_levels) if side == HIGH else (self._low_prices, self._low_levels)
        index = bisect_right(prices, entry_price) if side == HIGH else bisect_left(prices, entry_price) - 1
        found: list[LiquidityLevel] = []
        while 0 <= index < len(prices):
            price = prices[index]
            start = index
            if side == HIGH:
                while index < len(prices) and prices[index] == price:
                    index += 1
                group = levels[start:index]
            else:
                while index >= 0 and prices[index] == price:
                    index -= 1
                group = levels[index + 1:start + 1]
            found.extend(level for level in group if self._eligible(level, entry_known_at))
            if len(found) >= count:
                break  # all later groups have strictly larger distance
        return tuple(sorted(found, key=lambda level: (abs((level.upper if side == HIGH else level.lower) - entry_price), level.formed_at, level.level_id))[:count])
