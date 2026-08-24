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
from heapq import nsmallest
from typing import Mapping

from tradelab.liquidity import HIGH, LOW, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus
from tradelab.liquidity_store import _NULL_TIME, _to_micros


@dataclass(frozen=True)
class _IndexedLevel:
    price: float
    level: LiquidityLevel
    effective_from: datetime
    effective_until: datetime | None


@dataclass(frozen=True)
class LiquidityTargetIndex:
    _high_prices: tuple[float, ...]
    _high_levels: tuple[_IndexedLevel, ...]
    _low_prices: tuple[float, ...]
    _low_levels: tuple[_IndexedLevel, ...]

    @classmethod
    def build(
        cls,
        swing_equal_results: Mapping[str, LiquidityDetectionResult],
        session_results: Mapping[str, LiquidityDetectionResult],
    ) -> "LiquidityTargetIndex":
        by_side: dict[str, list[_IndexedLevel]] = {HIGH: [], LOW: []}
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

                revisions_by_id: dict[str, list[tuple[datetime, LiquidityLevel]]] = {}
                for revision in result.level_revisions:
                    revisions_by_id.setdefault(revision.level.level_id, []).append((revision.effective_at, revision.level))
                for level in result.levels:
                    revisions = revisions_by_id.get(level.level_id, [(level.formed_at, level)])
                    revisions.sort(key=lambda item: item[0])
                    for index, (effective_from, version) in enumerate(revisions):
                        price = version.upper if version.side == HIGH else version.lower
                        if price <= 0:
                            raise ValueError(f"LiquidityLevel {version.level_id!r} has non-positive target price {price!r}")
                        effective_until = revisions[index + 1][0] if index + 1 < len(revisions) else None
                        by_side[version.side].append(_IndexedLevel(price, version, effective_from, effective_until))
        high = sorted(by_side[HIGH], key=lambda item: (item.price, item.effective_from, item.level.level_id))
        low = sorted(by_side[LOW], key=lambda item: (item.price, item.effective_from, item.level.level_id))
        return cls(tuple(item.price for item in high), tuple(high), tuple(item.price for item in low), tuple(low))

    @staticmethod
    def _eligible(indexed: _IndexedLevel, known_at: datetime) -> bool:
        level = indexed.level
        return (
            indexed.effective_from <= known_at
            and (indexed.effective_until is None or known_at < indexed.effective_until)
            and level.status != LiquidityStatus.ABSORBED_IN_EQUAL_CLUSTER
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
                group_start, group_stop = start, index
            else:
                while index >= 0 and prices[index] == price:
                    index -= 1
                group_start, group_stop = index + 1, start + 1
            # A single price tick can contain thousands of historical levels,
            # while a caller can observe at most ``count`` (three) of them.
            # Both the eligible-level list AND the equal-price group itself
            # are therefore streamed: ``nsmallest`` over a generator that
            # indexes ``levels`` directly costs O(group * log count) time and
            # O(count) memory, whereas slicing ``levels[group_start:
            # group_stop]`` first would allocate a fresh tuple of every
            # pointer in the group -- measured at up to ~640 KB per call on
            # real MNQ data, repeated for millions of entries (see
            # docs/dense-class-memory-diagnostic.md).
            remaining = count - len(found)
            found.extend(
                nsmallest(
                    remaining,
                    (
                        levels[position].level
                        for position in range(group_start, group_stop)
                        if self._eligible(levels[position], entry_known_at)
                    ),
                    key=lambda level: (level.formed_at, level.level_id),
                )
            )
            if len(found) >= count:
                break  # all later groups have strictly larger distance
        return tuple(sorted(found, key=lambda level: (abs((level.upper if side == HIGH else level.lower) - entry_price), level.formed_at, level.level_id))[:count])


class MappedLiquidityTargetIndex:
    """``LiquidityTargetIndex`` over a memory-mapped store instead of objects.

    Same public surface (``nearest``), same rules, same tie-breaks. The
    difference is where the data lives: this holds no ``LiquidityLevel`` and no
    ``_IndexedLevel``, only the reader and two mapped arrays it does not own.
    On the real artifact that is 785,102 level versions the worker never
    materialises.

    **Eligibility and ranking run on primitives.** ``_eligible`` in the object
    index reads only ``effective_from``, ``effective_until``, ``status``,
    ``formed_at`` and ``taken_at``; all five are numeric columns here, so a
    candidate is accepted or rejected without building anything. Only the
    survivors -- at most ``count`` (three) per call -- become real levels.

    **The tie-break is preserved, not approximated.** The object index orders
    equal candidates by ``level_id``, a string. This orders them by the store's
    numeric level ordinal, which is assigned in sorted ``level_id`` order, so
    the two comparisons induce the same order. See
    ``tradelab/liquidity_store.py``.
    """

    __slots__ = ("_reader", "_absorbed_status", "_taken_status")

    def __init__(self, reader) -> None:
        self._reader = reader
        self._absorbed_status = reader.absorbed_status_id
        self._taken_status = reader.taken_status_id

    def _eligible(self, version: int, known_at_us: int) -> bool:
        formed_at, taken_at, effective_from, effective_until = self._reader.times(version)
        if effective_from > known_at_us:
            return False
        if effective_until != _NULL_TIME and known_at_us >= effective_until:
            return False
        status = self._reader.status_id(version)
        if status == self._absorbed_status:
            return False
        if formed_at > known_at_us:
            return False
        if status == self._taken_status and taken_at <= known_at_us:
            return False
        return True

    def nearest(self, *, side: str, entry_price: float, entry_known_at, count: int = 3):
        if side not in {HIGH, LOW}:
            raise ValueError(f"Unsupported liquidity side {side!r}")
        if count <= 0:
            return ()
        reader = self._reader
        if side == HIGH:
            prices, order = reader.high_prices, reader.high_order
        else:
            prices, order = reader.low_prices, reader.low_order
        total = len(prices)
        if total == 0:
            return ()
        known_at_us = _to_micros(entry_known_at)
        index = bisect_right(prices, entry_price) if side == HIGH else bisect_left(prices, entry_price) - 1

        found: list[int] = []
        while 0 <= index < total:
            price = prices[index]
            start = index
            if side == HIGH:
                while index < total and prices[index] == price:
                    index += 1
                group_start, group_stop = start, index
            else:
                while index >= 0 and prices[index] == price:
                    index -= 1
                group_start, group_stop = index + 1, start + 1
            # Streamed exactly as the object index streams it: at most ``count``
            # of a group that can hold thousands of historical levels is ever
            # observable, so neither the group nor the eligible subset is
            # materialised.
            remaining = count - len(found)
            found.extend(
                nsmallest(
                    remaining,
                    (
                        order[position]
                        for position in range(group_start, group_stop)
                        if self._eligible(order[position], known_at_us)
                    ),
                    key=lambda version: (reader.times(version)[0], reader.level_ord(version)),
                )
            )
            if len(found) >= count:
                break  # all later groups have strictly larger distance
            if group_start == group_stop:
                break  # defensive: an empty group would not advance the cursor

        def final_key(version: int):
            lower, upper = reader.prices(version)
            price_at = upper if side == HIGH else lower
            return (abs(price_at - entry_price), reader.times(version)[0], reader.level_ord(version))

        return tuple(reader.level_at(version) for version in sorted(found, key=final_key)[:count])
