"""Exact first-hit index over immutable 1-minute bars.

The backtester repeatedly asks for the first bar in ``[lo, hi)`` whose high
or low reaches a fixed TP/SL price. A chronological scan is exact but too
expensive across hundreds of thousands of variants.

This implementation stores two flat segment trees (range maximum of highs
and range minimum of lows). Build cost and memory are O(n); range extrema
and first-hit searches are O(log n). Earlier TradeLab versions used sparse
tables, which consumed O(n log n) memory. On the real ~1.06M-bar MNQ input,
the flat ``array('d')`` trees replace tens of millions of Python objects with
about ``4 * next_power_of_two(n)`` doubles in total.

This is an acceleration index, not look-ahead. It returns the same earliest
bar the reference chronological scan reaches; randomized equivalence tests
compare every public query with independent brute force.
"""
from __future__ import annotations

import bisect
from array import array
from dataclasses import dataclass
from datetime import datetime
from math import inf
from typing import Callable, Sequence


def _tree_size(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _build_tree(
    values: Sequence[float],
    *,
    size: int,
    is_max: bool,
    on_progress: Callable[[int, int], None] | None = None,
) -> array:
    identity = -inf if is_max else inf
    n = len(values)
    tree = array("d", [identity]) * size
    tree.extend(float(value) for value in values)
    if n < size:
        tree.extend(array("d", [identity]) * (size - n))

    # Meaningful cells exclude the deliberately unused element at index 0.
    cells_built = size
    if on_progress is not None:
        on_progress(0, cells_built)
    level = 1
    level_start = size // 2
    level_end = size
    while level_start:
        if is_max:
            for index in range(level_start, level_end):
                left = tree[index * 2]
                right = tree[index * 2 + 1]
                tree[index] = left if left >= right else right
        else:
            for index in range(level_start, level_end):
                left = tree[index * 2]
                right = tree[index * 2 + 1]
                tree[index] = left if left <= right else right
        cells_built += level_end - level_start
        if on_progress is not None:
            on_progress(level, cells_built)
        level += 1
        level_end = level_start
        level_start //= 2
    return tree


def _range_extreme(tree: array, size: int, lo: int, hi: int, *, is_max: bool) -> float:
    if lo >= hi:
        raise ValueError("range must be non-empty")
    left_value = -inf if is_max else inf
    right_value = left_value
    lo += size
    hi += size
    while lo < hi:
        if lo & 1:
            value = tree[lo]
            left_value = max(left_value, value) if is_max else min(left_value, value)
            lo += 1
        if hi & 1:
            hi -= 1
            value = tree[hi]
            right_value = max(value, right_value) if is_max else min(value, right_value)
        lo //= 2
        hi //= 2
    return max(left_value, right_value) if is_max else min(left_value, right_value)


def _first_qualifying(
    tree: array,
    size: int,
    bar_count: int,
    lo: int,
    hi: int,
    threshold: float,
    *,
    high_at_least: bool,
) -> int | None:
    if lo >= hi:
        return None

    def qualifies(value: float) -> bool:
        return value >= threshold if high_at_least else value <= threshold

    # Decompose [lo, hi) into chronological canonical tree nodes. The first
    # qualifying node contains the answer; a left-first descent then reaches
    # its earliest qualifying leaf. This avoids traversing unrelated branches.
    left_nodes: list[int] = []
    right_nodes: list[int] = []
    left = lo + size
    right = hi + size
    while left < right:
        if left & 1:
            left_nodes.append(left)
            left += 1
        if right & 1:
            right -= 1
            right_nodes.append(right)
        left //= 2
        right //= 2
    for node in (*left_nodes, *reversed(right_nodes)):
        if not qualifies(tree[node]):
            continue
        while node < size:
            child = node * 2
            node = child if qualifies(tree[child]) else child + 1
        index = node - size
        return index if index < bar_count else None
    return None


@dataclass(frozen=True)
class BarRangeIndex:
    """Reusable, immutable segment-tree index for one ordered bar series."""

    bar_count: int
    timestamps: tuple[datetime, ...]
    _size: int
    _highs_max: array
    _lows_min: array

    @staticmethod
    def build(
        bars: Sequence,
        *,
        on_progress: Callable[[str, int, int, int], None] | None = None,
    ) -> "BarRangeIndex":
        # A memory-mapped bar series offers its columns directly. Taking them
        # avoids materialising one Bar1m per bar just to read three fields off
        # it, and -- more importantly -- avoids retaining a tuple of a million
        # datetimes for the lifetime of the index. The values are the same
        # numbers either way; only the route to them differs.
        as_columns = getattr(bars, "as_columns", None)
        if as_columns is not None:
            timestamps, highs_source, lows_source = as_columns()
        else:
            timestamps = tuple(bar.timestamp for bar in bars)
            highs_source = [bar.high for bar in bars]
            lows_source = [bar.low for bar in bars]
        n = len(bars)
        size = _tree_size(n)
        if n == 0:
            return BarRangeIndex(0, timestamps, size, array("d", [-inf, -inf]), array("d", [inf, inf]))

        meaningful_cells_per_tree = 2 * size - 1
        cells_total = meaningful_cells_per_tree * 2
        highs_progress = None
        lows_progress = None
        if on_progress is not None:
            highs_progress = lambda level, built: on_progress("highs", level, built, cells_total)  # noqa: E731
            lows_progress = lambda level, built: on_progress(  # noqa: E731
                "lows", level, meaningful_cells_per_tree + built, cells_total
            )

        highs = _build_tree(highs_source, size=size, is_max=True, on_progress=highs_progress)
        lows = _build_tree(lows_source, size=size, is_max=False, on_progress=lows_progress)
        return BarRangeIndex(n, timestamps, size, highs, lows)

    def find_bar_at_or_after(self, known_at: datetime, start_from: int = 0) -> int | None:
        index = bisect.bisect_left(self.timestamps, known_at, lo=start_from)
        return index if index < self.bar_count else None

    def range_max_high(self, lo: int, hi: int) -> float:
        return _range_extreme(self._highs_max, self._size, lo, hi, is_max=True)

    def range_min_low(self, lo: int, hi: int) -> float:
        return _range_extreme(self._lows_min, self._size, lo, hi, is_max=False)

    def first_high_at_least(self, lo: int, hi: int, threshold: float) -> int | None:
        return _first_qualifying(
            self._highs_max, self._size, self.bar_count, lo, hi, threshold, high_at_least=True
        )

    def first_low_at_most(self, lo: int, hi: int, threshold: float) -> int | None:
        return _first_qualifying(
            self._lows_min, self._size, self.bar_count, lo, hi, threshold, high_at_least=False
        )
