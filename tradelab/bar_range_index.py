"""Exact, no-look-ahead execution-acceleration primitives (task 045).

``simulate_market_execution`` (``tradelab.backtest_v2``) previously
walked the full 1m bar sequence one bar at a time for nearly every
strategy variant -- including long stretches where no position was even
open, and, once a position *was* open, one candle at a time checking
whether that candle's own high/low crossed the position's fixed TP/SL
price. A real preflight measured 1,091,880 strategy-variant simulations
at 0.0009323/sec against the real ~1,057,182-bar dataset -- implying
roughly 37 years on one process.

This module is the fix: a reusable, immutable range-query index built
**once** over a fixed 1m bar sequence's own high/low/timestamp arrays
(sparse tables for O(1) range-max/range-min queries), answering:

- "the first bar at index >= ``lo`` (and < ``hi``) whose ``high`` is
  >= a threshold" -- ``first_high_at_least`` -- in O(log n).
- "the first bar at index >= ``lo`` (and < ``hi``) whose ``low`` is
  <= a threshold" -- ``first_low_at_most`` -- in O(log n).
- "the first bar at or after a given timestamp" -- ``find_bar_at_or_
  after`` -- in O(log n), via the same sorted timestamps array every
  entry-fill lookup already needs (previously rebuilt, redundantly, on
  every single ``simulate_market_execution`` call).

**This is execution acceleration, not look-ahead.** Every answer comes
from data the reference (unaccelerated) per-bar loop already reads --
``context.bars_1m``'s own already-known high/low/timestamp values --
reorganized once for fast range queries instead of a linear scan.
Nothing here is computed from, or reveals, any bar the reference
implementation would not eventually have read too; a TP/SL exit found
at index ``k`` here is provably the exact same bar the reference
implementation's own bar-by-bar ``_check_exit`` loop would have stopped
at first (see ``tests/test_tradelab_backtest_v2.py``'s randomized
slow-reference-vs-accelerated equivalence test).

**Built once, reused many times.** The O(n log n) sparse-table build
cost is paid exactly once per bar sequence (``BarRangeIndex.build``),
then amortized across however many (equivalence class, exit variant)
simulations reuse it for the lifetime of one run -- callers should
build one ``BarRangeIndex`` per ``PreparedBacktestContext.bars_1m`` and
thread it through every simulation, never rebuild it per call.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence


def _sparse_table_cell_count(n: int) -> int:
    """The exact total number of cells a sparse table over ``n`` values
    will hold, summed across every level -- level 0 has ``n`` cells,
    level ``j`` (``j >= 1``) has ``n - 2**j + 1`` -- computed in O(log n)
    (simulating the level sizes, never materializing them) so real
    "cells built / cells total" progress (task 046) can be reported
    honestly, with a real, known denominator, before construction even
    starts."""
    if n == 0:
        return 0
    total = n
    j = 1
    while (1 << j) <= n:
        total += n - (1 << j) + 1
        j += 1
    return total


def _build_sparse_table(
    values: Sequence[float], combine: Callable[[float, float], float], *, on_progress: Callable[[int, int], None] | None = None
) -> tuple[tuple[float, ...], ...]:
    """Level 0 is ``values`` itself; level ``j`` (for ``j >= 1``) holds,
    at index ``i``, ``combine`` over the window ``[i, i + 2**j)`` --
    computed from two overlapping level-``(j-1)`` windows, exactly the
    standard sparse-table construction. Valid because ``max``/``min``
    are idempotent (overlap never double-counts, unlike e.g. sum).

    ``on_progress`` (task 046), if given, is called with ``(level,
    cells_built_so_far)`` once per level as it completes -- real,
    already-happened progress (every level really did just finish being
    built), never a fabricated estimate. Each level costs close to
    O(n), so this gives roughly evenly-spaced progress across the whole
    O(n log n) build, not just a single callback at the start or end."""
    n = len(values)
    levels: list[tuple[float, ...]] = [tuple(values)]
    cells_built = n
    if on_progress is not None:
        on_progress(0, cells_built)
    if n == 0:
        return tuple(levels)
    j = 1
    while (1 << j) <= n:
        prev = levels[j - 1]
        half = 1 << (j - 1)
        length = n - (1 << j) + 1
        levels.append(tuple(combine(prev[i], prev[i + half]) for i in range(length)))
        cells_built += length
        if on_progress is not None:
            on_progress(j, cells_built)
        j += 1
    return tuple(levels)


def _query_sparse_table(levels: tuple[tuple[float, ...], ...], lo: int, hi: int, combine: Callable[[float, float], float]) -> float:
    """``combine`` (``max`` or ``min``) over ``[lo, hi)`` in O(1): the two
    (possibly overlapping) level-``j`` windows ``[lo, lo+2**j)`` and
    ``[hi-2**j, hi)`` together exactly cover ``[lo, hi)`` for ``j =
    floor(log2(hi - lo))``."""
    length = hi - lo
    j = length.bit_length() - 1
    row = levels[j]
    return combine(row[lo], row[hi - (1 << j)])


@dataclass(frozen=True)
class BarRangeIndex:
    """Immutable range-query index over one bar sequence's own high/low/
    timestamp arrays -- see module docstring. Never mutated after
    ``build()``; every query method is a pure function of ``self`` and
    its arguments."""

    bar_count: int
    timestamps: tuple[datetime, ...]
    _highs_max: tuple[tuple[float, ...], ...]
    _lows_min: tuple[tuple[float, ...], ...]

    @staticmethod
    def build(bars: Sequence, *, on_progress: Callable[[str, int, int, int], None] | None = None) -> "BarRangeIndex":
        """``bars`` is any sequence of objects with ``.timestamp``/
        ``.high``/``.low`` (i.e. ``Bar1m``) -- typically ``context.
        bars_1m``. O(n log n) time and O(n log n) space; call this
        exactly once per bar sequence and reuse the result.

        ``on_progress`` (task 046), if given, is called with ``(stage,
        level, cells_built, cells_total)`` as construction proceeds --
        ``stage`` is ``"highs"`` while the range-max table is being
        built, then ``"lows"`` for the range-min table; ``level`` is the
        current sparse-table level just completed; ``cells_built``/
        ``cells_total`` are real, exact counts (summed across *both*
        tables, so ``cells_built == cells_total`` only on the very last
        call) -- never an estimate. This exists precisely because
        building the index is itself real, sometimes multi-second work
        with no other progress signal (task 046's own root problem: a
        dashboard polling every second would otherwise see nothing
        change here at all)."""
        timestamps = tuple(b.timestamp for b in bars)
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        per_table_cells = _sparse_table_cell_count(len(bars))
        cells_total = per_table_cells * 2

        highs_progress = None
        lows_progress = None
        if on_progress is not None:
            highs_progress = lambda level, cells_built: on_progress("highs", level, cells_built, cells_total)  # noqa: E731
            lows_progress = lambda level, cells_built: on_progress("lows", level, per_table_cells + cells_built, cells_total)  # noqa: E731

        highs_max = _build_sparse_table(highs, max, on_progress=highs_progress)
        lows_min = _build_sparse_table(lows, min, on_progress=lows_progress)
        return BarRangeIndex(bar_count=len(bars), timestamps=timestamps, _highs_max=highs_max, _lows_min=lows_min)

    def find_bar_at_or_after(self, known_at: datetime, start_from: int = 0) -> int | None:
        """Index of the first bar at or after ``known_at``, searching
        only from ``start_from`` onward -- O(log n), via ``bisect`` over
        the already-built ``timestamps`` array (never rebuilt per call,
        unlike the reference implementation's own local list)."""
        idx = bisect.bisect_left(self.timestamps, known_at, lo=start_from)
        return idx if idx < self.bar_count else None

    def range_max_high(self, lo: int, hi: int) -> float:
        return _query_sparse_table(self._highs_max, lo, hi, max)

    def range_min_low(self, lo: int, hi: int) -> float:
        return _query_sparse_table(self._lows_min, lo, hi, min)

    def first_high_at_least(self, lo: int, hi: int, threshold: float) -> int | None:
        """The smallest index ``k`` in ``[lo, hi)`` with ``high[k] >=
        threshold``, or ``None`` if no such index exists -- O(log n).
        Binary descent using O(1) range-max queries: at each step,
        exactly one half of the current window is known to contain the
        answer (since the whole window was already confirmed to
        qualify), so this never revisits a bar the equivalent reference
        per-bar scan would not eventually have reached, and never misses
        the *first* one."""
        if lo >= hi:
            return None
        if self.range_max_high(lo, hi) < threshold:
            return None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.range_max_high(lo, mid) >= threshold:
                hi = mid
            else:
                lo = mid
        return lo

    def first_low_at_most(self, lo: int, hi: int, threshold: float) -> int | None:
        """The smallest index ``k`` in ``[lo, hi)`` with ``low[k] <=
        threshold``, or ``None`` -- the mirror of ``first_high_at_least``
        for the opposite comparison direction."""
        if lo >= hi:
            return None
        if self.range_min_low(lo, hi) > threshold:
            return None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.range_min_low(lo, mid) <= threshold:
                hi = mid
            else:
                lo = mid
        return lo
