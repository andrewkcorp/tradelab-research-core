"""Exact, no-look-ahead entry-target caching (task 047).

**The real bottleneck task 045's ``BarRangeIndex`` didn't touch.** A
real accelerated preflight sample measured 0.0013789123 variants/sec
against the real ~1,057,182-bar MNQ dataset -- ``BarRangeIndex`` (task
045) improved the prior, unaccelerated result by only ~1.5x, meaning
the dominant cost was elsewhere. Profiling (see ``tradelab.backtest_v2
.run_stage_diagnostic``) confirmed it: ``tradelab.tp_targets.detect_tp_
target_candidates``/``tradelab.sl_targets.detect_sl_target_candidates``
each do a full linear scan (with per-level validation) over every
liquidity level on every native timeframe -- real data has 285,509
liquidity levels -- and this scan is genuinely O(levels), not O(log
levels); ``BarRangeIndex`` only ever accelerated *exit* resolution
(scanning bars for a TP/SL price crossing), never *target* resolution
(deciding what those TP/SL prices even are in the first place).

**The redundancy this module eliminates.** ``tradelab.backtest_v2``
simulates one (equivalence class, exit variant) pair at a time -- 20
exit variants per class -- but ``detect_tp_target_candidates``'s own
result for one observed entry is a function of *only* ``(direction,
entry_price, entry_known_at)`` plus the fixed, per-context liquidity/
data-wick detection results: **never of which exit variant is asking**.
``detect_sl_target_candidates``'s result additionally depends on the
FVG confluence, which itself only ever takes one of three forms per
entry (empty, for the liquidity-rank SL family; or the entry's own
linked FVG, for whichever *one* of the two FVG-boundary timeframes it
actually matches, if any). Task 045 already proved -- and this module
relies on that same invariant -- that ``simulate_market_execution``'s
own fill resolution is bar-for-bar identical for the same observed
entry across every exit variant (``bar_cursor`` is always reset to
exactly that entry's own fill index after processing it, regardless of
what happened to any position, and ``bisect_left`` is insensitive to a
smaller-but-valid starting bound). So the *same* expensive O(levels)
computation was being redundantly re-run once per exit variant -- up to
20x for TP, up to 12x for the liquidity-rank SL family, up to 4x for
each FVG-boundary SL timeframe -- for a result that never changes.

**Correctness, not approximation.** This cache never changes *which*
target a given exit variant resolves to, nor its price, distance, rank,
or source -- it only avoids recomputing an already-known answer. The
cache key is explicit and contains every input that can affect the
result: the *observed entry's own stable id* (``ObservedEntry.entry_id``
-- fully determines ``direction``/``entry_price``/``entry_known_at``
once resolved to its fill bar, which is itself variant-invariant, see
above) for TP, plus a confluence key (``None`` for the empty/liquidity-
rank confluence, or the linked FVG's own timeframe string) for SL.
Nothing here uses any information not already available to, and used
by, the reference (uncached) simulation at the exact same point in its
own execution -- see ``tests/test_tradelab_backtest_v2.py``'s randomized
cached-vs-uncached equivalence test for the proof.

**Scope and memory.** Production builds one bounded cache per equivalence
class, not one cache for the full run. This is intentional: a dense class is
scanned from its first entry for every exit variant, so a run-wide LRU smaller
than that class would evict all useful entries before the next pass revisits
them. The production bound covers every target family an entry can request in
that class; the whole cache is then released before the next class. Direct
callers retain the unbounded default for simple unit use.

Only ever used by ``tradelab.backtest_v2``'s own ``_resolve_tp_
candidate``/``_resolve_sl_candidate`` -- this module has no dependency
on ``tradelab.backtest_v2`` itself, only on ``tradelab.tp_targets``/
``tradelab.sl_targets``, matching this package's own convention of
small, independently testable primitives.
"""
from __future__ import annotations

from collections import OrderedDict

from tradelab.sl_targets import SelectedFVGConfluence, SLTargetCandidateResult, detect_sl_target_candidates
from tradelab.tp_targets import MarketEntryContext, TPTargetCandidateResult, detect_tp_target_candidates
from tradelab.liquidity_target_index import LiquidityTargetIndex


class EntryTargetCache:
    """Memoizes ``detect_tp_target_candidates``/``detect_sl_target_
    candidates`` results, keyed by everything that can affect them --
    see module docstring. Callers already have a cheap-to-construct
    ``MarketEntryContext``/``SelectedFVGConfluence`` in hand by the time
    they call in (constructing those dataclasses is not the expensive
    part -- the O(levels) scan inside ``detect_*`` is), so this takes
    them directly rather than lazily-evaluated factories.

    Not thread-safe (never needs to be: one cache is built and consumed
    synchronously within a single equivalence class's own exit-variant
    loop, on one thread, exactly like ``tradelab.bar_range_index.
    BarRangeIndex`` is built once and reused, but never concurrently)."""

    def __init__(
        self,
        *,
        max_entries: int | None = None,
        include_data_wick_targets: bool = True,
    ) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("EntryTargetCache.max_entries must be positive or None")
        self._max_entries = max_entries
        self._include_data_wick_targets = include_data_wick_targets
        self._tp: dict[str, TPTargetCandidateResult] = {}
        self._sl: dict[tuple[str, str | None, int | None], SLTargetCandidateResult] = {}
        # The linked-FVG stop *scan* (which zone an FVG-boundary SL resolves
        # to) is memoised separately from the detector result it feeds: it
        # depends only on the observed entry and the SL timeframe, never on
        # which exit variant is asking, yet it used to run once per exit
        # variant. On a dense 1m class that is a full pass over up to ~2,000
        # active zones, repeated ~6x per entry for nothing -- see
        # docs/dense-class-memory-diagnostic.md.
        self._fvg_stop: dict[tuple[str, str], object] = {}
        # One LRU order across all namespaces. Eviction is strictly a
        # performance/memory decision: a missing item is recomputed by the
        # same exact detector, never approximated.
        self._recency: OrderedDict[tuple[str, object], None] = OrderedDict()

    _POOLS = {"tp": "_tp", "sl": "_sl", "fvg_stop": "_fvg_stop"}

    def _touch(self, key: tuple[str, object]) -> None:
        self._recency.pop(key, None)
        self._recency[key] = None
        if self._max_entries is not None:
            while len(self._recency) > self._max_entries:
                namespace, evicted_key = self._recency.popitem(last=False)[0]
                getattr(self, self._POOLS[namespace]).pop(evicted_key, None)

    def fvg_stop_candidate(self, entry_id: str, timeframe: str, compute):
        """Memoise one entry's chosen FVG-boundary stop zone per timeframe.

        ``compute`` is a zero-argument callable invoked only on a miss. A
        legitimate ``None`` result (no eligible zone) is cached as such, and
        is deliberately distinguishable from an absent key -- recomputing a
        known-empty scan is exactly the waste this exists to remove.
        """
        key = (entry_id, timeframe)
        if key in self._fvg_stop:
            self._touch(("fvg_stop", key))
            return self._fvg_stop[key]
        value = compute()
        self._fvg_stop[key] = value
        self._touch(("fvg_stop", key))
        return value

    def tp_candidates(
        self,
        market_entry: MarketEntryContext,
        swing_equal_results,
        session_results,
        data_wick_result,
        *,
        liquidity_index: LiquidityTargetIndex | None = None,
    ) -> TPTargetCandidateResult:
        cached = self._tp.get(market_entry.entry_id)
        if cached is not None:
            self._touch(("tp", market_entry.entry_id))
            return cached
        result = detect_tp_target_candidates(
            market_entry, swing_equal_results, session_results, data_wick_result,
            liquidity_index=liquidity_index,
            include_data_wick_targets=self._include_data_wick_targets,
        )
        self._tp[market_entry.entry_id] = result
        self._touch(("tp", market_entry.entry_id))
        return result

    def sl_candidates(
        self,
        market_entry: MarketEntryContext,
        confluence_key: str | None,
        confluence: SelectedFVGConfluence,
        swing_equal_results,
        session_results,
        *,
        liquidity_index: LiquidityTargetIndex | None = None,
        max_fvg_targets: int | None = None,
    ) -> SLTargetCandidateResult:
        key = (market_entry.entry_id, confluence_key, max_fvg_targets)
        cached = self._sl.get(key)
        if cached is not None:
            self._touch(("sl", key))
            return cached
        result = detect_sl_target_candidates(
            market_entry,
            confluence,
            swing_equal_results,
            session_results,
            liquidity_index=liquidity_index,
            max_fvg_targets=max_fvg_targets,
        )
        self._sl[key] = result
        self._touch(("sl", key))
        return result

    def __len__(self) -> int:
        """Total number of memoized entries (TP + SL + FVG stop scans) --
        purely informational, e.g. for a diagnostic's own memory-estimate
        reporting."""
        return len(self._tp) + len(self._sl) + len(self._fvg_stop)
