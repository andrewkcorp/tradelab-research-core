"""Data-driven, exhaustive-on-real-events strategy space generator.

Builds every ``StrategySpec`` (primary entry x required confluence
subset x bias policy x TP choice x SL choice) that could ever actually
fire against a given, already-prepared ``PreparedBacktestContext`` --
never an abstract combinatorial sweep over the full 438-selector
canonical catalog.

**"Every possible combination on real events" is not ``2**438``.** The
canonical selector catalog (``tradelab.selector_catalog.ALL_CANONICAL_
ENTRY_SELECTORS``) names 438 selector *families* this package's
detectors can ever produce -- it says nothing about which of them were
ever simultaneously *active* in a specific dataset. A required-
confluence subset is only ever a real strategy candidate if every one
of its selectors was genuinely active, in the same direction, at some
real primary signal's own ``known_at``. **There is still no artificial
cap on how many required confluences a generated strategy may carry**
(this module never sets ``max_required_confluences`` in the sense
``tradelab.strategy_spec.iter_strategy_specs`` uses that parameter) --
the only bound is what the data itself ever actually produced, gated by
the same minimum-trade-frequency threshold (``min_trades_per_window``,
default 100, in every one of two explicit full-calendar-year windows)
``tradelab.research_runner`` already applies downstream.

**Why frequent-itemset mining, not subset enumeration.** Real market
data does not keep active pools small: signals stay valid for the
*whole trading day* (``tradelab.entry_signal_timeline``'s own
persistence rule), so at any one instant, dozens to over a hundred of
the 438 selectors can be simultaneously active (measured on a real MNQ
slice: average ~58, max 137 concurrently active selectors). Enumerating
"every subset" of a 58-selector pool is ``2**58`` -- computationally
impossible, not merely slow -- so this module never enumerates subsets
directly. Instead, ``discover_observed_universe`` mines required-
confluence subsets as **frequent itemsets**: a subset is only generated
if its support (the count of real primary occurrences whose own active
pool is a superset of it) meets ``min_trades_per_window`` in *every*
explicit window -- exactly the same "raw co-occurrence count is a safe
upper bound on real trade count" prefilter principle ``tradelab.
research_runner`` already uses (a confluence-restricted combo can never
accumulate more real trades than its own raw co-occurrence count, so a
combo that can't even clear the threshold *before* a backtest runs
could never clear it after one either). Mining uses Apriori-style
level-wise search with anti-monotonic pruning (support can only shrink,
never grow, as an itemset gains members, so once a subset fails the
threshold no superset of it is ever even considered) -- proven complete
against brute-force enumeration on synthetic data (``tests/test_
tradelab_observed_strategy_universe.py``'s ``FrequentItemsetMining
CompletenessTests``).

**Known limitation, not yet solved: real-MNQ-scale tractability.**
Apriori's pruning assumes support decays quickly as itemsets grow --
true for sparse, independent data, but real MNQ confluence selectors
are *densely correlated* (a single price-structure event routinely
trips many selectors, timeframes, and variants at once, and whole-day
persistence keeps them co-active for hours), so support decays far more
slowly than classical Apriori assumes. Measured on a real 10k-bar
slice at the default threshold (100): one single ``(primary,
direction)`` pair alone already has 92 frequent single-item confluence
selectors; at a relaxed threshold of 50, 134 frequent items produce
8,911 candidate pairs, of which 6,078 (68%) are *still* frequent at the
next level. A follow-up investigation into closed-itemset deduplication
(collapsing itemsets that share an identical occurrence-ID set into one
canonical representative) found the same data does not rescue this
either: frequent items collapse into only 86 distinct occurrence-sets
out of 92 (correlated, but not identical), and at the next level 2,011
distinct candidate pairs converge to 2,011 distinct occurrence-sets --
zero convergence. Mining is therefore *correct* (verified against real,
independently-tested O(N log N) discovery and, separately, brute-force
completeness on synthetic data) but **not yet proven tractable at full
real-MNQ scale** -- this PR does not attempt a full real-data combo
discovery run; that remains open follow-up work.

**Two phases, deliberately split, mirroring ``tradelab.backtest``'s own
"preparation vs. execution" split:**

1. ``discover_observed_universe(context)`` -- a single, eager pass over
   every real ``EntrySignal`` in ``context.timeline`` (via ``tradelab.
   entry_signal_timeline.iter_active_signal_snapshots``, one linear
   forward sweep, ``O(N log N)`` for ``N`` signals -- see that
   function's own docstring; this pass alone was confirmed on a real
   100k-bar MNQ slice: 107,162 signals swept in 9.7s, no ``query_
   active_signals`` calls at all), collecting each ``(primary,
   direction)`` pair's own real occurrences, then mining the frequent
   required-confluence itemsets for each pair (see above), plus the
   empirical FVG-boundary-SL variant bounds (see below) and a few audit
   counts. Returns an ``ObservedUniverse``: the set of every really-
   observed, frequent ``(direction, primary_entry_selector, required_
   confluence_selectors)`` combination (an ``ObservedConfluenceCombo``,
   still no TP/SL/bias attached).
2. ``iter_observed_strategy_specs(universe)`` -- a lazy generator over
   that already-discovered, bounded set of combos, expanding each one
   across all three ``BiasPolicy`` values, all four TP choices
   (liquidity rank 1/2/3, nearest data-wick), and every SL choice
   (liquidity rank 1/2/3, plus one ``FVG_BOUNDARY`` variant per
   qualifying own fvg-5m/15m selector up to its own empirical bound) --
   this is the layer that can be astronomically larger than step 1's own
   combo count, and it is never materialized: calling this function runs
   none of its body, and each ``StrategySpec`` is only constructed the
   instant a caller asks for the next one.

``StrategyUniverseGenerator`` (the stateful, SQLite-backed class) wraps
both phases with exact-once, resumable checkpointing of *emitted*
``strategy_id``s -- see its own docstring for the full contract. Its
``.run()`` yields plain ``StrategySpec`` objects, so it can be passed
directly as the ``strategy_specs`` argument to ``tradelab.research_
journal.ResearchJournal.run(strategy_specs, context)`` unchanged.

**Empirical FVG-boundary-SL variant bounds** (this task's own
requirement): ``SLChoice(source_type=FVG_BOUNDARY, fvg_variant_number=N)``
has no *artificial* global cap on ``N`` -- but generating every ``N``
from 1 to infinity would be nonsensical, so this module computes the
*exact* maximum ``N`` that could ever actually matter, not an
approximation. For each ``(fvg_link_selector, direction)`` pair, at
every real instant where that selector is genuinely active (as this
occurrence's own primary or as a confluence), this module calls
``tradelab.sl_targets.detect_sl_target_candidates`` itself -- with the
real entry price a backtest would actually use (``tradelab.backtest``'s
own contract: the 1-minute bar whose own ``timestamp`` equals the
signal's ``known_at``, its own ``open``), the real ``direction``, and
exactly the FVG candidates genuinely active for that selector at that
instant -- and reads ``len(sl_result.fvg_targets)`` directly: the *true*
number of boundaries that call would accept, already filtered by
``detect_sl_target_candidates``'s own side-relative-to-entry-price rule.
The running maximum of that true count, across every real occurrence,
*is* the bound -- never an approximation from merely counting active
signals (which can overcount, since not every active FVG candidate ends
up on the correct side of a given real entry price). A variant number
above this bound could never produce a qualifying boundary anywhere in
this context; every variant number up to and including it was genuinely
observed to qualify at least once.

**Never filters by historical results.** Exactly like ``tradelab.
strategy_spec.iter_strategy_specs``, nothing here inspects P&L, win
rate, or trade count -- only whether a combination is structurally
*reachable* given what the timeline actually contains. That filtering
(minimum trade frequency, ranking) is entirely ``tradelab.research_
runner``/``tradelab.research_journal``'s job, downstream and unchanged.

**Reuses, never duplicates, existing validation and ID construction.**
Every generated ``StrategySpec`` is built and validated by the exact
same ``StrategySpec.__post_init__`` and ``tradelab.strategy_spec.
_build_strategy_id`` that ``iter_strategy_specs`` itself uses internally
(cross-module private-helper reuse, the same established convention
this package already uses elsewhere, e.g. ``tradelab.fvg_touch``
importing ``tradelab.liquidity._validate_chronology``) -- this module
adds no second selector-validation or strategy-id scheme.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from tradelab.backtest import PreparedBacktestContext
from tradelab.entry_signal_timeline import EntrySignal, iter_active_signal_snapshots
from tradelab.research_runner import ResearchWindow, _count_per_window
from tradelab.selector_catalog import ALL_CANONICAL_ENTRY_SELECTORS, selector_fixed_direction, selector_timeframe
from tradelab.sl_targets import (
    FVG_SL_TIMEFRAMES,
    MAX_LIQUIDITY_SL_TARGETS,
    MarketEntryContext,
    SelectedFVGConfluence,
    SLSourceType,
    detect_sl_target_candidates,
)
from tradelab.strategy_spec import BiasPolicy, SLChoice, StrategySpec, TPChoice, _build_strategy_id
from tradelab.tp_targets import MAX_LIQUIDITY_TARGETS, TPSourceType

UNIVERSE_SCHEMA_VERSION = "1.0"
GENERATOR_POLICY_VERSION = "observed_strategy_universe_v1"

ALL_BIAS_POLICIES: tuple[BiasPolicy, ...] = tuple(BiasPolicy)
ALL_TP_CHOICES: tuple[TPChoice, ...] = tuple(
    TPChoice(source_type=TPSourceType.LIQUIDITY, rank=rank) for rank in range(1, MAX_LIQUIDITY_TARGETS + 1)
) + (TPChoice(source_type=TPSourceType.DATA_WICK, rank=None),)

# The same two explicit, full, half-open calendar-year windows
# tradelab.profile_research_context.PROFILE_WINDOWS already uses, and the
# same default tradelab.research_runner.ResearchConfig.min_trades_per_window
# already uses -- reused here as the *default* pruning threshold for
# frequent-confluence-itemset mining (see discover_observed_universe's own
# docstring). Both remain explicit, overridable parameters, never silently
# hard-coded assumptions a caller cannot see or change.
DEFAULT_PRUNING_WINDOWS: tuple[ResearchWindow, ...] = (
    ResearchWindow("2024", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc)),
    ResearchWindow("2025", datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc)),
)
DEFAULT_MIN_TRADES_PER_WINDOW = 100

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS universe_manifest (
    run_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    context_artifact_id TEXT NOT NULL,
    selector_catalog_fingerprint TEXT NOT NULL,
    generator_policy_version TEXT NOT NULL,
    min_trades_per_window INTEGER NOT NULL,
    pruning_windows_json TEXT NOT NULL,
    primary_pair_count INTEGER NOT NULL,
    primary_occurrence_count INTEGER NOT NULL,
    combo_count INTEGER NOT NULL,
    fvg_boundary_bounds_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS combo_support (
    direction TEXT NOT NULL,
    primary_entry_selector TEXT NOT NULL,
    required_confluence_selectors_json TEXT NOT NULL,
    window_id TEXT NOT NULL,
    support INTEGER NOT NULL,
    PRIMARY KEY (direction, primary_entry_selector, required_confluence_selectors_json, window_id)
);

CREATE TABLE IF NOT EXISTS emitted_strategies (
    strategy_id TEXT PRIMARY KEY,
    committed_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ObservedConfluenceCombo:
    """One really-observed, *frequent* ``(direction, primary_entry_
    selector, required_confluence_selectors)`` triple -- every selector
    in ``required_confluence_selectors`` was genuinely active, in this
    same ``direction``, at ``dict(support_by_window)[window_id]`` many
    real primary signal occurrences of ``primary_entry_selector`` in
    each explicit window (never fewer than the mining threshold that
    admitted it -- see ``discover_observed_universe``). ``support_by_
    window`` is a sorted tuple of ``(window_id, support)`` pairs, not a
    plain ``dict``, so this dataclass -- like every other frozen
    dataclass in this package -- stays genuinely hashable. No TP/SL/bias
    attached yet -- that expansion is ``iter_observed_strategy_specs``'s
    job."""

    direction: str
    primary_entry_selector: str
    required_confluence_selectors: tuple[str, ...]
    support_by_window: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ObservedUniverse:
    """The result of one ``discover_observed_universe`` pass: the
    deduplicated, deterministically-ordered set of every observed,
    frequent confluence combo, the empirical FVG-boundary-SL bounds
    keyed by ``(fvg_link_selector, direction)``, the pruning parameters
    that produced this set, and a few audit counts. Small and bounded by
    what the data actually contains -- never the full TP/SL/bias-
    expanded strategy space (see module docstring)."""

    combos: tuple[ObservedConfluenceCombo, ...]
    fvg_boundary_bounds: Mapping[tuple[str, str], int]
    windows: tuple[ResearchWindow, ...]
    min_trades_per_window: int
    primary_pair_count: int
    primary_occurrence_count: int
    distinct_pool_count: int


def discover_observed_universe(
    context: PreparedBacktestContext,
    *,
    windows: tuple[ResearchWindow, ...] = DEFAULT_PRUNING_WINDOWS,
    min_trades_per_window: int = DEFAULT_MIN_TRADES_PER_WINDOW,
) -> ObservedUniverse:
    """Walk every real ``EntrySignal`` in ``context.timeline`` exactly
    once and return the bounded ``ObservedUniverse`` it actually
    supports -- see the module docstring for why this eager pass is safe
    (it is the small, real-data-bounded layer, not the combinatorial
    TP/SL/bias expansion), for the ``O(N log N)`` sweep this uses instead
    of one ``query_active_signals`` call per signal, for the empirical
    FVG-boundary-SL bound derivation, and for why required-confluence
    subsets are *mined*, not enumerated (real active pools routinely
    exceed 100 simultaneously-active selectors, whose full power set is
    computationally impossible to enumerate; frequent-itemset mining
    with anti-monotonic pruning stays tractable without ever capping how
    large a qualifying combo may be).

    ``windows``/``min_trades_per_window`` are always explicit (never
    silently assumed) but default to the same two full calendar-year
    windows and the same ``100``-per-window threshold already used
    elsewhere in this package (``tradelab.profile_research_context.
    PROFILE_WINDOWS``, ``tradelab.research_runner.ResearchConfig.
    min_trades_per_window``'s own default).
    """
    if not isinstance(context, PreparedBacktestContext):
        raise ValueError(f"context must be a PreparedBacktestContext, got {context!r}")
    if not windows:
        raise ValueError("windows must not be empty")
    if min_trades_per_window < 1:
        raise ValueError(f"min_trades_per_window must be a positive integer, got {min_trades_per_window!r}")

    timeline = context.timeline
    bars_by_timestamp = {bar.timestamp: bar for bar in context.bars_1m}

    fvg_bounds: dict[tuple[str, str], int] = {}
    primary_pairs_seen: set[tuple[str, str]] = set()
    primary_occurrence_count = 0
    distinct_pools_seen: dict[tuple[str, str], set[frozenset[str]]] = {}
    # Phase 1 output: for every (primary, direction) pair, its own real
    # occurrences as (known_at, pool) -- pool already excludes primary
    # itself. This is the bounded "transaction database" frequent-
    # itemset mining (phase 2) runs against, one pair at a time.
    occurrences_by_pair: dict[tuple[str, str], list[tuple[datetime, frozenset[str]]]] = {}

    for snapshot in iter_active_signal_snapshots(timeline):
        signal = snapshot.signal
        primary = signal.selector
        direction = signal.direction

        # A direction-pinned liquidity selector (selector_fixed_direction)
        # can never really occur with a mismatched direction (tradelab.
        # liquidity_entry always assigns direction from a fixed
        # side/entry_mode table), but this module still never trusts that
        # invariant blindly -- a signal here whose own direction disagrees
        # with its own selector's fixed direction is never treated as a
        # primary occurrence of that selector.
        fixed_direction = selector_fixed_direction(primary)
        if fixed_direction is not None and fixed_direction != direction:
            continue

        active_by_selector = snapshot.active_by_selector

        primary_pairs_seen.add((primary, direction))
        primary_occurrence_count += 1

        # Empirical FVG-boundary-SL bound: the TRUE maximum count of
        # qualifying SL boundaries detect_sl_target_candidates could ever
        # return for (selector, direction), probed with the real entry
        # price a backtest would actually use at every real instant where
        # that fvg-5m/15m selector is genuinely active (as this
        # occurrence's own primary or as a confluence). Never an
        # approximation from merely counting active signals. Unaffected
        # by the mining change below -- every real occurrence is still
        # probed, not just ones that end up part of a frequent itemset.
        entry_bar = bars_by_timestamp.get(signal.known_at)
        if entry_bar is not None:
            for selector, active_signals in active_by_selector.items():
                if not (selector.startswith("fvg-") and selector_timeframe(selector) in FVG_SL_TIMEFRAMES):
                    continue
                key = (selector, direction)
                true_count = _true_fvg_boundary_count(
                    direction=direction,
                    known_at=signal.known_at,
                    entry_price=entry_bar.open,
                    active_signals=active_signals,
                )
                if true_count > fvg_bounds.get(key, 0):
                    fvg_bounds[key] = true_count

        pool = frozenset(active_by_selector.keys()) - {primary}
        distinct_pools_seen.setdefault((primary, direction), set()).add(pool)
        occurrences_by_pair.setdefault((primary, direction), []).append((signal.known_at, pool))

    # Phase 2: per (primary, direction) pair, mine every required-
    # confluence itemset whose support meets min_trades_per_window in
    # EVERY window -- never enumerate subsets of a pool directly.
    combos: list[ObservedConfluenceCombo] = []
    for (primary, direction), occurrences in occurrences_by_pair.items():
        known_ats = [known_at for known_at, _ in occurrences]

        # Safe prefilter, exactly mirroring tradelab.research_runner's own
        # "raw signal count is an upper bound on real trade count" logic:
        # if even the primary alone (the empty-confluence combo) never
        # reaches min_trades_per_window raw occurrences in every window,
        # no confluence-restricted subset of it ever could either (adding
        # a required confluence can only ever keep or shrink the
        # qualifying occurrence count) -- skip mining this pair entirely.
        primary_support = _count_per_window(known_ats, windows)
        if not all(primary_support[w.window_id] >= min_trades_per_window for w in windows):
            continue

        pools = [pool for _, pool in occurrences]
        for itemset, support in _mine_frequent_confluence_itemsets(known_ats, pools, windows, min_trades_per_window):
            combos.append(ObservedConfluenceCombo(direction, primary, tuple(sorted(itemset)), tuple(sorted(support.items()))))

    combos.sort(key=lambda combo: (combo.direction, combo.primary_entry_selector, combo.required_confluence_selectors))
    distinct_pool_count = sum(len(pools) for pools in distinct_pools_seen.values())
    return ObservedUniverse(
        combos=tuple(combos),
        fvg_boundary_bounds=dict(fvg_bounds),
        windows=windows,
        min_trades_per_window=min_trades_per_window,
        primary_pair_count=len(primary_pairs_seen),
        primary_occurrence_count=primary_occurrence_count,
        distinct_pool_count=distinct_pool_count,
    )


def _mine_frequent_confluence_itemsets(
    occurrence_known_ats: list[datetime],
    occurrence_pools: list[frozenset[str]],
    windows: tuple[ResearchWindow, ...],
    min_support_per_window: int,
) -> list[tuple[frozenset[str], dict[str, int]]]:
    """Deterministic frequent-itemset mining (Apriori-style: level-wise
    candidate generation with anti-monotonic pruning) over one
    ``(primary, direction)`` pair's own real occurrences.

    Returns every itemset (subset of the selector universe observed
    across ``occurrence_pools``) -- paired with its own per-window
    support -- whose support (the count of occurrences, restricted to a
    given window, whose own active pool is a *superset* of the itemset)
    meets ``min_support_per_window`` in *every* one of ``windows``.
    Always includes the empty itemset first (support = every occurrence
    in each window; the caller already confirmed this passes before
    calling, via the same safe prefilter ``tradelab.research_runner``
    itself uses).

    **Never enumerates a candidate whose own subset has already failed**
    (the Apriori/anti-monotonicity property: adding a member to an
    itemset can only ever keep its support the same or shrink it, since
    "pool is a superset of S" is strictly harder to satisfy for a larger
    S -- so if some ``(k-1)``-subset of a size-``k`` candidate already
    fails the threshold in some window, the size-``k`` candidate cannot
    possibly pass either, in that same window). This is what keeps
    mining tractable even when a single occurrence's own raw active pool
    is far too large (real MNQ data: routinely 50-140 simultaneously
    active selectors) to enumerate every subset of directly -- real
    per-selector support sparsity prunes the vast majority of the space
    away before it is ever built, deterministically and without
    approximation (no itemset that could pass the threshold is ever
    skipped -- see this function's own completeness test).
    """
    assert len(occurrence_known_ats) == len(occurrence_pools)

    window_indices: dict[str, set[int]] = {w.window_id: set() for w in windows}
    for index, known_at in enumerate(occurrence_known_ats):
        for window in windows:
            if window.start_utc <= known_at < window.end_utc:
                window_indices[window.window_id].add(index)
                break

    def support_by_window(tids: set[int]) -> dict[str, int]:
        return {window.window_id: len(tids & window_indices[window.window_id]) for window in windows}

    def passes(support: dict[str, int]) -> bool:
        return all(count >= min_support_per_window for count in support.values())

    all_indices = set(range(len(occurrence_pools)))
    result: list[tuple[frozenset[str], dict[str, int]]] = [(frozenset(), support_by_window(all_indices))]

    item_tids: dict[str, set[int]] = {}
    for index, pool in enumerate(occurrence_pools):
        for item in pool:
            item_tids.setdefault(item, set()).add(index)

    current_level: dict[frozenset[str], set[int]] = {}
    for item in sorted(item_tids):
        tids = item_tids[item]
        support = support_by_window(tids)
        if passes(support):
            itemset = frozenset((item,))
            current_level[itemset] = tids
            result.append((itemset, support))

    size = 2
    while current_level:
        ordered = sorted(current_level.keys(), key=lambda itemset: tuple(sorted(itemset)))
        next_level: dict[frozenset[str], set[int]] = {}
        for a_index in range(len(ordered)):
            for b_index in range(a_index + 1, len(ordered)):
                a, b = ordered[a_index], ordered[b_index]
                union = a | b
                if len(union) != size or union in next_level:
                    continue
                # Apriori join+prune: every (size-1)-subset of this
                # candidate must already be a frequent itemset at the
                # previous level, or the candidate cannot be either.
                if not all(frozenset(union - {member}) in current_level for member in union):
                    continue
                tids = current_level[a] & current_level[b]
                support = support_by_window(tids)
                if passes(support):
                    next_level[union] = tids
                    result.append((union, support))
        current_level = next_level
        size += 1

    return result


_EMPTY_LIQUIDITY_RESULTS: dict = {}


def _true_fvg_boundary_count(
    *,
    direction: str,
    known_at: datetime,
    entry_price: float,
    active_signals: tuple[EntrySignal, ...],
) -> int:
    """The exact number of qualifying ``FVG_BOUNDARY`` SL candidates
    ``tradelab.sl_targets.detect_sl_target_candidates`` returns for a
    real entry at ``entry_price``/``known_at``/``direction``, linked to
    exactly the FVG candidates genuinely active for one selector at that
    instant -- never an approximation.

    Deliberately calls ``detect_sl_target_candidates`` with **empty**
    ``swing_equal_results``/``session_results`` rather than the context's
    real (and, for a full real dataset, very large) liquidity detection
    results. This is not an approximation: reading that function's own
    body shows ``fvg_targets`` is produced exclusively by ``_fvg_sl_
    candidates(entry, selected_fvg)``, which never looks at either
    liquidity dict at all -- only ``liquidity_targets`` (which this
    probe never reads) depends on them. The only place both lists ever
    interact is a cross-list duplicate-``sl_target_id`` check, and
    liquidity/FVG ids use disjoint prefixes (``"sl-liquidity-"`` vs.
    ``"sl-fvg_boundary-"``) by construction, so that check can never
    actually fire across them. Passing empty dicts therefore yields a
    ``fvg_targets`` result byte-for-byte identical to passing the real
    ones, while skipping a full liquidity-level scan (potentially
    hundreds of thousands of levels for a real multi-year context) on
    every single probe call -- this is what makes the bound computation
    tractable at real-data scale without ever approximating the count.
    """
    entry = MarketEntryContext(entry_id="fvg-sl-bound-probe", direction=direction, entry_price=entry_price, entry_known_at=known_at)
    selected_fvg = SelectedFVGConfluence(entry_id="fvg-sl-bound-probe", fvg_candidates=tuple(signal.source for signal in active_signals))
    sl_result = detect_sl_target_candidates(entry, selected_fvg, _EMPTY_LIQUIDITY_RESULTS, _EMPTY_LIQUIDITY_RESULTS)
    return len(sl_result.fvg_targets)


def _sl_choices_for_combo(combo: ObservedConfluenceCombo, fvg_boundary_bounds: Mapping[tuple[str, str], int]) -> Iterator[SLChoice]:
    for rank in range(1, MAX_LIQUIDITY_SL_TARGETS + 1):
        yield SLChoice(source_type=SLSourceType.LIQUIDITY, rank=rank)

    candidate_selectors = (combo.primary_entry_selector,) + combo.required_confluence_selectors
    seen: set[str] = set()
    for selector in candidate_selectors:
        if selector in seen:
            continue
        seen.add(selector)
        if not selector.startswith("fvg-") or selector_timeframe(selector) not in FVG_SL_TIMEFRAMES:
            continue
        bound = fvg_boundary_bounds.get((selector, combo.direction), 0)
        for variant in range(1, bound + 1):
            yield SLChoice(source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector=selector, fvg_variant_number=variant)


def iter_observed_strategy_specs(universe: ObservedUniverse) -> Iterator[StrategySpec]:
    """Lazily yield every valid ``StrategySpec`` obtainable by expanding
    ``universe.combos`` across all bias policies, TP choices, and SL
    choices (liquidity ranks + empirically-bounded FVG boundary
    variants) -- a generator function: calling it runs none of its body,
    and materializes nothing beyond ``universe`` itself (already small
    and bounded -- see ``discover_observed_universe``). Structurally
    invalid combinations (same validation ``StrategySpec.__post_init__``
    already enforces) are silently skipped, exactly like ``tradelab.
    strategy_spec.iter_strategy_specs`` does for its own combinations --
    this is never a historical-results filter.
    """
    for combo in universe.combos:
        for bias_policy in ALL_BIAS_POLICIES:
            for tp_choice in ALL_TP_CHOICES:
                for sl_choice in _sl_choices_for_combo(combo, universe.fvg_boundary_bounds):
                    try:
                        strategy_id = _build_strategy_id(
                            combo.direction, combo.primary_entry_selector, combo.required_confluence_selectors, bias_policy, tp_choice, sl_choice
                        )
                        spec = StrategySpec(
                            strategy_id=strategy_id,
                            direction=combo.direction,
                            primary_entry_selector=combo.primary_entry_selector,
                            required_confluence_selectors=combo.required_confluence_selectors,
                            bias_policy=bias_policy,
                            tp_choice=tp_choice,
                            sl_choice=sl_choice,
                        )
                    except ValueError:
                        continue
                    yield spec


@dataclass(frozen=True)
class UniverseManifest:
    """Everything that identifies *this* universe-generation run -- see
    the module docstring for what each field means. ``fvg_boundary_
    bounds`` is stored as ``{"selector|direction": bound}`` (plain,
    JSON-stable string keys) rather than a dict keyed by tuples, which
    JSON cannot represent natively."""

    run_id: str
    schema_version: str
    context_artifact_id: str
    selector_catalog_fingerprint: str
    generator_policy_version: str
    min_trades_per_window: int
    pruning_windows: tuple[dict, ...]
    primary_pair_count: int
    primary_occurrence_count: int
    combo_count: int
    fvg_boundary_bounds: dict[str, int]


@dataclass(frozen=True)
class UniverseProgress:
    run_id: str
    emitted_count: int
    last_committed_at: str | None


class StrategyUniverseGenerator:
    """A resumable, crash-safe SQLite checkpoint layer over ``iter_
    observed_strategy_specs``. Construction runs ``discover_observed_
    universe(context)`` once (see that function's own docstring for why
    this bounded, eager pass is safe) and writes (fresh run) or verifies
    (resume) a single ``universe_manifest`` row identifying this run --
    the discovered universe's own counts and FVG bounds are folded into
    the manifest's own identity fields, so resuming against a
    meaningfully different context is blocked exactly like a mismatched
    ``context_artifact_id`` would be, even if a caller forgot to bump
    that id themselves.

    ``.run()`` streams ``StrategySpec`` objects, committing each newly-
    seen ``strategy_id`` to the ``emitted_strategies`` table in its own
    transaction *before* yielding it -- so a crash can only ever lose
    work that was never durably checkpointed, and a resumed run never
    re-emits an already-registered ``strategy_id``, exactly mirroring
    ``tradelab.research_journal.ResearchJournal``'s own checkpoint
    contract. The result is a plain iterator of ``StrategySpec``, so it
    can be passed directly as ``ResearchJournal.run(strategy_specs,
    context)``'s own ``strategy_specs`` argument, unchanged.

    Never persists ``context`` itself or any market data -- only counts,
    bounds, and the emitted ``strategy_id`` strings.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        context_artifact_id: str,
        context: PreparedBacktestContext,
        windows: tuple[ResearchWindow, ...] = DEFAULT_PRUNING_WINDOWS,
        min_trades_per_window: int = DEFAULT_MIN_TRADES_PER_WINDOW,
    ) -> None:
        if not context_artifact_id:
            raise ValueError("context_artifact_id must be non-empty")
        if not isinstance(context, PreparedBacktestContext):
            raise ValueError(f"context must be a PreparedBacktestContext, got {context!r}")

        self._db_path = Path(db_path)
        self._context_artifact_id = context_artifact_id
        self._universe = discover_observed_universe(context, windows=windows, min_trades_per_window=min_trades_per_window)

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

        self._manifest = self._open_or_verify_manifest()

    @property
    def run_id(self) -> str:
        return self._manifest.run_id

    @property
    def manifest(self) -> UniverseManifest:
        return self._manifest

    @property
    def universe(self) -> ObservedUniverse:
        return self._universe

    def run(self) -> Iterator[StrategySpec]:
        committed = self._load_committed_ids()
        for spec in iter_observed_strategy_specs(self._universe):
            if spec.strategy_id in committed:
                continue
            self._commit(spec.strategy_id)
            committed.add(spec.strategy_id)
            yield spec

    def progress(self) -> UniverseProgress:
        row = self._conn.execute("SELECT COUNT(*), MAX(committed_at) FROM emitted_strategies").fetchone()
        count, last_committed_at = row
        return UniverseProgress(run_id=self.run_id, emitted_count=count or 0, last_committed_at=last_committed_at)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StrategyUniverseGenerator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _load_committed_ids(self) -> set[str]:
        return {row[0] for row in self._conn.execute("SELECT strategy_id FROM emitted_strategies").fetchall()}

    def _commit(self, strategy_id: str) -> str:
        committed_at = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute("INSERT INTO emitted_strategies (strategy_id, committed_at) VALUES (?, ?)", (strategy_id, committed_at))
        return committed_at

    def _open_or_verify_manifest(self) -> UniverseManifest:
        selector_fingerprint = _selector_catalog_fingerprint()
        fvg_bounds_dict = {f"{selector}|{direction}": bound for (selector, direction), bound in sorted(self._universe.fvg_boundary_bounds.items())}
        pruning_windows = tuple(
            {"window_id": w.window_id, "start_utc": w.start_utc.isoformat(), "end_utc": w.end_utc.isoformat()}
            for w in sorted(self._universe.windows, key=lambda w: w.window_id)
        )
        fields = {
            "schema_version": UNIVERSE_SCHEMA_VERSION,
            "context_artifact_id": self._context_artifact_id,
            "selector_catalog_fingerprint": selector_fingerprint,
            "generator_policy_version": GENERATOR_POLICY_VERSION,
            "min_trades_per_window": self._universe.min_trades_per_window,
            "pruning_windows": list(pruning_windows),
            "primary_pair_count": self._universe.primary_pair_count,
            "primary_occurrence_count": self._universe.primary_occurrence_count,
            "combo_count": len(self._universe.combos),
            "fvg_boundary_bounds": fvg_bounds_dict,
        }
        computed_run_id = _compute_run_id(fields)

        row = self._conn.execute(
            "SELECT run_id, schema_version, context_artifact_id, selector_catalog_fingerprint, generator_policy_version, "
            "min_trades_per_window, pruning_windows_json, primary_pair_count, primary_occurrence_count, combo_count, "
            "fvg_boundary_bounds_json FROM universe_manifest"
        ).fetchone()

        if row is None:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO universe_manifest (run_id, schema_version, context_artifact_id, selector_catalog_fingerprint, "
                    "generator_policy_version, min_trades_per_window, pruning_windows_json, primary_pair_count, "
                    "primary_occurrence_count, combo_count, fvg_boundary_bounds_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        computed_run_id,
                        UNIVERSE_SCHEMA_VERSION,
                        self._context_artifact_id,
                        selector_fingerprint,
                        GENERATOR_POLICY_VERSION,
                        fields["min_trades_per_window"],
                        json.dumps(fields["pruning_windows"], sort_keys=True),
                        fields["primary_pair_count"],
                        fields["primary_occurrence_count"],
                        fields["combo_count"],
                        json.dumps(fvg_bounds_dict, sort_keys=True),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self._write_combo_support()
            return UniverseManifest(run_id=computed_run_id, pruning_windows=pruning_windows, **{k: v for k, v in fields.items() if k != "pruning_windows"})

        (
            stored_run_id,
            stored_schema,
            stored_artifact,
            stored_fingerprint,
            stored_policy,
            stored_min_trades,
            stored_windows_json,
            stored_pp,
            stored_po,
            stored_cc,
            stored_bounds_json,
        ) = row
        stored_pruning_windows = json.loads(stored_windows_json)
        stored_fields = {
            "schema_version": stored_schema,
            "context_artifact_id": stored_artifact,
            "selector_catalog_fingerprint": stored_fingerprint,
            "generator_policy_version": stored_policy,
            "min_trades_per_window": stored_min_trades,
            "pruning_windows": stored_pruning_windows,
            "primary_pair_count": stored_pp,
            "primary_occurrence_count": stored_po,
            "combo_count": stored_cc,
            "fvg_boundary_bounds": json.loads(stored_bounds_json),
        }
        mismatched = sorted(key for key in fields if fields[key] != stored_fields[key])
        if mismatched:
            raise ValueError(
                f"cannot resume observed strategy universe at {self._db_path}: manifest mismatch in field(s) {mismatched} "
                f"against the existing run (run_id={stored_run_id!r}) -- refusing to resume with a different context/universe"
            )
        return UniverseManifest(
            run_id=stored_run_id,
            pruning_windows=tuple(stored_pruning_windows),
            **{k: v for k, v in stored_fields.items() if k != "pruning_windows"},
        )

    def _write_combo_support(self) -> None:
        """Persists every observed combo's per-window support -- the
        deterministic frequent-itemset mining audit trail requirement --
        as one row per (combo, window). Only ever called once, from
        inside the same transaction that inserts the fresh manifest row
        (never on resume, where the existing rows are already correct
        for the already-verified-matching universe)."""
        for combo in self._universe.combos:
            confluence_json = json.dumps(list(combo.required_confluence_selectors), sort_keys=True)
            for window_id, support in sorted(combo.support_by_window):
                self._conn.execute(
                    "INSERT OR REPLACE INTO combo_support "
                    "(direction, primary_entry_selector, required_confluence_selectors_json, window_id, support) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (combo.direction, combo.primary_entry_selector, confluence_json, window_id, support),
                )


def _selector_catalog_fingerprint() -> str:
    return hashlib.sha256("|".join(ALL_CANONICAL_ENTRY_SELECTORS).encode("utf-8")).hexdigest()


def _compute_run_id(manifest_fields: dict) -> str:
    canonical = json.dumps(manifest_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
