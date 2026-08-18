#!/usr/bin/env python3
"""tradelab.observed_strategy_universe tests: required-confluence
subsets are mined as *frequent itemsets* (support >= min_trades_per_
window in every explicit window), never enumerated exhaustively -- every
subset that reaches the threshold is generated (including large ones,
with no artificial size cap), a subset that never co-occurs frequently
enough is never generated, and this is proven complete against a brute-
force reference on a small, tractable universe; 1m/5m/15m selectors
cooperate freely; direction-pinned liquidity selectors never generate
the wrong direction while direction-agnostic fvg selectors follow
whatever real directions were observed; every bias/TP/SL variant is
produced for a combo; empirical FVG-boundary-SL bounds are the *true*
qualifying count from ``tradelab.sl_targets.detect_sl_target_candidates``
(never an active-signal-count approximation) and never truncate a real
variant; discovery no longer calls ``query_active_signals`` once per
primary signal (the O(N^2) pattern) and scales accordingly; the
StrategySpec expansion is lazy; the SQLite-backed generator dedups
exactly once, resumes without re-emitting, and persists combo support
per window; and the manifest -- and the discovery pass behind it -- is
fully deterministic. No real market data anywhere: every timeline is
built directly from hand-constructed ``EntrySignal`` objects, exactly
like ``tests/test_tradelab_profile_research_context.py`` already does; a
handful of tests that exercise the FVG-boundary-SL bound also supply
real ``Bar1m`` bars and real ``FVGEntryCandidate`` sources, since the
true bound now genuinely calls ``detect_sl_target_candidates``."""
from __future__ import annotations

import inspect
import itertools
import random
import tempfile
import time as time_module
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tradelab.bar import Bar1m
from tradelab.bias import HTFBiasResult
from tradelab.backtest import PreparedBacktestContext
from tradelab.data_wicks import DataWickDetectionResult
from tradelab.entry_signal_timeline import EntrySignal, EntrySignalTimeline, SourceKind
from tradelab.fvg_entry import FVGEntryCandidate, FVGEntryCandidateResult
from tradelab.fvg_entry import SourceType as FVGSourceType
from tradelab.liquidity_entry import LiquidityEntryCandidateResult
from tradelab.observed_strategy_universe import (
    ALL_BIAS_POLICIES,
    ALL_TP_CHOICES,
    DEFAULT_MIN_TRADES_PER_WINDOW,
    DEFAULT_PRUNING_WINDOWS,
    ObservedConfluenceCombo,
    ObservedUniverse,
    StrategyUniverseGenerator,
    _mine_frequent_confluence_itemsets,
    discover_observed_universe,
    iter_observed_strategy_specs,
)
from tradelab.research_runner import ResearchWindow
from tradelab.sl_targets import MAX_LIQUIDITY_SL_TARGETS, SLSourceType
from tradelab.strategy_spec import LONG, SHORT

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

# A single, deliberately broad window used by tests that don't care about
# the two-window AND-semantics specifically -- avoids needing fabricated
# 2025 data everywhere just to satisfy DEFAULT_PRUNING_WINDOWS.
ALL_TIME_WINDOW = ResearchWindow("all", datetime(2000, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC))
ALL_TIME_WINDOWS = (ALL_TIME_WINDOW,)

P_SHORT = "liquidity-1m-swing_equal-swing_high-take-reversal"  # side=HIGH, mode=REVERSAL -> fixed SHORT
C_SHORT = "liquidity-1m-swing_equal-swing_low-take-continuation"  # side=LOW, mode=CONTINUATION -> fixed SHORT
A_5M = "fvg-5m-fvg_touch"  # direction-agnostic
B_15M = "fvg-15m-ifvg"  # direction-agnostic
D_1M = "fvg-1m-fvg_touch"  # direction-agnostic


def sig(
    selector: str,
    direction: str,
    known_at: datetime,
    *,
    candidate_suffix: str = "",
) -> EntrySignal:
    timeframe = selector.split("-")[1]
    return EntrySignal(
        candidate_id=f"{selector}-{direction}-{known_at.isoformat()}{candidate_suffix}",
        selector=selector,
        source_kind=SourceKind.FVG if selector.startswith("fvg-") else SourceKind.LIQUIDITY,
        timeframe=timeframe,
        direction=direction,
        timestamp=known_at,
        known_at=known_at,
        trading_day_id="2024-06-01",
        source=None,
    )


def invalidator(selector: str, opposite_of: str, known_at: datetime) -> EntrySignal:
    """A signal built purely to invalidate an earlier ``opposite_of``-
    direction signal on ``selector``'s timeframe (persist-until-
    invalidated model: there is no more time-based expiry, so tests that
    need a confluence to stop being active at some later instant must
    invalidate it explicitly with an opposite-direction, same-timeframe
    signal instead)."""
    other_direction = LONG if opposite_of == SHORT else SHORT
    return sig(selector, other_direction, known_at)


def repeated(selector: str, direction: str, base: datetime, count: int, *, spacing: timedelta = timedelta(minutes=10)) -> list[EntrySignal]:
    """``count`` distinct real occurrences of ``selector``, spaced apart
    -- the standard way these tests hit a given ``min_trades_per_window``
    support threshold without relying on a single repeated instant."""
    return [sig(selector, direction, base + spacing * i) for i in range(count)]


def fvg_sig(zone_id: str, direction: str, known_at: datetime, *, timeframe: str = "5m", lower: float = 90.0, upper: float = 95.0) -> EntrySignal:
    """Like ``sig()``, but with a real ``FVGEntryCandidate`` as its
    ``.source`` -- required for any test exercising the true FVG-
    boundary-SL bound, which calls ``detect_sl_target_candidates``
    against exactly this candidate."""
    candidate = FVGEntryCandidate(
        candidate_id=f"{timeframe}-fvg_touch-{zone_id}",
        source_type=FVGSourceType.FVG_TOUCH,
        direction=direction,
        timeframe=timeframe,
        zone_id=zone_id,
        lower=lower,
        upper=upper,
        contact_type=None,
        timestamp=known_at - timedelta(minutes=1),
        known_at=known_at,
    )
    return EntrySignal(
        candidate_id=candidate.candidate_id,
        selector=f"fvg-{timeframe}-fvg_touch",
        source_kind=SourceKind.FVG,
        timeframe=timeframe,
        direction=direction,
        timestamp=candidate.timestamp,
        known_at=known_at,
        trading_day_id="2024-06-01",
        source=candidate,
    )


def build_timeline(signals: list[EntrySignal]) -> EntrySignalTimeline:
    by_selector: dict[str, list[EntrySignal]] = {}
    for signal in signals:
        by_selector.setdefault(signal.selector, []).append(signal)
    ordered = sorted(signals, key=lambda s: (s.known_at, s.candidate_id))
    return EntrySignalTimeline(signals=ordered, signals_by_selector=by_selector)


def build_context(timeline: EntrySignalTimeline, *, bars: tuple[Bar1m, ...] = ()) -> PreparedBacktestContext:
    return PreparedBacktestContext(
        bars_1m=bars,
        news_events=(),
        tick_size=0.25,
        pivot_strength=3,
        reclaim_window_bars=5,
        fvg_results={},
        fvg_touch_results={},
        fvg_entry_result=FVGEntryCandidateResult(candidates=[]),
        swing_equal_liquidity_results={},
        session_liquidity_results={},
        liquidity_entry_result=LiquidityEntryCandidateResult(candidates=[]),
        data_wick_result=DataWickDetectionResult(events=[]),
        htf_bias_result=HTFBiasResult(timeframe_changes={}),
        timeline=timeline,
    )


def combos_for_primary(universe: ObservedUniverse, primary: str, direction: str | None = None) -> list[ObservedConfluenceCombo]:
    return [c for c in universe.combos if c.primary_entry_selector == primary and (direction is None or c.direction == direction)]


def combo(direction: str, primary: str, confluence: tuple[str, ...] = (), support: int = 1) -> ObservedConfluenceCombo:
    """A hand-built ``ObservedConfluenceCombo`` for tests that exercise
    ``iter_observed_strategy_specs`` directly, bypassing discovery."""
    return ObservedConfluenceCombo(
        direction=direction, primary_entry_selector=primary, required_confluence_selectors=confluence, support_by_window=(("all", support),)
    )


class AllActiveSubsetsGeneratedTests(unittest.TestCase):
    def test_every_frequent_subset_of_the_real_active_pool_is_generated(self):
        # 3 occurrences of P, all with the identical active pool {A, B}
        # (both stay valid across the whole span) -- every subset of
        # {A, B} reaches support 3, so all 4 subsets must be generated.
        threshold = 3
        occurrences = repeated(P_SHORT, SHORT, T0, threshold)
        signals = list(occurrences) + [
            sig(A_5M, SHORT, T0),
            sig(B_15M, SHORT, T0),
        ]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=threshold)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        observed_subsets = {frozenset(c.required_confluence_selectors) for c in combos}
        self.assertEqual(observed_subsets, {frozenset(), frozenset({A_5M}), frozenset({B_15M}), frozenset({A_5M, B_15M})})
        for c in combos:
            self.assertEqual(dict(c.support_by_window)["all"], threshold)

    def test_no_duplicate_combos_across_repeated_identical_pools(self):
        threshold = 3
        signals = list(repeated(P_SHORT, SHORT, T0, threshold, spacing=timedelta(minutes=30)))
        signals += [sig(A_5M, SHORT, T0)]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=threshold)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        self.assertEqual({frozenset(c.required_confluence_selectors) for c in combos}, {frozenset(), frozenset({A_5M})})
        self.assertEqual(len(combos), 2)  # not more -- deduplicated, one row per distinct itemset


class NeverOccurringComboNotGeneratedTests(unittest.TestCase):
    def test_selectors_that_never_co_occur_frequently_are_never_combined(self):
        # {A, B} both reach the threshold together at 3 occurrences; C
        # separately reaches the threshold at 3 different occurrences
        # where A/B are not active (invalidated by an opposite-direction,
        # same-timeframe signal shortly after group1 -- there is no more
        # time-based expiry, so "no longer active" must come from an
        # explicit invalidating event instead). A+C, B+C, A+B+C never
        # co-occur even once and must never appear, regardless of each
        # item's own individual frequency.
        threshold = 3
        group1 = repeated(P_SHORT, SHORT, T0, threshold, spacing=timedelta(minutes=10))
        group2 = repeated(P_SHORT, SHORT, T0 + timedelta(hours=5), threshold, spacing=timedelta(minutes=10))
        signals = list(group1) + list(group2)
        signals += [sig(A_5M, SHORT, T0)]
        signals += [sig(B_15M, SHORT, T0)]
        signals += [invalidator(A_5M, opposite_of=SHORT, known_at=T0 + timedelta(minutes=25))]
        signals += [invalidator(B_15M, opposite_of=SHORT, known_at=T0 + timedelta(minutes=25))]
        signals += [sig(C_SHORT, SHORT, T0 + timedelta(hours=5))]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=threshold)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        observed_subsets = {frozenset(c.required_confluence_selectors) for c in combos}
        self.assertEqual(
            observed_subsets,
            {frozenset(), frozenset({A_5M}), frozenset({B_15M}), frozenset({A_5M, B_15M}), frozenset({C_SHORT})},
        )
        never_seen = {frozenset({A_5M, C_SHORT}), frozenset({B_15M, C_SHORT}), frozenset({A_5M, B_15M, C_SHORT})}
        self.assertEqual(observed_subsets & never_seen, set())

    def test_a_subset_below_threshold_is_not_generated_even_though_the_primary_alone_is(self):
        # P alone reaches the threshold (5 occurrences); A is (re-)signaled
        # right at occurrences 0 and 1 and each instance is invalidated
        # (opposite-direction, same-timeframe) shortly after, before the
        # next occurrence's query -- so A is active alongside only 2 of
        # them, below the threshold of 3 -- so {A} must not be generated
        # even though the empty combo is.
        threshold = 3
        occurrences = repeated(P_SHORT, SHORT, T0, 5, spacing=timedelta(minutes=30))
        signals = list(occurrences)
        signals += [sig(A_5M, SHORT, occurrences[0].known_at)]
        signals += [invalidator(A_5M, opposite_of=SHORT, known_at=occurrences[0].known_at + timedelta(minutes=1))]
        signals += [sig(A_5M, SHORT, occurrences[1].known_at, candidate_suffix="-b")]
        signals += [invalidator(A_5M, opposite_of=SHORT, known_at=occurrences[1].known_at + timedelta(minutes=1))]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=threshold)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        observed_subsets = {frozenset(c.required_confluence_selectors) for c in combos}
        self.assertEqual(observed_subsets, {frozenset()})


class CooperationAcrossTimeframesTests(unittest.TestCase):
    def test_1m_primary_with_5m_and_15m_confluence_together_is_generated_when_frequent(self):
        threshold = 3
        occurrences = repeated(P_SHORT, SHORT, T0, threshold)
        signals = list(occurrences) + [
            sig(A_5M, SHORT, T0),
            sig(B_15M, SHORT, T0),
        ]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=threshold)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        subsets = {frozenset(c.required_confluence_selectors) for c in combos}
        self.assertIn(frozenset({A_5M, B_15M}), subsets)


class DirectionConsistencyTests(unittest.TestCase):
    def test_direction_pinned_selector_never_generates_the_wrong_direction(self):
        # P_SHORT is structurally pinned to SHORT (selector_fixed_direction);
        # rogue LONG-labelled signals for it must never produce a LONG combo.
        threshold = 2
        signals = list(repeated(P_SHORT, SHORT, T0, threshold)) + list(repeated(P_SHORT, LONG, T0 + timedelta(hours=5), threshold))
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=threshold)

        self.assertEqual(combos_for_primary(universe, P_SHORT, LONG), [])
        self.assertGreater(len(combos_for_primary(universe, P_SHORT, SHORT)), 0)

    def test_direction_agnostic_fvg_selector_generates_both_observed_directions(self):
        threshold = 2
        signals = list(repeated(D_1M, LONG, T0, threshold)) + list(repeated(D_1M, SHORT, T0 + timedelta(hours=5), threshold))
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=threshold)

        self.assertGreater(len(combos_for_primary(universe, D_1M, LONG)), 0)
        self.assertGreater(len(combos_for_primary(universe, D_1M, SHORT)), 0)


class AllTPSLBiasVariantsTests(unittest.TestCase):
    def test_combo_without_fvg_selectors_gets_exactly_liquidity_sl_bias_tp_product(self):
        universe = ObservedUniverse(
            combos=(combo(SHORT, P_SHORT),),
            fvg_boundary_bounds={},
            windows=ALL_TIME_WINDOWS,
            min_trades_per_window=1,
            primary_pair_count=1,
            primary_occurrence_count=1,
            distinct_pool_count=1,
        )

        specs = list(iter_observed_strategy_specs(universe))
        self.assertEqual(len(specs), len(ALL_BIAS_POLICIES) * len(ALL_TP_CHOICES) * MAX_LIQUIDITY_SL_TARGETS)
        self.assertEqual({s.bias_policy for s in specs}, set(ALL_BIAS_POLICIES))
        self.assertEqual({(s.tp_choice.source_type, s.tp_choice.rank) for s in specs}, {(c.source_type, c.rank) for c in ALL_TP_CHOICES})
        self.assertEqual({s.sl_choice.rank for s in specs}, {1, 2, 3})
        self.assertTrue(all(s.sl_choice.source_type == SLSourceType.LIQUIDITY for s in specs))

    def test_combo_with_fvg_selector_adds_exactly_bound_many_fvg_boundary_variants(self):
        universe = ObservedUniverse(
            combos=(combo(SHORT, A_5M),),
            fvg_boundary_bounds={(A_5M, SHORT): 2},
            windows=ALL_TIME_WINDOWS,
            min_trades_per_window=1,
            primary_pair_count=1,
            primary_occurrence_count=1,
            distinct_pool_count=1,
        )
        specs = list(iter_observed_strategy_specs(universe))
        expected_sl_count = MAX_LIQUIDITY_SL_TARGETS + 2
        self.assertEqual(len(specs), len(ALL_BIAS_POLICIES) * len(ALL_TP_CHOICES) * expected_sl_count)

        fvg_variants = sorted(s.sl_choice.fvg_variant_number for s in specs if s.sl_choice.source_type == SLSourceType.FVG_BOUNDARY)
        self.assertEqual(set(fvg_variants), {1, 2})


class EmpiricalFVGBoundsTests(unittest.TestCase):
    """The bound is the TRUE count ``tradelab.sl_targets.detect_sl_
    target_candidates`` would actually accept -- computed with the real
    entry price a backtest would use (the bar whose own ``timestamp``
    equals the signal's ``known_at``, its own ``open``) -- never an
    approximation from merely counting active signals. Bound probing
    runs on every real occurrence regardless of the mining threshold, so
    these tests use ``min_trades_per_window=1`` purely so the combo
    itself also appears (for the ``iter_observed_strategy_specs``
    assertions) -- the bound computation itself is unaffected either way."""

    def test_bound_counts_only_signals_that_actually_qualify_as_sl_boundaries(self):
        # Three simultaneously-active fvg-5m LONG signals at one real
        # primary occurrence, entry_price (bar.open) = 100: only zone Z1's
        # lower bound (90) is on the correct (below-entry) side for a LONG
        # SL; Z2 (105) and Z3 (101) are both >= entry_price and never
        # qualify. The true bound must be exactly 1, not 3.
        entry_known_at = T0
        primary = fvg_sig("P", LONG, entry_known_at, timeframe="1m", lower=200.0, upper=205.0)
        z1 = fvg_sig("Z1", LONG, entry_known_at, lower=90.0, upper=95.0)  # qualifies: 90 < 100
        z2 = fvg_sig("Z2", LONG, entry_known_at, lower=105.0, upper=110.0)  # wrong side: 105 >= 100
        z3 = fvg_sig("Z3", LONG, entry_known_at, lower=101.0, upper=103.0)  # wrong side: 101 >= 100
        bars = (Bar1m(timestamp=entry_known_at, open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0, contract="MNQ"),)
        context = build_context(build_timeline([primary, z1, z2, z3]), bars=bars)

        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=1)
        self.assertEqual(universe.fvg_boundary_bounds[(A_5M, LONG)], 1)

        specs = list(iter_observed_strategy_specs(universe))
        fvg_variants = {
            s.sl_choice.fvg_variant_number for s in specs if s.sl_choice.source_type == SLSourceType.FVG_BOUNDARY and s.sl_choice.fvg_link_selector == A_5M
        }
        self.assertEqual(fvg_variants, {1})

    def test_bound_reaches_the_full_count_when_every_active_signal_qualifies(self):
        entry_known_at = T0
        primary = fvg_sig("P", LONG, entry_known_at, timeframe="1m", lower=200.0, upper=205.0)
        z1 = fvg_sig("Z1", LONG, entry_known_at, lower=90.0, upper=95.0)
        z2 = fvg_sig("Z2", LONG, entry_known_at, lower=80.0, upper=85.0)
        z3 = fvg_sig("Z3", LONG, entry_known_at, lower=70.0, upper=75.0)
        bars = (Bar1m(timestamp=entry_known_at, open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0, contract="MNQ"),)
        context = build_context(build_timeline([primary, z1, z2, z3]), bars=bars)

        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=1)
        self.assertEqual(universe.fvg_boundary_bounds[(A_5M, LONG)], 3)

    def test_bound_is_the_max_true_count_across_multiple_real_occurrences(self):
        t1 = T0
        t2 = T0 + timedelta(minutes=10)
        primary1 = fvg_sig("P1", LONG, t1, timeframe="1m", lower=300.0, upper=305.0)
        primary2 = fvg_sig("P2", LONG, t2, timeframe="1m", lower=300.0, upper=305.0)
        z1 = fvg_sig("Z1", LONG, t1, lower=90.0, upper=95.0)
        z2 = fvg_sig("Z2", LONG, t1, lower=150.0, upper=155.0)
        bars = (
            Bar1m(timestamp=t1, open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0, contract="MNQ"),
            Bar1m(timestamp=t2, open=200.0, high=201.0, low=199.0, close=200.5, volume=1.0, contract="MNQ"),
        )
        context = build_context(build_timeline([primary1, primary2, z1, z2]), bars=bars)
        universe = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=1)
        self.assertEqual(universe.fvg_boundary_bounds[(A_5M, LONG)], 2)

    def test_bound_never_truncates_a_variant_that_could_really_occur(self):
        universe = ObservedUniverse(
            combos=(combo(SHORT, P_SHORT, (A_5M,)),),
            fvg_boundary_bounds={(A_5M, SHORT): 3},
            windows=ALL_TIME_WINDOWS,
            min_trades_per_window=1,
            primary_pair_count=1,
            primary_occurrence_count=1,
            distinct_pool_count=1,
        )
        specs = list(iter_observed_strategy_specs(universe))
        fvg_variants = {
            s.sl_choice.fvg_variant_number
            for s in specs
            if s.sl_choice.source_type == SLSourceType.FVG_BOUNDARY and s.sl_choice.fvg_link_selector == A_5M
        }
        self.assertEqual(fvg_variants, {1, 2, 3})

    def test_bound_is_zero_and_no_fvg_boundary_variants_generated_when_never_observed(self):
        universe = ObservedUniverse(
            combos=(combo(SHORT, P_SHORT, (A_5M,)),),
            fvg_boundary_bounds={},
            windows=ALL_TIME_WINDOWS,
            min_trades_per_window=1,
            primary_pair_count=1,
            primary_occurrence_count=1,
            distinct_pool_count=1,
        )
        specs = list(iter_observed_strategy_specs(universe))
        self.assertTrue(all(s.sl_choice.source_type == SLSourceType.LIQUIDITY for s in specs))


class FrequentItemsetMiningCompletenessTests(unittest.TestCase):
    """The core correctness proof requested in review: the Apriori-
    pruned miner must return *exactly* the same set of frequent itemsets
    a brute-force scan of every subset of the item universe would --
    proving no potentially-qualifying subset is ever dropped by the
    pruning (and no spurious one is ever added), on randomized
    transaction data across two windows."""

    @staticmethod
    def _brute_force(
        known_ats: list[datetime], pools: list[frozenset[str]], windows: tuple[ResearchWindow, ...], min_support: int
    ) -> set[frozenset[str]]:
        universe = sorted(set().union(*pools)) if pools else []
        window_indices = {w.window_id: set() for w in windows}
        for index, known_at in enumerate(known_ats):
            for w in windows:
                if w.start_utc <= known_at < w.end_utc:
                    window_indices[w.window_id].add(index)
                    break

        result: set[frozenset[str]] = set()
        for size in range(len(universe) + 1):
            for candidate in itertools.combinations(universe, size):
                itemset = frozenset(candidate)
                tids = {i for i, pool in enumerate(pools) if itemset <= pool}
                if all(len(tids & window_indices[w.window_id]) >= min_support for w in windows):
                    result.add(itemset)
        return result

    def test_mined_itemsets_exactly_match_brute_force_on_randomized_data(self):
        rng = random.Random(20260818)
        items = [f"item-{i}" for i in range(9)]  # 2^9 = 512 subsets -- tractable to brute force directly
        windows = (
            ResearchWindow("2024", datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
            ResearchWindow("2025", datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
        )
        min_support = 8

        known_ats: list[datetime] = []
        pools: list[frozenset[str]] = []
        for _ in range(400):
            year = rng.choice([2024, 2025])
            known_at = datetime(year, rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59), tzinfo=UTC)
            # Biased sampling (not uniform) so some items/pairs are
            # naturally more frequent than others -- a more realistic,
            # non-degenerate test of the support threshold than uniform
            # random inclusion would give.
            pool_size = rng.choice([0, 1, 1, 2, 2, 3, 4, 5])
            pool = frozenset(rng.sample(items, k=min(pool_size, len(items))))
            known_ats.append(known_at)
            pools.append(pool)

        mined = _mine_frequent_confluence_itemsets(known_ats, pools, windows, min_support)
        mined_itemsets = {itemset for itemset, _support in mined}

        expected = self._brute_force(known_ats, pools, windows, min_support)
        self.assertEqual(mined_itemsets, expected)
        self.assertGreater(len(expected), 1)  # sanity: the scenario actually exercises more than just the empty set

    def test_mined_support_matches_true_support_per_window(self):
        rng = random.Random(7)
        items = ["x", "y", "z"]
        windows = (
            ResearchWindow("2024", datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
            ResearchWindow("2025", datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
        )
        known_ats = []
        pools = []
        for _ in range(200):
            year = rng.choice([2024, 2025])
            known_at = datetime(year, rng.randint(1, 12), rng.randint(1, 28), tzinfo=UTC)
            pool = frozenset(rng.sample(items, k=rng.randint(0, 3)))
            known_ats.append(known_at)
            pools.append(pool)

        mined = dict(_mine_frequent_confluence_itemsets(known_ats, pools, windows, 5))
        for itemset, support in mined.items():
            for w in windows:
                true_support = sum(
                    1 for known_at, pool in zip(known_ats, pools) if itemset <= pool and w.start_utc <= known_at < w.end_utc
                )
                self.assertEqual(support[w.window_id], true_support)


class LazyGenerationTests(unittest.TestCase):
    def test_iter_observed_strategy_specs_returns_a_generator(self):
        universe = ObservedUniverse(
            combos=(), fvg_boundary_bounds={}, windows=ALL_TIME_WINDOWS, min_trades_per_window=1, primary_pair_count=0, primary_occurrence_count=0, distinct_pool_count=0
        )
        self.assertTrue(inspect.isgenerator(iter_observed_strategy_specs(universe)))

    def test_generator_run_returns_a_generator_without_materializing(self):
        signals = list(repeated(P_SHORT, SHORT, T0, 2))
        context = build_context(build_timeline(signals))
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            gen = StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=context, windows=ALL_TIME_WINDOWS, min_trades_per_window=2)
            result = gen.run()
            self.assertTrue(inspect.isgenerator(result))
            gen.close()


class ScalabilityTests(unittest.TestCase):
    """Discovery must not call ``query_active_signals`` once per primary
    signal (the O(N^2) pattern this fixed) -- proven structurally
    (the function is no longer even imported into this module) and,
    as a secondary sanity check, empirically (a 4x larger synthetic
    timeline must not take anywhere near 4x-squared as long)."""

    def test_module_no_longer_imports_query_active_signals(self):
        import tradelab.observed_strategy_universe as module

        self.assertFalse(hasattr(module, "query_active_signals"))
        self.assertTrue(hasattr(module, "iter_active_signal_snapshots"))

    def _build_moderate_timeline_context(self, count: int) -> PreparedBacktestContext:
        # Alternates between two liquidity selectors on the same 1m
        # timeframe, and periodically flips the direction label -- an
        # invalidating event under the persist-until-invalidated model
        # that wipes the whole growing same-timeframe pool, exactly as a
        # real market's own direction reversals naturally bound the
        # active pool despite unbounded validity. Without any such flip
        # the pool would grow to size N (nothing ever invalidates a
        # same-direction-forever timeline), which is a real property of
        # that degenerate dataset, not a complexity regression in the
        # sweep itself -- isolates the O(N log N) sweep, not the
        # (inherently different-complexity) FVG-SL bound probe, which is
        # never triggered here (no fvg-5m/15m selectors, no bars_1m).
        signals = [
            sig(P_SHORT if i % 2 == 0 else C_SHORT, SHORT if (i // 20) % 2 == 0 else LONG, T0 + timedelta(minutes=i)) for i in range(count)
        ]
        return build_context(build_timeline(signals))

    def test_discovery_time_does_not_grow_quadratically_with_signal_count(self):
        small_ctx = self._build_moderate_timeline_context(1000)
        large_ctx = self._build_moderate_timeline_context(4000)

        start = time_module.perf_counter()
        discover_observed_universe(small_ctx, windows=ALL_TIME_WINDOWS, min_trades_per_window=2)
        small_time = time_module.perf_counter() - start

        start = time_module.perf_counter()
        discover_observed_universe(large_ctx, windows=ALL_TIME_WINDOWS, min_trades_per_window=2)
        large_time = time_module.perf_counter() - start

        # True O(N^2) would show ~16x growth for a 4x size increase;
        # O(N log N) shows barely more than 4x. A generous 10x/0.5s floor
        # keeps this from flaking on a slow CI machine while still
        # catching a real quadratic regression.
        self.assertLess(large_time, max(small_time * 10, 0.5))


class ExactOnceDedupAndResumeTests(unittest.TestCase):
    def _context(self):
        threshold = 3
        occurrences = repeated(P_SHORT, SHORT, T0, threshold)
        signals = list(occurrences) + [
            sig(A_5M, SHORT, T0),
            sig(B_15M, SHORT, T0),
        ]
        return build_context(build_timeline(signals))

    def _generator(self, db_path, context_artifact_id="ctx-1", context=None):
        return StrategyUniverseGenerator(
            db_path, context_artifact_id=context_artifact_id, context=context or self._context(), windows=ALL_TIME_WINDOWS, min_trades_per_window=3
        )

    def test_full_run_has_no_duplicate_strategy_ids(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            gen = self._generator(db_path)
            specs = list(gen.run())
            ids = [s.strategy_id for s in specs]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertGreater(len(ids), 0)
            gen.close()

    def test_calling_run_again_on_the_same_open_generator_emits_nothing_new(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            gen = self._generator(db_path)
            first = list(gen.run())
            second = list(gen.run())
            self.assertEqual(second, [])
            self.assertGreater(len(first), 0)
            gen.close()

    def test_crash_mid_run_then_resume_yields_exactly_the_missing_strategies_once(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            context = self._context()

            gen1 = self._generator(db_path, context=context)
            full_total = len(list(iter_observed_strategy_specs(gen1.universe)))
            gen1.close()

            gen2 = self._generator(db_path, context=context)
            real_commit = StrategyUniverseGenerator._commit
            call_count = {"n": 0}

            def flaky_commit(self, strategy_id):
                call_count["n"] += 1
                if call_count["n"] == 5:
                    raise RuntimeError("simulated crash before commit")
                return real_commit(self, strategy_id)

            interrupted_ids = []
            with mock.patch.object(StrategyUniverseGenerator, "_commit", flaky_commit):
                with self.assertRaises(RuntimeError):
                    for spec in gen2.run():
                        interrupted_ids.append(spec.strategy_id)
            self.assertEqual(len(interrupted_ids), 4)
            self.assertEqual(gen2.progress().emitted_count, 4)
            gen2.close()

            gen3 = self._generator(db_path, context=context)
            resumed_ids = [spec.strategy_id for spec in gen3.run()]
            gen3.close()

            self.assertEqual(len(set(interrupted_ids) & set(resumed_ids)), 0)
            self.assertEqual(len(interrupted_ids) + len(resumed_ids), full_total)

    def test_manifest_mismatch_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            context = self._context()
            gen1 = self._generator(db_path, context=context)
            gen1.close()

            with self.assertRaises(ValueError):
                self._generator(db_path, context_artifact_id="ctx-2", context=context)

    def test_different_min_trades_per_window_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            context = self._context()
            gen1 = self._generator(db_path, context=context)
            gen1.close()

            with self.assertRaises(ValueError):
                StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=context, windows=ALL_TIME_WINDOWS, min_trades_per_window=4)

    def test_combo_support_is_persisted_and_queryable(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            gen = self._generator(db_path)
            rows = gen._conn.execute("SELECT direction, primary_entry_selector, window_id, support FROM combo_support").fetchall()
            gen.close()
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(window_id == "all" for (_, _, window_id, _) in rows))
            self.assertTrue(all(support >= 3 for (_, _, _, support) in rows))


class DeterministicManifestTests(unittest.TestCase):
    def _context(self):
        threshold = 2
        signals = list(repeated(P_SHORT, SHORT, T0, threshold)) + [
            sig(A_5M, SHORT, T0),
            sig(B_15M, SHORT, T0),
        ]
        return build_context(build_timeline(signals))

    def test_discovery_counts_are_identical_across_repeated_calls(self):
        context = self._context()

        universe_a = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=2)
        universe_b = discover_observed_universe(context, windows=ALL_TIME_WINDOWS, min_trades_per_window=2)

        self.assertEqual(len(universe_a.combos), len(universe_b.combos))
        self.assertEqual(universe_a.primary_pair_count, universe_b.primary_pair_count)
        self.assertEqual(universe_a.primary_occurrence_count, universe_b.primary_occurrence_count)
        self.assertEqual(dict(universe_a.fvg_boundary_bounds), dict(universe_b.fvg_boundary_bounds))
        self.assertEqual(set(universe_a.combos), set(universe_b.combos))

    def test_two_fresh_runs_with_identical_context_produce_the_same_run_id(self):
        context = self._context()

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            gen_a = StrategyUniverseGenerator(tmp / "a.sqlite3", context_artifact_id="ctx-1", context=context, windows=ALL_TIME_WINDOWS, min_trades_per_window=2)
            gen_b = StrategyUniverseGenerator(tmp / "b.sqlite3", context_artifact_id="ctx-1", context=context, windows=ALL_TIME_WINDOWS, min_trades_per_window=2)
            self.assertEqual(gen_a.run_id, gen_b.run_id)
            gen_a.close()
            gen_b.close()

    def test_defaults_match_the_documented_100_2024_2025_convention(self):
        self.assertEqual(DEFAULT_MIN_TRADES_PER_WINDOW, 100)
        self.assertEqual({w.window_id for w in DEFAULT_PRUNING_WINDOWS}, {"2024", "2025"})
        window_2024 = next(w for w in DEFAULT_PRUNING_WINDOWS if w.window_id == "2024")
        window_2025 = next(w for w in DEFAULT_PRUNING_WINDOWS if w.window_id == "2025")
        self.assertEqual(window_2024.start_utc, datetime(2024, 1, 1, tzinfo=UTC))
        self.assertEqual(window_2024.end_utc, datetime(2025, 1, 1, tzinfo=UTC))
        self.assertEqual(window_2025.start_utc, datetime(2025, 1, 1, tzinfo=UTC))
        self.assertEqual(window_2025.end_utc, datetime(2026, 1, 1, tzinfo=UTC))


if __name__ == "__main__":
    unittest.main()
