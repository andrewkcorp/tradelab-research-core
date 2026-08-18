#!/usr/bin/env python3
"""tradelab.observed_strategy_universe tests: every real subset of
actually-co-active confluence is generated; a confluence combination
that never co-occurred in the data is never generated; 1m/5m/15m
selectors cooperate freely; direction-pinned liquidity selectors never
generate the wrong direction while direction-agnostic fvg selectors
follow whatever real directions were observed; every bias/TP/SL variant
is produced for a combo; empirical FVG-boundary-SL bounds are derived
from real concurrent activity and never truncate a real variant; the
StrategySpec expansion is lazy; the SQLite-backed generator dedups
exactly once and resumes without re-emitting; and the manifest -- and
the discovery pass behind it -- is fully deterministic. No real market
data anywhere: every timeline is built directly from hand-constructed
``EntrySignal`` objects, exactly like ``tests/test_tradelab_profile_
research_context.py`` already does."""
from __future__ import annotations

import inspect
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tradelab.bias import HTFBiasResult
from tradelab.backtest import PreparedBacktestContext
from tradelab.data_wicks import DataWickDetectionResult
from tradelab.entry_signal_timeline import EntrySignal, EntrySignalTimeline, SourceKind
from tradelab.fvg_entry import FVGEntryCandidateResult
from tradelab.liquidity_entry import LiquidityEntryCandidateResult
from tradelab.observed_strategy_universe import (
    ALL_BIAS_POLICIES,
    ALL_TP_CHOICES,
    ObservedConfluenceCombo,
    ObservedUniverse,
    StrategyUniverseGenerator,
    discover_observed_universe,
    iter_observed_strategy_specs,
)
from tradelab.sl_targets import MAX_LIQUIDITY_SL_TARGETS, SLSourceType
from tradelab.strategy_spec import LONG, SHORT

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

P_SHORT = "liquidity-1m-swing_equal-swing_high-take-reversal"  # side=HIGH, mode=REVERSAL -> fixed SHORT
C_SHORT = "liquidity-1m-swing_equal-swing_low-take-continuation"  # side=LOW, mode=CONTINUATION -> fixed SHORT
A_5M = "fvg-5m-fvg_touch"  # direction-agnostic
B_15M = "fvg-15m-ifvg"  # direction-agnostic
D_1M = "fvg-1m-fvg_touch"  # direction-agnostic


def sig(selector: str, direction: str, known_at: datetime, *, valid_until: datetime | None = None, candidate_suffix: str = "") -> EntrySignal:
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
        valid_until=valid_until if valid_until is not None else known_at + timedelta(hours=1),
        source=None,
    )


def build_timeline(signals: list[EntrySignal]) -> EntrySignalTimeline:
    by_selector: dict[str, list[EntrySignal]] = {}
    for signal in signals:
        by_selector.setdefault(signal.selector, []).append(signal)
    ordered = sorted(signals, key=lambda s: (s.known_at, s.candidate_id))
    return EntrySignalTimeline(signals=ordered, signals_by_selector=by_selector)


def build_context(timeline: EntrySignalTimeline) -> PreparedBacktestContext:
    return PreparedBacktestContext(
        bars_1m=(),
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
    return [
        c
        for c in universe.combos
        if c.primary_entry_selector == primary and (direction is None or c.direction == direction)
    ]


class AllActiveSubsetsGeneratedTests(unittest.TestCase):
    def test_every_subset_of_the_real_active_pool_is_generated(self):
        signals = [sig(P_SHORT, SHORT, T0), sig(A_5M, SHORT, T0), sig(B_15M, SHORT, T0)]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        observed_subsets = {frozenset(c.required_confluence_selectors) for c in combos}
        self.assertEqual(observed_subsets, {frozenset(), frozenset({A_5M}), frozenset({B_15M}), frozenset({A_5M, B_15M})})

    def test_no_duplicate_combos_across_repeated_identical_pools(self):
        # Three separate real occurrences of P, all with the identical
        # active pool {A} -- must still only produce each distinct
        # confluence subset once.
        signals = [sig(P_SHORT, SHORT, T0 + timedelta(minutes=i * 30)) for i in range(3)]
        signals += [sig(A_5M, SHORT, T0, valid_until=T0 + timedelta(hours=2))]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        self.assertEqual({frozenset(c.required_confluence_selectors) for c in combos}, {frozenset(), frozenset({A_5M})})
        self.assertEqual(len(combos), 2)  # not 2 * 3 -- deduplicated


class NeverOccurringComboNotGeneratedTests(unittest.TestCase):
    def test_selectors_never_simultaneously_active_never_form_a_combo_together(self):
        # Occurrence 1 (T0): A and B are active. Occurrence 2 (T0+20min):
        # A/B have already expired, only C is active. A+C, B+C, A+B+C
        # were never simultaneously active and must never appear.
        signals = [
            sig(P_SHORT, SHORT, T0),
            sig(A_5M, SHORT, T0, valid_until=T0 + timedelta(minutes=5)),
            sig(B_15M, SHORT, T0, valid_until=T0 + timedelta(minutes=5)),
            sig(P_SHORT, SHORT, T0 + timedelta(minutes=20)),
            sig(C_SHORT, SHORT, T0 + timedelta(minutes=20)),
        ]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        observed_subsets = {frozenset(c.required_confluence_selectors) for c in combos}
        self.assertEqual(
            observed_subsets,
            {frozenset(), frozenset({A_5M}), frozenset({B_15M}), frozenset({A_5M, B_15M}), frozenset({C_SHORT})},
        )
        never_seen = {frozenset({A_5M, C_SHORT}), frozenset({B_15M, C_SHORT}), frozenset({A_5M, B_15M, C_SHORT})}
        self.assertEqual(observed_subsets & never_seen, set())


class CooperationAcrossTimeframesTests(unittest.TestCase):
    def test_1m_primary_with_5m_and_15m_confluence_together_is_generated(self):
        signals = [sig(P_SHORT, SHORT, T0), sig(A_5M, SHORT, T0), sig(B_15M, SHORT, T0)]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context)

        combos = combos_for_primary(universe, P_SHORT, SHORT)
        subsets = {frozenset(c.required_confluence_selectors) for c in combos}
        self.assertIn(frozenset({A_5M, B_15M}), subsets)


class DirectionConsistencyTests(unittest.TestCase):
    def test_direction_pinned_selector_never_generates_the_wrong_direction(self):
        # P_SHORT is structurally pinned to SHORT (selector_fixed_direction);
        # a rogue LONG-labelled signal for it must never produce a LONG combo.
        signals = [sig(P_SHORT, SHORT, T0), sig(P_SHORT, LONG, T0 + timedelta(minutes=5))]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context)

        self.assertEqual(combos_for_primary(universe, P_SHORT, LONG), [])
        self.assertGreater(len(combos_for_primary(universe, P_SHORT, SHORT)), 0)

    def test_direction_agnostic_fvg_selector_generates_both_observed_directions(self):
        signals = [sig(D_1M, LONG, T0), sig(D_1M, SHORT, T0 + timedelta(minutes=30))]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context)

        self.assertGreater(len(combos_for_primary(universe, D_1M, LONG)), 0)
        self.assertGreater(len(combos_for_primary(universe, D_1M, SHORT)), 0)


class AllTPSLBiasVariantsTests(unittest.TestCase):
    def test_combo_without_fvg_selectors_gets_exactly_liquidity_sl_bias_tp_product(self):
        combo = ObservedConfluenceCombo(direction=SHORT, primary_entry_selector=P_SHORT, required_confluence_selectors=())
        universe = ObservedUniverse(combos=(combo,), fvg_boundary_bounds={}, primary_pair_count=1, primary_occurrence_count=1, distinct_pool_count=1)

        specs = list(iter_observed_strategy_specs(universe))
        self.assertEqual(len(specs), len(ALL_BIAS_POLICIES) * len(ALL_TP_CHOICES) * MAX_LIQUIDITY_SL_TARGETS)
        self.assertEqual({s.bias_policy for s in specs}, set(ALL_BIAS_POLICIES))
        self.assertEqual({(s.tp_choice.source_type, s.tp_choice.rank) for s in specs}, {(c.source_type, c.rank) for c in ALL_TP_CHOICES})
        self.assertEqual({s.sl_choice.rank for s in specs}, {1, 2, 3})
        self.assertTrue(all(s.sl_choice.source_type == SLSourceType.LIQUIDITY for s in specs))

    def test_combo_with_fvg_selector_adds_exactly_bound_many_fvg_boundary_variants(self):
        combo = ObservedConfluenceCombo(direction=SHORT, primary_entry_selector=A_5M, required_confluence_selectors=())
        universe = ObservedUniverse(
            combos=(combo,), fvg_boundary_bounds={(A_5M, SHORT): 2}, primary_pair_count=1, primary_occurrence_count=1, distinct_pool_count=1
        )
        specs = list(iter_observed_strategy_specs(universe))
        expected_sl_count = MAX_LIQUIDITY_SL_TARGETS + 2
        self.assertEqual(len(specs), len(ALL_BIAS_POLICIES) * len(ALL_TP_CHOICES) * expected_sl_count)

        fvg_variants = sorted(
            s.sl_choice.fvg_variant_number for s in specs if s.sl_choice.source_type == SLSourceType.FVG_BOUNDARY
        )
        self.assertEqual(set(fvg_variants), {1, 2})


class EmpiricalFVGBoundsTests(unittest.TestCase):
    def test_bound_equals_max_concurrently_active_same_selector_signals(self):
        # Three distinct, simultaneously-active fvg-5m signals (different
        # zones) of the same direction at one real primary occurrence.
        signals = [
            sig(P_SHORT, SHORT, T0),
            sig(A_5M, SHORT, T0, candidate_suffix="-zoneA"),
            sig(A_5M, SHORT, T0, candidate_suffix="-zoneB"),
            sig(A_5M, SHORT, T0, candidate_suffix="-zoneC"),
        ]
        context = build_context(build_timeline(signals))
        universe = discover_observed_universe(context)
        self.assertEqual(universe.fvg_boundary_bounds[(A_5M, SHORT)], 3)

    def test_bound_never_truncates_a_variant_that_could_really_occur(self):
        combo = ObservedConfluenceCombo(direction=SHORT, primary_entry_selector=P_SHORT, required_confluence_selectors=(A_5M,))
        universe = ObservedUniverse(
            combos=(combo,), fvg_boundary_bounds={(A_5M, SHORT): 3}, primary_pair_count=1, primary_occurrence_count=1, distinct_pool_count=1
        )
        specs = list(iter_observed_strategy_specs(universe))
        fvg_variants = {
            s.sl_choice.fvg_variant_number
            for s in specs
            if s.sl_choice.source_type == SLSourceType.FVG_BOUNDARY and s.sl_choice.fvg_link_selector == A_5M
        }
        self.assertEqual(fvg_variants, {1, 2, 3})

    def test_bound_is_zero_and_no_fvg_boundary_variants_generated_when_never_observed(self):
        combo = ObservedConfluenceCombo(direction=SHORT, primary_entry_selector=P_SHORT, required_confluence_selectors=(A_5M,))
        universe = ObservedUniverse(combos=(combo,), fvg_boundary_bounds={}, primary_pair_count=1, primary_occurrence_count=1, distinct_pool_count=1)
        specs = list(iter_observed_strategy_specs(universe))
        self.assertTrue(all(s.sl_choice.source_type == SLSourceType.LIQUIDITY for s in specs))


class LazyGenerationTests(unittest.TestCase):
    def test_iter_observed_strategy_specs_returns_a_generator(self):
        universe = ObservedUniverse(combos=(), fvg_boundary_bounds={}, primary_pair_count=0, primary_occurrence_count=0, distinct_pool_count=0)
        self.assertTrue(inspect.isgenerator(iter_observed_strategy_specs(universe)))

    def test_generator_run_returns_a_generator_without_materializing(self):
        signals = [sig(P_SHORT, SHORT, T0)]
        context = build_context(build_timeline(signals))
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            gen = StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=context)
            result = gen.run()
            self.assertTrue(inspect.isgenerator(result))
            gen.close()


class ExactOnceDedupAndResumeTests(unittest.TestCase):
    def _context(self):
        signals = [sig(P_SHORT, SHORT, T0), sig(A_5M, SHORT, T0), sig(B_15M, SHORT, T0)]
        return build_context(build_timeline(signals))

    def test_full_run_has_no_duplicate_strategy_ids(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            gen = StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=self._context())
            specs = list(gen.run())
            ids = [s.strategy_id for s in specs]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertGreater(len(ids), 0)
            gen.close()

    def test_calling_run_again_on_the_same_open_generator_emits_nothing_new(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            gen = StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=self._context())
            first = list(gen.run())
            second = list(gen.run())
            self.assertEqual(second, [])
            self.assertGreater(len(first), 0)
            gen.close()

    def test_crash_mid_run_then_resume_yields_exactly_the_missing_strategies_once(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            context = self._context()

            gen1 = StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=context)
            full_total = len(list(iter_observed_strategy_specs(gen1.universe)))
            gen1.close()

            gen2 = StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=context)
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

            gen3 = StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=context)
            resumed_ids = [spec.strategy_id for spec in gen3.run()]
            gen3.close()

            self.assertEqual(len(set(interrupted_ids) & set(resumed_ids)), 0)
            self.assertEqual(len(interrupted_ids) + len(resumed_ids), full_total)

    def test_manifest_mismatch_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            db_path = Path(tmp_name) / "universe.sqlite3"
            context = self._context()
            gen1 = StrategyUniverseGenerator(db_path, context_artifact_id="ctx-1", context=context)
            gen1.close()

            with self.assertRaises(ValueError):
                StrategyUniverseGenerator(db_path, context_artifact_id="ctx-2", context=context)


class DeterministicManifestTests(unittest.TestCase):
    def test_discovery_counts_are_identical_across_repeated_calls(self):
        signals = [sig(P_SHORT, SHORT, T0), sig(A_5M, SHORT, T0), sig(B_15M, SHORT, T0)]
        context = build_context(build_timeline(signals))

        universe_a = discover_observed_universe(context)
        universe_b = discover_observed_universe(context)

        self.assertEqual(len(universe_a.combos), len(universe_b.combos))
        self.assertEqual(universe_a.primary_pair_count, universe_b.primary_pair_count)
        self.assertEqual(universe_a.primary_occurrence_count, universe_b.primary_occurrence_count)
        self.assertEqual(dict(universe_a.fvg_boundary_bounds), dict(universe_b.fvg_boundary_bounds))
        self.assertEqual(set(universe_a.combos), set(universe_b.combos))

    def test_two_fresh_runs_with_identical_context_produce_the_same_run_id(self):
        signals = [sig(P_SHORT, SHORT, T0), sig(A_5M, SHORT, T0)]
        context = build_context(build_timeline(signals))

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            gen_a = StrategyUniverseGenerator(tmp / "a.sqlite3", context_artifact_id="ctx-1", context=context)
            gen_b = StrategyUniverseGenerator(tmp / "b.sqlite3", context_artifact_id="ctx-1", context=context)
            self.assertEqual(gen_a.run_id, gen_b.run_id)
            gen_a.close()
            gen_b.close()


if __name__ == "__main__":
    unittest.main()
