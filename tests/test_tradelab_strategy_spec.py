#!/usr/bin/env python3
"""Strategy spec tests: selector reuse/validation, TP/SL choice rules
(including data wick rejected as SL and disallowed FVG-SL timeframes),
StrategySpec construction validation (mandatory primary, no primary/
confluence duplication, direction/selector consistency, exactly one TP
and one SL, FVG boundary SL linked only to the strategy's own selector),
JSON-compatible serialization, and the iter_strategy_specs generator:
laziness, deterministic ordering/strategy_id, full coverage of a small
synthetic combination space, and no duplicate strategy_id."""
from __future__ import annotations

import inspect
import itertools
import json
import types
import unittest
from unittest import mock

from tradelab.sl_targets import SLSourceType
from tradelab.tp_targets import TPSourceType
from tradelab.strategy_spec import (
    LONG,
    SHORT,
    BiasPolicy,
    SLChoice,
    StrategySpec,
    TPChoice,
    iter_strategy_specs,
    selector_fixed_direction,
    selector_timeframe,
)

# fvg-* selectors are direction-agnostic; timeframe must be 1m/5m/15m.
FVG_1M_TOUCH = "fvg-1m-fvg_touch"
FVG_5M_IFVG = "fvg-5m-ifvg"
FVG_15M_TOUCH = "fvg-15m-fvg_touch"

# liquidity-* selectors are pinned to exactly one direction by side+entry_mode.
LIQ_SWING_HIGH_REVERSAL = "liquidity-1m-swing_equal-swing_high-take-reversal"  # HIGH+REVERSAL -> SHORT
LIQ_SWING_LOW_REVERSAL = "liquidity-1m-swing_equal-swing_low-take-reversal"  # LOW+REVERSAL -> LONG
LIQ_SWING_HIGH_CONTINUATION = "liquidity-1m-swing_equal-swing_high-take-continuation"  # HIGH+CONTINUATION -> LONG


def tp_liq(rank: int = 1) -> TPChoice:
    return TPChoice(source_type=TPSourceType.LIQUIDITY, rank=rank)


def tp_wick() -> TPChoice:
    return TPChoice(source_type=TPSourceType.DATA_WICK)


def sl_liq(rank: int = 1) -> SLChoice:
    return SLChoice(source_type=SLSourceType.LIQUIDITY, rank=rank)


def sl_fvg(selector: str = FVG_5M_IFVG) -> SLChoice:
    return SLChoice(source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector=selector)


def spec(
    strategy_id: str = "s1",
    direction: str = LONG,
    primary: str = FVG_1M_TOUCH,
    confluence: tuple = (),
    bias_policy: BiasPolicy = BiasPolicy.IGNORE,
    tp_choice: TPChoice = None,
    sl_choice: SLChoice = None,
) -> StrategySpec:
    return StrategySpec(
        strategy_id=strategy_id,
        direction=direction,
        primary_entry_selector=primary,
        required_confluence_selectors=confluence,
        bias_policy=bias_policy,
        tp_choice=tp_choice if tp_choice is not None else tp_liq(),
        sl_choice=sl_choice if sl_choice is not None else sl_liq(),
    )


class SelectorHelperTests(unittest.TestCase):
    def test_fvg_selector_timeframe(self):
        self.assertEqual(selector_timeframe(FVG_5M_IFVG), "5m")
        self.assertEqual(selector_timeframe(FVG_1M_TOUCH), "1m")

    def test_liquidity_selector_timeframe(self):
        self.assertEqual(selector_timeframe(LIQ_SWING_HIGH_REVERSAL), "1m")

    def test_fvg_selector_direction_is_agnostic(self):
        self.assertIsNone(selector_fixed_direction(FVG_1M_TOUCH))
        self.assertIsNone(selector_fixed_direction(FVG_5M_IFVG))

    def test_liquidity_selector_direction_is_pinned(self):
        self.assertEqual(selector_fixed_direction(LIQ_SWING_HIGH_REVERSAL), SHORT)
        self.assertEqual(selector_fixed_direction(LIQ_SWING_LOW_REVERSAL), LONG)
        self.assertEqual(selector_fixed_direction(LIQ_SWING_HIGH_CONTINUATION), LONG)

    def test_malformed_selector_prefix_rejected(self):
        with self.assertRaises(ValueError):
            selector_timeframe("bogus-1m-x")

    def test_fvg_selector_wrong_token_count_rejected(self):
        with self.assertRaises(ValueError):
            selector_timeframe("fvg-1m-fvg_touch-extra")

    def test_fvg_selector_unsupported_timeframe_rejected(self):
        with self.assertRaises(ValueError):
            selector_timeframe("fvg-1H-fvg_touch")  # 1H never produces an FVG entry candidate

    def test_fvg_selector_unknown_source_type_rejected(self):
        with self.assertRaises(ValueError):
            selector_timeframe("fvg-1m-bogus_type")

    def test_liquidity_selector_unsupported_timeframe_rejected(self):
        with self.assertRaises(ValueError):
            selector_timeframe("liquidity-1H-swing_equal-swing_high-take-reversal")

    def test_liquidity_selector_unknown_entry_mode_rejected(self):
        with self.assertRaises(ValueError):
            selector_timeframe("liquidity-1m-swing_equal-swing_high-take-sideways")


class TPChoiceTests(unittest.TestCase):
    def test_liquidity_requires_rank_1_2_or_3(self):
        for rank in (1, 2, 3):
            TPChoice(source_type=TPSourceType.LIQUIDITY, rank=rank)  # no raise
        with self.assertRaises(ValueError):
            TPChoice(source_type=TPSourceType.LIQUIDITY, rank=None)
        with self.assertRaises(ValueError):
            TPChoice(source_type=TPSourceType.LIQUIDITY, rank=4)

    def test_data_wick_must_have_no_rank(self):
        TPChoice(source_type=TPSourceType.DATA_WICK)  # no raise
        with self.assertRaises(ValueError):
            TPChoice(source_type=TPSourceType.DATA_WICK, rank=1)

    def test_wrong_type_source_type_rejected(self):
        with self.assertRaises(ValueError):
            TPChoice(source_type="liquidity", rank=1)


class SLChoiceTests(unittest.TestCase):
    def test_liquidity_requires_rank_and_no_link(self):
        SLChoice(source_type=SLSourceType.LIQUIDITY, rank=2)  # no raise
        with self.assertRaises(ValueError):
            SLChoice(source_type=SLSourceType.LIQUIDITY, rank=None)
        with self.assertRaises(ValueError):
            SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1, fvg_link_selector=FVG_5M_IFVG)

    def test_fvg_boundary_requires_link_and_no_rank(self):
        SLChoice(source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector=FVG_5M_IFVG)  # no raise
        with self.assertRaises(ValueError):
            SLChoice(source_type=SLSourceType.FVG_BOUNDARY, rank=1, fvg_link_selector=FVG_5M_IFVG)
        with self.assertRaises(ValueError):
            SLChoice(source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector=None)

    def test_fvg_boundary_rejects_disallowed_timeframe(self):
        with self.assertRaises(ValueError):
            SLChoice(source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector=FVG_1M_TOUCH)

    def test_fvg_boundary_accepts_5m_and_15m(self):
        SLChoice(source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector=FVG_5M_IFVG)
        SLChoice(source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector=FVG_15M_TOUCH)

    def test_data_wick_rejected_as_sl_source_type(self):
        with self.assertRaises(ValueError):
            SLChoice(source_type=TPSourceType.DATA_WICK)


class StrategySpecValidationTests(unittest.TestCase):
    def test_valid_minimal_spec_constructs(self):
        s = spec()
        self.assertEqual(s.primary_entry_selector, FVG_1M_TOUCH)
        self.assertEqual(s.required_confluence_selectors, ())

    def test_primary_entry_is_mandatory(self):
        with self.assertRaises(ValueError):
            spec(primary="")

    def test_primary_must_not_duplicate_as_required_confluence(self):
        with self.assertRaises(ValueError):
            spec(primary=FVG_1M_TOUCH, confluence=(FVG_1M_TOUCH,))

    def test_required_confluence_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            spec(confluence=(FVG_5M_IFVG, FVG_5M_IFVG))

    def test_required_confluence_must_be_a_tuple(self):
        with self.assertRaises(ValueError):
            spec(confluence=[FVG_5M_IFVG])

    def test_long_strategy_rejects_short_pinned_primary(self):
        with self.assertRaises(ValueError):
            spec(direction=LONG, primary=LIQ_SWING_HIGH_REVERSAL)

    def test_short_strategy_rejects_long_pinned_primary(self):
        with self.assertRaises(ValueError):
            spec(direction=SHORT, primary=LIQ_SWING_LOW_REVERSAL)

    def test_long_strategy_accepts_long_pinned_primary(self):
        spec(direction=LONG, primary=LIQ_SWING_LOW_REVERSAL)  # no raise

    def test_direction_agnostic_fvg_primary_accepted_for_either_direction(self):
        spec(direction=LONG, primary=FVG_1M_TOUCH)
        spec(direction=SHORT, primary=FVG_1M_TOUCH)

    def test_invalid_direction_rejected(self):
        with self.assertRaises(ValueError):
            spec(direction="up")

    def test_bias_policy_must_be_enum_member(self):
        with self.assertRaises(ValueError):
            spec(bias_policy="ignore")

    def test_bias_policy_has_exactly_three_members(self):
        self.assertEqual(
            {p.value for p in BiasPolicy},
            {"ignore", "require_aligned", "require_neutral"},
        )

    def test_tp_choice_must_be_tp_choice_instance(self):
        with self.assertRaises(ValueError):
            spec(tp_choice=sl_liq())

    def test_sl_choice_must_be_sl_choice_instance(self):
        with self.assertRaises(ValueError):
            spec(sl_choice=tp_liq())

    def test_fvg_boundary_sl_must_link_to_primary_or_confluence(self):
        with self.assertRaises(ValueError):
            spec(primary=FVG_1M_TOUCH, confluence=(), sl_choice=sl_fvg(FVG_5M_IFVG))

    def test_fvg_boundary_sl_linked_to_primary_is_accepted(self):
        spec(primary=FVG_5M_IFVG, confluence=(), sl_choice=sl_fvg(FVG_5M_IFVG))  # no raise

    def test_fvg_boundary_sl_linked_to_required_confluence_is_accepted(self):
        spec(primary=FVG_1M_TOUCH, confluence=(FVG_5M_IFVG,), sl_choice=sl_fvg(FVG_5M_IFVG))  # no raise

    def test_strategy_id_must_be_non_empty(self):
        with self.assertRaises(ValueError):
            spec(strategy_id="")


class SerializationTests(unittest.TestCase):
    def test_to_dict_is_json_serializable_and_round_trips_values(self):
        s = spec(
            strategy_id="s-json",
            direction=SHORT,
            primary=FVG_1M_TOUCH,
            confluence=(FVG_5M_IFVG,),
            bias_policy=BiasPolicy.REQUIRE_NEUTRAL,
            tp_choice=tp_wick(),
            sl_choice=sl_fvg(FVG_5M_IFVG),
        )
        d = s.to_dict()
        raw = json.dumps(d)  # must not raise
        round_tripped = json.loads(raw)
        self.assertEqual(round_tripped["strategy_id"], "s-json")
        self.assertEqual(round_tripped["direction"], "short")
        self.assertEqual(round_tripped["required_confluence_selectors"], [FVG_5M_IFVG])
        self.assertEqual(round_tripped["bias_policy"], "require_neutral")
        self.assertEqual(round_tripped["tp_choice"], {"source_type": "data_wick", "rank": None})
        self.assertEqual(
            round_tripped["sl_choice"],
            {"source_type": "fvg_boundary", "rank": None, "fvg_link_selector": FVG_5M_IFVG},
        )


class GeneratorLazinessTests(unittest.TestCase):
    def _kwargs(self, **overrides):
        base = dict(
            directions=[LONG, SHORT],
            primary_entry_selectors=[FVG_1M_TOUCH, LIQ_SWING_LOW_REVERSAL],
            confluence_selector_pool=[FVG_5M_IFVG],
            max_required_confluences=1,
            bias_policies=[BiasPolicy.IGNORE, BiasPolicy.REQUIRE_ALIGNED],
            tp_choices=[tp_liq(1), tp_wick()],
            sl_choices=[sl_liq(1), sl_fvg(FVG_5M_IFVG)],
        )
        base.update(overrides)
        return base

    def test_is_a_generator_function(self):
        self.assertTrue(inspect.isgeneratorfunction(iter_strategy_specs))

    def test_call_returns_a_generator_not_a_list(self):
        result = iter_strategy_specs(**self._kwargs())
        self.assertIsInstance(result, types.GeneratorType)
        self.assertNotIsInstance(result, list)

    def test_next_can_be_pulled_one_at_a_time(self):
        result = iter_strategy_specs(**self._kwargs())
        first = next(result)
        self.assertIsInstance(first, StrategySpec)

    def test_negative_max_required_confluences_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_strategy_specs(**self._kwargs(max_required_confluences=-1)))

    def test_empty_pool_with_positive_max_yields_only_empty_confluence(self):
        specs = list(iter_strategy_specs(**self._kwargs(confluence_selector_pool=(), max_required_confluences=3)))
        self.assertTrue(specs)
        for s in specs:
            self.assertEqual(s.required_confluence_selectors, ())

    def test_first_result_pulled_without_enumerating_the_full_confluence_combination_space(self):
        """Proves the fix for the eager confluence_combos materialization
        bug: a 30-selector pool with max_required_confluences=15 alone has
        C(30, 15) ~ 155 million size-15 subsets. If iter_strategy_specs
        still built the full combination set (any size) before yielding
        its first result, this would have to actually construct an
        enormous number of itertools.combinations items up front. We wrap
        the module's own `combinations` with a counting proxy and assert
        only a handful of items were ever produced to reach result #1."""
        real_combinations = itertools.combinations
        yielded = {"count": 0}

        def counting_combinations(iterable, r):
            for item in real_combinations(iterable, r):
                yielded["count"] += 1
                yield item

        large_pool = tuple(f"pool-item-{i}" for i in range(30))
        with mock.patch("tradelab.strategy_spec.combinations", counting_combinations):
            gen = iter_strategy_specs(
                directions=[LONG],
                primary_entry_selectors=[FVG_1M_TOUCH],
                confluence_selector_pool=large_pool,
                max_required_confluences=15,
                bias_policies=[BiasPolicy.IGNORE],
                tp_choices=[tp_liq(1)],
                sl_choices=[sl_liq(1)],
            )
            first = next(gen)

        self.assertEqual(first.required_confluence_selectors, ())
        # Only the trivial size-0 combination (the empty tuple) was needed
        # to reach the first result -- nowhere near the full C(30, 15)
        # (or even C(30, 1)) combination space.
        self.assertLess(yielded["count"], 10)


class GeneratorDeterminismTests(unittest.TestCase):
    def _kwargs(self):
        return dict(
            directions=[LONG, SHORT],
            primary_entry_selectors=[FVG_1M_TOUCH, LIQ_SWING_LOW_REVERSAL],
            confluence_selector_pool=[FVG_5M_IFVG],
            max_required_confluences=1,
            bias_policies=[BiasPolicy.IGNORE, BiasPolicy.REQUIRE_ALIGNED],
            tp_choices=[tp_liq(1), tp_wick()],
            sl_choices=[sl_liq(1), sl_fvg(FVG_5M_IFVG)],
        )

    def test_repeated_calls_yield_identical_ordered_strategy_ids(self):
        ids_1 = [s.strategy_id for s in iter_strategy_specs(**self._kwargs())]
        ids_2 = [s.strategy_id for s in iter_strategy_specs(**self._kwargs())]
        self.assertEqual(ids_1, ids_2)
        self.assertTrue(len(ids_1) > 0)

    def test_no_duplicate_strategy_ids(self):
        ids = [s.strategy_id for s in iter_strategy_specs(**self._kwargs())]
        self.assertEqual(len(ids), len(set(ids)))

    def test_strategy_id_is_stable_readable_string(self):
        s = next(iter_strategy_specs(**self._kwargs()))
        self.assertIsInstance(s.strategy_id, str)
        self.assertTrue(s.strategy_id.startswith("strategy-"))


class GeneratorFullCoverageTests(unittest.TestCase):
    """A small, synthetic parameter space, brute-force-verified against
    StrategySpec's own validation (the same public contract the
    generator itself relies on) rather than re-deriving the rules."""

    def _spaces(self):
        directions = (LONG, SHORT)
        primaries = (FVG_1M_TOUCH, LIQ_SWING_LOW_REVERSAL)
        pool = (FVG_5M_IFVG,)
        max_required = 1
        bias_policies = (BiasPolicy.IGNORE, BiasPolicy.REQUIRE_ALIGNED)
        tp_choices = (tp_liq(1), tp_wick())
        sl_choices = (sl_liq(1), sl_fvg(FVG_5M_IFVG))
        return directions, primaries, pool, max_required, bias_policies, tp_choices, sl_choices

    def _expected_ids(self):
        directions, primaries, pool, max_required, bias_policies, tp_choices, sl_choices = self._spaces()
        confluence_combos = []
        for k in range(0, max_required + 1):
            confluence_combos.extend(itertools.combinations(pool, k))

        expected = set()
        for direction, primary, confluence, bias_policy, tp_choice, sl_choice in itertools.product(
            directions, primaries, confluence_combos, bias_policies, tp_choices, sl_choices
        ):
            if primary in confluence:
                continue
            try:
                s = StrategySpec(
                    strategy_id="probe",
                    direction=direction,
                    primary_entry_selector=primary,
                    required_confluence_selectors=confluence,
                    bias_policy=bias_policy,
                    tp_choice=tp_choice,
                    sl_choice=sl_choice,
                )
            except ValueError:
                continue
            expected.add((direction, primary, confluence, bias_policy, tp_choice.source_type, tp_choice.rank, sl_choice.source_type, sl_choice.rank, sl_choice.fvg_link_selector))
            del s
        return expected

    def test_generator_produces_exactly_every_valid_combination(self):
        directions, primaries, pool, max_required, bias_policies, tp_choices, sl_choices = self._spaces()
        generated = list(
            iter_strategy_specs(
                directions=directions,
                primary_entry_selectors=primaries,
                confluence_selector_pool=pool,
                max_required_confluences=max_required,
                bias_policies=bias_policies,
                tp_choices=tp_choices,
                sl_choices=sl_choices,
            )
        )
        self.assertTrue(generated)

        generated_keys = {
            (
                s.direction,
                s.primary_entry_selector,
                s.required_confluence_selectors,
                s.bias_policy,
                s.tp_choice.source_type,
                s.tp_choice.rank,
                s.sl_choice.source_type,
                s.sl_choice.rank,
                s.sl_choice.fvg_link_selector,
            )
            for s in generated
        }
        self.assertEqual(generated_keys, self._expected_ids())
        # every generated spec is independently valid (constructible again)
        for s in generated:
            StrategySpec(
                strategy_id=s.strategy_id,
                direction=s.direction,
                primary_entry_selector=s.primary_entry_selector,
                required_confluence_selectors=s.required_confluence_selectors,
                bias_policy=s.bias_policy,
                tp_choice=s.tp_choice,
                sl_choice=s.sl_choice,
            )


if __name__ == "__main__":
    unittest.main()
