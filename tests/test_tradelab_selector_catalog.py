#!/usr/bin/env python3
"""Canonical entry selector catalog tests: exact counts (6 FVG / 432
liquidity / 438 total, no duplicates), every catalog entry passes
StrategySpec construction, semantically-impossible selectors (a
swing/equal family paired with a session type, or vice versa) are
rejected by both is_canonical_selector and StrategySpec, every selector
a synthetic set of real FVGEntryCandidate/LiquidityEntryCandidate
objects actually produces (via the real build_entry_signal_timeline)
is exactly the canonical set -- neither more nor fewer -- a
cross-timeframe StrategySpec (primary on one timeframe, confluence on
others) remains valid, and the catalog's own construction order is
reproducible/deterministic."""
from __future__ import annotations

import itertools
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.entry_signal_timeline import build_entry_signal_timeline
from tradelab.fvg_entry import FVGEntryCandidate, FVGEntryCandidateResult
from tradelab.fvg_entry import SourceType as FVGSourceType
from tradelab.liquidity import HIGH, LOW, LiquidityType
from tradelab.liquidity_entry import CandidatePhase, EntryMode, LiquidityEntryCandidate, LiquidityEntryCandidateResult, SourceFamily
from tradelab.selector_catalog import (
    ALL_CANONICAL_ENTRY_SELECTORS,
    CANONICAL_FVG_SELECTORS,
    CANONICAL_LIQUIDITY_SELECTORS,
    is_canonical_selector,
    selector_fixed_direction,
    selector_timeframe,
)
from tradelab.sl_targets import SLSourceType
from tradelab.strategy_spec import LONG, SHORT, BiasPolicy, SLChoice, StrategySpec, TPChoice
from tradelab.tp_targets import TPSourceType

UTC = timezone.utc
T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)  # 09:30 NY -- well clear of the daily 17:00-18:00 NY break

_TIMEFRAMES = ("1m", "5m", "15m")
_SWING_EQUAL_TYPES = (LiquidityType.SWING_HIGH, LiquidityType.SWING_LOW, LiquidityType.EQUAL_HIGHS, LiquidityType.EQUAL_LOWS)
_SESSION_TYPES = tuple(lt for lt in LiquidityType if lt.value.startswith("session_"))
_TYPE_SIDE = {
    LiquidityType.SWING_HIGH: HIGH,
    LiquidityType.SWING_LOW: LOW,
    LiquidityType.EQUAL_HIGHS: HIGH,
    LiquidityType.EQUAL_LOWS: LOW,
}
for _lt in _SESSION_TYPES:
    _TYPE_SIDE[_lt] = HIGH if _lt.value.endswith("_high") else LOW


class CatalogCountTests(unittest.TestCase):
    def test_six_fvg_selectors(self):
        self.assertEqual(len(CANONICAL_FVG_SELECTORS), 6)
        self.assertEqual(len(set(CANONICAL_FVG_SELECTORS)), 6)

    def test_432_liquidity_selectors(self):
        self.assertEqual(len(CANONICAL_LIQUIDITY_SELECTORS), 432)
        self.assertEqual(len(set(CANONICAL_LIQUIDITY_SELECTORS)), 432)

    def test_438_total_selectors_no_overlap(self):
        self.assertEqual(len(ALL_CANONICAL_ENTRY_SELECTORS), 438)
        self.assertEqual(len(set(ALL_CANONICAL_ENTRY_SELECTORS)), 438)
        self.assertEqual(set(ALL_CANONICAL_ENTRY_SELECTORS), set(CANONICAL_FVG_SELECTORS) | set(CANONICAL_LIQUIDITY_SELECTORS))
        self.assertTrue(set(CANONICAL_FVG_SELECTORS).isdisjoint(CANONICAL_LIQUIDITY_SELECTORS))


def _minimal_strategy_for(selector: str, strategy_id: str) -> StrategySpec:
    direction = selector_fixed_direction(selector) or LONG
    return StrategySpec(
        strategy_id=strategy_id,
        direction=direction,
        primary_entry_selector=selector,
        required_confluence_selectors=(),
        bias_policy=BiasPolicy.IGNORE,
        tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY, rank=1),
        sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1),
    )


class EveryCatalogEntryIsAcceptedByStrategySpecTests(unittest.TestCase):
    def test_every_canonical_selector_constructs_a_valid_strategy_spec_as_primary(self):
        for index, selector in enumerate(ALL_CANONICAL_ENTRY_SELECTORS):
            with self.subTest(selector=selector):
                spec = _minimal_strategy_for(selector, f"s-{index}")
                self.assertEqual(spec.primary_entry_selector, selector)


class ImpossibleSelectorRejectionTests(unittest.TestCase):
    def test_swing_high_filed_under_session_family_is_rejected(self):
        selector = "liquidity-1m-session-swing_high-take-reversal"
        self.assertFalse(is_canonical_selector(selector))
        with self.assertRaises(ValueError):
            selector_timeframe(selector)
        with self.assertRaises(ValueError):
            _minimal_strategy_for(selector, "s-bad-1")

    def test_session_type_filed_under_swing_equal_family_is_rejected(self):
        selector = "liquidity-5m-swing_equal-session_london_high-take-reversal"
        self.assertFalse(is_canonical_selector(selector))
        with self.assertRaises(ValueError):
            selector_timeframe(selector)
        with self.assertRaises(ValueError):
            _minimal_strategy_for(selector, "s-bad-2")

    def test_every_swing_equal_type_rejects_session_family(self):
        for lt in _SWING_EQUAL_TYPES:
            selector = f"liquidity-1m-session-{lt.value}-take-reversal"
            with self.subTest(selector=selector):
                self.assertFalse(is_canonical_selector(selector))

    def test_every_session_type_rejects_swing_equal_family(self):
        for lt in _SESSION_TYPES:
            selector = f"liquidity-1m-swing_equal-{lt.value}-take-reversal"
            with self.subTest(selector=selector):
                self.assertFalse(is_canonical_selector(selector))

    def test_unknown_timeframe_rejected(self):
        self.assertFalse(is_canonical_selector("liquidity-1H-swing_equal-swing_high-take-reversal"))
        self.assertFalse(is_canonical_selector("fvg-4H-fvg_touch"))

    def test_unknown_trigger_phase_or_entry_mode_rejected(self):
        self.assertFalse(is_canonical_selector("liquidity-1m-swing_equal-swing_high-pending-reversal"))
        self.assertFalse(is_canonical_selector("liquidity-1m-swing_equal-swing_high-take-sideways"))


def _fvg_candidate(index: int, timeframe: str, source_type: FVGSourceType) -> FVGEntryCandidate:
    direction = LONG if index % 2 == 0 else SHORT
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{source_type.value}-zone-{index}",
        source_type=source_type,
        direction=direction,
        timeframe=timeframe,
        zone_id=f"zone-{index}",
        lower=100.0,
        upper=102.0,
        contact_type=None,
        timestamp=T0 - timedelta(minutes=1),
        known_at=T0,
    )


def _liquidity_candidate(
    index: int, timeframe: str, source_family: SourceFamily, liquidity_type: LiquidityType, trigger_phase: CandidatePhase, entry_mode: EntryMode
) -> LiquidityEntryCandidate:
    side = _TYPE_SIDE[liquidity_type]
    direction = {
        (HIGH, EntryMode.REVERSAL): SHORT,
        (HIGH, EntryMode.CONTINUATION): LONG,
        (LOW, EntryMode.REVERSAL): LONG,
        (LOW, EntryMode.CONTINUATION): SHORT,
    }[(side, entry_mode)]
    level_id = f"{timeframe}-{liquidity_type.value}-{index}"
    return LiquidityEntryCandidate(
        candidate_id=f"{timeframe}-{source_family.value}-{level_id}-{trigger_phase.value}-{entry_mode.value}",
        source_family=source_family,
        timeframe=timeframe,
        level_id=level_id,
        liquidity_type=liquidity_type,
        side=side,
        lower=100.0,
        upper=100.0,
        trigger_phase=trigger_phase,
        entry_mode=entry_mode,
        direction=direction,
        timestamp=T0,
        known_at=T0,
    )


class RealCandidateProducedSelectorsMatchCatalogTests(unittest.TestCase):
    """Builds one real FVGEntryCandidate/LiquidityEntryCandidate per
    catalog position (bypassing full bar-level detection, exactly like
    tests/test_tradelab_entry_signal_timeline.py's own factories do),
    feeds them through the real, public build_entry_signal_timeline, and
    proves the resulting selector set is exactly the canonical one --
    neither a selector the catalog claims but the real pipeline can't
    produce, nor one the real pipeline produces that the catalog
    doesn't recognize."""

    def test_synthetic_candidates_produce_exactly_the_canonical_selector_set(self):
        fvg_candidates = [
            _fvg_candidate(i, timeframe, source_type)
            for i, (timeframe, source_type) in enumerate(itertools.product(_TIMEFRAMES, FVGSourceType))
        ]

        pairs = [(SourceFamily.SWING_EQUAL, lt) for lt in _SWING_EQUAL_TYPES] + [(SourceFamily.SESSION, lt) for lt in _SESSION_TYPES]
        liquidity_candidates = [
            _liquidity_candidate(i, timeframe, source_family, liquidity_type, trigger_phase, entry_mode)
            for i, (timeframe, (source_family, liquidity_type), trigger_phase, entry_mode) in enumerate(
                itertools.product(_TIMEFRAMES, pairs, CandidatePhase, EntryMode)
            )
        ]
        self.assertEqual(len(fvg_candidates), 6)
        self.assertEqual(len(liquidity_candidates), 432)

        timeline = build_entry_signal_timeline(
            FVGEntryCandidateResult(candidates=fvg_candidates), LiquidityEntryCandidateResult(candidates=liquidity_candidates)
        )
        produced_selectors = {signal.selector for signal in timeline.signals}
        self.assertEqual(produced_selectors, set(ALL_CANONICAL_ENTRY_SELECTORS))

    def test_every_produced_selector_is_canonical(self):
        fvg_candidates = [_fvg_candidate(0, "1m", FVGSourceType.FVG_TOUCH)]
        liquidity_candidates = [
            _liquidity_candidate(0, "5m", SourceFamily.SESSION, LiquidityType.SESSION_LONDON_LOW, CandidatePhase.RECLAIM_CLOSE, EntryMode.CONTINUATION)
        ]
        timeline = build_entry_signal_timeline(
            FVGEntryCandidateResult(candidates=fvg_candidates), LiquidityEntryCandidateResult(candidates=liquidity_candidates)
        )
        for signal in timeline.signals:
            self.assertTrue(is_canonical_selector(signal.selector))


class CrossTimeframeStrategySpecTests(unittest.TestCase):
    def test_primary_1m_with_5m_and_15m_confluence_is_valid(self):
        spec = StrategySpec(
            strategy_id="s-cross-tf",
            direction=LONG,
            primary_entry_selector="fvg-1m-fvg_touch",
            required_confluence_selectors=("fvg-5m-ifvg", "liquidity-15m-swing_equal-swing_low-take-reversal"),
            bias_policy=BiasPolicy.IGNORE,
            tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY, rank=1),
            sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1),
        )
        self.assertEqual(spec.primary_entry_selector, "fvg-1m-fvg_touch")
        self.assertEqual(
            set(spec.required_confluence_selectors), {"fvg-5m-ifvg", "liquidity-15m-swing_equal-swing_low-take-reversal"}
        )

    def test_confluence_across_all_three_timeframes_with_a_liquidity_primary(self):
        spec = StrategySpec(
            strategy_id="s-cross-tf-2",
            direction=SHORT,
            primary_entry_selector="liquidity-1m-swing_equal-swing_high-take-reversal",
            required_confluence_selectors=("fvg-5m-fvg_touch", "liquidity-15m-session-session_overnight_high-take-reversal"),
            bias_policy=BiasPolicy.IGNORE,
            tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY, rank=1),
            sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1),
        )
        self.assertEqual({selector_timeframe(s) for s in (spec.primary_entry_selector, *spec.required_confluence_selectors)}, {"1m", "5m", "15m"})


class DeterministicOrderTests(unittest.TestCase):
    def test_fvg_catalog_order_matches_independent_reconstruction(self):
        expected = tuple(f"fvg-{tf}-{st.value}" for tf in _TIMEFRAMES for st in FVGSourceType)
        self.assertEqual(CANONICAL_FVG_SELECTORS, expected)

    def test_liquidity_catalog_order_matches_independent_reconstruction(self):
        pairs = [("swing_equal", lt.value) for lt in _SWING_EQUAL_TYPES] + [("session", lt.value) for lt in _SESSION_TYPES]
        expected = tuple(
            f"liquidity-{tf}-{family}-{lt}-{phase.value}-{mode.value}"
            for tf in _TIMEFRAMES
            for family, lt in pairs
            for phase in CandidatePhase
            for mode in EntryMode
        )
        self.assertEqual(CANONICAL_LIQUIDITY_SELECTORS, expected)

    def test_repeated_import_yields_the_identical_ordered_tuple(self):
        import importlib

        import tradelab.selector_catalog as module

        reloaded = importlib.reload(module)
        self.assertEqual(reloaded.ALL_CANONICAL_ENTRY_SELECTORS, ALL_CANONICAL_ENTRY_SELECTORS)
        importlib.reload(module)  # restore a clean module state for any test that runs after this one


if __name__ == "__main__":
    unittest.main()
