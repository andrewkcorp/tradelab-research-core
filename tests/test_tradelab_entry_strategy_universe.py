#!/usr/bin/env python3
"""Entry strategy universe tests: all 5 primary trigger kinds, all 3
execution timeframes, all 4^3 = 64 HTF bias requirement patterns, empty
and multi-selector FVG context subsets, no look-ahead, the 2024/2025
eligibility boundary, exact equivalence-class deduplication (identical
sequences merge, distinct ones never do), sweep->FVG/iFVG setups using
only SweepFVGSetupEvent (never a raw historical sweep), and a randomized
small-fixture comparison against a straightforward brute-force reference
enumerator."""
from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta, timezone
from itertools import combinations, product

from tradelab.bias import BEARISH as BIAS_BEARISH
from tradelab.bias import BULLISH as BIAS_BULLISH
from tradelab.bias import NEUTRAL as BIAS_NEUTRAL
from tradelab.bias import HTFBiasResult, TimeframeBiasChange
from tradelab.entry_signal_timeline import LONG, SHORT, EntryEventKind, iter_candidate_entry_snapshots
from tradelab.entry_strategy_universe import (
    ANY_BIAS,
    DEFAULT_MIN_ENTRIES_PER_YEAR,
    EXECUTION_TIMEFRAMES,
    FVG_CONTEXT_SELECTOR_UNIVERSE,
    SYNTACTIC_SPEC_COUNT,
    BiasRequirementSpec,
    EntryStrategySpec,
    PrimaryTriggerKind,
    PrimaryTriggerSpec,
    _ALL_BIAS_REQUIREMENTS,
    _ALL_CONTEXT_SUBSETS,
    discover_entry_strategy_universe,
)
from tradelab.fvg import BEARISH as FVG_BEARISH
from tradelab.fvg import BULLISH as FVG_BULLISH
from tradelab.fvg import FVGDetectionResult, FVGStatus, FVGZone
from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, FVGEntryCandidateResult, SourceType
from tradelab.liquidity import HIGH, LOW, LiquidityType
from tradelab.liquidity_entry import CandidatePhase, EntryMode, LiquidityEntryCandidate, LiquidityEntryCandidateResult, SourceFamily
from tradelab.sweep_fvg_setup import SweepFVGSetupEvent

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)

_FVG_DIRECTION = {LONG: FVG_BULLISH, SHORT: FVG_BEARISH}
_TYPE_SIDE = {LiquidityType.SWING_HIGH: HIGH, LiquidityType.SWING_LOW: LOW}
_ENTRY_DIRECTION = {
    (HIGH, EntryMode.REVERSAL): SHORT,
    (HIGH, EntryMode.CONTINUATION): LONG,
    (LOW, EntryMode.REVERSAL): LONG,
    (LOW, EntryMode.CONTINUATION): SHORT,
}


def fvg_zone(zone_id: str, timeframe: str, direction: str, formed_at: datetime) -> FVGZone:
    fvg_dir = _FVG_DIRECTION[direction]
    return FVGZone(
        zone_id=zone_id, timeframe=timeframe, direction=fvg_dir, lower=100.0, upper=101.5,
        candle_1_timestamp=formed_at - timedelta(minutes=3), candle_2_timestamp=formed_at - timedelta(minutes=2),
        candle_3_timestamp=formed_at - timedelta(minutes=1), formed_at=formed_at, status=FVGStatus.ACTIVE,
        resolved_at=None, produced_ifvg=False,
    )


def fvg_touch(zone_id: str, timeframe: str, direction: str, known_at: datetime) -> FVGEntryCandidate:
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{SourceType.FVG_TOUCH.value}-{zone_id}", source_type=SourceType.FVG_TOUCH, direction=direction,
        timeframe=timeframe, zone_id=zone_id, lower=100.0, upper=102.0, contact_type=FVGContactType.PARTIAL_TOUCH,
        timestamp=known_at, known_at=known_at,
    )


def fvg_ifvg(zone_id: str, timeframe: str, direction: str, known_at: datetime) -> FVGEntryCandidate:
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{SourceType.IFVG.value}-{zone_id}", source_type=SourceType.IFVG, direction=direction,
        timeframe=timeframe, zone_id=zone_id, lower=100.0, upper=102.0, contact_type=None, timestamp=known_at, known_at=known_at,
    )


def liquidity_candidate(
    level_id: str, timeframe: str, liquidity_type: LiquidityType = LiquidityType.SWING_HIGH,
    entry_mode: EntryMode = EntryMode.REVERSAL, known_at: datetime = T0,
) -> LiquidityEntryCandidate:
    side = _TYPE_SIDE[liquidity_type]
    direction = _ENTRY_DIRECTION[(side, entry_mode)]
    return LiquidityEntryCandidate(
        candidate_id=f"{timeframe}-{SourceFamily.SWING_EQUAL.value}-{level_id}-{CandidatePhase.TAKE.value}-{entry_mode.value}",
        source_family=SourceFamily.SWING_EQUAL, timeframe=timeframe, level_id=level_id, liquidity_type=liquidity_type, side=side,
        lower=100.0, upper=100.0, trigger_phase=CandidatePhase.TAKE, entry_mode=entry_mode, direction=direction,
        timestamp=known_at, known_at=known_at,
    )


def fvg_res(*zones: FVGZone, timeframe: str = "1m") -> dict[str, FVGDetectionResult]:
    by_tf: dict[str, list[FVGZone]] = {}
    for z in zones:
        by_tf.setdefault(z.timeframe, []).append(z)
    return {tf: FVGDetectionResult(zones=zs, events=[]) for tf, zs in by_tf.items()}


EMPTY_BIAS = HTFBiasResult(timeframe_changes={"1H": [], "4H": [], "1D": []})


def repeated_touch_candidates(direction: str, timeframe: str, year: int, count: int, id_prefix: str) -> tuple[list[FVGEntryCandidate], list[FVGZone]]:
    """``count`` distinct zones, each touched once, spread across ``year``
    (one per day starting Jan 1) -- enough real entries to clear the
    yearly eligibility bar deterministically."""
    candidates = []
    zones = []
    base = datetime(year, 1, 1, tzinfo=UTC)
    for i in range(count):
        known_at = base + timedelta(days=i, hours=12)
        zone_id = f"{id_prefix}-{i}"
        zones.append(fvg_zone(zone_id, timeframe, direction, known_at - timedelta(minutes=1)))
        candidates.append(fvg_touch(zone_id, timeframe, direction, known_at))
    return candidates, zones


class GrammarSizeTests(unittest.TestCase):
    def test_primary_trigger_catalog_has_exactly_3036_entries(self):
        from tradelab.entry_strategy_universe import ALL_PRIMARY_TRIGGERS

        self.assertEqual(len(ALL_PRIMARY_TRIGGERS), 12 + 432 + 2 * 432 * 3)
        self.assertEqual(len(set(ALL_PRIMARY_TRIGGERS)), len(ALL_PRIMARY_TRIGGERS))  # no duplicates

    def test_context_subset_universe_has_exactly_8_subsets_including_empty(self):
        self.assertEqual(len(_ALL_CONTEXT_SUBSETS), 8)
        self.assertIn(frozenset(), _ALL_CONTEXT_SUBSETS)
        self.assertIn(frozenset(FVG_CONTEXT_SELECTOR_UNIVERSE), _ALL_CONTEXT_SUBSETS)

    def test_all_4_cubed_bias_requirement_patterns_are_enumerable(self):
        self.assertEqual(len(_ALL_BIAS_REQUIREMENTS), 64)
        self.assertEqual(len({(p.tf_1h, p.tf_4h, p.tf_1d) for p in _ALL_BIAS_REQUIREMENTS}), 64)
        values = (ANY_BIAS, BIAS_BULLISH, BIAS_BEARISH, BIAS_NEUTRAL)
        expected = {(h1, h4, d1) for h1 in values for h4 in values for d1 in values}
        self.assertEqual({(p.tf_1h, p.tf_4h, p.tf_1d) for p in _ALL_BIAS_REQUIREMENTS}, expected)

    def test_syntactic_spec_count_is_the_fixed_structural_constant(self):
        self.assertEqual(SYNTACTIC_SPEC_COUNT, 3036 * 8 * 64)
        self.assertEqual(SYNTACTIC_SPEC_COUNT, 1_554_432)

    def test_no_short_strategy_is_silently_restricted_to_bearish_bias(self):
        """A short strategy requiring bullish 1D bias must be representable
        -- this module never applies a directional bias heuristic."""
        pattern = BiasRequirementSpec(tf_1h=ANY_BIAS, tf_4h=ANY_BIAS, tf_1d=BIAS_BULLISH)
        primary = PrimaryTriggerSpec(kind=PrimaryTriggerKind.FVG_TOUCH, direction=SHORT, timeframe="1m")
        spec = EntryStrategySpec(spec_id="x", primary=primary, required_fvg_context_selectors=frozenset(), bias_requirement=pattern)
        self.assertEqual(spec.primary.direction, SHORT)
        self.assertEqual(spec.bias_requirement.tf_1d, BIAS_BULLISH)


class PrimaryTriggerKindTests(unittest.TestCase):
    def test_both_yearly_eligibility_combines_natural_long_and_short_sides(self):
        candidates = []
        zones = []
        for year in (2024, 2025):
            base = datetime(year, 1, 1, tzinfo=UTC)
            for direction, count, prefix in ((LONG, 60, "long"), (SHORT, 40, "short")):
                for i in range(count):
                    known_at = base + timedelta(days=i, hours=12)
                    zone_id = f"{year}-{prefix}-{i}"
                    zones.append(fvg_zone(zone_id, "1m", direction, known_at - timedelta(minutes=1)))
                    candidates.append(fvg_touch(zone_id, "1m", direction, known_at))

        directional = discover_entry_strategy_universe(
            fvg_res(*zones), FVGEntryCandidateResult(candidates=candidates),
            LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
            min_entries_per_year=100,
        )
        combined = discover_entry_strategy_universe(
            fvg_res(*zones), FVGEntryCandidateResult(candidates=candidates),
            LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
            min_entries_per_year=100, pair_yearly_eligibility_for_both=True,
        )

        self.assertEqual(directional.eligible_spec_count, 0)
        self.assertGreater(combined.eligible_spec_count, 0)
        directions = {
            spec.primary.direction
            for equivalence_class in combined.equivalence_classes
            for spec in equivalence_class.member_specs
        }
        self.assertEqual(directions, {LONG, SHORT})

    def test_fvg_touch_primary_produces_eligible_entries(self):
        year = 2024
        candidates, zones = repeated_touch_candidates(LONG, "1m", year, DEFAULT_MIN_ENTRIES_PER_YEAR, "ta")
        candidates_2025, zones_2025 = repeated_touch_candidates(LONG, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "tb")

        result = discover_entry_strategy_universe(
            fvg_res(*zones, *zones_2025),
            FVGEntryCandidateResult(candidates=candidates + candidates_2025),
            LiquidityEntryCandidateResult(candidates=[]),
            EMPTY_BIAS,
        )
        kinds = {cls.representative_spec.primary.kind for cls in result.equivalence_classes}
        self.assertIn(PrimaryTriggerKind.FVG_TOUCH, kinds)

    def test_ifvg_primary_produces_eligible_entries(self):
        year = 2024
        base_2024 = datetime(2024, 1, 1, tzinfo=UTC)
        base_2025 = datetime(2025, 1, 1, tzinfo=UTC)
        candidates = []
        zones = []
        for i in range(DEFAULT_MIN_ENTRIES_PER_YEAR):
            known_at = base_2024 + timedelta(days=i, hours=12)
            zone_id = f"ia-{i}"
            zones.append(fvg_zone(zone_id, "5m", LONG, known_at - timedelta(minutes=5)))
            candidates.append(fvg_ifvg(zone_id, "5m", LONG, known_at))
        for i in range(DEFAULT_MIN_ENTRIES_PER_YEAR):
            known_at = base_2025 + timedelta(days=i, hours=12)
            zone_id = f"ib-{i}"
            zones.append(fvg_zone(zone_id, "5m", LONG, known_at - timedelta(minutes=5)))
            candidates.append(fvg_ifvg(zone_id, "5m", LONG, known_at))

        result = discover_entry_strategy_universe(
            fvg_res(*zones), FVGEntryCandidateResult(candidates=candidates), LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS
        )
        kinds = {cls.representative_spec.primary.kind for cls in result.equivalence_classes}
        self.assertIn(PrimaryTriggerKind.IFVG, kinds)

    def _liquidity_2024_2025(self, timeframe: str = "1m", liquidity_type: LiquidityType = LiquidityType.SWING_HIGH):
        candidates = []
        for year in (2024, 2025):
            base = datetime(year, 1, 1, tzinfo=UTC)
            for i in range(DEFAULT_MIN_ENTRIES_PER_YEAR):
                candidates.append(liquidity_candidate(f"lv-{year}-{i}", timeframe, liquidity_type, EntryMode.REVERSAL, base + timedelta(days=i, hours=12)))
        return candidates

    def test_liquidity_sweep_primary_produces_eligible_entries_with_full_selector_provenance(self):
        candidates = self._liquidity_2024_2025()
        result = discover_entry_strategy_universe(
            {}, FVGEntryCandidateResult(candidates=[]), LiquidityEntryCandidateResult(candidates=candidates), EMPTY_BIAS
        )
        sweep_classes = [c for c in result.equivalence_classes if c.representative_spec.primary.kind == PrimaryTriggerKind.LIQUIDITY_SWEEP]
        self.assertEqual(len(sweep_classes), 1)
        primary = sweep_classes[0].representative_spec.primary
        self.assertIsNotNone(primary.liquidity_selector)
        self.assertTrue(primary.liquidity_selector.startswith("liquidity-1m-"))
        self.assertEqual(primary.direction, SHORT)  # SWING_HIGH + REVERSAL -> short

    def test_sweep_fvg_touch_setup_and_sweep_ifvg_setup_primaries_produce_eligible_entries(self):
        # Two SEPARATE sweeps per day (different timeframes -> different
        # selectors, different pending-sweep slots -- tradelab.sweep_fvg_
        # setup consumes a pending sweep exactly once, so pairing both a
        # touch and an iFVG off the SAME sweep would starve whichever
        # arrives second): a 1m sweep paired with a compatible 5m FVG
        # touch, and a separate 5m sweep paired with a compatible 15m iFVG.
        liquidity_candidates = []
        fvg_candidates = []
        zones = []
        for year in (2024, 2025):
            base = datetime(year, 1, 1, tzinfo=UTC)
            for i in range(DEFAULT_MIN_ENTRIES_PER_YEAR):
                sweep_at = base + timedelta(days=i, hours=12)
                liquidity_candidates.append(liquidity_candidate(f"sv-{year}-{i}", "1m", LiquidityType.SWING_LOW, EntryMode.REVERSAL, sweep_at))
                touch_at = sweep_at + timedelta(minutes=1)
                zone_id = f"stz-{year}-{i}"
                zones.append(fvg_zone(zone_id, "5m", LONG, touch_at - timedelta(minutes=1)))
                fvg_candidates.append(fvg_touch(zone_id, "5m", LONG, touch_at))

                ifvg_sweep_at = sweep_at + timedelta(minutes=30)
                liquidity_candidates.append(liquidity_candidate(f"iv-{year}-{i}", "5m", LiquidityType.SWING_LOW, EntryMode.REVERSAL, ifvg_sweep_at))
                ifvg_at = ifvg_sweep_at + timedelta(minutes=1)
                izone_id = f"siz-{year}-{i}"
                zones.append(fvg_zone(izone_id, "15m", LONG, ifvg_at - timedelta(minutes=1)))
                fvg_candidates.append(fvg_ifvg(izone_id, "15m", LONG, ifvg_at))

        result = discover_entry_strategy_universe(
            fvg_res(*zones), FVGEntryCandidateResult(candidates=fvg_candidates),
            LiquidityEntryCandidateResult(candidates=liquidity_candidates), EMPTY_BIAS,
        )
        kinds = {cls.representative_spec.primary.kind for cls in result.equivalence_classes}
        self.assertIn(PrimaryTriggerKind.SWEEP_FVG_TOUCH_SETUP, kinds)
        self.assertIn(PrimaryTriggerKind.SWEEP_IFVG_SETUP, kinds)
        for cls in result.equivalence_classes:
            if cls.representative_spec.primary.kind in (PrimaryTriggerKind.SWEEP_FVG_TOUCH_SETUP, PrimaryTriggerKind.SWEEP_IFVG_SETUP):
                for entry in cls.observed_entries:
                    self.assertIsInstance(entry.source, SweepFVGSetupEvent)
                    self.assertIn("::", entry.entry_id)  # setup_id, never a bare raw sweep candidate_id


class ExecutionTimeframeTests(unittest.TestCase):
    def test_all_3_execution_timeframes_are_representable_for_fvg_touch(self):
        for tf in EXECUTION_TIMEFRAMES:
            with self.subTest(tf=tf):
                candidates, zones = repeated_touch_candidates(LONG, tf, 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, f"z24-{tf}")
                candidates2, zones2 = repeated_touch_candidates(LONG, tf, 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, f"z25-{tf}")
                result = discover_entry_strategy_universe(
                    fvg_res(*zones, *zones2), FVGEntryCandidateResult(candidates=candidates + candidates2),
                    LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
                )
                tfs_seen = {cls.representative_spec.primary.timeframe for cls in result.equivalence_classes}
                self.assertIn(tf, tfs_seen)


class FVGContextSubsetTests(unittest.TestCase):
    def test_empty_and_full_context_subsets_are_both_discoverable(self):
        """A zone active the whole time on every execution timeframe, plus
        a primary touch elsewhere -- the empty-context spec and the full-
        context spec must both survive (their observed sequence is
        identical here, since the context is always present -- they must
        land in the SAME equivalence class, both retained)."""
        candidates = []
        zones = []
        for year in (2024, 2025):
            base = datetime(year, 1, 1, tzinfo=UTC)
            for tf in EXECUTION_TIMEFRAMES:
                zones.append(fvg_zone(f"ctx-{year}-{tf}", tf, LONG, base - timedelta(days=1)))
            for i in range(DEFAULT_MIN_ENTRIES_PER_YEAR):
                known_at = base + timedelta(days=i, hours=12)
                zone_id = f"prim-{year}-{i}"
                zones.append(fvg_zone(zone_id, "1m", LONG, known_at - timedelta(minutes=1)))
                candidates.append(fvg_touch(zone_id, "1m", LONG, known_at))

        result = discover_entry_strategy_universe(
            fvg_res(*zones), FVGEntryCandidateResult(candidates=candidates), LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS
        )
        matching = [c for c in result.equivalence_classes if c.representative_spec.primary.kind == PrimaryTriggerKind.FVG_TOUCH and c.representative_spec.primary.timeframe == "1m" and c.representative_spec.primary.direction == LONG]
        self.assertEqual(len(matching), 1)
        cls = matching[0]
        context_sets = {s.required_fvg_context_selectors for s in cls.member_specs}
        self.assertIn(frozenset(), context_sets)
        self.assertIn(frozenset(FVG_CONTEXT_SELECTOR_UNIVERSE), context_sets)
        # every one of the 8 possible subsets must be present, since the
        # context selectors are ALWAYS active -- every subset requirement holds
        self.assertEqual(len(context_sets), 8)

    def test_never_uses_raw_htf_fvg_zones_as_required_context(self):
        candidates, zones = repeated_touch_candidates(LONG, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, "h24")
        candidates2, zones2 = repeated_touch_candidates(LONG, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "h25")
        htf_zone = fvg_zone("htf-1", "1H", LONG, datetime(2023, 1, 1, tzinfo=UTC))

        result = discover_entry_strategy_universe(
            fvg_res(*zones, *zones2, htf_zone), FVGEntryCandidateResult(candidates=candidates + candidates2),
            LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
        )
        for cls in result.equivalence_classes:
            for spec in cls.member_specs:
                for selector in spec.required_fvg_context_selectors:
                    self.assertIn(selector, FVG_CONTEXT_SELECTOR_UNIVERSE)
                    self.assertNotIn("1H", selector)
                    self.assertNotIn("4H", selector)
                    self.assertNotIn("1D", selector)


class NoLookaheadTests(unittest.TestCase):
    def test_a_later_context_zone_never_grants_eligibility_to_an_earlier_entry(self):
        base = datetime(2024, 1, 1, tzinfo=UTC)
        candidates, zones = repeated_touch_candidates(LONG, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, "nl24")
        candidates2, zones2 = repeated_touch_candidates(LONG, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "nl25")
        # a 5m context zone that forms AFTER every 2024 touch but before every 2025 touch
        late_zone = fvg_zone("late-ctx", "5m", LONG, datetime(2024, 6, 1, tzinfo=UTC))

        result = discover_entry_strategy_universe(
            fvg_res(*zones, *zones2, late_zone), FVGEntryCandidateResult(candidates=candidates + candidates2),
            LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
        )
        matching = [
            c for c in result.equivalence_classes
            if c.representative_spec.primary.kind == PrimaryTriggerKind.FVG_TOUCH and c.representative_spec.primary.timeframe == "1m"
        ]
        for cls in matching:
            for spec in cls.member_specs:
                if "fvg-5m-fvg_touch" in spec.required_fvg_context_selectors:
                    # every entry using this requirement must be from AFTER the late zone formed (2025 side only)
                    for entry in cls.observed_entries:
                        self.assertGreaterEqual(entry.known_at, late_zone.formed_at)


class YearlyEligibilityBoundaryTests(unittest.TestCase):
    def test_exactly_99_entries_in_one_year_is_rejected(self):
        candidates_24, zones_24 = repeated_touch_candidates(LONG, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR - 1, "b24")
        candidates_25, zones_25 = repeated_touch_candidates(LONG, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "b25")
        result = discover_entry_strategy_universe(
            fvg_res(*zones_24, *zones_25), FVGEntryCandidateResult(candidates=candidates_24 + candidates_25),
            LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
        )
        self.assertEqual(result.equivalence_classes, ())
        self.assertEqual(result.eligible_spec_count, 0)

    def test_exactly_100_entries_in_both_years_is_accepted(self):
        candidates_24, zones_24 = repeated_touch_candidates(LONG, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, "e24")
        candidates_25, zones_25 = repeated_touch_candidates(LONG, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "e25")
        result = discover_entry_strategy_universe(
            fvg_res(*zones_24, *zones_25), FVGEntryCandidateResult(candidates=candidates_24 + candidates_25),
            LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
        )
        self.assertGreater(result.eligible_spec_count, 0)
        for cls in result.equivalence_classes:
            self.assertGreaterEqual(cls.entry_count_2024, DEFAULT_MIN_ENTRIES_PER_YEAR)
            self.assertGreaterEqual(cls.entry_count_2025, DEFAULT_MIN_ENTRIES_PER_YEAR)

    def test_entries_only_in_2023_and_2026_never_count_toward_2024_or_2025(self):
        base_2023 = datetime(2023, 1, 1, tzinfo=UTC)
        base_2026 = datetime(2026, 1, 1, tzinfo=UTC)
        candidates = []
        zones = []
        for i in range(300):
            for base, prefix in ((base_2023, "y23"), (base_2026, "y26")):
                known_at = base + timedelta(days=i, hours=12)
                zone_id = f"{prefix}-{i}"
                zones.append(fvg_zone(zone_id, "1m", LONG, known_at - timedelta(minutes=1)))
                candidates.append(fvg_touch(zone_id, "1m", LONG, known_at))
        result = discover_entry_strategy_universe(
            fvg_res(*zones), FVGEntryCandidateResult(candidates=candidates), LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS
        )
        self.assertEqual(result.equivalence_classes, ())


class EquivalenceClassDedupTests(unittest.TestCase):
    def test_identical_entry_sequences_deduplicate_but_retain_all_source_specs(self):
        """Requiring the ALWAYS-present context selector changes nothing
        about which events match -- both the empty-context spec and the
        always-present-context spec must land in the same class."""
        candidates, zones = repeated_touch_candidates(LONG, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, "d24")
        candidates2, zones2 = repeated_touch_candidates(LONG, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "d25")
        always_zone = fvg_zone("always", "5m", LONG, datetime(2023, 1, 1, tzinfo=UTC))

        result = discover_entry_strategy_universe(
            fvg_res(*zones, *zones2, always_zone), FVGEntryCandidateResult(candidates=candidates + candidates2),
            LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
        )
        matching = [
            c for c in result.equivalence_classes
            if c.representative_spec.primary.kind == PrimaryTriggerKind.FVG_TOUCH and c.representative_spec.primary.timeframe == "1m"
        ]
        self.assertEqual(len(matching), 1)
        cls = matching[0]
        self.assertGreaterEqual(len(cls.member_specs), 2)
        context_sets = {s.required_fvg_context_selectors for s in cls.member_specs}
        self.assertIn(frozenset(), context_sets)
        self.assertIn(frozenset({"fvg-5m-fvg_touch"}), context_sets)
        # deterministic representative
        self.assertEqual(cls.representative_spec, min(cls.member_specs, key=lambda s: s.spec_id))
        self.assertEqual(cls.class_id, cls.representative_spec.spec_id)

    def test_distinct_entry_sequences_never_merge(self):
        candidates_long, zones_long = repeated_touch_candidates(LONG, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, "long24")
        candidates_long2, zones_long2 = repeated_touch_candidates(LONG, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "long25")
        candidates_short, zones_short = repeated_touch_candidates(SHORT, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, "short24")
        candidates_short2, zones_short2 = repeated_touch_candidates(SHORT, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "short25")

        result = discover_entry_strategy_universe(
            fvg_res(*zones_long, *zones_long2, *zones_short, *zones_short2),
            FVGEntryCandidateResult(candidates=candidates_long + candidates_long2 + candidates_short + candidates_short2),
            LiquidityEntryCandidateResult(candidates=[]), EMPTY_BIAS,
        )
        sequences = [tuple(e.entry_id for e in c.observed_entries) for c in result.equivalence_classes]
        self.assertEqual(len(sequences), len(set(sequences)))  # every class has a UNIQUE sequence
        directions = {c.representative_spec.primary.direction for c in result.equivalence_classes}
        self.assertEqual(directions, {LONG, SHORT})


class OnProgressCallbackTests(unittest.TestCase):
    """task 044: on_progress is purely observational -- it reports
    (primaries_processed, primaries_total) once per real primary in the
    discovery loop, in order, with no gaps or duplicates, and never
    changes the returned EntryStrategyUniverseResult in any way."""

    def test_callback_reports_every_primary_in_order_and_never_changes_the_result(self):
        candidates_long, zones_long = repeated_touch_candidates(LONG, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, "long24")
        candidates_long2, zones_long2 = repeated_touch_candidates(LONG, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "long25")
        candidates_short, zones_short = repeated_touch_candidates(SHORT, "1m", 2024, DEFAULT_MIN_ENTRIES_PER_YEAR, "short24")
        candidates_short2, zones_short2 = repeated_touch_candidates(SHORT, "1m", 2025, DEFAULT_MIN_ENTRIES_PER_YEAR, "short25")

        fvg_results = fvg_res(*zones_long, *zones_long2, *zones_short, *zones_short2)
        fvg_entry_result = FVGEntryCandidateResult(candidates=candidates_long + candidates_long2 + candidates_short + candidates_short2)
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=[])

        without_callback = discover_entry_strategy_universe(fvg_results, fvg_entry_result, liquidity_entry_result, EMPTY_BIAS)

        calls = []
        with_callback = discover_entry_strategy_universe(
            fvg_results, fvg_entry_result, liquidity_entry_result, EMPTY_BIAS,
            on_progress=lambda done, total: calls.append((done, total)),
        )

        self.assertGreater(len(calls), 0)
        totals = {total for _done, total in calls}
        self.assertEqual(len(totals), 1)  # the total never changes mid-loop
        total = totals.pop()
        self.assertEqual([done for done, _total in calls], list(range(1, total + 1)))  # exactly once per primary, in order, no gaps/dupes

        self.assertEqual(without_callback.equivalence_class_count, with_callback.equivalence_class_count)
        self.assertEqual(without_callback.eligible_spec_count, with_callback.eligible_spec_count)
        self.assertEqual(
            [c.class_id for c in without_callback.equivalence_classes], [c.class_id for c in with_callback.equivalence_classes]
        )


class BruteForceReferenceComparisonTests(unittest.TestCase):
    """Independently re-derives eligible-entry sequences for a small,
    randomly generated fixture via a naive O(primaries x cells x events)
    brute-force scan (no signature/lookup optimization), and checks
    agreement with ``discover_entry_strategy_universe``'s own,
    efficient result."""

    def test_matches_brute_force_on_a_small_random_fixture(self):
        rng = random.Random(90210)
        base_2024 = datetime(2024, 1, 1, tzinfo=UTC)
        base_2025 = datetime(2025, 1, 1, tzinfo=UTC)

        zones = []
        fvg_candidates = []
        liquidity_candidates = []

        # A handful of FVG zones/touches/iFVGs across timeframes/directions,
        # spread across BOTH eligibility years (alternating) so at least
        # some (kind, direction, timeframe) combinations recur in both.
        for i in range(80):
            tf = rng.choice(EXECUTION_TIMEFRAMES)
            direction = rng.choice([LONG, SHORT])
            base = base_2024 if i % 2 == 0 else base_2025
            known_at = base + timedelta(hours=(i // 2) * 3)
            zone_id = f"bz-{i}"
            zones.append(fvg_zone(zone_id, tf, direction, known_at - timedelta(minutes=1)))
            if rng.random() < 0.5:
                fvg_candidates.append(fvg_touch(zone_id, tf, direction, known_at))
            else:
                fvg_candidates.append(fvg_ifvg(zone_id, tf, direction, known_at))

        for i in range(30):
            tf = rng.choice(EXECUTION_TIMEFRAMES)
            liq_type = rng.choice([LiquidityType.SWING_HIGH, LiquidityType.SWING_LOW])
            mode = rng.choice([EntryMode.REVERSAL, EntryMode.CONTINUATION])
            base = base_2024 if i % 2 == 0 else base_2025
            known_at = base + timedelta(hours=(i // 2) * 5)
            liquidity_candidates.append(liquidity_candidate(f"bl-{i}", tf, liq_type, mode, known_at))

        fvg_entry_result = FVGEntryCandidateResult(candidates=fvg_candidates)
        liquidity_entry_result = LiquidityEntryCandidateResult(candidates=liquidity_candidates)
        fvg_results = fvg_res(*zones)

        MIN_PER_YEAR = 1  # tiny fixture -- lower the real 100/year bar for testability
        result = discover_entry_strategy_universe(fvg_results, fvg_entry_result, liquidity_entry_result, EMPTY_BIAS, min_entries_per_year=MIN_PER_YEAR)

        # --- naive brute-force reference ---
        market_state_by_id = {
            snap.event.event_id: snap.market_state
            for snap in iter_candidate_entry_snapshots(fvg_results, fvg_entry_result, liquidity_entry_result, EMPTY_BIAS)
        }
        from tradelab.entry_signal_timeline import build_entry_events
        from tradelab.sweep_fvg_setup import build_sweep_fvg_setup_events

        all_events = build_entry_events(fvg_entry_result, liquidity_entry_result)
        all_setups = build_sweep_fvg_setup_events(fvg_entry_result, liquidity_entry_result)

        # primaries actually present in this tiny fixture, derived directly (not via the module's own key-building code)
        primaries_present = set()
        for e in all_events:
            if e.kind == EntryEventKind.LIQUIDITY_SWEEP:
                primaries_present.add((PrimaryTriggerKind.LIQUIDITY_SWEEP, e.direction, e.timeframe, e.selector))
            else:
                kind = PrimaryTriggerKind.IFVG if e.kind == EntryEventKind.IFVG else PrimaryTriggerKind.FVG_TOUCH
                primaries_present.add((kind, e.direction, e.timeframe, None))
        for s in all_setups:
            kind = PrimaryTriggerKind.SWEEP_FVG_TOUCH_SETUP if s.fvg_kind == EntryEventKind.FVG_TOUCH else PrimaryTriggerKind.SWEEP_IFVG_SETUP
            primaries_present.add((kind, s.direction, s.fvg_timeframe, s.sweep_event.selector))

        brute_sequences: dict[tuple, tuple] = {}
        for kind, direction, timeframe, liquidity_selector in primaries_present:
            if kind == PrimaryTriggerKind.LIQUIDITY_SWEEP:
                matching = [
                    (e.event_id, e.known_at, e.direction, market_state_by_id[e.event_id])
                    for e in all_events if e.kind == EntryEventKind.LIQUIDITY_SWEEP and e.direction == direction and e.timeframe == timeframe and e.selector == liquidity_selector
                ]
            elif kind in (PrimaryTriggerKind.FVG_TOUCH, PrimaryTriggerKind.IFVG):
                want_kind = EntryEventKind.IFVG if kind == PrimaryTriggerKind.IFVG else EntryEventKind.FVG_TOUCH
                matching = [
                    (e.event_id, e.known_at, e.direction, market_state_by_id[e.event_id])
                    for e in all_events if e.kind == want_kind and e.direction == direction and e.timeframe == timeframe
                ]
            else:
                want_fvg_kind = EntryEventKind.FVG_TOUCH if kind == PrimaryTriggerKind.SWEEP_FVG_TOUCH_SETUP else EntryEventKind.IFVG
                matching = [
                    (s.setup_id, s.known_at, s.direction, market_state_by_id[s.fvg_event.event_id])
                    for s in all_setups if s.fvg_kind == want_fvg_kind and s.direction == direction and s.fvg_timeframe == timeframe and s.sweep_event.selector == liquidity_selector
                ]
            matching.sort(key=lambda t: (t[1], t[0]))

            for req in _ALL_CONTEXT_SUBSETS:
                for pattern in _ALL_BIAS_REQUIREMENTS:
                    seq = []
                    for entry_id, known_at, entry_direction, market_state in matching:
                        active = market_state.active_fvg_selectors_by_direction[entry_direction] & set(FVG_CONTEXT_SELECTOR_UNIVERSE)
                        if not req <= active:
                            continue
                        observed_states = tuple(market_state.bias.timeframe_states[tf2] for tf2 in ("1H", "4H", "1D"))
                        if not pattern.matches(observed_states):
                            continue
                        seq.append((entry_id, known_at, entry_direction))
                    count_2024 = sum(1 for e in seq if e[1].year == 2024)
                    count_2025 = sum(1 for e in seq if e[1].year == 2025)
                    if count_2024 < MIN_PER_YEAR or count_2025 < MIN_PER_YEAR:
                        continue
                    brute_sequences[(kind, direction, timeframe, liquidity_selector, req, (pattern.tf_1h, pattern.tf_4h, pattern.tf_1d))] = tuple(seq)

        module_sequences: dict[tuple, tuple] = {}
        for cls in result.equivalence_classes:
            for spec in cls.member_specs:
                key = (
                    spec.primary.kind, spec.primary.direction, spec.primary.timeframe, spec.primary.liquidity_selector,
                    spec.required_fvg_context_selectors, (spec.bias_requirement.tf_1h, spec.bias_requirement.tf_4h, spec.bias_requirement.tf_1d),
                )
                module_sequences[key] = tuple((e.entry_id, e.known_at, e.direction) for e in cls.observed_entries)

        self.assertEqual(module_sequences, brute_sequences)
        self.assertGreater(len(module_sequences), 0)


if __name__ == "__main__":
    unittest.main()
