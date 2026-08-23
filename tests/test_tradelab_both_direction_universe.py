from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tradelab.bias import BEARISH, BULLISH
from tradelab.both_direction_universe import discover_both_direction_universe
from tradelab.entry_signal_timeline import LONG, SHORT
from tradelab.entry_strategy_universe import (
    BiasRequirementSpec, EntryStrategyEquivalenceClass, EntryStrategySpec,
    EntryStrategyUniverseResult, ObservedEntry, PrimaryTriggerKind, PrimaryTriggerSpec,
)

UTC = timezone.utc


def _spec(direction: str, bias: BiasRequirementSpec | None = None) -> EntryStrategySpec:
    primary = PrimaryTriggerSpec(PrimaryTriggerKind.FVG_TOUCH, direction, "1m")
    return EntryStrategySpec(
        spec_id=f"fvg_touch|{direction}|1m|-|ctx=-|bias=1H:any,4H:any,1D:any",
        primary=primary, required_fvg_context_selectors=frozenset(),
        bias_requirement=bias or BiasRequirementSpec("any", "any", "any"),
    )


def _class(spec: EntryStrategySpec, entry: ObservedEntry) -> EntryStrategyEquivalenceClass:
    return EntryStrategyEquivalenceClass(spec.spec_id, (entry,), (spec,), spec, 1, 0)


class BothDirectionUniverseTests(unittest.TestCase):
    @staticmethod
    def _universe(long: EntryStrategySpec, short: EntryStrategySpec) -> EntryStrategyUniverseResult:
        t0 = datetime(2024, 6, 3, 14, tzinfo=UTC)
        return EntryStrategyUniverseResult(
            equivalence_classes=(
                _class(long, ObservedEntry("long", t0, LONG, object())),
                _class(short, ObservedEntry("short", t0.replace(minute=1), SHORT, object())),
            ),
            syntactically_enumerated_spec_count=2, eligible_spec_count=2,
            specs_rejected_by_yearly_eligibility_count=0, equivalence_class_count=2,
            spec_counts_by_primary_kind={}, spec_counts_by_direction={},
            spec_counts_by_execution_timeframe={}, spec_counts_by_bias_requirement_pattern={},
        )

    def test_pairs_direction_relative_mirrored_bias_not_same_literal_bias(self):
        long = _spec(LONG, BiasRequirementSpec(BULLISH, BEARISH, "neutral"))
        mirrored_short = _spec(SHORT, BiasRequirementSpec(BEARISH, BULLISH, "neutral"))
        same_literal_short = _spec(SHORT, BiasRequirementSpec(BULLISH, BEARISH, "neutral"))
        self.assertEqual(discover_both_direction_universe(self._universe(long, mirrored_short)).paired_spec_count, 1)
        self.assertEqual(discover_both_direction_universe(self._universe(long, same_literal_short)).paired_spec_count, 0)

    def test_pairs_symmetric_fvg_long_short_and_merges_entries(self):
        t0 = datetime(2024, 6, 3, 14, tzinfo=UTC)
        long, short = _spec(LONG), _spec(SHORT)
        universe = EntryStrategyUniverseResult(
            equivalence_classes=(
                _class(long, ObservedEntry("long", t0, LONG, object())),
                _class(short, ObservedEntry("short", t0.replace(minute=1), SHORT, object())),
            ),
            syntactically_enumerated_spec_count=2, eligible_spec_count=2,
            specs_rejected_by_yearly_eligibility_count=0, equivalence_class_count=2,
            spec_counts_by_primary_kind={}, spec_counts_by_direction={},
            spec_counts_by_execution_timeframe={}, spec_counts_by_bias_requirement_pattern={},
        )
        both = discover_both_direction_universe(universe)
        self.assertEqual((both.paired_spec_count, both.equivalence_class_count), (1, 1))
        result = both.equivalence_classes[0]
        self.assertEqual(result.representative_spec.long_spec.primary.direction, LONG)
        self.assertEqual(result.representative_spec.short_spec.primary.direction, SHORT)
        self.assertEqual([entry.entry_id for entry in result.observed_entries], ["long", "short"])

    def test_same_instant_opposing_entries_are_excluded_from_both_strategy(self):
        t0 = datetime(2024, 6, 3, 14, tzinfo=UTC)
        long, short = _spec(LONG), _spec(SHORT)
        universe = EntryStrategyUniverseResult(
            equivalence_classes=(
                _class(long, ObservedEntry("long", t0, LONG, object())),
                _class(short, ObservedEntry("short", t0, SHORT, object())),
            ),
            syntactically_enumerated_spec_count=2, eligible_spec_count=2,
            specs_rejected_by_yearly_eligibility_count=0, equivalence_class_count=2,
            spec_counts_by_primary_kind={}, spec_counts_by_direction={},
            spec_counts_by_execution_timeframe={}, spec_counts_by_bias_requirement_pattern={},
        )
        both = discover_both_direction_universe(universe)
        self.assertEqual(both.equivalence_classes[0].observed_entries, ())


class SharedEntryStreamTests(unittest.TestCase):
    """Pairing no longer copies an ObservedEntry per pair just to stamp the
    recipe's FVG link selectors onto it.

    The selectors are a property of the recipe, so they live on
    ``BothDirectionEquivalenceClass.fvg_link_selectors`` and are passed into
    execution. These tests pin the two things that change: the entries really
    are the shared, uncopied source objects, and each class still reports the
    selectors its entries would previously have carried."""

    @staticmethod
    def _universe_with_two_context_variants():
        t0 = datetime(2024, 6, 3, 14, tzinfo=UTC)
        # Two long recipes differing only in required FVG context, so both
        # resolve to distinct classes but identical entry payloads, and two
        # matching shorts. This is exactly the shape that makes an
        # identity-keyed cache dangerous if the key omitted the selectors.
        specs = []
        classes = []
        # The 5m primary always contributes "fvg-5m-fvg_touch"; the second
        # variant additionally requires a 15m context, so the two recipes end
        # up with genuinely different link-selector tuples.
        for index, context in enumerate((frozenset(), frozenset({"fvg-15m-fvg_touch"}))):
            for direction, entry_id, minute in ((LONG, f"long{index}", 0), (SHORT, f"short{index}", 1)):
                primary = PrimaryTriggerSpec(PrimaryTriggerKind.FVG_TOUCH, direction, "5m")
                spec = EntryStrategySpec(
                    spec_id=f"fvg_touch|{direction}|5m|-|ctx={sorted(context)}|bias=any",
                    primary=primary, required_fvg_context_selectors=context,
                    bias_requirement=BiasRequirementSpec("any", "any", "any"),
                )
                specs.append(spec)
                classes.append(_class(spec, ObservedEntry(entry_id, t0.replace(minute=minute), direction, object())))
        return EntryStrategyUniverseResult(
            equivalence_classes=tuple(classes),
            syntactically_enumerated_spec_count=len(specs), eligible_spec_count=len(specs),
            specs_rejected_by_yearly_eligibility_count=0, equivalence_class_count=len(classes),
            spec_counts_by_primary_kind={}, spec_counts_by_direction={},
            spec_counts_by_execution_timeframe={}, spec_counts_by_bias_requirement_pattern={},
        )

    def test_recipes_with_different_link_selectors_are_kept_distinct_on_the_class(self):
        universe = self._universe_with_two_context_variants()
        both = discover_both_direction_universe(universe)
        self.assertEqual(both.paired_spec_count, 2)
        # Two recipes, two different selector sets -- the distinction the old
        # per-entry stamping carried is preserved, just at class level.
        self.assertEqual(len({c.fvg_link_selectors for c in both.equivalence_classes}), 2)

    def test_entries_are_the_shared_source_objects_not_per_pair_copies(self):
        universe = self._universe_with_two_context_variants()
        source_ids = {
            id(entry)
            for equivalence_class in universe.equivalence_classes
            for entry in equivalence_class.observed_entries
        }
        both = discover_both_direction_universe(universe)
        paired_entries = [
            entry
            for equivalence_class in both.equivalence_classes
            for entry in equivalence_class.long_entries + equivalence_class.short_entries
        ]
        self.assertGreater(len(paired_entries), 0)
        # Identity, not equality: a copy per pair is exactly what this change
        # removed, and it is invisible to an equality check.
        for entry in paired_entries:
            self.assertIn(id(entry), source_ids)

    def test_repeated_discovery_is_deterministic(self):
        universe = self._universe_with_two_context_variants()
        first = discover_both_direction_universe(universe)
        second = discover_both_direction_universe(universe)
        self.assertEqual(
            [(c.class_id, [e.entry_id for e in c.observed_entries], c.fvg_link_selectors) for c in first.equivalence_classes],
            [(c.class_id, [e.entry_id for e in c.observed_entries], c.fvg_link_selectors) for c in second.equivalence_classes],
        )
