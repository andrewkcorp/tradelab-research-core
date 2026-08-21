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
