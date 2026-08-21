"""Exact symmetric ``both`` direction strategies.

The existing entry universe has one directional primary per spec.  This
module pairs only the natural opposite side of the *same* recipe: FVG/iFVG
primaries share kind/timeframe/context/bias; liquidity and sweep+FVG setup
primaries share every field except the high/low liquidity side.  It never
forms a Cartesian product of unrelated long and short systems.

Each paired setup produces one merged chronological entry stream suitable for
``simulate_both_direction_market_execution``. Simultaneous opposing signals
are deliberately removed from that stream by its conservative conflict rule.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from tradelab.entry_signal_timeline import LONG, SHORT
from tradelab.entry_strategy_universe import (
    EntryStrategyEquivalenceClass,
    EntryStrategySpec,
    EntryStrategyUniverseResult,
    ObservedEntry,
    PrimaryTriggerKind,
    mirror_bias_requirement,
)
from tradelab.selector_catalog import opposite_liquidity_selector


@dataclass(frozen=True)
class BothDirectionSpec:
    spec_id: str
    long_spec: EntryStrategySpec
    short_spec: EntryStrategySpec


@dataclass(frozen=True)
class BothDirectionEquivalenceClass:
    class_id: str
    observed_entries: tuple[ObservedEntry, ...]
    # Preserve the two original streams as well as the conflict-free stream
    # used for equivalence grouping.  Execution needs the originals so it can
    # emit an explicit audit record when opposite signals share an instant.
    long_entries: tuple[ObservedEntry, ...]
    short_entries: tuple[ObservedEntry, ...]
    member_specs: tuple[BothDirectionSpec, ...]
    representative_spec: BothDirectionSpec
    entry_count_2024: int
    entry_count_2025: int
    fvg_link_selectors: tuple[str, ...] = ()


@dataclass(frozen=True)
class BothDirectionUniverseResult:
    equivalence_classes: tuple[BothDirectionEquivalenceClass, ...]
    paired_spec_count: int
    equivalence_class_count: int


def discover_both_direction_universe(universe: EntryStrategyUniverseResult) -> BothDirectionUniverseResult:
    """Pair every structurally symmetric eligible long/short recipe once."""
    class_by_spec_id: dict[str, EntryStrategyEquivalenceClass] = {}
    specs: list[EntryStrategySpec] = []
    for equivalence_class in universe.equivalence_classes:
        for spec in equivalence_class.member_specs:
            class_by_spec_id[spec.spec_id] = equivalence_class
            specs.append(spec)

    long_by_key: dict[tuple, EntryStrategySpec] = {}
    short_by_key: dict[tuple, EntryStrategySpec] = {}
    for spec in specs:
        key = _symmetric_key(spec)
        (long_by_key if spec.primary.direction == LONG else short_by_key)[key] = spec

    pairs: list[BothDirectionSpec] = []
    for key in sorted(set(long_by_key) & set(short_by_key), key=repr):
        long_spec, short_spec = long_by_key[key], short_by_key[key]
        pairs.append(BothDirectionSpec(
            spec_id=f"both|{long_spec.spec_id}|{short_spec.spec_id}", long_spec=long_spec, short_spec=short_spec,
        ))

    grouped: dict[tuple, list[BothDirectionSpec]] = {}
    entries_by_key: dict[tuple, tuple[ObservedEntry, ...]] = {}
    streams_by_key: dict[tuple, tuple[tuple[ObservedEntry, ...], tuple[ObservedEntry, ...]]] = {}
    for pair in pairs:
        fvg_link_selectors = _fvg_link_selectors(pair.long_spec)
        long_entries = tuple(
            replace(entry, strategy_fvg_link_selectors=fvg_link_selectors)
            for entry in class_by_spec_id[pair.long_spec.spec_id].observed_entries
        )
        short_entries = tuple(
            replace(entry, strategy_fvg_link_selectors=fvg_link_selectors)
            for entry in class_by_spec_id[pair.short_spec.spec_id].observed_entries
        )
        entries = _merge_for_key(long_entries, short_entries)
        sequence_key = (
            tuple((entry.entry_id, entry.known_at, entry.direction) for entry in entries),
            fvg_link_selectors,
        )
        grouped.setdefault(sequence_key, []).append(pair)
        entries_by_key[sequence_key] = entries
        streams_by_key[sequence_key] = (long_entries, short_entries)

    classes: list[BothDirectionEquivalenceClass] = []
    for sequence_key, members in grouped.items():
        ordered_members = tuple(sorted(members, key=lambda spec: spec.spec_id))
        entries = entries_by_key[sequence_key]
        long_entries, short_entries = streams_by_key[sequence_key]
        classes.append(BothDirectionEquivalenceClass(
            class_id=ordered_members[0].spec_id, observed_entries=entries,
            long_entries=long_entries, short_entries=short_entries, member_specs=ordered_members,
            representative_spec=ordered_members[0],
            entry_count_2024=sum(1 for entry in entries if entry.known_at.year == 2024),
            entry_count_2025=sum(1 for entry in entries if entry.known_at.year == 2025),
            fvg_link_selectors=_fvg_link_selectors(ordered_members[0].long_spec),
        ))
    classes.sort(key=lambda value: value.class_id)
    return BothDirectionUniverseResult(tuple(classes), len(pairs), len(classes))


def _symmetric_key(spec: EntryStrategySpec) -> tuple:
    primary = spec.primary
    if primary.kind in (PrimaryTriggerKind.FVG_TOUCH, PrimaryTriggerKind.IFVG):
        primary_key = (primary.kind.value, primary.timeframe, None)
    else:
        assert primary.liquidity_selector is not None
        opposite = opposite_liquidity_selector(primary.liquidity_selector)
        primary_key = (primary.kind.value, primary.timeframe, min(primary.liquidity_selector, opposite))
    bias = spec.bias_requirement if spec.primary.direction == LONG else mirror_bias_requirement(spec.bias_requirement)
    return (
        primary_key,
        tuple(sorted(spec.required_fvg_context_selectors)),
        bias.tf_1h,
        bias.tf_4h,
        bias.tf_1d,
    )


def _fvg_link_selectors(spec: EntryStrategySpec) -> tuple[str, ...]:
    """Exact 5m/15m FVG selectors this recipe is allowed to use for SL."""
    selectors = {
        selector for selector in spec.required_fvg_context_selectors
        if selector.startswith(("fvg-5m-", "fvg-15m-"))
    }
    primary = spec.primary
    kind_by_primary = {
        PrimaryTriggerKind.FVG_TOUCH: "fvg_touch",
        PrimaryTriggerKind.IFVG: "ifvg",
        PrimaryTriggerKind.SWEEP_FVG_TOUCH_SETUP: "fvg_touch",
        PrimaryTriggerKind.SWEEP_IFVG_SETUP: "ifvg",
    }
    kind = kind_by_primary.get(primary.kind)
    if kind is not None and primary.timeframe in ("5m", "15m"):
        selectors.add(f"fvg-{primary.timeframe}-{kind}")
    return tuple(sorted(selectors))


def _merge_for_key(long_entries: tuple[ObservedEntry, ...], short_entries: tuple[ObservedEntry, ...]) -> tuple[ObservedEntry, ...]:
    """Same conservative conflict policy as execution, without importing it.

    Keeping this private duplicate avoids a circular dependency from universe
    discovery back into the execution engine. The tiny rule is intentionally
    repeated and covered by tests at both boundaries.
    """
    by_time: dict[datetime, list[ObservedEntry]] = {}
    for entry in long_entries + short_entries:
        by_time.setdefault(entry.known_at, []).append(entry)
    merged: list[ObservedEntry] = []
    for known_at, entries in sorted(by_time.items()):
        if {entry.direction for entry in entries} == {LONG, SHORT}:
            continue
        merged.extend(sorted(entries, key=lambda entry: entry.entry_id))
    return tuple(merged)
