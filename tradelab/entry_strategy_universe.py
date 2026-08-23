"""Entry strategy universe: the exhaustive generator of every valid
ENTRY-only strategy variant on top of the atomic ``EntryEvent`` /
persistent ``MarketStateSnapshot`` model (task 036) and the sweep -> FVG/
iFVG setup chains (task 038, ``tradelab.sweep_fvg_setup``).

**Entry logic only.** This module never simulates TP, SL, contract
sizing, fees, positions, P&L, or a full backtest -- see
``tradelab.strategy_spec``/``tradelab.tp_targets``/``tradelab.sl_targets``
for that, later, separate machinery (built on the *older* ``EntrySignal``
model, not reused here: this module's whole point is the exact,
one-shot-event observed-entry semantics task 036 introduced, which
``tradelab.strategy_spec`` predates).

**The strategy grammar, exactly**:

1. **One primary trigger** -- exactly one of five kinds:
   atomic FVG touch, atomic iFVG, atomic liquidity sweep, a sweep -> FVG-
   touch setup, or a sweep -> iFVG setup (``PrimaryTriggerKind``). Sweep-
   involving primaries are *only* ever built from ``tradelab.sweep_fvg_
   setup.build_sweep_fvg_setup_events`` -- this module never recreates,
   approximates, or re-derives that pairing logic itself.
2. **Zero or more required, persistent execution-FVG context selectors**
   -- any subset (including empty) of the *original*-FVG-zone selectors
   on the three execution timeframes, in the strategy's own direction
   (``fvg-1m-fvg_touch`` / ``fvg-5m-fvg_touch`` / ``fvg-15m-fvg_touch``
   -- never an ``-ifvg`` selector, which per task 036/038 is never
   persistent state at all). 2^3 = 8 possible subsets, always, no cap.
3. **Independent 1H/4H/1D bias requirements** -- each of the three HTF
   timeframes independently requires ``any``/``bullish``/``bearish``/
   ``neutral``. 4^3 = 64 possible patterns, always representable --
   including, deliberately, a ``short`` strategy that requires
   ``bullish`` 1D bias, or any other combination a naive "long needs
   bullish" heuristic would have silently excluded. This module never
   applies such a heuristic.

**Why 1H/4H/1D are bias-only, never direct FVG entry context.** This
package's own FVG detector runs on all six timeframes it supports
(``tradelab.fvg.SUPPORTED_TIMEFRAMES``), including 1H/4H/1D -- and those
HTF zones *do* appear in ``MarketStateSnapshot.active_fvg_selectors_by_
direction`` (task 036's own persistent-state pool is built from every
timeframe present in the ``fvg_results`` mapping it is given, with no
timeframe restriction of its own). This module explicitly intersects
that pool with only the three execution-timeframe selectors
(``FVG_CONTEXT_SELECTOR_UNIVERSE``) before ever treating anything as
"required FVG context" -- an HTF FVG zone can still influence a
strategy, but *only* through the bias state it already, separately,
feeds into (``tradelab.bias.compute_htf_bias``), never as a raw,
directly-selectable execution-context selector. Getting this wrong (by
skipping the intersection) would silently let a strategy require, say,
``fvg-1H-fvg_touch`` as execution context -- exactly what this task
explicitly forbids.

**Exact, no-look-ahead observed-entry semantics.** For a given primary
event or setup event, "eligible" means: at that event's own ``known_at``
(never any other instant), every required FVG-context selector is
actually present in ``MarketStateSnapshot.active_fvg_selectors_by_
direction[direction]`` (task 036's own, already-audited, no-look-ahead
persistent state), and every non-``any`` bias requirement matches
``MarketStateSnapshot.bias.timeframe_states`` at that same instant. This
module never recomputes state itself -- it joins each real primary/setup
event with the market state ``tradelab.entry_signal_timeline.iter_
candidate_entry_snapshots`` already computed for it (one shared
``event_id -> MarketStateSnapshot`` lookup, built once), so no-look-ahead
is inherited, not re-proven.

**Exhaustive enumeration, without duplicate backtests.** Every
syntactically valid ``(primary, required_fvg_context_selectors, bias_
requirement)`` combination is a distinct ``EntryStrategySpec`` --
3036 primaries (12 FVG touch/iFVG + 432 liquidity sweep + 2 x 432 x 3
sweep setups, the last two spanning every sweep-selector x FVG-execution-
timeframe pairing) x 8 context subsets x 64 bias patterns = **1,554,432**
total, a fixed structural constant, never a data-dependent count. Two
specs whose *observed*, ordered eligible-entry sequences (exact
``entry_id``/``known_at``/``direction`` triples) are identical are
grouped into one ``EntryStrategyEquivalenceClass`` for future
backtesting -- this is **exact deduplication, not approximation**: it
only merges specs that would, provably, produce the literal same trade
sequence (e.g. a redundant FVG-context requirement that happens to
always already hold whenever the primary fires, or a bias requirement
pinned to the one state that always holds), never specs whose sequences
merely *overlap* or are merely *similar*. Every original spec is kept
inside its class (``member_specs``) for full auditability -- nothing is
silently discarded. A deterministic representative (lexicographically
smallest ``spec_id``) is chosen per class.

**The annual 100-entries rule is an explicit research eligibility rule,
not a heuristic or a cap.** A class only enters the discovered universe
if its *observed* sequence has at least ``min_entries_per_year``
(business default: 100) eligible entries in each of the two complete
calendar years 2024 and 2025 (UTC half-open windows, matching
``tradelab.profile_research_context.PROFILE_WINDOWS`` exactly) -- a
strategy that barely ever fires is not researchable, this is not a
performance-driven truncation. ``min_entries_per_year`` is overridable
purely for small-fixture testing/diagnostics; production discovery must
use the default.

See ``docs/entry-strategy-universe-v1.md`` for the full architecture
writeup and real-data results.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations, product
from typing import Callable, Mapping

from tradelab.bias import BEARISH, BULLISH, HTFBiasResult, NEUTRAL
from tradelab.bias import SUPPORTED_TIMEFRAMES as _BIAS_TIMEFRAMES
from tradelab.entry_signal_timeline import LONG, SHORT, EntryEvent, EntryEventKind, MarketStateSnapshot, build_entry_events, iter_candidate_entry_snapshots
from tradelab.fvg import FVGDetectionResult, FVGZone
from tradelab.fvg_entry import FVGEntryCandidateResult
from tradelab.liquidity_entry import LiquidityEntryCandidateResult
from tradelab.selector_catalog import (
    CANONICAL_FVG_SELECTORS,
    CANONICAL_LIQUIDITY_SELECTORS,
    opposite_liquidity_selector,
    selector_fixed_direction,
    selector_timeframe,
)
from tradelab.sweep_fvg_setup import SweepFVGSetupEvent, build_sweep_fvg_setup_events

_UTC = timezone.utc
_DIRECTIONS = (LONG, SHORT)

# The three execution timeframes (spec, this task) -- never the HTF
# bias-only ones (1H/4H/1D), which participate only through bias.
EXECUTION_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m")

# Original-FVG-zone selectors only, on execution timeframes only -- never
# an -ifvg selector (never persistent state, see module docstring) and
# never an HTF (1H/4H/1D) one, even though those DO appear in
# MarketStateSnapshot.active_fvg_selectors_by_direction (see module
# docstring's own explanation of why this intersection is required).
FVG_CONTEXT_SELECTOR_UNIVERSE: tuple[str, ...] = tuple(
    s for s in CANONICAL_FVG_SELECTORS if selector_timeframe(s) in EXECUTION_TIMEFRAMES and s.endswith("-fvg_touch")
)
_FVG_CONTEXT_SELECTOR_UNIVERSE_SET = frozenset(FVG_CONTEXT_SELECTOR_UNIVERSE)

ANY_BIAS = "any"
_BIAS_REQUIREMENT_VALUES: tuple[str, ...] = (ANY_BIAS, BULLISH, BEARISH, NEUTRAL)
_BIAS_OBSERVED_STATES: tuple[str, ...] = (BULLISH, BEARISH, NEUTRAL)
_BIAS_TIMEFRAME_ORDER: tuple[str, ...] = ("1H", "4H", "1D")
if frozenset(_BIAS_TIMEFRAME_ORDER) != _BIAS_TIMEFRAMES:
    raise AssertionError(  # pragma: no cover -- would only fire if tradelab.bias's own HTF timeframes ever changed
        f"_BIAS_TIMEFRAME_ORDER {_BIAS_TIMEFRAME_ORDER!r} no longer matches tradelab.bias.SUPPORTED_TIMEFRAMES "
        f"({sorted(_BIAS_TIMEFRAMES)!r}) -- update this module"
    )

# The same half-open UTC calendar-year windows tradelab.profile_research_
# context.PROFILE_WINDOWS already uses -- never re-derived differently.
_YEAR_2024_START = datetime(2024, 1, 1, tzinfo=_UTC)
_YEAR_2025_START = datetime(2025, 1, 1, tzinfo=_UTC)
_YEAR_2026_START = datetime(2026, 1, 1, tzinfo=_UTC)

DEFAULT_MIN_ENTRIES_PER_YEAR = 100


class PrimaryTriggerKind(str, Enum):
    FVG_TOUCH = "fvg_touch"
    IFVG = "ifvg"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    SWEEP_FVG_TOUCH_SETUP = "sweep_fvg_touch_setup"
    SWEEP_IFVG_SETUP = "sweep_ifvg_setup"


_FVG_FAMILY_PRIMARY_KINDS = frozenset({PrimaryTriggerKind.FVG_TOUCH, PrimaryTriggerKind.IFVG})
_SETUP_PRIMARY_KINDS = frozenset({PrimaryTriggerKind.SWEEP_FVG_TOUCH_SETUP, PrimaryTriggerKind.SWEEP_IFVG_SETUP})


@dataclass(frozen=True, order=True)
class PrimaryTriggerSpec:
    """One primary-trigger definition -- full detector provenance,
    never just a bare selector string. ``timeframe`` is always the
    primary's *own* triggering timeframe: the FVG/iFVG's own execution
    timeframe for ``FVG_TOUCH``/``IFVG`` and both setup kinds, or the
    sweep's own timeframe for ``LIQUIDITY_SWEEP``. ``liquidity_selector``
    is the full canonical liquidity selector of whichever sweep is
    involved -- ``None`` for pure ``FVG_TOUCH``/``IFVG`` (which have no
    liquidity provenance at all); for a setup kind, this is the *sweep's*
    selector, and ``timeframe`` is separately the *FVG/iFVG's* own
    timeframe -- both fields together are exactly "the sweep selector
    and FVG/iFVG timeframe/kind" this task's own grammar requires (kind
    itself is ``self.kind``)."""

    kind: PrimaryTriggerKind
    direction: str
    timeframe: str
    liquidity_selector: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"PrimaryTriggerSpec.direction must be one of {_DIRECTIONS}, got {self.direction!r}")
        if self.timeframe not in EXECUTION_TIMEFRAMES:
            raise ValueError(f"PrimaryTriggerSpec.timeframe must be one of {EXECUTION_TIMEFRAMES}, got {self.timeframe!r}")
        if self.kind in _FVG_FAMILY_PRIMARY_KINDS:
            if self.liquidity_selector is not None:
                raise ValueError(f"PrimaryTriggerSpec.liquidity_selector must be None for {self.kind.value}")
        else:
            if self.liquidity_selector is None:
                raise ValueError(f"PrimaryTriggerSpec.liquidity_selector is required for {self.kind.value}")
            if self.liquidity_selector not in CANONICAL_LIQUIDITY_SELECTORS:
                raise ValueError(f"PrimaryTriggerSpec.liquidity_selector {self.liquidity_selector!r} is not a canonical liquidity selector")
            if selector_fixed_direction(self.liquidity_selector) != self.direction:
                raise ValueError(
                    f"PrimaryTriggerSpec.direction {self.direction!r} does not match liquidity_selector "
                    f"{self.liquidity_selector!r}'s own fixed direction"
                )


@dataclass(frozen=True, order=True)
class BiasRequirementSpec:
    """Independent per-HTF-timeframe requirement -- each of ``tf_1h``/
    ``tf_4h``/``tf_1d`` is one of ``ANY_BIAS``/``BULLISH``/``BEARISH``/
    ``NEUTRAL``, entirely independently (this module never restricts
    ``long`` to bullish-only or ``short`` to bearish-only requirements --
    see module docstring)."""

    tf_1h: str
    tf_4h: str
    tf_1d: str

    def __post_init__(self) -> None:
        for field_name, value in (("tf_1h", self.tf_1h), ("tf_4h", self.tf_4h), ("tf_1d", self.tf_1d)):
            if value not in _BIAS_REQUIREMENT_VALUES:
                raise ValueError(f"BiasRequirementSpec.{field_name} must be one of {_BIAS_REQUIREMENT_VALUES}, got {value!r}")

    def matches(self, observed_states: tuple[str, str, str]) -> bool:
        return all(
            requirement == ANY_BIAS or requirement == observed
            for requirement, observed in zip((self.tf_1h, self.tf_4h, self.tf_1d), observed_states)
        )

    def as_label(self) -> str:
        return f"1H:{self.tf_1h},4H:{self.tf_4h},1D:{self.tf_1d}"


def mirror_bias_requirement(pattern: BiasRequirementSpec) -> BiasRequirementSpec:
    """Direction-relative mirror used by one coherent ``both`` strategy.

    A bullish requirement for the long side corresponds to a bearish
    requirement for the short side. ``any`` and ``neutral`` are symmetric.
    """
    opposite = {ANY_BIAS: ANY_BIAS, BULLISH: BEARISH, BEARISH: BULLISH, NEUTRAL: NEUTRAL}
    return BiasRequirementSpec(
        tf_1h=opposite[pattern.tf_1h],
        tf_4h=opposite[pattern.tf_4h],
        tf_1d=opposite[pattern.tf_1d],
    )


def _build_spec_id(primary: PrimaryTriggerSpec, required_fvg_context_selectors: frozenset[str], bias_requirement: BiasRequirementSpec) -> str:
    context_label = ",".join(sorted(required_fvg_context_selectors)) if required_fvg_context_selectors else "-"
    liquidity_label = primary.liquidity_selector or "-"
    return (
        f"{primary.kind.value}|{primary.direction}|{primary.timeframe}|{liquidity_label}"
        f"|ctx={context_label}|bias={bias_requirement.as_label()}"
    )


@dataclass(frozen=True)
class EntryStrategySpec:
    """One canonical, immutable entry-strategy definition -- see module
    docstring for the full grammar. ``spec_id`` is deterministic and
    human-readable, built by :func:`_build_spec_id` from the other
    fields (never independently chosen)."""

    spec_id: str
    primary: PrimaryTriggerSpec
    required_fvg_context_selectors: frozenset[str]
    bias_requirement: BiasRequirementSpec

    def __post_init__(self) -> None:
        extra = self.required_fvg_context_selectors - _FVG_CONTEXT_SELECTOR_UNIVERSE_SET
        if extra:
            raise ValueError(
                f"EntryStrategySpec.required_fvg_context_selectors contains non-execution-timeframe or non-original-FVG "
                f"selector(s) {sorted(extra)} -- must be a subset of {FVG_CONTEXT_SELECTOR_UNIVERSE}"
            )


@dataclass(frozen=True)
class ObservedEntry:
    """One real, exact eligible entry -- either an ``EntryEvent`` (FVG
    touch, iFVG, or liquidity sweep) or a ``SweepFVGSetupEvent``, kept in
    full as ``source`` for complete provenance/audit."""

    entry_id: str
    known_at: datetime
    direction: str
    source: EntryEvent | SweepFVGSetupEvent
    # Legacy/eager representation kept for small fixtures and backwards
    # compatibility.  Real discovery must prefer the two shared references
    # below: copying every active FVGZone object into every observed event
    # makes a dense 1m dataset consume tens of GB before simulation begins.
    active_fvg_zones_by_selector: Mapping[str, tuple[FVGZone, ...]] | None = None
    active_fvg_zone_ids_by_selector: Mapping[str, tuple[str, ...]] | None = None
    fvg_zone_by_id: Mapping[str, FVGZone] | None = None
    strategy_fvg_link_selectors: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntryStrategyEquivalenceClass:
    """Every ``EntryStrategySpec`` whose *exact* observed, ordered entry
    sequence is identical, grouped for future backtesting without
    duplicate work -- see module docstring's "exact deduplication"
    paragraph. ``representative_spec`` is the deterministic (smallest
    ``spec_id``) choice among ``member_specs``; every original spec is
    kept, never discarded."""

    class_id: str
    observed_entries: tuple[ObservedEntry, ...]
    member_specs: tuple[EntryStrategySpec, ...]
    representative_spec: EntryStrategySpec
    entry_count_2024: int
    entry_count_2025: int


@dataclass(frozen=True)
class EntryStrategyUniverseResult:
    equivalence_classes: tuple[EntryStrategyEquivalenceClass, ...]
    syntactically_enumerated_spec_count: int
    eligible_spec_count: int
    specs_rejected_by_yearly_eligibility_count: int
    equivalence_class_count: int
    spec_counts_by_primary_kind: Mapping[str, int]
    spec_counts_by_direction: Mapping[str, int]
    spec_counts_by_execution_timeframe: Mapping[str, int]
    spec_counts_by_bias_requirement_pattern: Mapping[str, int]


def _fvg_family_primaries() -> tuple[PrimaryTriggerSpec, ...]:
    """The 12 fixed FVG_TOUCH/IFVG primaries: 3 execution timeframes x 2
    directions x 2 kinds. A selector is direction-agnostic (see
    ``tradelab.selector_catalog.selector_fixed_direction``), so both
    directions are always representable for each."""
    primaries = []
    for selector in CANONICAL_FVG_SELECTORS:
        timeframe = selector_timeframe(selector)
        if timeframe not in EXECUTION_TIMEFRAMES:
            continue
        kind = PrimaryTriggerKind.IFVG if selector.endswith("-ifvg") else PrimaryTriggerKind.FVG_TOUCH
        for direction in _DIRECTIONS:
            primaries.append(PrimaryTriggerSpec(kind=kind, direction=direction, timeframe=timeframe, liquidity_selector=None))
    return tuple(primaries)


def _liquidity_sweep_primaries() -> tuple[PrimaryTriggerSpec, ...]:
    """432 fixed LIQUIDITY_SWEEP primaries, one per canonical liquidity
    selector -- each pinned to its own single fixed direction."""
    return tuple(
        PrimaryTriggerSpec(
            kind=PrimaryTriggerKind.LIQUIDITY_SWEEP,
            direction=selector_fixed_direction(selector),
            timeframe=selector_timeframe(selector),
            liquidity_selector=selector,
        )
        for selector in CANONICAL_LIQUIDITY_SELECTORS
    )


def _sweep_setup_primaries() -> tuple[PrimaryTriggerSpec, ...]:
    """2 x 432 x 3 = 2592 fixed setup primaries: every canonical
    liquidity selector x every FVG execution timeframe x both setup
    kinds -- every sweep-timeframe x FVG/iFVG-timeframe pairing, per
    this task's own requirement."""
    primaries = []
    for selector in CANONICAL_LIQUIDITY_SELECTORS:
        direction = selector_fixed_direction(selector)
        for fvg_timeframe in EXECUTION_TIMEFRAMES:
            for kind in (PrimaryTriggerKind.SWEEP_FVG_TOUCH_SETUP, PrimaryTriggerKind.SWEEP_IFVG_SETUP):
                primaries.append(PrimaryTriggerSpec(kind=kind, direction=direction, timeframe=fvg_timeframe, liquidity_selector=selector))
    return tuple(primaries)


# The full, fixed primary-trigger catalog -- 12 + 432 + 2592 = 3036,
# never data-dependent (see module docstring).
ALL_PRIMARY_TRIGGERS: tuple[PrimaryTriggerSpec, ...] = _fvg_family_primaries() + _liquidity_sweep_primaries() + _sweep_setup_primaries()

# The 8 FVG-context subsets (powerset of the 3-element universe) and the
# 64 bias-requirement patterns -- both fixed, computed once.
_ALL_CONTEXT_SUBSETS: tuple[frozenset[str], ...] = tuple(
    frozenset(combo) for r in range(len(FVG_CONTEXT_SELECTOR_UNIVERSE) + 1) for combo in combinations(FVG_CONTEXT_SELECTOR_UNIVERSE, r)
)
_ALL_BIAS_REQUIREMENTS: tuple[BiasRequirementSpec, ...] = tuple(
    BiasRequirementSpec(tf_1h=h1, tf_4h=h4, tf_1d=d1)
    for h1 in _BIAS_REQUIREMENT_VALUES
    for h4 in _BIAS_REQUIREMENT_VALUES
    for d1 in _BIAS_REQUIREMENT_VALUES
)

SYNTACTIC_SPEC_COUNT: int = len(ALL_PRIMARY_TRIGGERS) * len(_ALL_CONTEXT_SUBSETS) * len(_ALL_BIAS_REQUIREMENTS)


def _build_context_subset_lookup() -> dict[frozenset[str], tuple[frozenset[str], ...]]:
    """For each of the 8 possible *observed* active-context sets S, the
    subsets of S that are themselves one of the 8 valid requirement
    subsets -- precomputed once (8x8 = 64 checks) so matching a real
    event's own S against every compatible requirement is an O(1) dict
    lookup, never recomputed per event."""
    return {S: tuple(req for req in _ALL_CONTEXT_SUBSETS if req <= S) for S in _ALL_CONTEXT_SUBSETS}


def _build_bias_pattern_lookup() -> dict[tuple[str, str, str], tuple[BiasRequirementSpec, ...]]:
    """For each of the 27 possible *observed* (1H, 4H, 1D) bias-state
    triples, the 8 requirement patterns compatible with it -- precomputed
    once (27x64 checks) for the same reason as ``_build_context_subset_
    lookup``."""
    observed_triples = tuple(product(_BIAS_OBSERVED_STATES, repeat=3))
    return {triple: tuple(pat for pat in _ALL_BIAS_REQUIREMENTS if pat.matches(triple)) for triple in observed_triples}


def _year_counts(observed_entries: list[ObservedEntry] | tuple[ObservedEntry, ...]) -> tuple[int, int]:
    count_2024 = sum(1 for o in observed_entries if _YEAR_2024_START <= o.known_at < _YEAR_2025_START)
    count_2025 = sum(1 for o in observed_entries if _YEAR_2025_START <= o.known_at < _YEAR_2026_START)
    return count_2024, count_2025


def _count_by(specs: list[EntryStrategySpec], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in specs:
        k = key(spec)
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def discover_entry_strategy_universe(
    fvg_results: Mapping[str, FVGDetectionResult],
    fvg_entry_result: FVGEntryCandidateResult,
    liquidity_entry_result: LiquidityEntryCandidateResult,
    htf_bias_result: HTFBiasResult,
    *,
    min_entries_per_year: int = DEFAULT_MIN_ENTRIES_PER_YEAR,
    require_yearly_eligibility: bool = True,
    pair_yearly_eligibility_for_both: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> EntryStrategyUniverseResult:
    """Enumerate the full, fixed 1,554,432-spec syntactic universe,
    determine each spec's exact, no-look-ahead observed eligible-entry
    sequence, normally keep only specs meeting ``min_entries_per_year`` in
    both 2024 and 2025 (the default, 100, is the real business rule), and group
    eligible specs by identical observed sequence into equivalence
    classes. One shared ``event_id -> MarketStateSnapshot`` lookup (built
    from ``iter_candidate_entry_snapshots``, already no-look-ahead and
    already ``O(N log N)``) is reused for every event and every setup
    (a setup's own market state is identical to its underlying FVG/iFVG
    event's own -- both share the exact same ``known_at``), so this
    function never recomputes persistent state itself. Only primaries
    with at least one real matching event/setup are ever processed --
    the vast majority of the 3036-primary catalog that never fires at
    all in a given dataset costs nothing beyond the fixed arithmetic
    ``SYNTACTIC_SPEC_COUNT`` total.

    ``require_yearly_eligibility=False`` is reserved for bounded engine
    benchmarks on a short real-data prefix. It does not alter the observed
    entries or the strategy grammar; it merely admits every non-empty
    observed sequence so the inner execution loop can be measured before two
    full calendar years are available. Production callers must retain the
    default ``True``.

    ``on_progress`` (task 044), if given, is called with ``(primaries_
    processed, primaries_total)`` once per real (already-fired) primary
    processed in the loop below -- purely observational, never read back
    into the returned ``EntryStrategyUniverseResult`` in any way. The
    caller is expected to pass an already time-throttled callback (e.g.
    ``tradelab.research_run.ProgressThrottle.maybe``).
    """
    market_state_by_event_id: dict[str, MarketStateSnapshot] = {
        snapshot.event.event_id: snapshot.market_state
        for snapshot in iter_candidate_entry_snapshots(fvg_results, fvg_entry_result, liquidity_entry_result, htf_bias_result)
    }
    fvg_zone_by_id = {
        zone.zone_id: zone
        for result in fvg_results.values()
        for zone in result.zones
    }

    # (ObservedEntry, S, B) tuples, grouped by their own PrimaryTriggerSpec
    # -- S is the observed active-FVG-context set (already intersected
    # with the execution-timeframe-only universe, see module docstring),
    # B is the observed (1H, 4H, 1D) bias-state triple.
    items_by_primary: dict[PrimaryTriggerSpec, list[tuple[ObservedEntry, frozenset[str], tuple[str, str, str]]]] = {}

    def _signature(direction: str, market_state: MarketStateSnapshot) -> tuple[frozenset[str], tuple[str, str, str]]:
        s = market_state.active_fvg_selectors_by_direction[direction] & _FVG_CONTEXT_SELECTOR_UNIVERSE_SET
        b = tuple(market_state.bias.timeframe_states[tf] for tf in _BIAS_TIMEFRAME_ORDER)
        return s, b

    for event in build_entry_events(fvg_entry_result, liquidity_entry_result):
        if event.kind == EntryEventKind.LIQUIDITY_SWEEP:
            key = PrimaryTriggerSpec(
                kind=PrimaryTriggerKind.LIQUIDITY_SWEEP, direction=event.direction, timeframe=event.timeframe, liquidity_selector=event.selector
            )
        else:
            kind = PrimaryTriggerKind.IFVG if event.kind == EntryEventKind.IFVG else PrimaryTriggerKind.FVG_TOUCH
            key = PrimaryTriggerSpec(kind=kind, direction=event.direction, timeframe=event.timeframe, liquidity_selector=None)
        market_state = market_state_by_event_id[event.event_id]
        s, b = _signature(event.direction, market_state)
        observed = ObservedEntry(
            entry_id=event.event_id, known_at=event.known_at, direction=event.direction, source=event,
            # Reuse the immutable timeline snapshot and the one global zone
            # index.  Resolving ids to FVGZone objects is deferred until a
            # selected FVG-boundary SL is actually requested.
            active_fvg_zone_ids_by_selector=market_state.active_fvg_zone_ids_by_direction_selector[event.direction],
            fvg_zone_by_id=fvg_zone_by_id,
        )
        items_by_primary.setdefault(key, []).append((observed, s, b))

    for setup in build_sweep_fvg_setup_events(fvg_entry_result, liquidity_entry_result):
        kind = PrimaryTriggerKind.SWEEP_FVG_TOUCH_SETUP if setup.fvg_kind == EntryEventKind.FVG_TOUCH else PrimaryTriggerKind.SWEEP_IFVG_SETUP
        key = PrimaryTriggerSpec(kind=kind, direction=setup.direction, timeframe=setup.fvg_timeframe, liquidity_selector=setup.sweep_event.selector)
        market_state = market_state_by_event_id[setup.fvg_event.event_id]
        s, b = _signature(setup.direction, market_state)
        observed = ObservedEntry(
            entry_id=setup.setup_id, known_at=setup.known_at, direction=setup.direction, source=setup,
            active_fvg_zone_ids_by_selector=market_state.active_fvg_zone_ids_by_direction_selector[setup.direction],
            fvg_zone_by_id=fvg_zone_by_id,
        )
        items_by_primary.setdefault(key, []).append((observed, s, b))

    context_subset_lookup = _build_context_subset_lookup()
    bias_pattern_lookup = _build_bias_pattern_lookup()

    eligible_specs: list[EntryStrategySpec] = []
    classes_by_sequence_key: dict[tuple, list[EntryStrategySpec]] = {}
    observed_by_sequence_key: dict[tuple, tuple[ObservedEntry, ...]] = {}

    def _cells_for(
        items: list[tuple[ObservedEntry, frozenset[str], tuple[str, str, str]]],
    ) -> dict[tuple[frozenset[str], BiasRequirementSpec], list[ObservedEntry]]:
        cells: dict[tuple[frozenset[str], BiasRequirementSpec], list[ObservedEntry]] = {}
        for observed, s, b in items:
            for req in context_subset_lookup[s]:
                for pattern in bias_pattern_lookup[b]:
                    cells.setdefault((req, pattern), []).append(observed)
        return cells

    def _accept_cell(
        primary: PrimaryTriggerSpec,
        req: frozenset[str],
        pattern: BiasRequirementSpec,
        observed_list: list[ObservedEntry],
    ) -> None:
        spec = EntryStrategySpec(
            spec_id=_build_spec_id(primary, req, pattern), primary=primary,
            required_fvg_context_selectors=req, bias_requirement=pattern,
        )
        eligible_specs.append(spec)
        sequence_key = tuple((o.entry_id, o.known_at, o.direction) for o in observed_list)
        classes_by_sequence_key.setdefault(sequence_key, []).append(spec)
        observed_by_sequence_key[sequence_key] = tuple(observed_list)

    def _opposite_primary(primary: PrimaryTriggerSpec) -> PrimaryTriggerSpec:
        direction = SHORT if primary.direction == LONG else LONG
        selector = primary.liquidity_selector
        if selector is not None:
            selector = opposite_liquidity_selector(selector)
        return PrimaryTriggerSpec(
            kind=primary.kind, direction=direction, timeframe=primary.timeframe,
            liquidity_selector=selector,
        )

    total_primaries = len(items_by_primary)
    progress_count = 0

    if pair_yearly_eligibility_for_both:
        # Production ranks one coherent long+short system.  Its coarse
        # discovery threshold must therefore be applied to the combined raw
        # stream, not independently to each side (99 long + 1 short is a
        # valid 100-entry recipe).  Process natural opposite primaries as a
        # pair so sub-threshold directional specs never need to be retained
        # globally; memory remains bounded by two 512-cell primary grids.
        processed: set[PrimaryTriggerSpec] = set()
        for primary, items in items_by_primary.items():
            if primary in processed:
                continue
            opposite = _opposite_primary(primary)
            processed.add(primary)
            paired_items = items_by_primary.get(opposite)
            if paired_items is None:
                progress_count += 1
                if on_progress is not None:
                    on_progress(progress_count, total_primaries)
                continue
            processed.add(opposite)
            long_primary, long_items = (primary, items) if primary.direction == LONG else (opposite, paired_items)
            short_primary, short_items = (opposite, paired_items) if primary.direction == LONG else (primary, items)
            long_cells = _cells_for(long_items)
            short_cells = _cells_for(short_items)
            for (req, long_pattern), long_observed in long_cells.items():
                short_pattern = mirror_bias_requirement(long_pattern)
                right_observed = short_cells.get((req, short_pattern))
                if right_observed is None:
                    continue
                count_2024_left, count_2025_left = _year_counts(long_observed)
                count_2024_right, count_2025_right = _year_counts(right_observed)
                if require_yearly_eligibility and (
                    count_2024_left + count_2024_right < min_entries_per_year
                    or count_2025_left + count_2025_right < min_entries_per_year
                ):
                    continue
                _accept_cell(long_primary, req, long_pattern, long_observed)
                _accept_cell(short_primary, req, short_pattern, right_observed)
            for _ in range(2):
                progress_count += 1
                if on_progress is not None:
                    on_progress(progress_count, total_primaries)
    else:
        for primary, items in items_by_primary.items():
            cells = _cells_for(items)

            for (req, pattern), observed_list in cells.items():
                count_2024, count_2025 = _year_counts(observed_list)
                if require_yearly_eligibility and (count_2024 < min_entries_per_year or count_2025 < min_entries_per_year):
                    continue
                _accept_cell(primary, req, pattern, observed_list)

            progress_count += 1
            if on_progress is not None:
                on_progress(progress_count, total_primaries)

    equivalence_classes: list[EntryStrategyEquivalenceClass] = []
    for sequence_key, specs in classes_by_sequence_key.items():
        member_specs = tuple(sorted(specs, key=lambda s: s.spec_id))
        representative = member_specs[0]
        observed_entries = observed_by_sequence_key[sequence_key]
        count_2024, count_2025 = _year_counts(observed_entries)
        equivalence_classes.append(
            EntryStrategyEquivalenceClass(
                class_id=representative.spec_id,
                observed_entries=observed_entries,
                member_specs=member_specs,
                representative_spec=representative,
                entry_count_2024=count_2024,
                entry_count_2025=count_2025,
            )
        )
    equivalence_classes.sort(key=lambda c: c.class_id)

    return EntryStrategyUniverseResult(
        equivalence_classes=tuple(equivalence_classes),
        syntactically_enumerated_spec_count=SYNTACTIC_SPEC_COUNT,
        eligible_spec_count=len(eligible_specs),
        specs_rejected_by_yearly_eligibility_count=SYNTACTIC_SPEC_COUNT - len(eligible_specs),
        equivalence_class_count=len(equivalence_classes),
        spec_counts_by_primary_kind=_count_by(eligible_specs, lambda s: s.primary.kind.value),
        spec_counts_by_direction=_count_by(eligible_specs, lambda s: s.primary.direction),
        spec_counts_by_execution_timeframe=_count_by(eligible_specs, lambda s: s.primary.timeframe),
        spec_counts_by_bias_requirement_pattern=_count_by(eligible_specs, lambda s: s.bias_requirement.as_label()),
    )


def summarize_entry_strategy_universe(result: EntryStrategyUniverseResult) -> dict:
    """A small, JSON-serializable summary of ``result`` -- counts only,
    never the full equivalence-class/spec detail (which can be large at
    real-data scale). Used by the diagnostic CLI below and reusable
    directly by any future profiling/reporting integration (e.g. a
    ``tradelab.profile_research_context`` section, not added in this
    task)."""
    class_sizes = sorted((len(cls.member_specs) for cls in result.equivalence_classes), reverse=True)
    return {
        "syntactically_enumerated_spec_count": result.syntactically_enumerated_spec_count,
        "eligible_spec_count": result.eligible_spec_count,
        "specs_rejected_by_yearly_eligibility_count": result.specs_rejected_by_yearly_eligibility_count,
        "equivalence_class_count": result.equivalence_class_count,
        "member_specs_per_class": {
            "min": class_sizes[-1] if class_sizes else None,
            "max": class_sizes[0] if class_sizes else None,
            "mean": (sum(class_sizes) / len(class_sizes)) if class_sizes else None,
        },
        "spec_counts_by_primary_kind": dict(result.spec_counts_by_primary_kind),
        "spec_counts_by_direction": dict(result.spec_counts_by_direction),
        "spec_counts_by_execution_timeframe": dict(result.spec_counts_by_execution_timeframe),
        "spec_counts_by_bias_requirement_pattern_count": len(result.spec_counts_by_bias_requirement_pattern),
    }


def _run_cli(argv: list[str] | None = None) -> int:
    """Diagnostic CLI: prepare a real ``PreparedBacktestContext`` from a
    market artifact + news CSV (reusing ``tradelab.backtest.prepare_
    backtest_context`` and ``tradelab.profile_research_context``'s own
    explicit, non-guessed parameters), run ``discover_entry_strategy_
    universe`` on it, and print ``summarize_entry_strategy_universe``'s
    output as JSON. Accepts an optional ``--prefix-bars`` to run on a
    deterministic prefix instead of the whole artifact (see
    ``tradelab.diagnose_context_preparation.select_deterministic_prefix``)
    -- this task's own instruction is a 100k-bar smoke run, never a full
    3-year discovery unless debugging a defect."""
    import argparse
    import json as _json
    from pathlib import Path

    from tradelab.backtest import prepare_backtest_context
    from tradelab.diagnose_context_preparation import select_deterministic_prefix
    from tradelab.market_data_artifact import load_market_data_artifact, load_market_data_series
    from tradelab.news_loaders import read_news_events_csv
    from tradelab.profile_research_context import PIVOT_STRENGTH, RECLAIM_WINDOW_BARS, TICK_SIZE_MNQ

    parser = argparse.ArgumentParser(description="Discover the entry strategy universe on a real market artifact.")
    parser.add_argument("--market-artifact", required=True, type=Path)
    parser.add_argument("--news-csv", required=True, type=Path)
    parser.add_argument("--prefix-bars", type=int, default=None)
    parser.add_argument("--min-entries-per-year", type=int, default=DEFAULT_MIN_ENTRIES_PER_YEAR)
    args = parser.parse_args(argv)

    artifact = load_market_data_artifact(args.market_artifact)
    series_result = load_market_data_series(artifact, base_dir=args.market_artifact.parent)
    bars = list(series_result.bars)
    if args.prefix_bars is not None:
        bars = select_deterministic_prefix(bars, args.prefix_bars)
    news_events = read_news_events_csv(args.news_csv)

    context = prepare_backtest_context(
        bars,
        news_events,
        tick_size=TICK_SIZE_MNQ,
        pivot_strength=PIVOT_STRENGTH,
        reclaim_window_bars=RECLAIM_WINDOW_BARS,
        special_schedule=series_result.special_session_schedule,
    )
    result = discover_entry_strategy_universe(
        context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result,
        min_entries_per_year=args.min_entries_per_year,
    )
    print(_json.dumps(summarize_entry_strategy_universe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
