"""Liquidity-sweep -> FVG/iFVG setup chains: a small, additive extension
of the atomic ``EntryEvent`` model (``tradelab.entry_signal_timeline``,
task 036) that lets a liquidity sweep act as a confluence condition for
a *later* FVG touch or confirmed iFVG, without ever restoring the old
"every historical sweep stays active forever" model this package has
already rejected once (see ``docs/market-state-vs-entry-events-v1.md``).

**A liquidity sweep remains a one-shot ``EntryEvent`` exactly as in task
036** -- this module never mutates, wraps, or replaces it. What it adds
is a second, independent, explicitly bounded fact: a *pending sweep
setup context*, keyed by the sweep's own liquidity ``selector``, that a
later, compatible (same-direction) FVG touch or confirmed iFVG may
consume exactly once to produce a ``SweepFVGSetupEvent``.

**Structurally bounded, not time-bounded** (this module's own title):
at most one pending context can exist per liquidity ``selector`` at any
instant -- a brand-new sweep on a selector that already has one always
supersedes it (see below), so the pending pool's size is bounded by the
fixed liquidity-selector catalog itself (finite: timeframe x
source_family x liquidity_type x trigger_phase x entry_mode), never by
how much history has accumulated. No fixed minutes/hours expiry, no max
pending-sweep cap, no sampling, no heuristic pruning of any kind.

**Rules**:

- A liquidity sweep (``EntryEventKind.LIQUIDITY_SWEEP``) always creates
  a new pending context for its own ``selector`` at its own
  ``known_at`` -- unconditionally, regardless of what (if anything) was
  already pending there.
- If a pending context already exists for that exact same ``selector``:
  an **opposite-direction** new sweep **invalidates** it; a **same-
  direction** new sweep **replaces** it. Either way the old context is
  gone -- never usable again -- and the new sweep's own context takes
  its place.
- A later FVG touch or confirmed iFVG (``EntryEventKind.FVG_TOUCH`` /
  ``IFVG``) whose own ``direction`` matches a pending context's
  ``direction`` **consumes** it, producing one ``SweepFVGSetupEvent``.
  Compatibility is direction-only -- **every** sweep-timeframe x FVG/
  iFVG-timeframe pairing is supported (1m/5m/15m x 1m/5m/15m, same-
  timeframe and cross-timeframe alike); this module never scopes
  consumption by timeframe or by the two families' own selector
  catalogs matching in any way.
- **Fan-out**: one FVG/iFVG event may consume *every* currently
  pending, direction-compatible context across *every* liquidity
  selector in the same pass -- each is a genuinely independent causal
  chain (a different liquidity pool being swept), so one FVG touch can
  legitimately confirm several of them at once, each its own
  ``SweepFVGSetupEvent``. (This fan-out is a deliberate design choice,
  not explicitly forced by the spec -- see the module's own PR
  description for the alternative considered and why fan-out was
  chosen.)
- **Same-instant pairing is explicitly allowed**: a sweep and a
  compatible FVG/iFVG event sharing the *exact same* ``known_at`` may
  still form a setup, because both are already known the instant that
  candle closes. This is a deliberate exception to this package's usual
  "batch members never affect each other" convention (see
  ``tradelab.entry_signal_timeline``'s own docstring) -- spelled out
  explicitly here so it is never mistaken for an oversight.
- **Ties are honored, never arbitrarily discarded**: if several
  compatible FVG/iFVG events share the exact same ``known_at`` (the
  first instant at which a given pending context becomes consumable),
  *all* of them consume that context and each produces its own
  ``SweepFVGSetupEvent`` -- consumption (removal from the pending pool)
  only happens once every event sharing that instant has already been
  matched. A pending context is therefore never usable in any *later*
  batch once it has been consumed, but it is never short-changed within
  its own consuming instant either.
- Once consumed, a pending context can never be used again -- not even
  by a later FVG/iFVG event on the same instant as some other,
  unrelated consumption. Only a *new* sweep on that same selector can
  create a fresh pending context afterward.
- **No look-ahead**: processing is a single forward pass over ``build_
  entry_events``' own chronological order (grouped into ``known_at``
  batches); a pending context is only ever matched against events
  already known at or before the batch being processed.

See ``docs/sweep-fvg-setup-chains-v1.md`` for the full rationale and
real-data results.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import groupby
from typing import Iterator

from tradelab.entry_signal_timeline import EntryEvent, EntryEventKind, build_entry_events
from tradelab.fvg_entry import FVGEntryCandidateResult
from tradelab.liquidity_entry import LiquidityEntryCandidateResult

_FVG_FAMILY_KINDS = frozenset({EntryEventKind.FVG_TOUCH, EntryEventKind.IFVG})


@dataclass(frozen=True)
class PendingSweepContext:
    """A liquidity sweep's own pending directional setup context, alive
    from ``sweep_event.known_at`` until it is superseded (a later sweep
    on the exact same ``selector``) or consumed (a compatible FVG/iFVG
    event) -- whichever happens first. At most one exists per
    ``selector`` at any instant (see module docstring)."""

    sweep_event: EntryEvent
    selector: str
    direction: str


@dataclass(frozen=True)
class SweepFVGSetupEvent:
    """One atomic, one-shot fact: ``fvg_event`` (an FVG touch or
    confirmed iFVG) consumed a still-pending liquidity-sweep context
    (``sweep_event``) of its own ``direction``. Full provenance of both
    source ``EntryEvent``s is carried verbatim (``sweep_event``/
    ``fvg_event``), plus flattened convenience fields for reporting
    without traversing either. ``known_at`` is when the *setup itself*
    becomes knowable -- always ``fvg_event.known_at`` (never earlier
    than the sweep's own ``known_at``, since a context can only ever be
    consumed at or after its own creation)."""

    setup_id: str
    sweep_event: EntryEvent
    fvg_event: EntryEvent
    direction: str
    sweep_timeframe: str
    fvg_timeframe: str
    fvg_kind: EntryEventKind
    sweep_known_at: datetime
    fvg_known_at: datetime
    known_at: datetime


class PendingSweepTransitionKind(str, Enum):
    CREATED = "created"
    REPLACED = "replaced"
    INVALIDATED = "invalidated"
    CONSUMED = "consumed"


@dataclass(frozen=True)
class PendingSweepTransition:
    """One audit record of a ``PendingSweepContext``'s lifecycle --
    diagnostic/reporting only (see ``tradelab.profile_research_context``'s
    ``sweep_fvg_setup_model`` section), mirroring this package's existing
    ``SignalSuspension``/``SignalRestoration`` audit-trail convention.
    ``caused_by`` is the new sweep ``EntryEvent`` that superseded
    ``context`` for ``REPLACED``/``INVALIDATED``; ``None`` for
    ``CREATED`` (nothing precedes a context's own creation) and for
    ``CONSUMED`` (a consumption can be caused by more than one tied
    FVG/iFVG event at once -- see ``SweepFVGSetupEvent`` for that full
    detail, cross-referenced by ``context.sweep_event.event_id``)."""

    kind: PendingSweepTransitionKind
    context: PendingSweepContext
    known_at: datetime
    caused_by: EntryEvent | None


def _build_setup_event(context: PendingSweepContext, fvg_event: EntryEvent) -> SweepFVGSetupEvent:
    sweep_event = context.sweep_event
    return SweepFVGSetupEvent(
        setup_id=f"{sweep_event.event_id}::{fvg_event.event_id}",
        sweep_event=sweep_event,
        fvg_event=fvg_event,
        direction=fvg_event.direction,
        sweep_timeframe=sweep_event.timeframe,
        fvg_timeframe=fvg_event.timeframe,
        fvg_kind=fvg_event.kind,
        sweep_known_at=sweep_event.known_at,
        fvg_known_at=fvg_event.known_at,
        known_at=fvg_event.known_at,
    )


def _apply_sweep_fvg_batch(
    sweep_batch: list[EntryEvent],
    fvg_batch: list[EntryEvent],
    pending_by_selector: dict[str, PendingSweepContext],
) -> tuple[list[PendingSweepTransition], list[SweepFVGSetupEvent]]:
    """Mutates ``pending_by_selector`` in place for one ``known_at``
    batch and returns ``(transitions, setups)`` -- the shared, pure
    per-batch engine both ``iter_sweep_fvg_setup_events`` and ``iter_
    sweep_pending_transitions`` independently sweep with (mirrors this
    package's existing ``_apply_fvg_batch`` pattern: one rule
    implementation, multiple independent O(N log N) passes over it).

    Step 1 (sweeps) always runs before step 2 (FVG/iFVG matching) within
    the *same* batch -- deliberately, so a sweep and a compatible FVG/
    iFVG event sharing this exact ``known_at`` can still form a setup
    (see module docstring). Step 2 reads, but never mutates,
    ``pending_by_selector`` (a context consumed by a tie is still
    visible to every other tied FVG/iFVG event in this same batch);
    consumption itself -- actually removing a used context -- is
    deferred to step 3, once every tied match in this batch is already
    recorded.
    """
    transitions: list[PendingSweepTransition] = []

    # 1. Every sweep in this batch always creates its own new pending
    #    context for its own selector, unconditionally. If that selector
    #    already had one, it is superseded first -- INVALIDATED (opposite
    #    direction) or REPLACED (same direction) -- and is never usable
    #    again. sweep_batch is already deterministically ordered (by
    #    event_id, via build_entry_events); when more than one sweep in
    #    this batch shares a selector, they are applied in that order, so
    #    only the last one's context survives to step 2.
    for sweep in sweep_batch:
        existing = pending_by_selector.get(sweep.selector)
        if existing is not None:
            kind = (
                PendingSweepTransitionKind.INVALIDATED
                if existing.direction != sweep.direction
                else PendingSweepTransitionKind.REPLACED
            )
            transitions.append(PendingSweepTransition(kind=kind, context=existing, known_at=sweep.known_at, caused_by=sweep))
        new_context = PendingSweepContext(sweep_event=sweep, selector=sweep.selector, direction=sweep.direction)
        pending_by_selector[sweep.selector] = new_context
        transitions.append(
            PendingSweepTransition(kind=PendingSweepTransitionKind.CREATED, context=new_context, known_at=sweep.known_at, caused_by=None)
        )

    # 2. Every FVG/iFVG event in this batch is matched, by direction
    #    only, against every context currently pending (post step 1) --
    #    fan-out: one event can match several different selectors' own
    #    contexts at once (see module docstring). Deterministic order:
    #    outer loop over fvg_batch (event_id order), inner loop over
    #    pending_by_selector (insertion order, itself fully deterministic
    #    -- see step 1).
    setups: list[SweepFVGSetupEvent] = []
    consumed_selectors: list[str] = []
    for fvg_event in fvg_batch:
        for selector, context in pending_by_selector.items():
            if context.direction != fvg_event.direction:
                continue
            setups.append(_build_setup_event(context, fvg_event))
            if selector not in consumed_selectors:
                consumed_selectors.append(selector)

    # 3. Consume: remove every context matched at least once in this
    #    batch -- every tied match at this exact known_at has already
    #    been recorded (step 2), so removing it now only ever affects
    #    strictly later batches.
    for selector in consumed_selectors:
        context = pending_by_selector.pop(selector)
        known_at = fvg_batch[0].known_at if fvg_batch else sweep_batch[0].known_at
        transitions.append(PendingSweepTransition(kind=PendingSweepTransitionKind.CONSUMED, context=context, known_at=known_at, caused_by=None))

    return transitions, setups


def _iter_batches(
    fvg_result: FVGEntryCandidateResult, liquidity_result: LiquidityEntryCandidateResult
) -> Iterator[tuple[list[EntryEvent], list[EntryEvent]]]:
    """Yields ``(sweep_batch, fvg_batch)`` per distinct ``known_at``, in
    chronological order -- ``build_entry_events`` already returns FVG-
    family and liquidity events merged into one ``(known_at, event_id)``-
    sorted list (task 036), so no separate merge step is needed here."""
    events = build_entry_events(fvg_result, liquidity_result)
    for _known_at, group_iter in groupby(events, key=lambda e: e.known_at):
        group = list(group_iter)
        sweep_batch = [e for e in group if e.kind == EntryEventKind.LIQUIDITY_SWEEP]
        fvg_batch = [e for e in group if e.kind in _FVG_FAMILY_KINDS]
        yield sweep_batch, fvg_batch


def iter_sweep_fvg_setup_events(
    fvg_result: FVGEntryCandidateResult, liquidity_result: LiquidityEntryCandidateResult
) -> Iterator[SweepFVGSetupEvent]:
    """Every ``SweepFVGSetupEvent``, in chronological order (by
    ``known_at``, then by the deterministic within-batch order
    ``_apply_sweep_fvg_batch`` produces) -- the primary, intended entry
    point for a future strategy generator to use these setups as
    additional primary triggers (spec, this task). One linear forward
    pass, ``O(N log N)``; memory ``O(size of the liquidity selector
    catalog)`` for the pending pool, never ``O(number of sweeps ever
    seen)`` (see module docstring: structurally bounded, not time-
    bounded)."""
    pending_by_selector: dict[str, PendingSweepContext] = {}
    for sweep_batch, fvg_batch in _iter_batches(fvg_result, liquidity_result):
        if not sweep_batch and not fvg_batch:
            continue
        _transitions, setups = _apply_sweep_fvg_batch(sweep_batch, fvg_batch, pending_by_selector)
        yield from setups


def iter_sweep_pending_transitions(
    fvg_result: FVGEntryCandidateResult, liquidity_result: LiquidityEntryCandidateResult
) -> Iterator[PendingSweepTransition]:
    """Every ``PendingSweepTransition`` (created/replaced/invalidated/
    consumed), in chronological order -- diagnostic/audit only (see
    ``tradelab.profile_research_context``'s ``sweep_fvg_setup_model``
    section). Independently re-sweeps ``_apply_sweep_fvg_batch`` (mirrors
    ``iter_fvg_lifecycle_transitions`` vs. ``iter_active_signal_
    snapshots`` in ``tradelab.entry_signal_timeline``); never changes
    ``iter_sweep_fvg_setup_events``'s own behavior."""
    pending_by_selector: dict[str, PendingSweepContext] = {}
    for sweep_batch, fvg_batch in _iter_batches(fvg_result, liquidity_result):
        if not sweep_batch and not fvg_batch:
            continue
        transitions, _setups = _apply_sweep_fvg_batch(sweep_batch, fvg_batch, pending_by_selector)
        yield from transitions


def build_sweep_fvg_setup_events(
    fvg_result: FVGEntryCandidateResult, liquidity_result: LiquidityEntryCandidateResult
) -> list[SweepFVGSetupEvent]:
    """Materialized ``list`` form of ``iter_sweep_fvg_setup_events`` --
    for callers (and tests) that want the whole, already-chronological
    result at once rather than iterating."""
    return list(iter_sweep_fvg_setup_events(fvg_result, liquidity_result))
