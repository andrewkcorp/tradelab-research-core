"""Entry signal timeline: a unified, audited, strategy-agnostic stream of
entry confluence from FVG/iFVG and liquidity, plus a validity mechanism a
future strategy enumerator will query -- still not a backtester, TP/SL,
or trade generator.

Inputs are ``tradelab.fvg_entry.FVGEntryCandidateResult`` and
``tradelab.liquidity_entry.LiquidityEntryCandidateResult`` -- already-
built, already-validated candidate sets. This module does not re-detect
anything and does not change either upstream module's semantics. Data
wicks are not an input and never appear here (spec 4.3: a future TP/SL
level source only).

**Validity: persist until invalidated, never a time-based expiry**:

- A signal becomes available exactly at its own ``known_at`` -- never
  earlier -- and then remains active **indefinitely**, across trading
  days without limit, until it is explicitly invalidated or suspended
  (see below). There is no daily, session, or calendar cutoff of any
  kind on an otherwise-still-valid confluence.
- ``EntrySignal.trading_day_id`` still exists (the ISO date of the NY
  trading day -- ``[18:00 NY, 17:00 NY next day)`` -- that owns a
  signal's own ``known_at``), but it is pure audit metadata: it plays no
  role whatsoever in whether that signal is active.
- A signal whose own ``known_at`` falls strictly *between* 17:00 and
  18:00 NY (the daily maintenance break) still belongs to no trading day
  at all and is rejected outright (``ValueError``) -- a market/session
  data-integrity check, not a validity-duration rule. ``known_at``
  exactly *at* 17:00 NY is attributed to the trading day that is
  *ending* at that instant, not rejected.
- All boundary math uses ``zoneinfo.ZoneInfo("America/New_York")``,
  never a fixed UTC offset, so DST is handled correctly for free.

**Two separate, non-interacting conflict mechanisms, by family** (this
task -- a semantic correction from an earlier version of this module,
which used one unified, same-timeframe, cross-family, *permanent*
invalidation rule for everything): FVG/iFVG conflict on one timeframe
never affects liquidity, and liquidity invalidation never affects
FVG/iFVG, in either direction.

**Liquidity: same-selector, permanent invalidation** (reverted to this
module's original model) -- a later, opposite-direction liquidity signal
on the **exact same** ``selector`` (never a different selector, even on
the same timeframe) invalidates an earlier one, permanently: once
invalidated, a signal never becomes active again, and a later signal of
the same direction never invalidates an earlier one (refreshes/adds to
the active state instead). Only signals whose own ``selector`` is in the
caller's chosen ``selectors`` ever participate (spec: "Sygnaly z
niewybranych selectorow nigdy nie uniewazniaja setupu"). Reported via
``SignalInvalidation``.

**FVG family (``FVG_TOUCH`` + ``IFVG``): same-timeframe, cross-selector,
*reversible* suspension** -- a later, opposite-direction FVG-family
candidate never permanently deletes an earlier one; it *suspends* it
(retaining full identity/provenance), and that suspension can later be
lifted, restoring the earlier signal to active. Modeled directly on
``tradelab.fvg``'s own zone lifecycle (``ACTIVE`` / ``MITIGATED``,
5.2) rather than reimplementing it:

- Only an ``FVG_TOUCH`` candidate with ``contact_type=PARTIAL_TOUCH`` --
  a zone that ``tradelab.fvg`` still classifies ``ACTIVE``, i.e. is
  genuinely still an ongoing, "currently valid FVG" -- ever *durably*
  suspends the opposite-direction pool on its own timeframe (every
  still-active opposite-direction FVG-family signal there, any
  selector). Reported via ``SignalSuspension``; a suspended signal is
  not in ``active_signals`` until it is restored (see below).
- ``FVG_TOUCH`` with ``contact_type=FULL_MITIGATION_NO_INVERSION`` is
  already a *resolved*, terminal fact the instant it is known
  (``tradelab.fvg``: ``MITIGATED`` without inversion is terminal, this
  exact same candle) -- it still challenges the opposite-direction pool
  (any new FVG-family confluence does), but that challenge is resolved
  in the very same instant it is raised: the earlier signal is never
  actually removed from ``active_signals`` (so it is never observably
  missing at any queryable instant), yet a ``SignalRestoration`` is
  still recorded for full audit transparency (spec: "the competing FVG
  becomes inactive without becoming iFVG -> restore"). ``IFVG``
  candidates never challenge anything -- they only restore an earlier
  suspension caused by their own zone's prior touch (see below) and
  become a new active confluence. All three kinds still become a new
  active confluence of their own direction, same as any other
  FVG-family signal.
- **Restoration**: a zone can produce at most one ``FVG_TOUCH`` event,
  ever (``tradelab.fvg_touch``'s own contract), so the *only* possible
  further event for a zone whose touch was ``PARTIAL_TOUCH`` is that
  same zone's own, later ``IFVG`` event -- and an inversion always
  carries the *opposite* direction of its own originating zone
  (``tradelab.fvg_entry``'s own contract), i.e. the same direction as
  whatever that zone's earlier touch had suspended. When that ``IFVG``
  arrives, every signal that zone's touch suspended is restored to
  active, and the ``IFVG`` itself is also a new active confluence in its
  own (matching) direction. If a zone that caused a suspension is never
  later inverted, that suspension simply persists -- absent evidence of
  resolution, the suspending FVG is treated as still valid, exactly as
  persist-until-invalidated already treats any other unresolved fact.
  Reported via ``SignalRestoration``.
- **A touch that inverts is itself a terminal, superseded fact from that
  instant on.** The ``IFVG`` does not merely restore what its own touch
  had suspended -- it also permanently retires that touch: the touch is
  removed from wherever it currently lives (``active_signals``, or
  suspended by some unrelated, still-unresolved challenger) and, if it
  was suspended, dropped from that challenger's own pending restoration
  list. A retired touch never appears in ``active_signals`` again and
  is never restored by anyone's later resolution -- including its own
  challenger's -- because the zone it represented is no longer a valid
  FVG in its original direction at all; only the ``IFVG`` itself (and
  whatever it restores) carries that zone's confluence forward.
- **Nesting** is handled deterministically: each suspension records
  *which* signal it was caused by, keyed by that signal's own zone; a
  chain of suspensions (A suspended by B, B suspended by C, ...) unwinds
  one link at a time, each only when *that* link's own suspending zone
  resolves -- independent of whether the signal in between is itself
  currently active or suspended. If a link's own touch is retired by its
  own inversion (previous bullet) before the outer challenger resolves,
  the outer challenger's own eventual resolution finds nothing left to
  restore for that link.
- **A zone that later, separately, becomes ``MITIGATED`` without
  inverting also resolves whatever its own touch suspended.**
  ``tradelab.fvg_touch`` only ever records a zone's *first* touch,
  whatever kind it is -- so a zone whose first touch was
  ``PARTIAL_TOUCH`` and which only *later*, on a different candle,
  fully traverses without a close beyond the far boundary produces no
  further *touch* event of its own. ``tradelab.fvg`` itself, however,
  still emits a standalone, timestamped, no-look-ahead
  ``FVGMitigationEvent`` for that exact transition (``tradelab.fvg_entry``
  forwards it as ``FVGEntryCandidateResult.zone_mitigations`` -- never a
  new entry candidate or a new selectable confluence, purely a
  resolution fact this module consumes). At that event's own
  ``known_at``, this module resolves the zone exactly as it would for
  an ``IFVG``: the zone's own earlier ``PARTIAL_TOUCH`` is permanently
  retired (same rule as the previous bullet -- never eligible for
  restoration afterward, by anyone), and everything that touch had
  suspended is restored. Unlike an ``IFVG``, a mitigation is never
  itself a new active confluence -- the zone simply closes out, with no
  replacement.
- Same-direction FVG-family signals never suspend or invalidate each
  other, on any timeframe -- they simply coexist.
- Never crosses timeframes, exactly like the base same-timeframe scoping
  this module has always used for FVG-family conflict.

``query_active_signals`` answers, for a caller-chosen set of selectors
(the strategy's own choice of which categories participate), a
direction, and an as-of instant: every signal in ``selectors`` of that
``direction``, already known by ``ts_utc``, not invalidated (liquidity)
and not currently suspended (FVG family) as of ``ts_utc``.

Not built here, on purpose (later steps): which selectors combine into a
concrete strategy, full combination enumeration, opening positions, the
news filter, TP/SL, or bias as a filter.

**Bulk snapshotting** (``iter_active_signal_snapshots``): ``query_active_
signals`` is a fresh, self-contained query -- fine for a handful of
caller-chosen instants, but a caller that needs the active state at
*every* real signal's own instant (e.g. ``tradelab.observed_strategy_
universe``) would otherwise have to call it once per signal, each call
itself scanning the whole timeline -- ``O(N^2)`` overall. ``iter_active_
signal_snapshots`` computes the exact same per-instant active state for
every signal in one linear forward sweep instead -- ``O(N log N)``
total. It never changes ``query_active_signals`` itself or any of its
semantics; it is a second, independently implemented way to obtain the
same answer at every real instant instead of a handful of caller-chosen
ones (both share only the small, pure per-batch state-transition
helpers, never any incrementally-carried state), and its own test suite
verifies the two agree exactly, including on randomized fixtures
exercising liquidity invalidation, FVG suspension/restoration, and
nested FVG conflicts together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from itertools import groupby
from typing import Iterator, Mapping
from zoneinfo import ZoneInfo

from tradelab.bias import BEARISH, BULLISH, NEUTRAL, BiasSnapshot, HTFBiasResult, TimeframeBiasChange
from tradelab.bias import SUPPORTED_TIMEFRAMES as _BIAS_TIMEFRAMES
from tradelab.fvg import FVGDetectionResult, FVGZone
from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, FVGEntryCandidateResult, FVGMitigationEvent
from tradelab.fvg_entry import SourceType as FVGSourceType
from tradelab.liquidity_entry import LiquidityEntryCandidate, LiquidityEntryCandidateResult

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

LONG = "long"
SHORT = "short"
_DIRECTIONS = frozenset({LONG, SHORT})
_OPPOSITE_DIRECTION = {LONG: SHORT, SHORT: LONG}

# The daily NY maintenance break: a signal known_at falling strictly
# inside here belongs to no trading day and is rejected outright -- a
# market/session data-integrity check (real candles never close in
# there), unrelated to how long a confluence stays valid. Open on *both*
# ends -- unlike every other window in this package, which is half-open
# -- because known_at == 17:00 NY exactly is not itself part of the
# break: it is the regular session's own close, i.e. the known_at of
# every trading day's genuinely last 1-minute candle. known_at == 18:00
# NY can never actually occur (every Candle.known_at is strictly after
# its own timestamp, and the earliest possible session-open timestamp is
# itself 18:00 NY), so only the true open interval is ever reachable.
_BREAK_START = time(17, 0)
_BREAK_END = time(18, 0)


class SourceKind(str, Enum):
    FVG = "fvg"
    LIQUIDITY = "liquidity"


@dataclass(frozen=True)
class EntrySignal:
    """One unified entry confluence fact, normalized from either an
    ``FVGEntryCandidate`` or a ``LiquidityEntryCandidate``.

    ``source`` is the original candidate object, kept in full for later
    TP/SL and audit -- this module never flattens or discards any of its
    fields. ``selector`` groups signals into stable, serializable
    categories (see module docstring); ``timeframe`` is what FVG-family
    suspension is scoped by (liquidity invalidation is scoped by
    ``selector`` instead -- see module docstring). ``trading_day_id`` is
    pure audit metadata (which NY trading day this confluence was born
    in) -- it plays no role in validity, which is unbounded except for
    invalidation/suspension (see module docstring)."""

    candidate_id: str
    selector: str
    source_kind: SourceKind
    timeframe: str
    direction: str  # "long" or "short"
    timestamp: datetime
    known_at: datetime
    trading_day_id: str  # ISO date the trading day started on (its own 18:00 NY) -- audit only
    source: FVGEntryCandidate | LiquidityEntryCandidate


@dataclass(frozen=True)
class EntrySignalTimeline:
    signals: list[EntrySignal]  # all signals, chronological by known_at
    signals_by_selector: dict[str, list[EntrySignal]]  # each list chronological by known_at
    # Pure zone-resolution facts (never a confluence signal, never in
    # `signals`/`signals_by_selector`, never selectable, never returned
    # by `active_signals`) -- used solely by the FVG suspend/restore
    # mechanic to learn when a zone's ACTIVE life ends without inverting,
    # even long after an earlier partial touch. Chronological by known_at.
    # Defaults to `()` so existing direct construction (bypassing
    # `build_entry_signal_timeline`, e.g. in tests) keeps working.
    zone_mitigations: list[FVGMitigationEvent] = field(default_factory=list)


@dataclass(frozen=True)
class SignalInvalidation:
    """Permanent, one-way invalidation -- **liquidity only** (this
    task): ``signal`` was invalidated by ``invalidated_by``, a later,
    opposite-direction signal on the exact same liquidity ``selector``
    (never a different selector, never cross-family with FVG/iFVG -- see
    module docstring). The FVG family never produces this record; see
    ``SignalSuspension``/``SignalRestoration`` instead."""

    signal: EntrySignal
    invalidated_by: EntrySignal


@dataclass(frozen=True)
class SignalSuspension:
    """**FVG family only** (this task): ``signal`` is currently
    suspended by ``suspended_by`` -- a later, opposite-direction
    ``FVG_TOUCH`` candidate with ``contact_type=PARTIAL_TOUCH`` (i.e.
    still a genuinely ongoing, "currently valid" FVG) on the exact same
    timeframe. Reversible, never permanent: ``signal`` is simply absent
    from ``active_signals`` while suspended, and reappears the moment
    ``suspended_by``'s own zone resolves (see ``SignalRestoration``)."""

    signal: EntrySignal
    suspended_by: EntrySignal


@dataclass(frozen=True)
class SignalRestoration:
    """**FVG family only** (this task): ``signal`` had been suspended
    (see ``SignalSuspension``) and has since been restored, because the
    zone that suspended it has itself now resolved via ``restored_by`` --
    either an ``IFVG`` candidate for that same zone (carrying the same
    direction as ``signal``, since a zone's inversion always flips its
    own direction, so the thing that suspended ``signal`` is, by
    resolving this way, no longer opposite to it), or a
    ``tradelab.fvg.FVGMitigationEvent`` for that same zone (the zone
    became ``MITIGATED`` *without* inverting -- never itself a new
    confluence, just a pure resolution fact). ``signal`` is active again
    and included in ``active_signals``."""

    signal: EntrySignal
    restored_by: EntrySignal | FVGMitigationEvent


@dataclass(frozen=True)
class EntrySignalQueryResult:
    active_signals: list[EntrySignal]
    invalidations: list[SignalInvalidation]
    suspensions: list[SignalSuspension]
    restorations: list[SignalRestoration]


def _trading_day_for(known_at: datetime) -> str:
    """The ``trading_day_id`` for a signal known at ``known_at`` -- pure
    audit metadata, not tied to any validity window (see module
    docstring). Raises if ``known_at`` falls strictly between 17:00 and
    18:00 NY, which no trading day owns (market/session data-integrity
    check, unrelated to validity duration). ``known_at`` exactly at
    17:00 NY -- the regular session's own close -- is not rejected: it
    is attributed to the trading day that is ending at that instant."""
    local = known_at.astimezone(NY)
    t = local.time()
    if _BREAK_START < t < _BREAK_END:
        raise ValueError(
            f"known_at {known_at!r} (NY local {local!r}) falls inside the daily NY maintenance break "
            "(17:00, 18:00) -- no trading day owns this instant, so it can never be a valid entry signal"
        )
    start_date = local.date() if t >= _BREAK_END else local.date() - timedelta(days=1)
    return start_date.isoformat()


def _fvg_selector(candidate: FVGEntryCandidate) -> str:
    return f"fvg-{candidate.timeframe}-{candidate.source_type.value}"


def _liquidity_selector(candidate: LiquidityEntryCandidate) -> str:
    return (
        f"liquidity-{candidate.timeframe}-{candidate.source_family.value}-{candidate.liquidity_type.value}-"
        f"{candidate.trigger_phase.value}-{candidate.entry_mode.value}"
    )


def _signal_from_fvg(candidate: FVGEntryCandidate) -> EntrySignal:
    return EntrySignal(
        candidate_id=candidate.candidate_id,
        selector=_fvg_selector(candidate),
        source_kind=SourceKind.FVG,
        timeframe=candidate.timeframe,
        direction=candidate.direction,
        timestamp=candidate.timestamp,
        known_at=candidate.known_at,
        trading_day_id=_trading_day_for(candidate.known_at),
        source=candidate,
    )


def _signal_from_liquidity(candidate: LiquidityEntryCandidate) -> EntrySignal:
    return EntrySignal(
        candidate_id=candidate.candidate_id,
        selector=_liquidity_selector(candidate),
        source_kind=SourceKind.LIQUIDITY,
        timeframe=candidate.timeframe,
        direction=candidate.direction,
        timestamp=candidate.timestamp,
        known_at=candidate.known_at,
        trading_day_id=_trading_day_for(candidate.known_at),
        source=candidate,
    )


def build_entry_signal_timeline(
    fvg_result: FVGEntryCandidateResult,
    liquidity_result: LiquidityEntryCandidateResult,
) -> EntrySignalTimeline:
    """Normalize both candidate result sets into one chronological
    ``EntrySignal`` timeline, indexed by selector."""
    signals = [_signal_from_fvg(c) for c in fvg_result.candidates] + [
        _signal_from_liquidity(c) for c in liquidity_result.candidates
    ]

    candidate_ids = [s.candidate_id for s in signals]
    if len(candidate_ids) != len(set(candidate_ids)):
        dupes = sorted({cid for cid in candidate_ids if candidate_ids.count(cid) > 1})
        raise ValueError(f"Duplicate candidate_id(s) across fvg_result and liquidity_result: {dupes}")

    signals.sort(key=lambda s: (s.known_at, s.candidate_id))

    signals_by_selector: dict[str, list[EntrySignal]] = {}
    for signal in signals:
        signals_by_selector.setdefault(signal.selector, []).append(signal)

    zone_mitigations = sorted(fvg_result.zone_mitigations, key=lambda m: (m.known_at, m.zone_id))

    return EntrySignalTimeline(signals=signals, signals_by_selector=signals_by_selector, zone_mitigations=zone_mitigations)


def _apply_liquidity_batch(
    batch: list[EntrySignal],
    active: dict[str, dict[str, EntrySignal]],
) -> list[SignalInvalidation]:
    """Mutates ``active`` (``selector -> {candidate_id: signal}``) in
    place for one ``known_at`` batch of liquidity signals, and returns
    the invalidations this batch caused. Liquidity invalidation is
    scoped to the exact same ``selector``, permanent (removed signals
    are never re-added). Batch members share this exact ``known_at`` and
    never invalidate each other (mirrors ``query_active_signals``'s own
    strict ``>`` comparison): targets are the pre-batch active pool.
    """
    invalidations: list[SignalInvalidation] = []
    for signal in batch:
        selector_pool = active.get(signal.selector)
        if not selector_pool:
            continue
        invalidated_ids = [cid for cid, other in selector_pool.items() if other.direction != signal.direction]
        for candidate_id in invalidated_ids:
            invalidated_signal = selector_pool.pop(candidate_id)
            invalidations.append(SignalInvalidation(signal=invalidated_signal, invalidated_by=signal))
        if not selector_pool:
            active.pop(signal.selector, None)

    for signal in batch:
        active.setdefault(signal.selector, {})[signal.candidate_id] = signal

    return invalidations


def _resolve_zone(
    zone_id: str,
    resolving_fact: EntrySignal | FVGMitigationEvent,
    active: dict[str, dict[str, dict[str, EntrySignal]]],
    suspended: dict[str, EntrySignal],
    suspended_by: dict[str, SignalSuspension],
    pending_resolution: dict[str, list[str]],
    touch_by_zone: dict[str, tuple[str, str]],
) -> list[SignalRestoration]:
    """Shared resolution step for a zone whose ``ACTIVE`` life has just
    ended -- whether via its own ``IFVG`` (inversion) or via a later,
    separate ``FVGMitigationEvent`` (full traversal without inversion).
    Permanently retires that zone's own earlier ``PARTIAL_TOUCH`` (if
    any) -- removed from wherever it currently lives (``active``, or
    suspended by some unrelated, still-unresolved challenger, in which
    case it is also dropped from that challenger's own pending
    restoration list) -- and restores everything that touch itself had
    suspended, attributing the restoration to ``resolving_fact``. Never
    adds anything new to ``active`` itself: that is the caller's own job
    for an ``IFVG`` signal (it is a new confluence in its own right, on
    top of resolving its zone), and never happens for a plain mitigation
    (never itself a new confluence).
    """
    restorations: list[SignalRestoration] = []

    touch_entry = touch_by_zone.pop(zone_id, None)
    if touch_entry is not None:
        touch_candidate_id, touch_direction = touch_entry
        if touch_candidate_id in suspended:
            suspended.pop(touch_candidate_id)
            challenger_zone = suspended_by.pop(touch_candidate_id).suspended_by.source.zone_id
            still_pending = [cid for cid in pending_resolution.get(challenger_zone, []) if cid != touch_candidate_id]
            if still_pending:
                pending_resolution[challenger_zone] = still_pending
            else:
                pending_resolution.pop(challenger_zone, None)
        else:
            tf_pool = active.get(resolving_fact.timeframe, {}).get(touch_direction, {})
            tf_pool.pop(touch_candidate_id, None)

    pending = pending_resolution.pop(zone_id, None)
    if pending:
        for candidate_id in pending:
            restored_signal = suspended.pop(candidate_id)
            suspended_by.pop(candidate_id, None)
            active.setdefault(restored_signal.timeframe, {}).setdefault(restored_signal.direction, {})[candidate_id] = restored_signal
            restorations.append(SignalRestoration(signal=restored_signal, restored_by=resolving_fact))

    return restorations


def _apply_fvg_batch(
    signal_batch: list[EntrySignal],
    mitigation_batch: list[FVGMitigationEvent],
    active: dict[str, dict[str, dict[str, EntrySignal]]],
    suspended: dict[str, EntrySignal],
    suspended_by: dict[str, SignalSuspension],
    pending_resolution: dict[str, list[str]],
    touch_by_zone: dict[str, tuple[str, str]],
) -> tuple[list[SignalSuspension], list[SignalRestoration]]:
    """Mutates ``active`` (``timeframe -> direction -> {candidate_id:
    signal}``), ``suspended`` (``candidate_id -> signal``, everything
    currently suspended) and ``suspended_by`` (``candidate_id -> its
    current SignalSuspension``) in place for one ``known_at`` batch --
    ``signal_batch`` (real FVG-family confluence signals: touches and
    IFVGs) and ``mitigation_batch`` (pure zone-resolution facts, never
    confluence signals) sharing that exact ``known_at`` -- and returns
    ``(suspensions, restorations)``, the transitions this batch caused
    (used by ``iter_fvg_lifecycle_transitions`` for diagnostics; the
    other two callers only need the final, mutated state and discard
    these return values). ``pending_resolution`` maps a
    zone_id to the candidate_ids its own ``PARTIAL_TOUCH`` suspended, so
    that zone's own later resolution (``IFVG`` or ``FVGMitigationEvent``)
    can restore exactly those. ``touch_by_zone`` maps a zone_id to
    ``(candidate_id, direction)`` of its own ``PARTIAL_TOUCH``, so that
    zone's own later resolution can retire that touch -- it is now a
    superseded, terminal fact (the zone is no longer a valid original-
    direction FVG at all), never eligible for restoration by anyone,
    regardless of whether it is currently active or suspended by some
    unrelated challenger. See module docstring for the full suspend/
    restore rules. Batch members share this exact ``known_at`` and never
    suspend, resolve, or restore each other: resolution only ever uses a
    strictly earlier zone's pending entry, and suspension targets are
    the active pool as it stood before this batch's own suspensions.
    """
    restorations: list[SignalRestoration] = []

    # 1. Resolve zones: an IFVG signal or a mitigation fact in this batch
    #    both retire their own zone's own, strictly earlier PARTIAL_TOUCH
    #    (if any) and restore whatever it suspended -- identical
    #    mechanism either way (see _resolve_zone); only an IFVG is also,
    #    separately, a brand new active confluence itself (step 3, since
    #    it is a real signal in signal_batch).
    for signal in signal_batch:
        candidate: FVGEntryCandidate = signal.source
        if candidate.source_type != FVGSourceType.IFVG:
            continue
        restorations.extend(_resolve_zone(candidate.zone_id, signal, active, suspended, suspended_by, pending_resolution, touch_by_zone))
    for mitigation in mitigation_batch:
        restorations.extend(_resolve_zone(mitigation.zone_id, mitigation, active, suspended, suspended_by, pending_resolution, touch_by_zone))

    # 2. Challenges: a still-open PARTIAL_TOUCH ("remains a valid FVG")
    #    durably suspends the opposite-direction pool on its own
    #    timeframe -- an unresolved challenge, until some future event
    #    resolves it (step 1, above, on a later batch). A
    #    FULL_MITIGATION_NO_INVERSION touch is already resolved the
    #    instant it is known (its own zone is immediately, terminally
    #    MITIGATED without inverting, per tradelab.fvg) -- it still
    #    challenges the opposite-direction pool (rule: "an opposite-
    #    direction FVG touch/entry confluence appears"), but that
    #    challenge is resolved in this exact same step: the target is
    #    never actually removed from ``active`` (so it is never
    #    observably missing at any queryable instant), yet a
    #    SignalRestoration is still recorded for full audit transparency
    #    ("the competing FVG became inactive without becoming iFVG ->
    #    restore"). IFVG candidates never challenge anything (see the
    #    module docstring); they only retire their own zone's touch and
    #    restore whatever it suspended (step 1) and become a new active
    #    confluence (step 3). A durable PARTIAL_TOUCH also records itself
    #    (candidate_id and its own direction) under ``touch_by_zone``, so
    #    its own zone's later resolution (if any) can find and retire it
    #    (step 1 above, on that later batch).
    #    Targets are the pre-batch active pool (captured before any of
    #    this batch's own challenges are applied), so batch members
    #    never challenge each other.
    challenge_actions: list[tuple[EntrySignal, EntrySignal, bool]] = []  # (target, challenger, is_durable)
    for signal in signal_batch:
        candidate = signal.source
        if candidate.source_type != FVGSourceType.FVG_TOUCH:
            continue
        if candidate.contact_type not in (FVGContactType.PARTIAL_TOUCH, FVGContactType.FULL_MITIGATION_NO_INVERSION):
            continue
        is_durable = candidate.contact_type == FVGContactType.PARTIAL_TOUCH
        if is_durable:
            touch_by_zone[candidate.zone_id] = (signal.candidate_id, signal.direction)
        opposite = _OPPOSITE_DIRECTION[signal.direction]
        for target in active.get(signal.timeframe, {}).get(opposite, {}).values():
            challenge_actions.append((target, signal, is_durable))

    suspensions: list[SignalSuspension] = []
    for target, challenger, is_durable in challenge_actions:
        tf_pool = active.get(target.timeframe, {}).get(target.direction, {})
        if target.candidate_id not in tf_pool:
            continue  # already suspended earlier in this same batch
        if not is_durable:
            restorations.append(SignalRestoration(signal=target, restored_by=challenger))
            continue
        del tf_pool[target.candidate_id]
        suspended[target.candidate_id] = target
        suspension = SignalSuspension(signal=target, suspended_by=challenger)
        suspended_by[target.candidate_id] = suspension
        suspensions.append(suspension)
        pending_resolution.setdefault(challenger.source.zone_id, []).append(target.candidate_id)

    # 3. Every real signal in this batch becomes (or remains) active,
    #    under its own timeframe/direction, regardless of source_type/
    #    contact_type -- FVG-family "invalidation" is replaced by
    #    "suspension"; a newly-known signal itself is never suspended.
    #    Mitigation facts are never added here: they are never
    #    themselves a confluence.
    for signal in signal_batch:
        active.setdefault(signal.timeframe, {}).setdefault(signal.direction, {})[signal.candidate_id] = signal

    return suspensions, restorations


def _merge_known_at_groups(
    signals: list[EntrySignal], mitigations: list[FVGMitigationEvent]
):
    """Yields ``(signal_batch, mitigation_batch)`` pairs, one per
    distinct ``known_at`` across both streams combined, in chronological
    order -- each batch internally ordered exactly like a plain
    ``sorted(signals, key=lambda s: (s.known_at, s.candidate_id))``
    would already be, so merging in the mitigation stream never
    perturbs pre-existing ordering among signals sharing a given
    ``known_at``."""
    tagged = [(s.known_at, s.candidate_id, 0, s) for s in signals] + [(m.known_at, m.zone_id, 1, m) for m in mitigations]
    tagged.sort(key=lambda t: (t[0], t[1]))
    for _known_at, group_iter in groupby(tagged, key=lambda t: t[0]):
        group = list(group_iter)
        yield [t[3] for t in group if t[2] == 0], [t[3] for t in group if t[2] == 1]


def _resolve_liquidity(signals: list[EntrySignal]) -> tuple[list[EntrySignal], list[SignalInvalidation]]:
    """Fresh, self-contained replay of liquidity invalidation over
    ``signals`` (already filtered to known-by-ts_utc and to the caller's
    chosen selectors) -- used by ``query_active_signals``."""
    ordered = sorted(signals, key=lambda s: (s.known_at, s.candidate_id))
    active: dict[str, dict[str, EntrySignal]] = {}
    invalidations: list[SignalInvalidation] = []
    for _known_at, batch_iter in groupby(ordered, key=lambda s: s.known_at):
        invalidations.extend(_apply_liquidity_batch(list(batch_iter), active))
    all_active = [s for pool in active.values() for s in pool.values()]
    return all_active, invalidations


def _resolve_fvg_conflicts(
    signals: list[EntrySignal],
    mitigations: list[FVGMitigationEvent],
) -> tuple[list[EntrySignal], list[SignalSuspension], list[SignalRestoration]]:
    """Fresh, self-contained replay of FVG-family suspend/restore over
    ``signals`` (already filtered to known-by-ts_utc and to the caller's
    chosen selectors) and ``mitigations`` (already filtered to known-by-
    ts_utc; never selector-filtered -- a mitigation event only ever
    resolves a zone whose own touch was itself already selector-
    filtered in, via ``touch_by_zone``/``pending_resolution``, so an
    unselected zone's mitigation is inherently a no-op) -- used by
    ``query_active_signals``."""
    active: dict[str, dict[str, dict[str, EntrySignal]]] = {}
    suspended: dict[str, EntrySignal] = {}
    suspended_by: dict[str, SignalSuspension] = {}
    pending_resolution: dict[str, list[str]] = {}
    touch_by_zone: dict[str, tuple[str, str]] = {}
    restorations: list[SignalRestoration] = []
    for signal_batch, mitigation_batch in _merge_known_at_groups(signals, mitigations):
        _batch_suspensions, batch_restorations = _apply_fvg_batch(
            signal_batch, mitigation_batch, active, suspended, suspended_by, pending_resolution, touch_by_zone
        )
        restorations.extend(batch_restorations)
    all_active = [s for tf_pools in active.values() for direction_pool in tf_pools.values() for s in direction_pool.values()]
    return all_active, list(suspended_by.values()), restorations


def query_active_signals(
    timeline: EntrySignalTimeline,
    selectors: set[str],
    direction: str,
    ts_utc: datetime,
) -> EntrySignalQueryResult:
    """Every signal that is known, restricted to ``selectors`` and
    ``direction``, and neither invalidated (liquidity) nor currently
    suspended (FVG family) as of ``ts_utc`` -- exactly the query
    contract a future strategy enumerator needs (see module docstring).
    Liquidity and the FVG family are resolved completely independently
    and never interact.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}, got {direction!r}")
    if ts_utc.tzinfo is None:
        raise ValueError(f"ts_utc must be timezone-aware: {ts_utc!r}")
    ts_utc = ts_utc.astimezone(UTC)

    candidate_signals = [
        signal for selector in selectors for signal in timeline.signals_by_selector.get(selector, []) if signal.known_at <= ts_utc
    ]
    liquidity_signals = [s for s in candidate_signals if s.source_kind == SourceKind.LIQUIDITY]
    fvg_signals = [s for s in candidate_signals if s.source_kind == SourceKind.FVG]
    # Never selector-filtered (see _resolve_fvg_conflicts's own docstring):
    # a mitigation is inert unless the zone it resolves was itself already
    # selector-filtered in via an actual touch signal above.
    mitigations = [m for m in timeline.zone_mitigations if m.known_at <= ts_utc]

    liquidity_active, invalidations = _resolve_liquidity(liquidity_signals)
    fvg_active, suspensions, restorations = _resolve_fvg_conflicts(fvg_signals, mitigations)

    active = [s for s in liquidity_active + fvg_active if s.direction == direction]
    invalidations = [inv for inv in invalidations if inv.signal.direction == direction]
    suspensions = [sus for sus in suspensions if sus.signal.direction == direction]
    restorations = [res for res in restorations if res.signal.direction == direction]

    active.sort(key=lambda s: (s.known_at, s.candidate_id))
    invalidations.sort(key=lambda inv: (inv.signal.known_at, inv.signal.candidate_id))
    suspensions.sort(key=lambda sus: (sus.signal.known_at, sus.signal.candidate_id))
    restorations.sort(key=lambda res: (res.signal.known_at, res.signal.candidate_id))
    return EntrySignalQueryResult(active_signals=active, invalidations=invalidations, suspensions=suspensions, restorations=restorations)


@dataclass(frozen=True)
class SignalSnapshot:
    """One real ``EntrySignal``'s own ``known_at`` instant, paired with
    the full, per-selector snapshot of every signal of that *same*
    ``signal.direction`` simultaneously active (known, not invalidated,
    not currently suspended) at that exact instant -- semantically
    identical to grouping ``query_active_signals(timeline,
    set(timeline.signals_by_selector), signal.direction,
    signal.known_at).active_signals`` by ``.selector``, but produced by
    ``iter_active_signal_snapshots`` in one linear forward sweep instead
    of a fresh query. ``active_by_selector`` always includes ``signal``
    itself under its own ``signal.selector`` key."""

    signal: EntrySignal
    active_by_selector: dict[str, tuple[EntrySignal, ...]]


def iter_active_signal_snapshots(timeline: EntrySignalTimeline) -> Iterator[SignalSnapshot]:
    """Yield one ``SignalSnapshot`` per real signal in ``timeline.signals``
    (the timeline's own chronological order), each equivalent to a fresh
    ``query_active_signals`` call at that signal's own ``known_at`` for
    its own ``direction`` (with ``selectors`` = every selector in the
    timeline), grouped by selector -- computed incrementally in
    ``O(N log N)`` total over ``N`` signals via a forward sweep, instead
    of ``O(N)`` fresh queries each themselves scanning the whole
    timeline. Never changes ``query_active_signals`` itself; see the
    module docstring's "Bulk snapshotting" section.

    Liquidity and the FVG family are tracked as two entirely separate,
    incrementally-carried states (mirroring ``query_active_signals``'s
    own independent resolution of each), combined only when flattening
    the outward ``active_by_selector`` snapshot.

    Signals are processed in ``known_at``-grouped batches, not one at a
    time: two (or more) signals sharing the exact same ``known_at`` must
    each see *all* of the others in that same group as already known
    when their own snapshot is taken, and never invalidate, suspend, or
    restore each other (mirrors ``query_active_signals``'s own strict
    ``>`` comparisons throughout). ``timeline.zone_mitigations`` are
    merged into the same chronological batching as the FVG family (see
    ``_merge_known_at_groups``) so a mitigation event still resolves
    zones at exactly its own ``known_at``, relative to every real
    signal -- it just never yields a ``SignalSnapshot`` of its own,
    since it is never itself a confluence.
    """
    liquidity_active: dict[str, dict[str, EntrySignal]] = {}
    fvg_active: dict[str, dict[str, dict[str, EntrySignal]]] = {}
    fvg_suspended: dict[str, EntrySignal] = {}
    fvg_suspended_by: dict[str, SignalSuspension] = {}
    fvg_pending_resolution: dict[str, list[str]] = {}
    fvg_touch_by_zone: dict[str, tuple[str, str]] = {}

    for real_signal_batch, mitigation_batch in _merge_known_at_groups(timeline.signals, timeline.zone_mitigations):
        liquidity_batch = [s for s in real_signal_batch if s.source_kind == SourceKind.LIQUIDITY]
        fvg_batch = [s for s in real_signal_batch if s.source_kind == SourceKind.FVG]

        if liquidity_batch:
            _apply_liquidity_batch(liquidity_batch, liquidity_active)
        if fvg_batch or mitigation_batch:
            _apply_fvg_batch(fvg_batch, mitigation_batch, fvg_active, fvg_suspended, fvg_suspended_by, fvg_pending_resolution, fvg_touch_by_zone)

        for signal in real_signal_batch:
            active_by_selector: dict[str, list[EntrySignal]] = {}
            for pool in liquidity_active.values():
                for other in pool.values():
                    if other.direction == signal.direction:
                        active_by_selector.setdefault(other.selector, []).append(other)
            for tf_pools in fvg_active.values():
                for other in tf_pools.get(signal.direction, {}).values():
                    active_by_selector.setdefault(other.selector, []).append(other)
            yield SignalSnapshot(signal=signal, active_by_selector={selector: tuple(sigs) for selector, sigs in active_by_selector.items()})


def iter_fvg_lifecycle_transitions(timeline: EntrySignalTimeline) -> Iterator[SignalSuspension | SignalRestoration]:
    """Every FVG-family suspension and restoration across the whole
    timeline, in the exact chronological order they actually happened --
    diagnostic/reporting only (e.g. ``tradelab.profile_research_context``'s
    active-duration statistics). Unlike ``EntrySignalQueryResult``'s audit
    lists, which only ever reflect state as of one caller-chosen
    ``ts_utc`` (so a signal that was suspended and later restored no
    longer appears in ``suspensions`` at all, and its original
    suspension time is lost), this surfaces the complete transition
    history: every suspension exactly when it happened, and every
    restoration exactly when it happened, so a consumer can pair them up
    by ``signal.candidate_id`` to compute how long each confluence was
    actually active.

    Reuses the exact same incremental ``_apply_fvg_batch`` sweep as
    ``iter_active_signal_snapshots`` -- ``O(N log N)`` total, one linear
    forward pass, never re-implements the suspend/restore rules. Never
    changes ``query_active_signals`` or ``iter_active_signal_snapshots``
    themselves; liquidity is untouched (it has no suspension/restoration
    concept, only permanent invalidation -- see module docstring).
    """
    active: dict[str, dict[str, dict[str, EntrySignal]]] = {}
    suspended: dict[str, EntrySignal] = {}
    suspended_by: dict[str, SignalSuspension] = {}
    pending_resolution: dict[str, list[str]] = {}
    touch_by_zone: dict[str, tuple[str, str]] = {}

    fvg_signals = [s for s in timeline.signals if s.source_kind == SourceKind.FVG]
    for signal_batch, mitigation_batch in _merge_known_at_groups(fvg_signals, timeline.zone_mitigations):
        batch_suspensions, batch_restorations = _apply_fvg_batch(
            signal_batch, mitigation_batch, active, suspended, suspended_by, pending_resolution, touch_by_zone
        )
        yield from batch_suspensions
        yield from batch_restorations


class _ActivePoolRefCounter:
    """Per-direction selector ref-counts and total instance counts for
    the currently-active pool -- the only state ``iter_active_pool_
    scalar_snapshots`` needs. Never stores an ``EntrySignal`` itself,
    only integers, so memory stays ``O(number of currently active
    unique selectors)`` rather than growing with the number of
    snapshots taken."""

    __slots__ = ("_selector_counts", "_instance_counts", "_selector_set_revision")

    def __init__(self) -> None:
        self._selector_counts: dict[str, dict[str, int]] = {}
        self._instance_counts: dict[str, int] = {}
        # Bumped only when a direction's *selector key set* actually changes
        # (a selector appears or disappears), never when a count merely moves
        # up or down. ``active_selectors`` is therefore reusable between two
        # snapshots taken across an unchanged key set -- see
        # ``selector_set_revision``.
        self._selector_set_revision: dict[str, int] = {}

    def _bump(self, direction: str) -> None:
        self._selector_set_revision[direction] = self._selector_set_revision.get(direction, 0) + 1

    def selector_set_revision(self, direction: str) -> int:
        """Monotonic revision of ``active_selectors(direction)``'s own value.

        Equal revisions guarantee an equal selector set, so a caller may
        safely reuse a previously built frozenset instead of allocating an
        identical one. Used by ``iter_candidate_entry_snapshots``, which
        would otherwise build ~1.2M identical frozensets on real MNQ data.
        """
        return self._selector_set_revision.get(direction, 0)

    def increment(self, direction: str, selector: str) -> None:
        sel_map = self._selector_counts.setdefault(direction, {})
        if selector not in sel_map:
            self._bump(direction)
        sel_map[selector] = sel_map.get(selector, 0) + 1
        self._instance_counts[direction] = self._instance_counts.get(direction, 0) + 1

    def decrement(self, direction: str, selector: str) -> None:
        sel_map = self._selector_counts[direction]
        remaining = sel_map[selector] - 1
        if remaining <= 0:
            del sel_map[selector]
            self._bump(direction)
        else:
            sel_map[selector] = remaining
        self._instance_counts[direction] -= 1

    def unique_selector_count(self, direction: str) -> int:
        return len(self._selector_counts.get(direction, {}))

    def instance_count(self, direction: str) -> int:
        return self._instance_counts.get(direction, 0)

    def active_selectors(self, direction: str) -> frozenset[str]:
        """The actual set of selector strings currently active in
        ``direction`` -- O(number of currently active unique selectors),
        never O(pool size): reads back the counter's own keys, never
        rescans any signal pool. Used by ``MarketStateSnapshot`` (see
        ``iter_candidate_entry_snapshots``), which needs real selector
        identities, not just a count."""
        return frozenset(self._selector_counts.get(direction, {}).keys())


class _RefCountingDict(dict):
    """A drop-in replacement for the plain ``{}`` used at every level of
    ``liquidity_active``/``fvg_active`` in ``_apply_liquidity_batch``/
    ``_apply_fvg_batch``/``_resolve_zone`` -- **reuses those functions
    completely unmodified**, so the active/suspended/restored/retired
    transition semantics can never drift from theirs (no rule is
    reimplemented here). It only adds a side effect on the dict
    mutation protocol those functions already use throughout
    (``__setitem__``, ``__delitem__``, ``pop``, ``setdefault``):
    whenever a *leaf* value (an ``EntrySignal``, never another dict) is
    inserted or removed, the shared ``_ActivePoolRefCounter`` is
    incremented/decremented for that signal's own ``(direction,
    selector)`` -- an O(1) update per transition, never a full rescan
    of the active pool. ``setdefault(key, {})`` (used throughout those
    functions purely to create an intermediate nesting level on first
    use, e.g. ``active.setdefault(timeframe, {})``) transparently
    substitutes another ``_RefCountingDict`` sharing the same counter
    for the literal ``{}``, so nesting stays uniform to any depth
    (liquidity is 2 levels deep, FVG is 3) without either caller
    needing to know about this class at all."""

    __slots__ = ("_counter", "key_set_revision")

    def __init__(self, counter: "_ActivePoolRefCounter") -> None:
        super().__init__()
        self._counter = counter
        # Monotonic revision of this dict's own *key set*, bumped on every
        # insert of a new key and every removal -- never when an existing key
        # is merely reassigned, which cannot change ``sorted(self)``. A
        # snapshot consumer that only reads keys (see
        # ``iter_candidate_entry_snapshots``) can therefore reuse a
        # previously built ``tuple(sorted(...))`` whenever the revision is
        # unchanged, instead of rebuilding an identical tuple per snapshot.
        self.key_set_revision = 0

    def setdefault(self, key, default):
        if key in self:
            return self[key]
        if isinstance(default, dict):
            default = _RefCountingDict(self._counter)
        dict.__setitem__(self, key, default)
        self.key_set_revision += 1
        return default

    def __setitem__(self, key, value) -> None:
        is_new = key not in self
        dict.__setitem__(self, key, value)
        if is_new:
            self.key_set_revision += 1
            if not isinstance(value, dict):
                self._counter.increment(value.direction, value.selector)

    def __delitem__(self, key) -> None:
        value = self[key]
        dict.__delitem__(self, key)
        self.key_set_revision += 1
        if not isinstance(value, dict):
            self._counter.decrement(value.direction, value.selector)

    def pop(self, key, *default):
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = self[key]
        dict.__delitem__(self, key)
        self.key_set_revision += 1
        if not isinstance(value, dict):
            self._counter.decrement(value.direction, value.selector)
        return value


@dataclass(frozen=True)
class ActivePoolScalarSnapshot:
    """One real ``EntrySignal``'s own ``known_at`` instant, paired with
    **only** the O(1) scalar active-pool-size counts in that signal's
    own direction -- never a materialized list or dict of active
    ``EntrySignal`` instances (see ``SignalSnapshot``/``iter_active_
    signal_snapshots`` for that). Built for ``tradelab.profile_research_
    context``'s ``active_pool_stats``: a real-MNQ run of that section
    via ``iter_active_signal_snapshots`` (rebuilding a full ``active_by_
    selector`` dict at every one of ~1.2M real signals) was the actual
    bottleneck -- >2GB RAM, over an hour. This carries only integers.

    ``active_selector_count_including_primary`` and ``active_signal_
    instance_count`` are both counted only over ``direction``'s own
    active pool, exactly like ``SignalSnapshot.active_by_selector``
    already restricts itself to. ``required_confluence_selector_count_
    excluding_primary`` is ``active_selector_count_including_primary``
    minus the primary signal's own selector -- guaranteed present (see
    ``iter_active_pool_scalar_snapshots``), so always exactly one less.
    """

    signal: EntrySignal
    active_selector_count_including_primary: int
    required_confluence_selector_count_excluding_primary: int
    active_signal_instance_count: int


def iter_active_pool_scalar_snapshots(timeline: EntrySignalTimeline) -> Iterator[ActivePoolScalarSnapshot]:
    """Yield one ``ActivePoolScalarSnapshot`` per real signal in
    ``timeline.signals``, in the timeline's own chronological order --
    the exact same transition sweep ``iter_active_signal_snapshots``
    already performs (same ``_merge_known_at_groups`` batching, the
    exact same, unmodified ``_apply_liquidity_batch``/``_apply_fvg_
    batch`` calls -- lifecycle semantics are reused, never reimplemented
    or approximated), except ``liquidity_active``/``fvg_active`` are
    ``_RefCountingDict`` instances instead of plain ``dict``s, so no
    per-snapshot ``active_by_selector`` is ever built: every snapshot's
    counts are read back from the shared ``_ActivePoolRefCounter`` in
    O(1). Time stays ``O(N log N)`` total (identical to ``iter_active_
    signal_snapshots``); memory stays ``O(number of currently active
    unique selectors/signals)``, never ``O(number of snapshots x active
    pool size)``.

    Proven exactly equivalent to ``iter_active_signal_snapshots`` (same
    per-signal ``direction``-scoped active-selector/instance counts) on
    randomized fixtures exercising liquidity invalidation, FVG
    suspension/restoration/mitigation-resolution, and nested FVG
    conflicts together -- see ``tests/test_tradelab_entry_signal_
    timeline.py``'s ``ActivePoolScalarSnapshotEquivalenceTests``. Never
    changes ``iter_active_signal_snapshots`` itself, which remains the
    audit/debug API for inspecting the actual active ``EntrySignal``
    instances at any given real signal's own instant.
    """
    counter = _ActivePoolRefCounter()
    liquidity_active: dict[str, dict[str, EntrySignal]] = _RefCountingDict(counter)
    fvg_active: dict[str, dict[str, dict[str, EntrySignal]]] = _RefCountingDict(counter)
    fvg_suspended: dict[str, EntrySignal] = {}
    fvg_suspended_by: dict[str, SignalSuspension] = {}
    fvg_pending_resolution: dict[str, list[str]] = {}
    fvg_touch_by_zone: dict[str, tuple[str, str]] = {}

    for real_signal_batch, mitigation_batch in _merge_known_at_groups(timeline.signals, timeline.zone_mitigations):
        liquidity_batch = [s for s in real_signal_batch if s.source_kind == SourceKind.LIQUIDITY]
        fvg_batch = [s for s in real_signal_batch if s.source_kind == SourceKind.FVG]

        if liquidity_batch:
            _apply_liquidity_batch(liquidity_batch, liquidity_active)
        if fvg_batch or mitigation_batch:
            _apply_fvg_batch(fvg_batch, mitigation_batch, fvg_active, fvg_suspended, fvg_suspended_by, fvg_pending_resolution, fvg_touch_by_zone)

        for signal in real_signal_batch:
            active_selector_count = counter.unique_selector_count(signal.direction)
            instance_count = counter.instance_count(signal.direction)
            yield ActivePoolScalarSnapshot(
                signal=signal,
                active_selector_count_including_primary=active_selector_count,
                required_confluence_selector_count_excluding_primary=active_selector_count - 1,
                active_signal_instance_count=instance_count,
            )


# ---------------------------------------------------------------------------
# Market state vs. entry events (this task): the ``EntrySignal``/``query_
# active_signals``/``iter_active_signal_snapshots`` model above treats every
# FVG touch, iFVG, and liquidity signal as "active until specifically
# invalidated" -- correct for the FVG *zone*'s own structural lifecycle
# (formalized on purpose, see module docstring), but semantically wrong when
# the same rule is applied to the *event* of a touch/sweep itself: a real-
# MNQ profile showed a median 219 simultaneously "active" required-
# confluence selectors and 255,886 "active" signal instances at real primary
# signal timestamps -- old, one-off historical occurrences that nothing
# later specifically invalidated, not a genuinely large simultaneous market
# state. This is not a performance bug (``iter_active_pool_scalar_snapshots``
# above already makes computing that number fast); it is that the number
# itself does not mean what a future strategy enumerator needs it to mean.
#
# This section adds a second, explicit, additive API -- ``EntryEvent``/
# ``MarketStateSnapshot``/``CandidateEntrySnapshot`` -- that never changes
# ``EntrySignal``/``query_active_signals``/``iter_active_signal_snapshots``/
# ``iter_active_pool_scalar_snapshots`` themselves (every existing consumer
# of those keeps working, unchanged). See ``docs/market-state-vs-entry-
# events-v1.md`` for the full rationale; in short:
#
# - A **persistent state fact** (an FVG zone's own active/suspended/
#   retired structural status, or the current HTF bias) genuinely does not
#   expire on its own -- it stays valid, unbounded, until its own explicit
#   structural terminal rule fires (a same-timeframe opposite challenger's
#   resolution, or its own inversion/mitigation; a bias change).
# - An **atomic entry event** (an FVG touch, a confirmed iFVG, or a
#   liquidity sweep) is known exactly once, at its own ``known_at`` -- it
#   may be a strategy's primary trigger *at that instant*, but there is no
#   function here that lets a caller ask "which events are still eligible
#   as of some later T" the way ``query_active_signals`` does for
#   ``EntrySignal``. A historical event that nothing later specifically
#   invalidated does not thereby remain an eligible primary indefinitely --
#   it simply is not offered again after its own instant.
#
# **A zone's persistent state must be visible from its own formation, not
# only from its first touch.** An earlier version of this section built FVG
# state solely from ``FVGEntryCandidateResult.candidates`` (touch/iFVG
# entry candidates) -- but a zone that has never been touched has no entry
# candidate at all, so it was invisible as context until *something*
# finally touched it, even though it was genuinely, structurally active
# (unmitigated) the whole time. A zone formed at 10:00 and never touched
# must still be visible in the ``MarketStateSnapshot`` of an unrelated
# entry event at 12:00. Fixed by tracking FVG zone state **by zone_id**,
# entering the pool at each zone's own ``formed_at`` (read straight from
# ``tradelab.fvg.FVGZone``, the real detector output -- never inferred from
# entry candidates), via a dedicated ``_apply_fvg_zone_batch`` that
# reimplements (necessarily -- the value model is zone-keyed, not
# candidate_id-keyed, so ``_apply_fvg_batch`` itself cannot be reused
# as-is) the exact same suspend/restore/retire/mitigation-resolution rules
# already proven correct above. ``EntryEvent``s themselves are unaffected
# by this: they are still built only from actual touch/iFVG/liquidity
# candidates (``build_entry_events``, unchanged) -- formation is never an
# ``EntryEvent``, only ever a persistent-state fact.
#
# **A confirmed iFVG is an atomic EntryEvent, never persistent state.** An
# earlier version of ``_apply_fvg_zone_batch`` retired an original zone on
# its own inversion (correct) but then re-inserted that same zone_id into
# ``zone_active`` under an ``ifvg`` selector, in the inverted direction --
# making the iFVG itself an indefinitely persistent confluence fact, the
# exact bug class this whole section exists to eliminate. Fixed: an iFVG is
# retired exactly like a plain, non-inverting mitigation (the original
# zone's presence is removed, whatever it had itself suspended is
# restored) and **nothing takes its place** -- ``EntryEventKind.IFVG``
# still fires as one atomic ``EntryEvent`` at its own ``known_at``,
# entirely independent of ``zone_active``. Persistent FVG state
# (``MarketStateSnapshot.active_fvg_selectors_by_direction``) therefore
# represents **only structurally active original FVG zones** -- bounded by
# 6 timeframes x 1 (original-FVG-only) = 6 possible selector strings per
# direction, not 12.
# ---------------------------------------------------------------------------


class EntryEventKind(str, Enum):
    FVG_TOUCH = "fvg_touch"
    IFVG = "ifvg"
    LIQUIDITY_SWEEP = "liquidity_sweep"


@dataclass(frozen=True)
class EntryEvent:
    """One atomic, one-shot market-entry trigger, known exactly once at
    its own ``known_at`` -- an FVG touch, a confirmed iFVG, or a
    liquidity sweep (never a data wick -- spec 4.3, TP-only, and this
    module is never given data wicks as an input to begin with). Never
    an everlasting/reusable confluence fact: ``iter_candidate_entry_
    snapshots`` yields exactly one ``CandidateEntrySnapshot`` per
    ``EntryEvent``, at its own ``known_at`` -- there is no query
    anywhere in this section that re-offers an ``EntryEvent`` as a
    fresh primary trigger at any later instant, unlike ``EntrySignal``/
    ``query_active_signals``'s persist-until-invalidated model above.

    ``event_id`` is the underlying candidate's own ``candidate_id`` --
    stable, deterministic provenance, identical to what ``EntrySignal``
    uses for the exact same fact (built via the exact same ``_signal_
    from_fvg``/``_signal_from_liquidity`` normalization, so ``selector``/
    ``timeframe``/``direction``/``known_at`` can never drift from what
    ``EntrySignal`` would report). ``zone_id``/``level_id`` are
    mutually exclusive FVG/liquidity provenance -- ``None`` for
    whichever family does not apply."""

    event_id: str
    kind: EntryEventKind
    selector: str
    timeframe: str
    direction: str
    timestamp: datetime
    known_at: datetime
    zone_id: str | None
    level_id: str | None
    source: FVGEntryCandidate | LiquidityEntryCandidate


@dataclass(frozen=True)
class MarketStateSnapshot:
    """Persistent, structural market state as of exactly one instant --
    never a list of historical events, never accumulated across
    ``EntryEvent``s. Two components, both genuinely, unboundedly
    persistent by design (a real structural fact, not a cap or
    approximation -- see ``docs/market-state-vs-entry-events-v1.md``):

    - ``active_fvg_selectors_by_direction``: for each direction, the
      set of **original-FVG** selectors (``"fvg-{timeframe}-fvg_touch"``
      -- never an ``ifvg`` string, see below) with at least one zone
      that is currently structurally ACTIVE in its own original
      direction -- not currently suspended by a same-timeframe
      opposite-direction challenger, and not permanently retired by its
      own resolution (inversion or mitigation). A zone contributes to
      its timeframe's ``fvg_touch`` selector from its own ``formed_at``
      onward -- **whether or not it has ever actually been touched**;
      the same suspend/restore/retire/mitigation-resolution rules
      ``_apply_fvg_batch`` implements for the older, signal-keyed model
      apply here too, just tracked by zone_id instead of candidate_id
      (see ``_apply_fvg_zone_batch``) so an untouched, unresolved zone
      is never invisible. **A confirmed iFVG is never represented
      here** -- when a zone inverts, it is retired from this set
      (exactly like a plain, non-inverting mitigation) and nothing
      takes its place; the inversion itself is reported solely as a
      one-shot ``EntryEvent`` (``EntryEventKind.IFVG``), never as
      persistent state (see that type's own docstring). At most 6
      timeframes x 1 (original-FVG-only) = 6 possible strings per
      direction, system-wide, ever -- bounded by the FVG-family
      selector catalog itself, never by how much history has
      accumulated.
    - ``bias``: the HTF bias snapshot (``tradelab.bias.BiasSnapshot``,
      identical to what ``tradelab.bias.bias_at`` would return at this
      same instant -- proven equivalent by randomized test, computed
      here incrementally instead of by calling ``bias_at`` per event,
      since that function's own linear rescan-from-the-start would
      reintroduce an ``O(events x bias changes so far)`` blowup at
      real-MNQ scale) as of this exact instant.

    Liquidity has **no persistent role here at all**, by design (spec,
    this task): a liquidity sweep is only ever a one-shot ``EntryEvent``,
    never reusable confluence context for a later, unrelated event --
    unlike the older ``EntrySignal``/``query_active_signals`` model,
    where a liquidity signal stayed "active" until a same-selector
    opposite signal specifically invalidated it, letting old sweeps
    silently become perpetual context."""

    as_of: datetime
    active_fvg_selectors_by_direction: Mapping[str, frozenset[str]]
    active_fvg_zone_ids_by_direction_selector: Mapping[str, Mapping[str, tuple[str, ...]]]
    bias: BiasSnapshot


@dataclass(frozen=True)
class CandidateEntrySnapshot:
    """The deterministic join of exactly one real ``EntryEvent`` with
    the ``MarketStateSnapshot`` known at that event's own ``known_at``
    -- the sole, explicit API for "what could this event's own strategy
    trigger see as context." One snapshot per real event, in the
    timeline's own chronological order -- never more, never re-derived
    later, so it is structurally impossible to accidentally treat an
    ``EntryEvent`` as an indefinitely active selector: there is no
    function anywhere in this section that returns "every event still
    eligible as of instant T" the way ``query_active_signals`` does for
    ``EntrySignal``."""

    event: EntryEvent
    market_state: MarketStateSnapshot


def _entry_event_from_signal(signal: EntrySignal) -> EntryEvent:
    if signal.source_kind == SourceKind.FVG:
        fvg_candidate: FVGEntryCandidate = signal.source
        kind = EntryEventKind.IFVG if fvg_candidate.source_type == FVGSourceType.IFVG else EntryEventKind.FVG_TOUCH
        zone_id, level_id = fvg_candidate.zone_id, None
    else:
        liquidity_candidate: LiquidityEntryCandidate = signal.source
        kind = EntryEventKind.LIQUIDITY_SWEEP
        zone_id, level_id = None, liquidity_candidate.level_id
    return EntryEvent(
        event_id=signal.candidate_id,
        kind=kind,
        selector=signal.selector,
        timeframe=signal.timeframe,
        direction=signal.direction,
        timestamp=signal.timestamp,
        known_at=signal.known_at,
        zone_id=zone_id,
        level_id=level_id,
        source=signal.source,
    )


def build_entry_events(
    fvg_result: FVGEntryCandidateResult, liquidity_result: LiquidityEntryCandidateResult
) -> list[EntryEvent]:
    """Every FVG touch, confirmed iFVG, and liquidity sweep, normalized
    into one atomic ``EntryEvent`` each -- chronological by ``known_at``
    (ties broken by ``event_id``, matching ``build_entry_signal_
    timeline``'s own ``EntrySignal`` ordering convention exactly).
    Reuses the exact same ``_signal_from_fvg``/``_signal_from_liquidity``
    normalization ``build_entry_signal_timeline`` uses, so an event's
    ``selector``/``timeframe``/``direction``/``known_at`` are always
    identical to what ``EntrySignal`` would report for the same
    underlying fact -- this function never re-derives them
    independently. Zone mitigations are never represented as an
    ``EntryEvent`` -- exactly like they are never an ``EntrySignal``
    either: a pure zone-resolution fact, never itself a new confluence
    (see ``tradelab.fvg.FVGMitigationEvent``'s own docstring)."""
    signals = [_signal_from_fvg(c) for c in fvg_result.candidates] + [
        _signal_from_liquidity(c) for c in liquidity_result.candidates
    ]
    events = [_entry_event_from_signal(s) for s in signals]
    events.sort(key=lambda e: (e.known_at, e.event_id))
    return events


def _merge_confluence_streams(
    formations: list[FVGZone],
    fvg_signals: list[EntrySignal],
    mitigations: list[FVGMitigationEvent],
    bias_changes: list[TimeframeBiasChange],
    liquidity_signals: list[EntrySignal],
):
    """Yields ``(formation_batch, fvg_signal_batch, mitigation_batch,
    bias_change_batch, liquidity_batch)`` tuples, one per distinct
    ``known_at``/``formed_at`` across all five streams combined,
    chronological -- generalizes ``_merge_known_at_groups`` to the
    additional streams ``iter_candidate_entry_snapshots`` needs (zone
    formations, bias changes, liquidity events), with the exact same
    guarantee: batch members sharing one instant never affect each
    other, only strictly earlier batches do."""
    tagged = (
        [(z.formed_at, z.zone_id, 0, z) for z in formations]
        + [(s.known_at, s.candidate_id, 1, s) for s in fvg_signals]
        + [(m.known_at, m.zone_id, 2, m) for m in mitigations]
        + [(c.known_at, c.timeframe, 3, c) for c in bias_changes]
        + [(s.known_at, s.candidate_id, 4, s) for s in liquidity_signals]
    )
    tagged.sort(key=lambda t: (t[0], t[2], t[1]))
    for _known_at, group_iter in groupby(tagged, key=lambda t: t[0]):
        group = list(group_iter)
        yield (
            [t[3] for t in group if t[2] == 0],
            [t[3] for t in group if t[2] == 1],
            [t[3] for t in group if t[2] == 2],
            [t[3] for t in group if t[2] == 3],
            [t[3] for t in group if t[2] == 4],
        )


@dataclass(frozen=True)
class _ZoneStateEntry:
    """Leaf value for the zone-keyed persistent-FVG-state pool
    ``_apply_fvg_zone_batch`` maintains -- carries exactly what
    ``_RefCountingDict`` needs (``.direction``, ``.selector``) plus the
    zone's own identity, so ``iter_candidate_entry_snapshots`` never
    needs to look anything up in ``fvg_result.candidates`` to answer
    "which selectors are currently active."""

    zone_id: str
    timeframe: str
    direction: str
    selector: str


def _fvg_zone_selector(timeframe: str, kind: str) -> str:
    """Identical string format to ``_fvg_selector`` (``"fvg-{timeframe}-
    {fvg_touch|ifvg}"``), built from a raw ``FVGZone`` (which has no
    ``source_type``) instead of an ``FVGEntryCandidate``."""
    return f"fvg-{timeframe}-{kind}"


# tradelab.bias re-exports tradelab.fvg's own BULLISH/BEARISH verbatim
# (see tradelab/bias.py), so this is the exact same direction vocabulary
# FVGZone.direction uses -- translated to this module's LONG/SHORT, the
# same translation tradelab.fvg_entry's own (private) candidate-building
# code already performs for touch/iFVG candidates.
_FVG_ZONE_DIRECTION = {BULLISH: LONG, BEARISH: SHORT}


def _apply_fvg_zone_batch(
    formation_batch: list[FVGZone],
    touch_batch: list[EntrySignal],
    mitigation_batch: list[FVGMitigationEvent],
    zone_active: dict[str, dict[str, dict[str, _ZoneStateEntry]]],
    zone_suspended: dict[str, _ZoneStateEntry],
    zone_suspended_by: dict[str, str],
    zone_entries: dict[str, _ZoneStateEntry],
    pending_resolution: dict[str, list[str]],
) -> None:
    """Mutates ``zone_active``/``zone_suspended``/``zone_suspended_by``/
    ``zone_entries``/``pending_resolution`` in place for one instant's
    batch -- the zone-keyed counterpart of ``_apply_fvg_batch``,
    covering every zone (touched or not) instead of only zones that
    have an ``EntrySignal``. Necessarily a **reimplementation**, not a
    reuse, of ``_apply_fvg_batch``: that function's value model is
    keyed by an entry candidate's own ``candidate_id``, which does not
    exist for an untouched zone at all, so it structurally cannot
    represent "zone formed, never touched, still active." The rules
    themselves are identical (same suspend/restore/retire/mitigation-
    resolution semantics, same same-timeframe/opposite-direction/
    durable-vs-immediate-resolution challenge logic) -- see this
    module's own docstring and ``tests/test_tradelab_entry_signal_
    timeline.py``'s ``FVGZoneStateEquivalenceTests`` for direct
    cross-checks against ``_apply_fvg_batch`` on every zone that *does*
    get touched, proving no semantic drift.

    **A confirmed iFVG is never (re-)inserted into ``zone_active``.**
    ``FVGEvent``/``EntryEventKind.IFVG`` is a one-shot, atomic
    ``EntryEvent`` (see this module's own top-of-section docstring) --
    never a persistent zone state. When an original zone's own iFVG
    arrives (step 1 below): (1) that zone's own, original-direction
    presence is permanently retired -- exactly like a plain,
    non-inverting mitigation, via the same ``_resolve`` call; (2)
    whatever that zone had itself suspended (as a challenger) is
    restored, per the existing suspend/restore rules, completely
    unaffected by this change; (3) ``build_entry_events``/``iter_
    candidate_entry_snapshots`` still emit exactly one atomic ``IFVG``
    ``EntryEvent`` for it, at its own ``known_at`` -- entirely
    independent of ``zone_active``, since ``EntryEvent`` construction
    never reads persistent state at all; (4) nothing is inserted back
    into ``zone_active`` in the zone's new, inverted direction --
    unlike an earlier version of this function, which incorrectly
    treated the iFVG as a *second*, indefinitely persistent confluence
    fact in its own right. Persistent FVG state therefore represents
    **only structurally active original FVG zones** -- an iFVG belongs
    solely to ``EntryEventKind.IFVG``, never to ``MarketStateSnapshot.
    active_fvg_selectors_by_direction``.

    ``zone_entries`` always holds a zone's current representation
    (timeframe/direction/selector) for every zone currently tracked,
    whether it lives in ``zone_active`` or ``zone_suspended`` --
    avoiding the need for a separate "which touch is this zone's own"
    lookup the way ``_apply_fvg_batch``'s ``touch_by_zone`` needs, since
    the zone itself (never a derived touch candidate_id) is the key
    throughout. ``zone_suspended_by`` maps a currently-suspended zone to
    its own challenger's zone_id, so that if the suspended zone is later
    separately retired (nested conflict), the challenger's own
    ``pending_resolution`` entry is cleaned up -- the exact same
    nested-conflict handling ``_apply_fvg_batch`` performs via
    ``suspended_by``.
    """

    def _resolve(zone_id: str) -> None:
        if zone_id in zone_entries:
            entry = zone_entries.pop(zone_id)
            if zone_id in zone_suspended:
                del zone_suspended[zone_id]
                challenger_zone_id = zone_suspended_by.pop(zone_id)
                still_pending = [zid for zid in pending_resolution.get(challenger_zone_id, []) if zid != zone_id]
                if still_pending:
                    pending_resolution[challenger_zone_id] = still_pending
                else:
                    pending_resolution.pop(challenger_zone_id, None)
            else:
                del zone_active[entry.timeframe][entry.direction][zone_id]

        pending = pending_resolution.pop(zone_id, None)
        if pending:
            for suspended_zone_id in pending:
                restored_entry = zone_suspended.pop(suspended_zone_id)
                zone_suspended_by.pop(suspended_zone_id, None)
                zone_entries[suspended_zone_id] = restored_entry
                zone_active.setdefault(restored_entry.timeframe, {}).setdefault(restored_entry.direction, {})[
                    suspended_zone_id
                ] = restored_entry

    # 1. Resolve zones whose own confirmed iFVG or mitigation is in this
    #    batch -- permanently retires that zone's (always original-
    #    direction, since a zone only ever lives in zone_active under
    #    its original direction -- see step 3's own comment) presence,
    #    and restores whatever it had itself suspended. A confirmed
    #    iFVG is resolved via this exact same call and nothing else --
    #    it never gets inserted anywhere afterward (see this function's
    #    own docstring: an iFVG is a one-shot EntryEvent, never a new
    #    persistent state fact). A zone's own FULL_MITIGATION_NO_
    #    INVERSION touch always has a matching FVGMitigationEvent at
    #    this exact known_at (tradelab.fvg._apply_lifecycle emits one
    #    unconditionally on every non-inverting resolution, first
    #    contact or not), so it is already retired here too -- no
    #    separate handling needed below.
    for signal in touch_batch:
        candidate: FVGEntryCandidate = signal.source
        if candidate.source_type != FVGSourceType.IFVG:
            continue
        _resolve(candidate.zone_id)
    for mitigation in mitigation_batch:
        _resolve(mitigation.zone_id)

    # 2. Challenges -- a still-open PARTIAL_TOUCH, or an immediately-
    #    resolved FULL_MITIGATION_NO_INVERSION touch, challenges the
    #    pre-batch (post step-1) active pool for the opposite direction,
    #    same timeframe -- every zone there, touched or not. A durable
    #    (PARTIAL_TOUCH) challenge actually suspends the target; a
    #    non-durable one is an immediate no-op for state (the target was
    #    never really removed -- see _apply_fvg_batch's own docstring
    #    for the identical rule).
    challenge_actions: list[tuple[str, str, bool]] = []
    for signal in touch_batch:
        candidate = signal.source
        if candidate.source_type != FVGSourceType.FVG_TOUCH:
            continue
        if candidate.contact_type not in (FVGContactType.PARTIAL_TOUCH, FVGContactType.FULL_MITIGATION_NO_INVERSION):
            continue
        is_durable = candidate.contact_type == FVGContactType.PARTIAL_TOUCH
        opposite = _OPPOSITE_DIRECTION[signal.direction]
        for target_zone_id in list(zone_active.get(signal.timeframe, {}).get(opposite, {}).keys()):
            challenge_actions.append((target_zone_id, candidate.zone_id, is_durable))

    for target_zone_id, challenger_zone_id, is_durable in challenge_actions:
        entry = zone_entries.get(target_zone_id)
        if entry is None:
            continue  # already resolved (retired) earlier in this batch
        tf_pool = zone_active.get(entry.timeframe, {}).get(entry.direction, {})
        if target_zone_id not in tf_pool:
            continue  # already suspended earlier in this same batch
        if not is_durable:
            continue
        del tf_pool[target_zone_id]
        zone_suspended[target_zone_id] = entry
        zone_suspended_by[target_zone_id] = challenger_zone_id
        pending_resolution.setdefault(challenger_zone_id, []).append(target_zone_id)

    # 3. New zone formations enter the pool, in their own (original)
    #    direction, as fvg_touch-kind -- structurally active from this
    #    instant on, whether or not anything ever touches them. This is
    #    the *only* way anything ever enters zone_active: a confirmed
    #    iFVG is deliberately never (re-)inserted here (see this
    #    function's own docstring) -- it is retired in step 1 above,
    #    exactly like a plain mitigation, and nothing takes its place.
    for zone in formation_batch:
        direction = _FVG_ZONE_DIRECTION[zone.direction]
        entry = _ZoneStateEntry(
            zone_id=zone.zone_id, timeframe=zone.timeframe, direction=direction,
            selector=_fvg_zone_selector(zone.timeframe, FVGSourceType.FVG_TOUCH.value),
        )
        zone_entries[zone.zone_id] = entry
        zone_active.setdefault(zone.timeframe, {}).setdefault(direction, {})[zone.zone_id] = entry


def iter_candidate_entry_snapshots(
    fvg_results: Mapping[str, FVGDetectionResult],
    fvg_result: FVGEntryCandidateResult,
    liquidity_result: LiquidityEntryCandidateResult,
    htf_bias_result: HTFBiasResult,
    *,
    _share_immutable_parts: bool = True,
) -> Iterator[CandidateEntrySnapshot]:
    """Yield one ``CandidateEntrySnapshot`` per real ``EntryEvent`` (see
    ``build_entry_events``), in chronological order -- the deterministic
    join of that event with the ``MarketStateSnapshot`` known at its own
    ``known_at``, and the primary, intended entry point for future
    strategy-discovery confluence semantics (spec, this task).

    ``_share_immutable_parts`` (default ``True``) lets consecutive snapshots
    reuse the *identical* immutable objects for the parts of
    ``MarketStateSnapshot`` that did not change -- active selector sets,
    per-(timeframe, direction) sorted zone-id tuples, their containing
    mappings, and the ``BiasSnapshot``. This never changes a value, an
    ordering, or a type: it only stops the generator from allocating ~1.2M
    identical copies on real MNQ data, all of which
    ``entry_strategy_universe.discover_entry_strategy_universe`` retains at
    once (measured: ~30 GB of physical footprint before a single backtest
    variant ran -- see ``docs/dense-class-memory-diagnostic.md``). Pass
    ``False`` to get the original, copy-per-snapshot reference behavior; it
    exists so ``tests/test_tradelab_entry_signal_timeline.py``'s
    ``SharedSnapshotPartsEquivalenceTests`` can prove the two produce equal
    values, and is never used in production. Consumers must treat every
    ``MarketStateSnapshot`` field as read-only, which they already do.

    ``fvg_results`` (keyed by timeframe, exactly ``PreparedBacktestContext
    .fvg_results`` -- the raw detector output, **not** ``fvg_result``'s
    own entry candidates) is what makes an untouched, structurally
    active zone visible: its own ``FVGZone.formed_at`` is read directly,
    never inferred from ``fvg_result.candidates`` (which only exist for
    zones that have actually been touched or inverted).

    One linear forward sweep, merging zone formations, FVG entry
    signals, zone mitigations, HTF bias changes, and liquidity signals
    by instant (``_merge_confluence_streams``). FVG zone state is
    tracked by ``_apply_fvg_zone_batch`` (zone-keyed, so an untouched
    zone is never invisible -- see this module's own docstring),
    wrapped in a ``_RefCountingDict`` so reading ``active_fvg_selectors_
    by_direction`` is ``O(1)`` per snapshot rather than rescanning the
    whole active pool (same technique as ``iter_active_pool_scalar_
    snapshots``). HTF bias is tracked incrementally (never by calling
    ``tradelab.bias.bias_at`` per event, which would rescan from the
    start of its own change list every time -- proven exactly
    equivalent to ``bias_at`` by randomized test instead). Liquidity
    signals never touch FVG or bias state at all (no persistent role --
    see ``MarketStateSnapshot``'s own docstring); they only ever appear
    as the primary event of their own, single snapshot. Total:
    ``O(N log N)``, memory ``O(number of currently active unique FVG
    zones)`` -- never ``O(number of snapshots x active pool size)``.
    """
    all_zones = [zone for result in fvg_results.values() for zone in result.zones]
    fvg_signals = [_signal_from_fvg(c) for c in fvg_result.candidates]
    liquidity_signals = [_signal_from_liquidity(c) for c in liquidity_result.candidates]
    bias_changes = [change for timeframe in _BIAS_TIMEFRAMES for change in htf_bias_result.timeframe_changes[timeframe]]

    counter = _ActivePoolRefCounter()
    zone_active: dict[str, dict[str, dict[str, _ZoneStateEntry]]] = _RefCountingDict(counter)
    zone_suspended: dict[str, _ZoneStateEntry] = {}
    zone_suspended_by: dict[str, str] = {}
    zone_entries: dict[str, _ZoneStateEntry] = {}
    pending_resolution: dict[str, list[str]] = {}

    bias_timeframe_states: dict[str, str] = {tf: NEUTRAL for tf in _BIAS_TIMEFRAMES}
    bias_global = NEUTRAL

    # Structural-sharing caches for the immutable parts of MarketStateSnapshot.
    #
    # Every snapshot used to rebuild, from scratch, a frozenset of active
    # selectors per direction, a sorted tuple of every active zone id per
    # (timeframe, direction), the dicts containing them, and a BiasSnapshot.
    # On real MNQ data that is ~1.2M snapshots x thousands of zone ids, all of
    # them retained simultaneously by
    # ``entry_strategy_universe.discover_entry_strategy_universe``'s own
    # ``market_state_by_event_id`` -- measured at ~30 GB of physical footprint
    # before a single backtest variant runs, which pushed the process into
    # swap and collapsed throughput (see docs/dense-class-memory-diagnostic.md).
    #
    # The values below are *immutable and read-only* for every consumer, and
    # consecutive snapshots overwhelmingly carry identical ones, so a snapshot
    # reuses the previously built object whenever the underlying key set is
    # provably unchanged (revision counters maintained by _RefCountingDict and
    # _ActivePoolRefCounter). This changes no value, ordering, or type -- only
    # how many identical copies of them exist at once.
    zone_tuple_cache: dict[tuple[str, str], tuple[int, tuple[str, ...]]] = {}
    zone_map_cache: dict[str, tuple[tuple, dict[str, tuple[str, ...]]]] = {}
    selector_set_cache: dict[str, tuple[int, frozenset[str]]] = {}
    outer_zone_map_cache: list = []  # [(long_map, short_map, outer_dict)]
    outer_selector_cache: list = []  # [(long_set, short_set, outer_dict)]
    bias_cache: list = []  # [(states_tuple, global_bias, BiasSnapshot)]

    def _active_selectors_shared(direction: str) -> frozenset[str]:
        if not _share_immutable_parts:
            return counter.active_selectors(direction)
        revision = counter.selector_set_revision(direction)
        cached = selector_set_cache.get(direction)
        if cached is not None and cached[0] == revision:
            return cached[1]
        built = counter.active_selectors(direction)
        selector_set_cache[direction] = (revision, built)
        return built

    def _active_zone_ids_shared(direction: str) -> dict[str, tuple[str, ...]]:
        # Signature is (timeframe, key_set_revision) for every non-empty pool,
        # in zone_active's own insertion order -- so an unchanged signature
        # guarantees both the same selectors and the same sorted zone ids, and
        # the rebuilt dict below preserves the original comprehension's key
        # order exactly.
        signature = tuple(
            (timeframe, getattr(zones, "key_set_revision", None))
            for timeframe, by_direction in zone_active.items()
            for zones in (by_direction.get(direction, ()),)
            if zones
        )
        cached = zone_map_cache.get(direction)
        if _share_immutable_parts and cached is not None and cached[0] == signature and all(rev is not None for _tf, rev in signature):
            return cached[1]
        built: dict[str, tuple[str, ...]] = {}
        for timeframe, revision in signature:
            zones = zone_active[timeframe][direction]
            cache_key = (timeframe, direction)
            entry = zone_tuple_cache.get(cache_key)
            if not _share_immutable_parts or revision is None or entry is None or entry[0] != revision:
                entry = (revision, tuple(sorted(zones)))
                zone_tuple_cache[cache_key] = entry
            built[_fvg_zone_selector(timeframe, FVGSourceType.FVG_TOUCH.value)] = entry[1]
        zone_map_cache[direction] = (signature, built)
        return built

    def _pair_shared(cache: list, long_value, short_value) -> dict:
        if _share_immutable_parts and cache and cache[0][0] is long_value and cache[0][1] is short_value:
            return cache[0][2]
        built = {LONG: long_value, SHORT: short_value}
        cache[:] = [(long_value, short_value, built)]
        return built

    def _bias_shared() -> BiasSnapshot:
        states = tuple(bias_timeframe_states.items())
        if _share_immutable_parts and bias_cache and bias_cache[0][0] == states and bias_cache[0][1] == bias_global:
            return bias_cache[0][2]
        built = BiasSnapshot(timeframe_states=dict(bias_timeframe_states), global_bias=bias_global)
        bias_cache[:] = [(states, bias_global, built)]
        return built

    for formation_batch, fvg_batch, mitigation_batch, bias_batch, liquidity_batch in _merge_confluence_streams(
        all_zones, fvg_signals, fvg_result.zone_mitigations, bias_changes, liquidity_signals
    ):
        if formation_batch or fvg_batch or mitigation_batch:
            _apply_fvg_zone_batch(
                formation_batch, fvg_batch, mitigation_batch, zone_active, zone_suspended, zone_suspended_by, zone_entries, pending_resolution
            )

        if bias_batch:
            for change in bias_batch:
                bias_timeframe_states[change.timeframe] = change.state
            # Mirrors tradelab.bias.bias_at's own combine rule exactly
            # (proven equivalent by randomized test) -- never re-derived
            # differently here.
            directions = set(bias_timeframe_states.values())
            if BULLISH in directions and BEARISH not in directions:
                bias_global = BULLISH
            elif BEARISH in directions and BULLISH not in directions:
                bias_global = BEARISH
            else:
                bias_global = NEUTRAL

        real_events = fvg_batch + liquidity_batch
        if not real_events:
            continue

        market_state = MarketStateSnapshot(
            as_of=real_events[0].known_at,
            active_fvg_selectors_by_direction=_pair_shared(
                outer_selector_cache, _active_selectors_shared(LONG), _active_selectors_shared(SHORT)
            ),
            active_fvg_zone_ids_by_direction_selector=_pair_shared(
                outer_zone_map_cache, _active_zone_ids_shared(LONG), _active_zone_ids_shared(SHORT)
            ),
            bias=_bias_shared(),
        )
        for signal in real_events:
            yield CandidateEntrySnapshot(event=_entry_event_from_signal(signal), market_state=market_state)
