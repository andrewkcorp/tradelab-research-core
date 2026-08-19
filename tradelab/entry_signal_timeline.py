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
- **Nesting** is handled deterministically: each suspension records
  *which* signal it was caused by, keyed by that signal's own zone; a
  chain of suspensions (A suspended by B, B suspended by C, ...) unwinds
  one link at a time, each only when *that* link's own suspending zone
  resolves -- independent of whether the signal in between is itself
  currently active or suspended.
- **Known scope limitation**: ``tradelab.fvg_touch`` only ever records a
  zone's *first* touch, whatever kind it is -- a zone whose first touch
  was ``PARTIAL_TOUCH`` and which only *later*, on a different candle,
  becomes ``MITIGATED`` without inverting produces **no further event at
  all** (this module does not re-detect anything, and that transition is
  not surfaced anywhere in ``FVGEntryCandidateResult``). A suspension
  such a zone caused therefore cannot be observed to resolve this way;
  it persists until/unless something else resolves it. The archetypal
  case the spec calls out -- a zone's *first* touch already being a full
  traversal without a close beyond the far boundary
  (``FULL_MITIGATION_NO_INVERSION``) -- is fully handled (see above);
  only the narrower "partial touch, then a later, separate, non-
  inverting full traversal" case is not observable at this module's
  input granularity.
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

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from itertools import groupby
from typing import Iterator
from zoneinfo import ZoneInfo

from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, FVGEntryCandidateResult
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
    always an ``IFVG`` candidate for that same zone, carrying the same
    direction as ``signal`` (a zone's inversion always flips its own
    direction, so the thing that suspended ``signal`` is, by resolving
    this way, no longer opposite to it). ``signal`` is active again and
    included in ``active_signals``."""

    signal: EntrySignal
    restored_by: EntrySignal


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

    return EntrySignalTimeline(signals=signals, signals_by_selector=signals_by_selector)


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


def _apply_fvg_batch(
    batch: list[EntrySignal],
    active: dict[str, dict[str, dict[str, EntrySignal]]],
    suspended: dict[str, EntrySignal],
    suspended_by: dict[str, SignalSuspension],
    pending_resolution: dict[str, list[str]],
) -> list[SignalRestoration]:
    """Mutates ``active`` (``timeframe -> direction -> {candidate_id:
    signal}``), ``suspended`` (``candidate_id -> signal``, everything
    currently suspended) and ``suspended_by`` (``candidate_id -> its
    current SignalSuspension``) in place for one ``known_at`` batch of
    FVG-family signals, and returns the restorations this batch caused.
    ``pending_resolution`` maps a zone_id to the candidate_ids its own
    ``PARTIAL_TOUCH`` suspended, so that zone's own later ``IFVG`` (if
    any) can restore exactly those. See module docstring for the full
    suspend/restore rules. Batch members share this exact ``known_at``
    and never suspend or restore each other: restoration only ever uses
    a strictly earlier zone's pending entry, and suspension targets are
    the active pool as it stood before this batch's own suspensions.
    """
    restorations: list[SignalRestoration] = []

    # 1. Restorations: an IFVG candidate in this batch resolves its own
    #    zone's own, strictly earlier PARTIAL_TOUCH (if any) -- a zone
    #    produces at most one touch event ever, so this can only ever be
    #    that same zone's own prior suspension, never someone else's.
    for signal in batch:
        candidate: FVGEntryCandidate = signal.source
        if candidate.source_type != FVGSourceType.IFVG:
            continue
        pending = pending_resolution.pop(candidate.zone_id, None)
        if not pending:
            continue
        for candidate_id in pending:
            restored_signal = suspended.pop(candidate_id)
            suspended_by.pop(candidate_id, None)
            active.setdefault(restored_signal.timeframe, {}).setdefault(restored_signal.direction, {})[candidate_id] = restored_signal
            restorations.append(SignalRestoration(signal=restored_signal, restored_by=signal))

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
    #    module docstring); they only restore (step 1) and become a new
    #    active confluence (step 3).
    #    Targets are the pre-batch active pool (captured before any of
    #    this batch's own challenges are applied), so batch members
    #    never challenge each other.
    challenge_actions: list[tuple[EntrySignal, EntrySignal, bool]] = []  # (target, challenger, is_durable)
    for signal in batch:
        candidate = signal.source
        if candidate.source_type != FVGSourceType.FVG_TOUCH:
            continue
        if candidate.contact_type not in (FVGContactType.PARTIAL_TOUCH, FVGContactType.FULL_MITIGATION_NO_INVERSION):
            continue
        is_durable = candidate.contact_type == FVGContactType.PARTIAL_TOUCH
        opposite = _OPPOSITE_DIRECTION[signal.direction]
        for target in active.get(signal.timeframe, {}).get(opposite, {}).values():
            challenge_actions.append((target, signal, is_durable))

    for target, challenger, is_durable in challenge_actions:
        tf_pool = active.get(target.timeframe, {}).get(target.direction, {})
        if target.candidate_id not in tf_pool:
            continue  # already suspended earlier in this same batch
        if not is_durable:
            restorations.append(SignalRestoration(signal=target, restored_by=challenger))
            continue
        del tf_pool[target.candidate_id]
        suspended[target.candidate_id] = target
        suspended_by[target.candidate_id] = SignalSuspension(signal=target, suspended_by=challenger)
        pending_resolution.setdefault(challenger.source.zone_id, []).append(target.candidate_id)

    # 3. Every signal in this batch becomes (or remains) active, under
    #    its own timeframe/direction, regardless of source_type/
    #    contact_type -- FVG-family "invalidation" is replaced by
    #    "suspension"; a newly-known signal itself is never suspended.
    for signal in batch:
        active.setdefault(signal.timeframe, {}).setdefault(signal.direction, {})[signal.candidate_id] = signal

    return restorations


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
) -> tuple[list[EntrySignal], list[SignalSuspension], list[SignalRestoration]]:
    """Fresh, self-contained replay of FVG-family suspend/restore over
    ``signals`` (already filtered to known-by-ts_utc and to the caller's
    chosen selectors) -- used by ``query_active_signals``."""
    ordered = sorted(signals, key=lambda s: (s.known_at, s.candidate_id))
    active: dict[str, dict[str, dict[str, EntrySignal]]] = {}
    suspended: dict[str, EntrySignal] = {}
    suspended_by: dict[str, SignalSuspension] = {}
    pending_resolution: dict[str, list[str]] = {}
    restorations: list[SignalRestoration] = []
    for _known_at, batch_iter in groupby(ordered, key=lambda s: s.known_at):
        restorations.extend(_apply_fvg_batch(list(batch_iter), active, suspended, suspended_by, pending_resolution))
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

    liquidity_active, invalidations = _resolve_liquidity(liquidity_signals)
    fvg_active, suspensions, restorations = _resolve_fvg_conflicts(fvg_signals)

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
    ``>`` comparisons throughout).
    """
    liquidity_active: dict[str, dict[str, EntrySignal]] = {}
    fvg_active: dict[str, dict[str, dict[str, EntrySignal]]] = {}
    fvg_suspended: dict[str, EntrySignal] = {}
    fvg_suspended_by: dict[str, SignalSuspension] = {}
    fvg_pending_resolution: dict[str, list[str]] = {}

    for _known_at, group_iter in groupby(timeline.signals, key=lambda s: s.known_at):
        group = list(group_iter)
        liquidity_batch = [s for s in group if s.source_kind == SourceKind.LIQUIDITY]
        fvg_batch = [s for s in group if s.source_kind == SourceKind.FVG]

        if liquidity_batch:
            _apply_liquidity_batch(liquidity_batch, liquidity_active)
        if fvg_batch:
            _apply_fvg_batch(fvg_batch, fvg_active, fvg_suspended, fvg_suspended_by, fvg_pending_resolution)

        for signal in group:
            active_by_selector: dict[str, list[EntrySignal]] = {}
            for pool in liquidity_active.values():
                for other in pool.values():
                    if other.direction == signal.direction:
                        active_by_selector.setdefault(other.selector, []).append(other)
            for tf_pools in fvg_active.values():
                for other in tf_pools.get(signal.direction, {}).values():
                    active_by_selector.setdefault(other.selector, []).append(other)
            yield SignalSnapshot(signal=signal, active_by_selector={selector: tuple(sigs) for selector, sigs in active_by_selector.items()})
