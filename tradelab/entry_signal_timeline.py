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

**Validity** (spec, this task):

- A signal becomes available exactly at its own ``known_at`` -- never
  earlier.
- It stays valid only through the end of its own NY trading day: 17:00
  ``America/New_York``, exclusive (querying exactly at 17:00 finds it
  already expired).
- An NY trading day runs ``[18:00 NY, 17:00 NY next day)`` -- the same
  23-hour shape as ``tradelab.session_calendar``'s regular session, and
  identified by ``trading_day_id``, the ISO date of the day it *started*
  on (18:00). A signal whose own ``known_at`` falls strictly *between*
  17:00 and 18:00 NY (open interval: excludes both boundaries) belongs
  to no trading day at all and is rejected outright (``ValueError``) --
  real candles never close in there, so this can only happen from
  malformed input. ``known_at`` exactly *at* 17:00 NY is different: it is
  the regular session's own close, and therefore the ``known_at`` of
  every trading day's genuinely last 1-minute candle (``timestamp``
  16:59, ``known_at`` 17:00) -- a completely ordinary, expected instant
  for real data, never malformed, so it belongs to the trading day that
  is *ending* at that instant rather than being rejected. Its own
  ``valid_until`` for that day is that same 17:00 NY instant, so
  ``[known_at, valid_until)`` is empty -- correct, not a bug: a signal
  known simultaneously with its own trading day's end has no remaining
  time left in that day to ever be found active.
- All boundary math uses ``zoneinfo.ZoneInfo("America/New_York")``, never
  a fixed UTC offset, so DST transitions are handled correctly for free.

**Strategy-dependent invalidation, not global** (spec, this task): there
is no single "current bias" this module maintains and no rule about
which selectors a strategy combines -- that is explicitly left to the
future enumerator. What this module *does* provide is a stable
``selector`` (a plain, serializable string) on every signal, identifying
which family it belongs to:

- FVG-derived: ``fvg-{timeframe}-{source_type}`` (e.g. ``fvg-1m-fvg_touch``
  or ``fvg-1m-ifvg``) -- one selector spans both directions, since a
  bullish and a bearish zone are still the same *kind* of trigger.
- Liquidity-derived: ``liquidity-{timeframe}-{source_family}-{liquidity_type}-{trigger_phase}-{entry_mode}``
  (e.g. ``liquidity-1m-swing_equal-swing_high-take-reversal``) --
  ``entry_mode`` is deliberately part of the key, so ``reversal`` and
  ``continuation`` for the very same take/reclaim are two *different*
  selectors with two fixed, opposite directions -- never the same
  category, so one can never invalidate the other (spec: "niewybrany
  wariant alternatywny nie może anulować wybranego wariantu").

``query_active_signals`` answers, for a caller-chosen set of selectors
(the strategy's own choice of which categories participate), a
direction, and an as-of instant:

- only signals whose ``selector`` is in the chosen set are ever
  considered -- a signal from an unselected selector can never
  invalidate anything (spec: "Sygnały z niewybranych selectorów nigdy
  nie unieważniają setupu");
- within *one* selector, a later signal of the *opposite* direction
  invalidates an earlier one of that same selector once it becomes known
  (mirrors the same persist-until-opposite-confluence pattern already
  used for HTF bias, 2.1, scoped to the selector instead of the
  timeframe) -- the query reports both which signals survived and which
  were invalidated, and by what, so a later trade stays fully
  explainable;
- a later signal of the *same* direction never invalidates an earlier
  one of the same selector -- both simply remain active side by side (no
  single "one active instance" to overwrite; spec: "może
  odświeżyć/uzupełnić stan, ale nie tworzy duplikatu").

Not built here, on purpose (later steps): which selectors combine into a
concrete strategy, full combination enumeration, opening positions, the
news filter, TP/SL, or bias as a filter.

**Bulk snapshotting** (``iter_active_signal_snapshots``): ``query_active_
signals`` is a fresh, self-contained query -- fine for a handful of
caller-chosen instants, but a caller that needs the active state at
*every* real signal's own instant (e.g. ``tradelab.observed_strategy_
universe``, discovering which confluence selectors were ever really
co-active) would otherwise have to call it once per signal, each call
itself scanning the whole timeline -- ``O(N)`` queries times ``O(N))``
work each, ``O(N^2)`` overall for ``N`` signals, which does not scale to
a real multi-year dataset's signal count. ``iter_active_signal_
snapshots`` computes the exact same per-instant active state for every
signal in one linear forward sweep instead: chronological order, an
expiry min-heap, and immediate (same-instant) invalidation removal --
``O(N log N)`` total. It never changes ``query_active_signals`` itself
or any of its semantics; it is a second, equivalent way to obtain the
same answer at every real instant instead of a handful of caller-chosen
ones, and its own test suite verifies the two agree exactly, including
on randomized fixtures.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from itertools import groupby
from typing import Iterator
from zoneinfo import ZoneInfo

from tradelab.fvg_entry import FVGEntryCandidate, FVGEntryCandidateResult
from tradelab.liquidity_entry import LiquidityEntryCandidate, LiquidityEntryCandidateResult

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

LONG = "long"
SHORT = "short"
_DIRECTIONS = frozenset({LONG, SHORT})

# The daily NY maintenance break: a signal known_at falling strictly
# inside here belongs to no trading day. Open on *both* ends -- unlike
# every other window in this package, which is half-open -- because
# known_at == 17:00 NY exactly is not itself part of the break: it is
# the regular session's own close, i.e. the known_at of every trading
# day's genuinely last 1-minute candle, and is handled separately below
# (attributed to the ending day, not rejected). known_at == 18:00 NY can
# never actually occur (every Candle.known_at is strictly after its own
# timestamp, and the earliest possible session-open timestamp is itself
# 18:00 NY), so only the true open interval is ever reachable there.
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
    categories (see module docstring); ``trading_day_id`` and
    ``valid_until`` implement the validity window.
    """

    candidate_id: str
    selector: str
    source_kind: SourceKind
    timeframe: str
    direction: str  # "long" or "short"
    timestamp: datetime
    known_at: datetime
    trading_day_id: str  # ISO date the trading day started on (its own 18:00 NY)
    valid_until: datetime  # UTC instant of that trading day's 17:00 NY end, exclusive
    source: FVGEntryCandidate | LiquidityEntryCandidate


@dataclass(frozen=True)
class EntrySignalTimeline:
    signals: list[EntrySignal]  # all signals, chronological by known_at
    signals_by_selector: dict[str, list[EntrySignal]]  # each list chronological by known_at


@dataclass(frozen=True)
class SignalInvalidation:
    """Audit record: ``signal`` was invalidated by ``invalidated_by``, a
    later, opposite-direction signal in the same selector."""

    signal: EntrySignal
    invalidated_by: EntrySignal


@dataclass(frozen=True)
class EntrySignalQueryResult:
    active_signals: list[EntrySignal]
    invalidations: list[SignalInvalidation]


def _trading_day_for(known_at: datetime) -> tuple[str, datetime]:
    """The ``(trading_day_id, valid_until_utc)`` pair for a signal known
    at ``known_at``. Raises if ``known_at`` falls strictly between 17:00
    and 18:00 NY, which no trading day owns. ``known_at`` exactly at
    17:00 NY -- the regular session's own close -- is not rejected: it
    is attributed to the trading day that is ending at that instant (see
    module docstring)."""
    local = known_at.astimezone(NY)
    t = local.time()
    if _BREAK_START < t < _BREAK_END:
        raise ValueError(
            f"known_at {known_at!r} (NY local {local!r}) falls inside the daily NY maintenance break "
            "(17:00, 18:00) -- no trading day owns this instant, so it can never be a valid entry signal"
        )
    start_date = local.date() if t >= _BREAK_END else local.date() - timedelta(days=1)
    end_date = start_date + timedelta(days=1)
    valid_until = datetime(end_date.year, end_date.month, end_date.day, 17, 0, tzinfo=NY).astimezone(UTC)
    return start_date.isoformat(), valid_until


def _fvg_selector(candidate: FVGEntryCandidate) -> str:
    return f"fvg-{candidate.timeframe}-{candidate.source_type.value}"


def _liquidity_selector(candidate: LiquidityEntryCandidate) -> str:
    return (
        f"liquidity-{candidate.timeframe}-{candidate.source_family.value}-{candidate.liquidity_type.value}-"
        f"{candidate.trigger_phase.value}-{candidate.entry_mode.value}"
    )


def _signal_from_fvg(candidate: FVGEntryCandidate) -> EntrySignal:
    trading_day_id, valid_until = _trading_day_for(candidate.known_at)
    return EntrySignal(
        candidate_id=candidate.candidate_id,
        selector=_fvg_selector(candidate),
        source_kind=SourceKind.FVG,
        timeframe=candidate.timeframe,
        direction=candidate.direction,
        timestamp=candidate.timestamp,
        known_at=candidate.known_at,
        trading_day_id=trading_day_id,
        valid_until=valid_until,
        source=candidate,
    )


def _signal_from_liquidity(candidate: LiquidityEntryCandidate) -> EntrySignal:
    trading_day_id, valid_until = _trading_day_for(candidate.known_at)
    return EntrySignal(
        candidate_id=candidate.candidate_id,
        selector=_liquidity_selector(candidate),
        source_kind=SourceKind.LIQUIDITY,
        timeframe=candidate.timeframe,
        direction=candidate.direction,
        timestamp=candidate.timestamp,
        known_at=candidate.known_at,
        trading_day_id=trading_day_id,
        valid_until=valid_until,
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


def query_active_signals(
    timeline: EntrySignalTimeline,
    selectors: set[str],
    direction: str,
    ts_utc: datetime,
) -> EntrySignalQueryResult:
    """Every signal that is known, unexpired, and not invalidated as of
    ``ts_utc``, restricted to ``selectors`` and ``direction`` -- exactly
    the query contract a future strategy enumerator needs (see module
    docstring for the invalidation rule).
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}, got {direction!r}")
    if ts_utc.tzinfo is None:
        raise ValueError(f"ts_utc must be timezone-aware: {ts_utc!r}")
    ts_utc = ts_utc.astimezone(UTC)

    active: list[EntrySignal] = []
    invalidations: list[SignalInvalidation] = []

    for selector in selectors:
        selector_signals = timeline.signals_by_selector.get(selector, [])
        for signal in selector_signals:
            if signal.direction != direction:
                continue
            if signal.known_at > ts_utc:
                continue  # not yet known
            if not (signal.known_at <= ts_utc < signal.valid_until):
                continue  # expired (its trading day already ended by ts_utc)

            invalidator = next(
                (
                    other
                    for other in selector_signals
                    if other.direction != direction and other.known_at > signal.known_at and other.known_at <= ts_utc
                ),
                None,
            )
            if invalidator is not None:
                invalidations.append(SignalInvalidation(signal=signal, invalidated_by=invalidator))
                continue

            active.append(signal)

    active.sort(key=lambda s: (s.known_at, s.candidate_id))
    invalidations.sort(key=lambda inv: (inv.signal.known_at, inv.signal.candidate_id))
    return EntrySignalQueryResult(active_signals=active, invalidations=invalidations)


@dataclass(frozen=True)
class SignalSnapshot:
    """One real ``EntrySignal``'s own ``known_at`` instant, paired with
    the full, per-selector snapshot of every signal of that *same*
    ``signal.direction`` simultaneously active (known, unexpired, not
    invalidated) at that exact instant -- semantically identical to
    grouping ``query_active_signals(timeline, set(timeline.signals_by_
    selector), signal.direction, signal.known_at).active_signals`` by
    ``.selector``, but produced by ``iter_active_signal_snapshots`` in
    one linear forward sweep instead of a fresh query. ``active_by_
    selector`` always includes ``signal`` itself under its own
    ``signal.selector`` key -- it is trivially known and unexpired at
    its own ``known_at``, exactly as ``query_active_signals`` would also
    report."""

    signal: EntrySignal
    active_by_selector: dict[str, tuple[EntrySignal, ...]]


def iter_active_signal_snapshots(timeline: EntrySignalTimeline) -> Iterator[SignalSnapshot]:
    """Yield one ``SignalSnapshot`` per real signal in ``timeline.signals``
    (the timeline's own chronological order), each equivalent to a fresh
    ``query_active_signals`` call at that signal's own ``known_at`` for
    its own ``direction``, grouped by selector -- computed incrementally
    in ``O(N log N)`` total over ``N`` signals via a forward sweep (an
    expiry min-heap plus immediate same-instant invalidation removal),
    instead of ``O(N)`` fresh queries each themselves scanning the whole
    timeline. Never changes ``query_active_signals`` itself; see the
    module docstring's "Bulk snapshotting" section.

    Mirrors every one of ``query_active_signals``'s rules exactly:
    direction filtering, ``known_at <= ts < valid_until``, invalidation
    by a strictly *later* opposite-direction signal of the same selector
    (two signals sharing the exact same ``known_at`` never invalidate
    each other -- this function's invalidation step only ever removes a
    *strictly earlier* opposite-direction signal, mirroring that
    function's own strict ``>`` comparison), and multiple simultaneously
    active same-selector signals never collapsing into one.

    Signals are processed in ``known_at``-grouped batches, not one at a
    time: two (or more) signals sharing the exact same ``known_at`` must
    each see *all* of the others in that same group as already known
    when their own snapshot is taken (``query_active_signals`` only ever
    checks ``known_at <= ts``, so equal ``known_at`` values are always
    mutually visible) -- processing them strictly one-by-one in
    ``timeline.signals``'s own tie-broken order would otherwise let an
    earlier-processed signal's snapshot miss a same-instant sibling that
    happens to sort after it.
    """
    active: dict[str, dict[str, EntrySignal]] = {}
    expiry_heap: list[tuple[datetime, str, str]] = []

    for known_at, group_iter in groupby(timeline.signals, key=lambda s: s.known_at):
        group = list(group_iter)

        # 1. Expire anything whose valid_until <= this instant (query
        #    semantics: known_at <= ts < valid_until, so valid_until == ts
        #    is already expired, not active).
        while expiry_heap and expiry_heap[0][0] <= known_at:
            _, expired_selector, expired_candidate_id = heapq.heappop(expiry_heap)
            selector_group = active.get(expired_selector)
            if selector_group is not None:
                selector_group.pop(expired_candidate_id, None)
                if not selector_group:
                    del active[expired_selector]

        # 2. Invalidate, for every signal in this group, against the
        #    pre-group active state: a still-active, *strictly earlier*
        #    (known_at <, never ==), opposite-direction signal of the
        #    same selector is permanently removed. Checked per group
        #    member (not merged) so a group mixing both directions for
        #    one selector still invalidates correctly on each side; group
        #    members never invalidate each other (their own known_at
        #    values are all equal, so the strict "<" never matches).
        for signal in group:
            selector_group = active.get(signal.selector)
            if not selector_group:
                continue
            invalidated_ids = [
                candidate_id
                for candidate_id, other in selector_group.items()
                if other.direction != signal.direction and other.known_at < known_at
            ]
            for candidate_id in invalidated_ids:
                del selector_group[candidate_id]
            if not selector_group:
                del active[signal.selector]

        # 3. Add every signal in this group.
        for signal in group:
            active.setdefault(signal.selector, {})[signal.candidate_id] = signal
            heapq.heappush(expiry_heap, (signal.valid_until, signal.selector, signal.candidate_id))

        # 4. Snapshot each group member now that the whole group (and
        #    only the whole group, nothing from the future) is active.
        for signal in group:
            active_by_selector = {
                selector: tuple(sig for sig in selector_group.values() if sig.direction == signal.direction)
                for selector, selector_group in active.items()
            }
            active_by_selector = {selector: sigs for selector, sigs in active_by_selector.items() if sigs}
            yield SignalSnapshot(signal=signal, active_by_selector=active_by_selector)
