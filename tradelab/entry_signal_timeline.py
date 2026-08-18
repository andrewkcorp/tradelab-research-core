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

**Validity: persist until invalidated, never a time-based expiry**
(spec, this task):

- A signal becomes available exactly at its own ``known_at`` -- never
  earlier -- and then remains active **indefinitely**, across trading
  days without limit, until it is explicitly invalidated (see below).
  There is no daily, session, or calendar cutoff of any kind on an
  otherwise-still-valid confluence -- a signal known on one trading day
  is exactly as active a month later as it was the instant it became
  known, unless something invalidated it in between.
- ``EntrySignal.trading_day_id`` still exists and is still computed
  exactly the same way (the ISO date of the NY trading day -- ``[18:00
  NY, 17:00 NY next day)`` -- that owns a signal's own ``known_at``),
  but it is now pure audit metadata: which trading day a confluence was
  actually born in. It plays no role whatsoever in whether that signal
  is active -- removing it would lose useful provenance for no benefit,
  so it stays, doing strictly less than it used to.
- A signal whose own ``known_at`` falls strictly *between* 17:00 and
  18:00 NY (the daily maintenance break, open interval: excludes both
  boundaries) still belongs to no trading day at all and is still
  rejected outright (``ValueError``) -- real candles never close in
  there, so this can only happen from malformed input; this is a market/
  session **data-integrity** check, not a validity-duration rule, and
  this task does not touch it. ``known_at`` exactly *at* 17:00 NY is
  still attributed to the trading day that is *ending* at that instant
  (the regular session's own close, and therefore the ``known_at`` of
  every trading day's genuinely last 1-minute candle) rather than
  rejected -- unchanged from before, since it was never actually about
  expiry, only about which ``trading_day_id`` a legitimate close belongs
  to.
- All boundary math for ``trading_day_id`` and the maintenance-break
  check uses ``zoneinfo.ZoneInfo("America/New_York")``, never a fixed
  UTC offset, so DST transitions are handled correctly for free.

**Invalidation: scoped by timeframe, across confluence families, never
by selector and never across timeframes** (spec, this task -- a
semantic correction from an earlier version of this module, which
scoped invalidation to the exact same *selector* and, as a side effect
of the old time-based expiry, implicitly never let signals from
different trading days interact at all):

- A **newly known** confluence invalidates every still-active, earlier
  confluence of the **opposite direction** on the exact **same
  timeframe** -- regardless of which family (FVG, iFVG, or liquidity)
  or which exact selector either one belongs to. A bearish ``1m``
  confluence -- whether it is an FVG touch, an iFVG, or any liquidity
  take/reclaim variant -- invalidates every still-active bullish ``1m``
  confluence of any family/selector; it never touches ``5m``/``15m``/
  ``1H``/``4H``/``1D`` confluence of either direction. Only the exact
  same timeframe is ever affected, in either direction.
- Only signals whose own ``selector`` is in the caller's chosen
  ``selectors`` set ever participate as an invalidator -- a signal from
  a selector the caller never selected still cannot invalidate anything
  (spec: "Sygnały z niewybranych selectorów nigdy nie unieważniają
  setupu"), exactly as before; what changed is only the *scope* of which
  *other* selectors on that timeframe are searched for an invalidator,
  not this "unselected selectors are inert" guarantee itself.
- A later signal of the **same** direction (on any timeframe, any
  family) never invalidates an earlier one -- both simply remain active
  side by side (spec: "może odświeżyć/uzupełnić stan, ale nie tworzy
  duplikatu").
- **A plain FVG touch is not itself a new opposite-direction event and
  therefore never invalidates anything by virtue of merely touching.**
  This is not a special case this module implements -- it falls
  straight out of ``tradelab.fvg_entry``'s own, unchanged contract: an
  ``FVG_TOUCH`` candidate always carries its *own* zone's original
  direction (a bullish zone's touch is a bullish/long confluence), never
  the opposite, so it can never be an "opposite-direction" invalidator
  relative to that same zone's own earlier state. A zone's touch (a
  partial touch, or even a full mitigation that does *not* close beyond
  the far boundary) leaves the zone's own FVG classification exactly as
  ``tradelab.fvg`` already computed it -- this module never overrides
  that lifecycle. Only a genuine inversion -- a full traversal *and* a
  close beyond the opposite boundary, already computed by ``tradelab.
  fvg`` as an ``FVGEvent`` and surfaced here as an ``IFVG``-family
  candidate -- carries the *opposite* direction of its own originating
  zone (``tradelab.fvg_entry``'s own, unchanged contract again), and is
  therefore a genuinely new, opposite-direction confluence on its own
  timeframe, subject to the same invalidation rule as anything else.
- **Once invalidated, a signal never becomes active again.** Invalidation
  is a permanent, one-way state transition for that specific signal
  instance (keyed by its own ``candidate_id``): a still-later signal of
  the direction that had been invalidated is a brand new confluence
  fact with its own ``candidate_id`` and its own ``known_at`` -- never a
  resurrection of the earlier, already-invalidated one. There is no
  mechanism anywhere in this module that ever re-adds a previously
  removed signal back into an active set.

``query_active_signals`` answers, for a caller-chosen set of selectors
(the strategy's own choice of which categories participate), a
direction, and an as-of instant: every signal in ``selectors`` of that
``direction``, already known by ``ts_utc``, and not invalidated as of
``ts_utc`` per the rule above.

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
signal in one linear forward sweep instead: chronological order and
immediate (same-instant) invalidation removal -- ``O(N log N)`` total
(no expiry bookkeeping at all now, since there is no more time-based
expiry to schedule). It never changes ``query_active_signals`` itself
or any of its semantics; it is a second, equivalent way to obtain the
same answer at every real instant instead of a handful of caller-chosen
ones, and its own test suite verifies the two agree exactly, including
on randomized fixtures.
"""
from __future__ import annotations

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
    categories (see module docstring); ``timeframe`` is what invalidation
    is actually scoped by. ``trading_day_id`` is pure audit metadata
    (which NY trading day this confluence was born in) -- it plays no
    role in validity, which is now unbounded except for invalidation
    (see module docstring)."""

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
    """Audit record: ``signal`` was invalidated by ``invalidated_by``, a
    later, opposite-direction signal on the same timeframe (any
    confluence family/selector on that timeframe -- see module
    docstring)."""

    signal: EntrySignal
    invalidated_by: EntrySignal


@dataclass(frozen=True)
class EntrySignalQueryResult:
    active_signals: list[EntrySignal]
    invalidations: list[SignalInvalidation]


def _trading_day_for(known_at: datetime) -> str:
    """The ``trading_day_id`` for a signal known at ``known_at`` -- pure
    audit metadata, no longer tied to any validity window (see module
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


def query_active_signals(
    timeline: EntrySignalTimeline,
    selectors: set[str],
    direction: str,
    ts_utc: datetime,
) -> EntrySignalQueryResult:
    """Every signal that is known and not invalidated as of ``ts_utc``,
    restricted to ``selectors`` and ``direction`` -- exactly the query
    contract a future strategy enumerator needs (see module docstring
    for the persist-until-invalidated validity model and the same-
    timeframe, cross-family invalidation rule). There is no expiry of
    any kind here any more: a known, not-yet-invalidated signal is
    active no matter how long ago it became known.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}, got {direction!r}")
    if ts_utc.tzinfo is None:
        raise ValueError(f"ts_utc must be timezone-aware: {ts_utc!r}")
    ts_utc = ts_utc.astimezone(UTC)

    # The invalidator search pool: every signal belonging to a caller-
    # chosen selector (selectors outside this set can never invalidate
    # anything, unchanged from before) -- scoped further, per candidate
    # signal, to the exact same timeframe (never the same selector,
    # never a different timeframe; see module docstring).
    candidate_signals = [signal for selector in selectors for signal in timeline.signals_by_selector.get(selector, [])]

    active: list[EntrySignal] = []
    invalidations: list[SignalInvalidation] = []

    for selector in selectors:
        selector_signals = timeline.signals_by_selector.get(selector, [])
        for signal in selector_signals:
            if signal.direction != direction:
                continue
            if signal.known_at > ts_utc:
                continue  # not yet known

            invalidator = next(
                (
                    other
                    for other in candidate_signals
                    if other.timeframe == signal.timeframe
                    and other.direction != direction
                    and other.known_at > signal.known_at
                    and other.known_at <= ts_utc
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
    ``signal.direction`` simultaneously active (known, not invalidated)
    at that exact instant -- semantically identical to grouping
    ``query_active_signals(timeline, set(timeline.signals_by_selector),
    signal.direction, signal.known_at).active_signals`` by ``.selector``,
    but produced by ``iter_active_signal_snapshots`` in one linear
    forward sweep instead of a fresh query. ``active_by_selector`` always
    includes ``signal`` itself under its own ``signal.selector`` key --
    it is trivially known at its own ``known_at``, exactly as ``query_
    active_signals`` would also report."""

    signal: EntrySignal
    active_by_selector: dict[str, tuple[EntrySignal, ...]]


def iter_active_signal_snapshots(timeline: EntrySignalTimeline) -> Iterator[SignalSnapshot]:
    """Yield one ``SignalSnapshot`` per real signal in ``timeline.signals``
    (the timeline's own chronological order), each equivalent to a fresh
    ``query_active_signals`` call at that signal's own ``known_at`` for
    its own ``direction`` (with ``selectors`` = every selector in the
    timeline), grouped by selector -- computed incrementally in
    ``O(N log N)`` total over ``N`` signals via a forward sweep (no
    expiry bookkeeping at all: validity is unbounded except for
    invalidation), instead of ``O(N)`` fresh queries each themselves
    scanning the whole timeline. Never changes ``query_active_signals``
    itself; see the module docstring's "Bulk snapshotting" section.

    Mirrors every one of ``query_active_signals``'s rules exactly:
    direction filtering, no time-based expiry, invalidation by a
    strictly *later* opposite-direction signal on the *same timeframe*
    (any confluence family/selector -- never the same selector only,
    never a different timeframe; two signals sharing the exact same
    ``known_at`` never invalidate each other -- this function's
    invalidation step only ever removes a *strictly earlier* opposite-
    direction signal, mirroring that function's own strict ``>``
    comparison), permanent invalidation (an invalidated signal is never
    reintroduced), and multiple simultaneously active same-selector
    signals never collapsing into one.

    Signals are processed in ``known_at``-grouped batches, not one at a
    time: two (or more) signals sharing the exact same ``known_at`` must
    each see *all* of the others in that same group as already known
    when their own snapshot is taken (``query_active_signals`` only ever
    checks ``known_at <= ts``, so equal ``known_at`` values are always
    mutually visible) -- processing them strictly one-by-one in
    ``timeline.signals``'s own tie-broken order would otherwise let an
    earlier-processed signal's snapshot miss a same-instant sibling that
    happens to sort after it.

    Internally, the active set is tracked **by timeframe**, not by
    selector (unlike the outward-facing ``active_by_selector`` snapshot)
    -- invalidation is a same-timeframe, cross-family concept, so a
    single per-timeframe pool (holding every still-active signal of any
    selector on that timeframe) is exactly the scope one new signal's
    invalidation check needs to search. The public snapshot is built by
    flattening those (at most six, one per native timeframe) per-
    timeframe pools back out by selector at yield time.
    """
    active_by_timeframe: dict[str, dict[str, EntrySignal]] = {}

    for known_at, group_iter in groupby(timeline.signals, key=lambda s: s.known_at):
        group = list(group_iter)

        # 1. Invalidate, for every signal in this group, against the
        #    pre-group active state: a still-active, *strictly earlier*
        #    (known_at <, never ==), opposite-direction signal on the
        #    SAME TIMEFRAME (any selector/family) is permanently
        #    removed. Checked per group member (not merged) so a group
        #    mixing both directions on one timeframe still invalidates
        #    correctly on each side; group members never invalidate each
        #    other (their own known_at values are all equal, so the
        #    strict "<" never matches).
        for signal in group:
            tf_group = active_by_timeframe.get(signal.timeframe)
            if not tf_group:
                continue
            invalidated_ids = [
                candidate_id for candidate_id, other in tf_group.items() if other.direction != signal.direction and other.known_at < known_at
            ]
            for candidate_id in invalidated_ids:
                del tf_group[candidate_id]
            if not tf_group:
                del active_by_timeframe[signal.timeframe]

        # 2. Add every signal in this group.
        for signal in group:
            active_by_timeframe.setdefault(signal.timeframe, {})[signal.candidate_id] = signal

        # 3. Snapshot each group member now that the whole group (and
        #    only the whole group, nothing from the future) is active.
        #    The active state is tracked by timeframe (for invalidation);
        #    the outward snapshot is grouped by selector (matching what
        #    query_active_signals itself returns), so every per-
        #    timeframe pool is flattened back out here.
        for signal in group:
            active_by_selector: dict[str, list[EntrySignal]] = {}
            for tf_group in active_by_timeframe.values():
                for other in tf_group.values():
                    if other.direction == signal.direction:
                        active_by_selector.setdefault(other.selector, []).append(other)
            yield SignalSnapshot(signal=signal, active_by_selector={selector: tuple(sigs) for selector, sigs in active_by_selector.items()})
