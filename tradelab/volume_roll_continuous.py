"""Deterministic, fully audited volume-front-month contract selection
for MNQ, turning a Databento parent-symbology record set (many
quarterly outright contracts trading in parallel, ``tradelab.
databento_parent_csv``) into the single, chronological, one-contract-
at-a-time ``list[Bar1m]`` that ``tradelab.continuous_futures.
build_back_adjusted_continuous_series`` already expects -- reusing
that module's own additive back-adjustment unchanged, never
duplicating it.

**The roll policy (v1, this task's own spec -- the only policy this
module implements; no alternative heuristic, threshold, or rollback of
any kind)**:

1. Only quarterly outright MNQ contracts participate -- symbols
   matching ``MNQ`` + a quarterly month code (``H``/``M``/``U``/``Z``)
   + a single year digit (e.g. ``MNQZ3``). Calendar spreads and any
   other root are excluded before anything else runs.
2. The CME session is ``America/New_York`` 18:00 to the next day's
   17:00 (the daily 17:00-18:00 break belongs to no session) -- reusing
   ``tradelab.entry_signal_timeline``'s own existing session
   classification (``_trading_day_for``) rather than a second
   implementation of the same calendar shape.
3. Each session selects exactly one contract: whichever quarterly
   outright had the single highest total volume in the **previous**
   completed session.
4. A tie is broken deterministically by ascending symbol and recorded
   in the audit trail (``SessionSelection.tied_symbols``) -- never a
   silent, arbitrary pick.
5. **No look-ahead**: a session's own selection depends only on the
   immediately preceding session's already-complete volume totals,
   never on anything in the current or a later session.
6. A session enters the research series only if it is itself **fully
   covered** by the file's own observed data range, and the
   immediately preceding session (whose volume it would use) is
   fully covered too. The file's coverage range is ``[earliest raw
   ts_event, latest raw ts_event + 1 minute)``. A session's true total
   volume is only known once the whole session has actually been
   observed -- an export that starts or ends mid-session must never
   let that partial volume drive (or be used to drive) a selection.
   In practice this can only ever affect the very first and/or very
   last session present in the file (any session strictly in between
   always has its own nominal bounds fully inside the file's overall
   data range, so it is always fully covered) -- but it is checked
   explicitly, never assumed. The very first session in the file is
   additionally always excluded regardless of its own coverage, since
   there is no preceding session in the file at all to compare it
   against. Every excluded session is recorded in the audit with one
   of three explicit reasons (``SessionSelection.exclusion_reason``):
   ``no_previous_session`` (the first session in the file), ``session_
   not_fully_covered`` (its own nominal span isn't fully inside the
   file's data range), or ``previous_session_not_fully_covered`` (the
   session itself is fine, but the previous session it would need to
   compare against is not).
7. When the selected contract changes between two consecutive
   sessions, a ``RollAdjustment`` is derived -- never guessed -- from
   the single latest minute, strictly before the rollover, at which
   *both* the outgoing and incoming contract have a record:
   ``adjustment_points = incoming_close - outgoing_close`` at that
   shared minute. Both prices and the reference minute are recorded in
   the audit trail (``RollTransition``).
8. ``effective_at`` is the new session's own nominal first minute
   (18:00 NY, converted to UTC) -- deterministic, independent of
   whichever minute the incoming contract's first *observed* bar
   happens to be.
9. The resulting ``list[Bar1m]``/``list[RollAdjustment]`` are handed,
   unchanged, straight to ``tradelab.continuous_futures.
   build_back_adjusted_continuous_series`` -- this module never
   reimplements additive back-adjustment or its own validation.

**Why this avoids look-ahead twice over.** The contract chosen for
session *N* is fixed entirely by session *N-1*'s already-closed volume
-- by the time session *N* starts, that decision could already have
been made without seeing a single bar of session *N* itself. And the
roll's own price reference (step 7) only ever looks at minutes
strictly before the rollover's own ``effective_at`` -- it can be
older than the immediately preceding session (the outgoing and
incoming contracts often trade in parallel for a while before the
front-month handoff), but it is never later than the roll.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.bar import Bar1m
from tradelab.continuous_futures import ContinuousSeriesResult, RollAdjustment, build_back_adjusted_continuous_series
from tradelab.databento_parent_csv import DatabentoParentRecord
from tradelab.entry_signal_timeline import _trading_day_for

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

# The single, versioned roll policy this module implements (see module
# docstring) -- the value recorded as MarketDataArtifact.normalization_
# method by callers (e.g. tradelab.materialize_research_artifact) so a
# resulting continuous/databento_parent_csv artifact's manifest always
# names exactly which policy version produced it.
POLICY_VERSION = "volume_front_month_v1"

SESSION_START_HOUR = 18
SESSION_END_HOUR = 17

# SessionSelection.exclusion_reason values -- see module docstring, policy point 6.
EXCLUSION_NO_PREVIOUS_SESSION = "no_previous_session"
EXCLUSION_SESSION_NOT_FULLY_COVERED = "session_not_fully_covered"
EXCLUSION_PREVIOUS_SESSION_NOT_FULLY_COVERED = "previous_session_not_fully_covered"

# Quarterly outright MNQ only: root + one quarterly month code + one year
# digit (e.g. MNQZ3). Calendar spreads (e.g. "MNQZ3-MNQH4") and any other
# root never match this -- requirement 1 of the roll policy.
_OUTRIGHT_SYMBOL_PATTERN = re.compile(r"^MNQ[HMUZ]\d$")


@dataclass(frozen=True)
class SymbolVolume:
    symbol: str
    total_volume: float

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "total_volume": self.total_volume}


@dataclass(frozen=True)
class SessionSelection:
    """One session's own contract-selection audit record. ``candidate_
    volumes``/``selected_symbol``/``tied_symbols``/``rejected_symbols``
    are all empty/``None`` when ``is_included`` is ``False`` -- this
    session never receives a selection at all and none of its bars
    enter ``raw_bars``. ``exclusion_reason`` is one of
    ``EXCLUSION_NO_PREVIOUS_SESSION``/``EXCLUSION_SESSION_NOT_FULLY_
    COVERED``/``EXCLUSION_PREVIOUS_SESSION_NOT_FULLY_COVERED`` when
    excluded, ``None`` when included (see module docstring, policy
    point 6)."""

    session_id: str
    session_start_utc: datetime
    session_end_utc: datetime
    previous_session_id: str | None
    is_included: bool
    exclusion_reason: str | None
    candidate_volumes: tuple[SymbolVolume, ...]
    selected_symbol: str | None
    tied_symbols: tuple[str, ...]
    rejected_symbols: tuple[str, ...]
    is_transition: bool

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_start_utc": self.session_start_utc.isoformat(),
            "session_end_utc": self.session_end_utc.isoformat(),
            "previous_session_id": self.previous_session_id,
            "is_included": self.is_included,
            "exclusion_reason": self.exclusion_reason,
            "candidate_volumes": [v.to_dict() for v in self.candidate_volumes],
            "selected_symbol": self.selected_symbol,
            "tied_symbols": list(self.tied_symbols),
            "rejected_symbols": list(self.rejected_symbols),
            "is_transition": self.is_transition,
        }


@dataclass(frozen=True)
class RollTransition:
    """One contract handoff's full audit record -- the exact minute and
    both prices the resulting ``RollAdjustment.adjustment_points`` was
    derived from, never merely the final number."""

    session_id: str
    outgoing_contract: str
    incoming_contract: str
    effective_at: datetime
    common_minute: datetime
    outgoing_close_at_common_minute: float
    incoming_close_at_common_minute: float
    adjustment_points: float

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "outgoing_contract": self.outgoing_contract,
            "incoming_contract": self.incoming_contract,
            "effective_at": self.effective_at.isoformat(),
            "common_minute": self.common_minute.isoformat(),
            "outgoing_close_at_common_minute": self.outgoing_close_at_common_minute,
            "incoming_close_at_common_minute": self.incoming_close_at_common_minute,
            "adjustment_points": self.adjustment_points,
        }


@dataclass(frozen=True)
class VolumeRollContinuousResult:
    """``raw_bars`` is the spliced-but-not-yet-adjusted series (one
    contiguous block per selected contract, exactly what ``tradelab.
    continuous_futures.build_back_adjusted_continuous_series`` was
    called with); ``continuous_series_result`` is that call's own,
    unmodified result. ``session_selections`` covers every session
    observed in the data, including every excluded one (see its own
    ``is_included``/``exclusion_reason``); ``transitions`` covers only
    the sessions where the selected contract actually changed.
    ``coverage_start``/``coverage_end_exclusive`` are the file's own
    observed data range (``[earliest raw ts_event, latest raw ts_event
    + 1 minute)``), used to decide which leading/trailing sessions are
    excluded as partially covered."""

    canonical_contract: str
    raw_bars: tuple[Bar1m, ...]
    session_selections: tuple[SessionSelection, ...]
    transitions: tuple[RollTransition, ...]
    excluded_non_outright_symbols: tuple[str, ...]
    coverage_start: datetime
    coverage_end_exclusive: datetime
    continuous_series_result: ContinuousSeriesResult

    def to_dict(self) -> dict:
        """A plain, JSON-compatible audit dict. Deliberately excludes
        ``raw_bars``/``continuous_series_result`` -- the bars themselves
        already live in whatever file a caller writes them to (e.g. a
        continuous CSV), and duplicating a potentially multi-year bar
        series into a second, redundant JSON file here would be wasteful;
        this covers only the audit trail (selections, transitions,
        exclusions, coverage). ``policy_version`` is always ``POLICY_
        VERSION`` -- this dict is only ever produced by this module's own
        ``build_volume_front_month_continuous_series``, so it can never
        actually vary, but it is included explicitly (not left implicit)
        so this file alone is self-contained: a reader never has to
        already know which module produced it to know which policy
        version did."""
        return {
            "policy_version": POLICY_VERSION,
            "canonical_contract": self.canonical_contract,
            "coverage_start": self.coverage_start.isoformat(),
            "coverage_end_exclusive": self.coverage_end_exclusive.isoformat(),
            "session_selections": [s.to_dict() for s in self.session_selections],
            "transitions": [t.to_dict() for t in self.transitions],
            "excluded_non_outright_symbols": list(self.excluded_non_outright_symbols),
        }


def build_volume_front_month_continuous_series(
    records: list[DatabentoParentRecord], canonical_contract: str
) -> VolumeRollContinuousResult:
    """Apply the volume-front-month roll policy (see module docstring)
    to ``records`` and hand the result to ``tradelab.continuous_futures
    .build_back_adjusted_continuous_series``. Never mutates ``records``
    or any record in it."""
    if not canonical_contract:
        raise ValueError("canonical_contract must be non-empty")

    records = list(records)
    if not records:
        raise ValueError("records must not be empty")
    _require_non_decreasing(records)

    # The file's own observed data range -- computed from every raw record,
    # before outright filtering, since it describes what the export itself
    # actually covers. records is already validated non-decreasing, so the
    # first/last entries are exactly the earliest/latest ts_event.
    coverage_start = records[0].ts_event
    coverage_end_exclusive = records[-1].ts_event + timedelta(minutes=1)

    outright_records, excluded_symbols = _filter_outright(records)
    if not outright_records:
        raise ValueError(
            "no quarterly outright MNQ records found after filtering (expected symbols matching "
            f"{_OUTRIGHT_SYMBOL_PATTERN.pattern!r}); excluded symbols: {sorted(excluded_symbols)}"
        )

    sessions = _group_by_session(outright_records)
    session_ids = sorted(sessions)
    if len(session_ids) < 2:
        raise ValueError(
            f"only {len(session_ids)} session(s) present after filtering -- at least one warm-up session plus "
            "one selectable session are required"
        )

    close_by_symbol_ts, timestamps_by_symbol = _build_price_lookup(outright_records)

    session_bounds = {session_id: _session_bounds(session_id) for session_id in session_ids}
    fully_covered = {
        session_id: (bounds[0] >= coverage_start and bounds[1] <= coverage_end_exclusive)
        for session_id, bounds in session_bounds.items()
    }

    session_selections: list[SessionSelection] = []
    transitions: list[RollTransition] = []
    roll_adjustments: list[RollAdjustment] = []
    raw_bars: list[Bar1m] = []
    previous_selected: str | None = None

    for index, session_id in enumerate(session_ids):
        session_start, session_end = session_bounds[session_id]

        if index == 0:
            # No preceding session exists in the file at all -- nothing
            # non-heuristic to compare against, regardless of this
            # session's own coverage.
            session_selections.append(
                _excluded_selection(session_id, session_start, session_end, None, EXCLUSION_NO_PREVIOUS_SESSION)
            )
            continue

        previous_session_id = session_ids[index - 1]

        if not fully_covered[session_id]:
            session_selections.append(
                _excluded_selection(session_id, session_start, session_end, previous_session_id, EXCLUSION_SESSION_NOT_FULLY_COVERED)
            )
            continue
        if not fully_covered[previous_session_id]:
            session_selections.append(
                _excluded_selection(
                    session_id, session_start, session_end, previous_session_id, EXCLUSION_PREVIOUS_SESSION_NOT_FULLY_COVERED
                )
            )
            continue

        volume_totals: dict[str, float] = {}
        for record in sessions[previous_session_id]:
            volume_totals[record.symbol] = volume_totals.get(record.symbol, 0.0) + record.volume
        candidate_volumes = tuple(SymbolVolume(symbol=symbol, total_volume=volume_totals[symbol]) for symbol in sorted(volume_totals))
        selected, tied, rejected = _select_symbol(volume_totals)

        current_session_symbols = {record.symbol for record in sessions[session_id]}
        if selected not in current_session_symbols:
            raise ValueError(
                f"session {session_id!r}: selected contract {selected!r} (highest total volume in previous "
                f"session {previous_session_id!r}) has no records in this session"
            )

        is_transition = previous_selected is not None and selected != previous_selected
        session_selections.append(
            SessionSelection(
                session_id=session_id,
                session_start_utc=session_start,
                session_end_utc=session_end,
                previous_session_id=previous_session_id,
                is_included=True,
                exclusion_reason=None,
                candidate_volumes=candidate_volumes,
                selected_symbol=selected,
                tied_symbols=tied,
                rejected_symbols=rejected,
                is_transition=is_transition,
            )
        )

        if is_transition:
            outgoing, incoming = previous_selected, selected
            common_minutes = timestamps_by_symbol.get(outgoing, set()) & timestamps_by_symbol.get(incoming, set())
            before_rollover = [ts for ts in common_minutes if ts < session_start]
            if not before_rollover:
                raise ValueError(
                    f"session {session_id!r}: no common minute between outgoing {outgoing!r} and incoming "
                    f"{incoming!r} found before the rollover at {session_start!r} -- cannot derive adjustment_points "
                    "without look-ahead"
                )
            reference_ts = max(before_rollover)
            outgoing_close = close_by_symbol_ts[(outgoing, reference_ts)]
            incoming_close = close_by_symbol_ts[(incoming, reference_ts)]
            adjustment_points = incoming_close - outgoing_close

            roll_adjustments.append(
                RollAdjustment(
                    outgoing_contract=outgoing,
                    incoming_contract=incoming,
                    effective_at=session_start,
                    adjustment_points=adjustment_points,
                )
            )
            transitions.append(
                RollTransition(
                    session_id=session_id,
                    outgoing_contract=outgoing,
                    incoming_contract=incoming,
                    effective_at=session_start,
                    common_minute=reference_ts,
                    outgoing_close_at_common_minute=outgoing_close,
                    incoming_close_at_common_minute=incoming_close,
                    adjustment_points=adjustment_points,
                )
            )

        # sessions[session_id] is already ts_event-ordered: it is a
        # stable sub-sequence of the caller's own non-decreasing input.
        for record in sessions[session_id]:
            if record.symbol == selected:
                raw_bars.append(
                    Bar1m(
                        timestamp=record.ts_event,
                        open=record.open,
                        high=record.high,
                        low=record.low,
                        close=record.close,
                        volume=record.volume,
                        contract=record.symbol,
                    )
                )

        previous_selected = selected

    # An empty raw_bars here means every session was legitimately excluded
    # (see session_selections for exactly why, per session) -- never raised
    # as an error in its own right: tradelab.continuous_futures.build_back_
    # adjusted_continuous_series already treats an empty bars/rolls pair as
    # a first-class, valid (if uninteresting) case, and discarding the full
    # per-session audit trail behind an opaque exception here would destroy
    # exactly the information a caller needs to diagnose a fragmented file.
    continuous_result = build_back_adjusted_continuous_series(raw_bars, roll_adjustments, canonical_contract)

    return VolumeRollContinuousResult(
        canonical_contract=canonical_contract,
        raw_bars=tuple(raw_bars),
        session_selections=tuple(session_selections),
        transitions=tuple(transitions),
        excluded_non_outright_symbols=tuple(sorted(excluded_symbols)),
        coverage_start=coverage_start,
        coverage_end_exclusive=coverage_end_exclusive,
        continuous_series_result=continuous_result,
    )


def _excluded_selection(
    session_id: str, session_start: datetime, session_end: datetime, previous_session_id: str | None, reason: str
) -> SessionSelection:
    return SessionSelection(
        session_id=session_id,
        session_start_utc=session_start,
        session_end_utc=session_end,
        previous_session_id=previous_session_id,
        is_included=False,
        exclusion_reason=reason,
        candidate_volumes=(),
        selected_symbol=None,
        tied_symbols=(),
        rejected_symbols=(),
        is_transition=False,
    )


def _require_non_decreasing(records: list[DatabentoParentRecord]) -> None:
    previous_ts: datetime | None = None
    for index, record in enumerate(records):
        if previous_ts is not None and record.ts_event < previous_ts:
            raise ValueError(
                f"records[{index}] ts_event {record.ts_event!r} is before the previous record's ts_event "
                f"{previous_ts!r} -- records must be non-decreasing in time"
            )
        previous_ts = record.ts_event


def _filter_outright(records: list[DatabentoParentRecord]) -> tuple[list[DatabentoParentRecord], set[str]]:
    outright: list[DatabentoParentRecord] = []
    excluded: set[str] = set()
    for record in records:
        if _OUTRIGHT_SYMBOL_PATTERN.match(record.symbol):
            outright.append(record)
        else:
            excluded.add(record.symbol)
    return outright, excluded


def _group_by_session(records: list[DatabentoParentRecord]) -> dict[str, list[DatabentoParentRecord]]:
    sessions: dict[str, list[DatabentoParentRecord]] = {}
    for record in records:
        session_id = _trading_day_for(record.ts_event)
        sessions.setdefault(session_id, []).append(record)
    return sessions


def _build_price_lookup(
    records: list[DatabentoParentRecord],
) -> tuple[dict[tuple[str, datetime], float], dict[str, set[datetime]]]:
    close_by_symbol_ts: dict[tuple[str, datetime], float] = {}
    timestamps_by_symbol: dict[str, set[datetime]] = {}
    for record in records:
        key = (record.symbol, record.ts_event)
        if key in close_by_symbol_ts:
            raise ValueError(f"duplicate record for symbol {record.symbol!r} at {record.ts_event!r}")
        close_by_symbol_ts[key] = record.close
        timestamps_by_symbol.setdefault(record.symbol, set()).add(record.ts_event)
    return close_by_symbol_ts, timestamps_by_symbol


def _select_symbol(volume_totals: dict[str, float]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    max_volume = max(volume_totals.values())
    tied = tuple(sorted(symbol for symbol, volume in volume_totals.items() if volume == max_volume))
    selected = tied[0]  # ascending symbol order -- deterministic (policy point 4)
    rejected = tuple(sorted(symbol for symbol in volume_totals if symbol != selected))
    return selected, tied, rejected


def _session_bounds(session_id: str) -> tuple[datetime, datetime]:
    """The literal UTC ``[start, end)`` of the session identified by
    ``session_id`` (the ISO date it started on, at 18:00 NY) -- a
    direct forward calculation from an already-known local date, not a
    second implementation of ``_trading_day_for``'s own instant-to-
    session classification, which remains the sole source of truth for
    *that*."""
    start_date = date.fromisoformat(session_id)
    start_local = datetime(start_date.year, start_date.month, start_date.day, SESSION_START_HOUR, 0, tzinfo=NY)
    end_date = start_date + timedelta(days=1)
    end_local = datetime(end_date.year, end_date.month, end_date.day, SESSION_END_HOUR, 0, tzinfo=NY)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)
