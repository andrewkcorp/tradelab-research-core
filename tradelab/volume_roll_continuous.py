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
6. The very first session observed in the data has no preceding
   session at all in the file -- there is nothing non-heuristic to
   compare against, so it is always warm-up and never enters the
   research series, regardless of whether the export happened to start
   exactly at that session's own open.
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

SESSION_START_HOUR = 18
SESSION_END_HOUR = 17

# Quarterly outright MNQ only: root + one quarterly month code + one year
# digit (e.g. MNQZ3). Calendar spreads (e.g. "MNQZ3-MNQH4") and any other
# root never match this -- requirement 1 of the roll policy.
_OUTRIGHT_SYMBOL_PATTERN = re.compile(r"^MNQ[HMUZ]\d$")


@dataclass(frozen=True)
class SymbolVolume:
    symbol: str
    total_volume: float


@dataclass(frozen=True)
class SessionSelection:
    """One session's own contract-selection audit record. ``candidate_
    volumes``/``selected_symbol``/``tied_symbols``/``rejected_symbols``
    are all empty/``None`` for the warm-up session, which never
    receives a selection at all (see the module docstring, policy
    point 6)."""

    session_id: str
    session_start_utc: datetime
    session_end_utc: datetime
    previous_session_id: str | None
    is_warmup: bool
    candidate_volumes: tuple[SymbolVolume, ...]
    selected_symbol: str | None
    tied_symbols: tuple[str, ...]
    rejected_symbols: tuple[str, ...]
    is_transition: bool


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


@dataclass(frozen=True)
class VolumeRollContinuousResult:
    """``raw_bars`` is the spliced-but-not-yet-adjusted series (one
    contiguous block per selected contract, exactly what ``tradelab.
    continuous_futures.build_back_adjusted_continuous_series`` was
    called with); ``continuous_series_result`` is that call's own,
    unmodified result. ``session_selections`` covers every session
    including the warm-up one; ``transitions`` covers only the
    sessions where the selected contract actually changed."""

    canonical_contract: str
    raw_bars: tuple[Bar1m, ...]
    session_selections: tuple[SessionSelection, ...]
    transitions: tuple[RollTransition, ...]
    excluded_non_outright_symbols: tuple[str, ...]
    continuous_series_result: ContinuousSeriesResult


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

    session_selections: list[SessionSelection] = []
    transitions: list[RollTransition] = []
    roll_adjustments: list[RollAdjustment] = []
    raw_bars: list[Bar1m] = []

    warmup_id = session_ids[0]
    warmup_start, warmup_end = _session_bounds(warmup_id)
    session_selections.append(
        SessionSelection(
            session_id=warmup_id,
            session_start_utc=warmup_start,
            session_end_utc=warmup_end,
            previous_session_id=None,
            is_warmup=True,
            candidate_volumes=(),
            selected_symbol=None,
            tied_symbols=(),
            rejected_symbols=(),
            is_transition=False,
        )
    )

    previous_selected: str | None = None
    for index in range(1, len(session_ids)):
        session_id = session_ids[index]
        previous_session_id = session_ids[index - 1]
        session_start, session_end = _session_bounds(session_id)

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
                is_warmup=False,
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

    continuous_result = build_back_adjusted_continuous_series(raw_bars, roll_adjustments, canonical_contract)

    return VolumeRollContinuousResult(
        canonical_contract=canonical_contract,
        raw_bars=tuple(raw_bars),
        session_selections=tuple(session_selections),
        transitions=tuple(transitions),
        excluded_non_outright_symbols=tuple(sorted(excluded_symbols)),
        continuous_series_result=continuous_result,
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
        session_id, _ = _trading_day_for(record.ts_event)
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
