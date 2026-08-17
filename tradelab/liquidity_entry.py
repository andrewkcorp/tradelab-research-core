"""Liquidity entry candidates: a neutral, audited adapter from already-
detected liquidity take/reclaim facts to every entry variant a future
strategy enumerator will test.

Not a backtester and not a confluence combiner: this module only
prepares candidate facts. It does not filter by HTF bias (spec 2.1), by
red-folder news (4.2), or by FVG/iFVG (5.x) -- those are separate, later
steps. Data wicks are never consumed here; they are a future TP/SL level
source only (4.3), never an entry candidate.

Reuses ``tradelab.liquidity.detect_liquidity`` (swing/equal-highs/lows)
and ``tradelab.session_liquidity.detect_session_liquidity`` (session
highs/lows) output directly -- it does not re-detect levels, takes, or
reclaims, and does not change either module's semantics.

**Timeframes: only ``1m``/``5m``/``15m``**, the same restriction already
enforced by ``tradelab.session_liquidity`` (spec 2: these are the setup/
trigger timeframes) -- ``1H``/``4H``/``1D`` liquidity never produces an
entry candidate here, even though swing/equal-highs liquidity is
detected on all six native timeframes.

**Trigger phases** (spec 6.4/6.5): a liquidity take, and separately its
eventual reclaim resolution, are each their own audited fact, known at
their own instant:

- ``TAKE``: known at ``LiquidityTakeEvent.known_at``.
- ``RECLAIM_WICK`` / ``RECLAIM_CLOSE`` / ``NO_RECLAIM_IN_WINDOW``: known
  at ``ReclaimResult.resolved_at``, whichever of the three the reclaim
  actually resolved to.
- ``PENDING`` never produces a candidate: the outcome is not yet known,
  so there is nothing to emit -- not even a provisional one.

A level with a resolved (non-``PENDING``) reclaim therefore contributes
*two* trigger phases, each becoming known at a different instant: TAKE
first, then whichever reclaim outcome later. A level whose reclaim is
still ``PENDING`` (or that has no reclaim recorded at all) contributes
only the TAKE phase.

**Direction: both are tested, never picked upfront.** For every single
trigger (whatever its phase), two disjoint entry-mode candidates are
generated:

===============  ============  ==============
Level side        reversal      continuation
===============  ============  ==============
high              short         long
low                long         short
===============  ============  ==============

A sweep of a liquidity high can mean rejection (reversal: short) or a
breakout (continuation: long) -- both are prepared here; the future
enumerator decides which (if either) survives. Liquidity low is the
mirror image.

No lookahead of its own: this module does not walk candles or reconcile
timing -- it only ever copies ``known_at``/``resolved_at`` already
computed by the upstream detectors, so its own correctness reduces to
theirs. It does, however, cross-validate that its two inputs are
mutually consistent rather than trusting malformed or mismatched input
silently:

- every take/reclaim references a real level, on the declared timeframe,
  filed under the correct source family;
- a take's ``liquidity_type``, ``side``, ``lower``/``upper``, and
  ``timestamp == known_at`` all agree with its own level (the contract
  every upstream detector already guarantees for real output -- checked
  here anyway, since this module cannot assume its input came from a
  real detector call);
- a reclaim's ``take_timestamp`` matches its matching take's own
  ``timestamp`` exactly;
- ``PENDING`` always has ``resolved_at is None``; every terminal outcome
  (``RECLAIM_WICK``/``RECLAIM_CLOSE``/``NO_RECLAIM_IN_WINDOW``) always
  has a ``resolved_at`` strictly after its take's ``known_at`` -- these
  are checked unconditionally, even for a ``PENDING`` reclaim that will
  never itself produce a candidate;
- no ``level_id``, take, or reclaim is duplicated within one source.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tradelab.liquidity import (  # noqa: F401  (re-exported)
    HIGH,
    LOW,
    LiquidityDetectionResult,
    LiquidityLevel,
    LiquidityTakeEvent,
    LiquidityType,
    ReclaimOutcome,
    ReclaimResult,
)
from tradelab.session_liquidity import SUPPORTED_TIMEFRAMES

LONG = "long"
SHORT = "short"

_SESSION_LIQUIDITY_TYPES = frozenset(lt for lt in LiquidityType if lt.value.startswith("session_"))


class SourceFamily(str, Enum):
    SWING_EQUAL = "swing_equal"
    SESSION = "session"


class CandidatePhase(str, Enum):
    TAKE = "take"
    RECLAIM_WICK = "reclaim_wick"
    RECLAIM_CLOSE = "reclaim_close"
    NO_RECLAIM_IN_WINDOW = "no_reclaim_in_window"


class EntryMode(str, Enum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


_PHASE_FROM_RECLAIM_OUTCOME = {
    ReclaimOutcome.RECLAIM_WICK: CandidatePhase.RECLAIM_WICK,
    ReclaimOutcome.RECLAIM_CLOSE: CandidatePhase.RECLAIM_CLOSE,
    ReclaimOutcome.NO_RECLAIM_IN_WINDOW: CandidatePhase.NO_RECLAIM_IN_WINDOW,
}

_DIRECTION = {
    (HIGH, EntryMode.REVERSAL): SHORT,
    (HIGH, EntryMode.CONTINUATION): LONG,
    (LOW, EntryMode.REVERSAL): LONG,
    (LOW, EntryMode.CONTINUATION): SHORT,
}


@dataclass(frozen=True)
class LiquidityEntryCandidate:
    """One fully-audited entry candidate: which level, which trigger
    phase, which of the two disjoint entry modes, and the resulting
    direction -- plus the level's own price bounds for later TP/SL.

    ``candidate_id`` is stable and deterministic: derived only from
    ``(timeframe, source_family, level_id, trigger_phase, entry_mode)``,
    never from list position or call order.
    """

    candidate_id: str
    source_family: SourceFamily
    timeframe: str
    level_id: str
    liquidity_type: LiquidityType
    side: str  # "high" or "low", copied from the level
    lower: float
    upper: float
    trigger_phase: CandidatePhase
    entry_mode: EntryMode
    direction: str  # "long" or "short"
    timestamp: datetime
    known_at: datetime


@dataclass(frozen=True)
class LiquidityEntryCandidateResult:
    candidates: list[LiquidityEntryCandidate]


def detect_liquidity_entry_candidates(
    swing_equal_results: dict[str, LiquidityDetectionResult],
    session_results: dict[str, LiquidityDetectionResult],
) -> LiquidityEntryCandidateResult:
    """Build every liquidity entry candidate from already-computed swing/
    equal-highs/lows results (keyed by timeframe, from
    ``tradelab.liquidity.detect_liquidity``) and session results (keyed
    by timeframe, from ``tradelab.session_liquidity.detect_session_liquidity``).

    Only ``1m``/``5m``/``15m`` keys are accepted in either dict; a
    timeframe absent from both simply contributes nothing.
    """
    candidates: list[LiquidityEntryCandidate] = []
    for source_family, results in (
        (SourceFamily.SWING_EQUAL, swing_equal_results),
        (SourceFamily.SESSION, session_results),
    ):
        for timeframe, result in results.items():
            candidates.extend(_candidates_for_source(source_family, timeframe, result))

    candidates.sort(key=lambda c: (c.known_at, c.candidate_id))
    return LiquidityEntryCandidateResult(candidates=candidates)


def _candidates_for_source(source_family: SourceFamily, timeframe: str, result: LiquidityDetectionResult) -> list[LiquidityEntryCandidate]:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported liquidity entry timeframe {timeframe!r} for source {source_family.value!r}; "
            f"expected one of {sorted(SUPPORTED_TIMEFRAMES)}"
        )

    levels_by_id: dict[str, LiquidityLevel] = {}
    for level in result.levels:
        if level.timeframe != timeframe:
            raise ValueError(f"LiquidityLevel {level.level_id!r} has timeframe {level.timeframe!r}, expected {timeframe!r}")
        is_session_type = level.liquidity_type in _SESSION_LIQUIDITY_TYPES
        if source_family is SourceFamily.SESSION and not is_session_type:
            raise ValueError(f"LiquidityLevel {level.level_id!r} ({level.liquidity_type!r}) is not a session liquidity_type but was filed under source_family=session")
        if source_family is SourceFamily.SWING_EQUAL and is_session_type:
            raise ValueError(f"LiquidityLevel {level.level_id!r} ({level.liquidity_type!r}) is a session liquidity_type but was filed under source_family=swing_equal")
        if level.level_id in levels_by_id:
            raise ValueError(f"Duplicate level_id {level.level_id!r} in {source_family.value!r} results for timeframe {timeframe!r}")
        levels_by_id[level.level_id] = level

    takes_by_level_id: dict[str, LiquidityTakeEvent] = {}
    for take in result.takes:
        if take.timeframe != timeframe:
            raise ValueError(f"LiquidityTakeEvent for level {take.level_id!r} has timeframe {take.timeframe!r}, expected {timeframe!r}")
        if take.level_id not in levels_by_id:
            raise ValueError(f"LiquidityTakeEvent references unknown level_id {take.level_id!r} in {source_family.value!r}/{timeframe!r}")
        level = levels_by_id[take.level_id]
        if take.liquidity_type != level.liquidity_type:
            raise ValueError(
                f"LiquidityTakeEvent for level {take.level_id!r} has liquidity_type {take.liquidity_type!r}, "
                f"expected {level.liquidity_type!r} (from its LiquidityLevel)"
            )
        if take.side != level.side:
            raise ValueError(
                f"LiquidityTakeEvent for level {take.level_id!r} has side {take.side!r}, expected {level.side!r} (from its LiquidityLevel)"
            )
        if take.lower != level.lower or take.upper != level.upper:
            raise ValueError(
                f"LiquidityTakeEvent for level {take.level_id!r} has bounds ({take.lower!r}, {take.upper!r}), "
                f"expected ({level.lower!r}, {level.upper!r}) (from its LiquidityLevel)"
            )
        if take.timestamp != take.known_at:
            raise ValueError(
                f"LiquidityTakeEvent for level {take.level_id!r} has timestamp {take.timestamp!r} != known_at "
                f"{take.known_at!r} -- violates the take-event contract (the two are always equal)"
            )
        if take.level_id in takes_by_level_id:
            raise ValueError(f"Duplicate take for level_id {take.level_id!r} in {source_family.value!r}/{timeframe!r}")
        takes_by_level_id[take.level_id] = take

    reclaims_by_level_id: dict[str, ReclaimResult] = {}
    for reclaim in result.reclaims:
        if reclaim.timeframe != timeframe:
            raise ValueError(f"ReclaimResult for level {reclaim.level_id!r} has timeframe {reclaim.timeframe!r}, expected {timeframe!r}")
        take = takes_by_level_id.get(reclaim.level_id)
        if take is None:
            raise ValueError(f"ReclaimResult references level_id {reclaim.level_id!r} with no matching take in {source_family.value!r}/{timeframe!r}")
        if reclaim.take_timestamp != take.timestamp:
            raise ValueError(
                f"ReclaimResult for level {reclaim.level_id!r} has take_timestamp {reclaim.take_timestamp!r}, "
                f"expected {take.timestamp!r} (its matching take's own timestamp)"
            )
        if reclaim.outcome == ReclaimOutcome.PENDING:
            if reclaim.resolved_at is not None:
                raise ValueError(
                    f"ReclaimResult for level {reclaim.level_id!r} is PENDING but has resolved_at {reclaim.resolved_at!r} "
                    "set -- PENDING means the outcome is not yet known, so resolved_at must be None"
                )
        else:
            if reclaim.resolved_at is None:
                raise ValueError(
                    f"ReclaimResult for level {reclaim.level_id!r} has terminal outcome {reclaim.outcome!r} but "
                    "resolved_at is None -- every terminal outcome must carry the instant it was resolved"
                )
            if reclaim.resolved_at <= take.known_at:
                raise ValueError(
                    f"ReclaimResult for level {reclaim.level_id!r} has resolved_at {reclaim.resolved_at!r} not "
                    f"strictly after its take's known_at {take.known_at!r} -- a reclaim cannot be known before or "
                    "at the moment of its own take"
                )
        if reclaim.level_id in reclaims_by_level_id:
            raise ValueError(f"Duplicate reclaim for level_id {reclaim.level_id!r} in {source_family.value!r}/{timeframe!r}")
        reclaims_by_level_id[reclaim.level_id] = reclaim

    out: list[LiquidityEntryCandidate] = []
    for level_id, take in takes_by_level_id.items():
        level = levels_by_id[level_id]
        out.extend(_candidate_pair(source_family, timeframe, level, CandidatePhase.TAKE, take.timestamp, take.known_at))

        reclaim = reclaims_by_level_id.get(level_id)
        if reclaim is not None and reclaim.outcome != ReclaimOutcome.PENDING:
            phase = _PHASE_FROM_RECLAIM_OUTCOME[reclaim.outcome]
            out.extend(_candidate_pair(source_family, timeframe, level, phase, reclaim.resolved_at, reclaim.resolved_at))

    return out


def _candidate_pair(
    source_family: SourceFamily,
    timeframe: str,
    level: LiquidityLevel,
    phase: CandidatePhase,
    timestamp: datetime,
    known_at: datetime,
) -> list[LiquidityEntryCandidate]:
    pair = []
    for entry_mode in (EntryMode.REVERSAL, EntryMode.CONTINUATION):
        pair.append(
            LiquidityEntryCandidate(
                candidate_id=f"{timeframe}-{source_family.value}-{level.level_id}-{phase.value}-{entry_mode.value}",
                source_family=source_family,
                timeframe=timeframe,
                level_id=level.level_id,
                liquidity_type=level.liquidity_type,
                side=level.side,
                lower=level.lower,
                upper=level.upper,
                trigger_phase=phase,
                entry_mode=entry_mode,
                direction=_DIRECTION[(level.side, entry_mode)],
                timestamp=timestamp,
                known_at=known_at,
            )
        )
    return pair
