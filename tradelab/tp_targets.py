"""TP target candidates: an audited, ordered take-profit candidate list
for a given market entry, sourced from liquidity levels and data wicks.

Not SL, not position management, not a news filter, not bias, not a
backtester -- those are separate, later steps. This module only answers
"given this one entry, what price levels could serve as TP?"

Reuses already-computed detection results directly:
``tradelab.liquidity.detect_liquidity`` (swing/equal-highs/lows, keyed by
timeframe), ``tradelab.session_liquidity.detect_session_liquidity``
(session highs/lows, keyed by timeframe -- the same two-dict convention
``tradelab.liquidity_entry`` already established for combining these two
detectors), and ``tradelab.data_wicks.detect_data_wicks`` (one flat
result, since that module has no timeframe concept -- data wicks are
always native 1m). Nothing here re-detects levels or wicks, and neither
upstream module's semantics are touched.

**Direction determines which side of price qualifies** (never the
other -- spec):

===========  ========================  ==============================
Direction    Liquidity TP               Data-wick TP
===========  ========================  ==============================
long         liquidity HIGH above       UPPER data wick above
             entry_price                entry_price
short        liquidity LOW below        LOWER data wick below
             entry_price                entry_price
===========  ========================  ==============================

A target strictly on the wrong side of ``entry_price``, or exactly at
it, is never a candidate.

**Liquidity is three disjoint source families** -- swing, equal, and
session -- derived from each level's own ``liquidity_type`` (swing and
equal both arrive bundled in one ``LiquidityDetectionResult`` per
timeframe, since ``detect_liquidity`` builds equal-highs/lows clusters
from swing levels in the same pass; session is always a separate
``LiquidityDetectionResult``, from the separate detector). At most the
**three nearest** qualifying liquidity targets are returned, sorted
ascending by distance from ``entry_price`` -- TP #1/#2/#3 candidates for
a future enumerator. Data wicks are a wholly separate TP source type
(never an entry condition here, spec 4.3): *every* qualifying data-wick
target is returned, sorted the same way, uncapped.

**Timing** (no lookahead): a liquidity level must be known
(``formed_at``) no later than ``entry_known_at``, and if it was already
``TAKEN``, that take must have happened *strictly after*
``entry_known_at`` -- a level taken before entry is stale, already-
consumed liquidity, not a forward-looking target; a level taken after
entry is exactly the case where the target gets hit. A level currently
``ABSORBED_IN_EQUAL_CLUSTER`` is never its own TP candidate (mirroring
``tradelab.liquidity_entry``: only the cluster it joined is a real,
independently tradeable level from that point on). A data wick must
likewise be known no later than ``entry_known_at`` -- it has no
take/reclaim lifecycle of its own, so no further timing check applies.

This module does not trust its input came from a real detector call: it
re-validates timeframe/source-family consistency, level/take internal
consistency (a ``TAKEN`` level must have a ``taken_at`` strictly after
its own ``formed_at``), positive prices, timezone-aware timestamps, and
target-id/level-id uniqueness, rather than assuming any of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from tradelab.data_wicks import DataWickDetectionResult, DataWickDirection, DataWickEvent
from tradelab.liquidity import HIGH, LOW, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType
from tradelab.liquidity import SUPPORTED_TIMEFRAMES as _SWING_EQUAL_TIMEFRAMES
from tradelab.session_liquidity import SUPPORTED_TIMEFRAMES as _SESSION_TIMEFRAMES
from tradelab.liquidity_target_index import LiquidityTargetIndex

LONG = "long"
SHORT = "short"
_DIRECTIONS = frozenset({LONG, SHORT})

_SWING_TYPES = frozenset({LiquidityType.SWING_HIGH, LiquidityType.SWING_LOW})
_EQUAL_TYPES = frozenset({LiquidityType.EQUAL_HIGHS, LiquidityType.EQUAL_LOWS})
_SESSION_TYPES = frozenset(lt for lt in LiquidityType if lt.value.startswith("session_"))

MAX_LIQUIDITY_TARGETS = 3
# 100 points = 400 MNQ ticks. This is deliberately a single research
# constraint, not a grid parameter: a data wick further away is not a
# plausible execution target for this intraday strategy and was responsible
# for the pathological outlier P&L seen in the diagnostic run.
MAX_DATA_WICK_TARGET_DISTANCE_POINTS = 100.0


class TPSourceType(str, Enum):
    LIQUIDITY = "liquidity"
    DATA_WICK = "data_wick"


class LiquiditySourceFamily(str, Enum):
    SWING = "swing"
    EQUAL = "equal"
    SESSION = "session"


def _liquidity_source_family(liquidity_type: LiquidityType) -> LiquiditySourceFamily:
    if liquidity_type in _SWING_TYPES:
        return LiquiditySourceFamily.SWING
    if liquidity_type in _EQUAL_TYPES:
        return LiquiditySourceFamily.EQUAL
    if liquidity_type in _SESSION_TYPES:
        return LiquiditySourceFamily.SESSION
    raise ValueError(f"Unrecognized liquidity_type {liquidity_type!r}")  # pragma: no cover -- exhaustive by construction


@dataclass(frozen=True)
class MarketEntryContext:
    """The one market entry TP candidates are being generated for."""

    entry_id: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_known_at: datetime

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValueError("MarketEntryContext.entry_id must be non-empty")
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"MarketEntryContext.direction must be one of {sorted(_DIRECTIONS)}, got {self.direction!r}")
        if self.entry_price <= 0:
            raise ValueError(f"MarketEntryContext.entry_price must be positive, got {self.entry_price!r}")
        if self.entry_known_at.tzinfo is None:
            raise ValueError(f"MarketEntryContext.entry_known_at must be timezone-aware: {self.entry_known_at!r}")
        object.__setattr__(self, "entry_known_at", self.entry_known_at.astimezone(timezone.utc))


@dataclass(frozen=True)
class TPTargetCandidate:
    """One fully-audited TP candidate for one entry.

    Fields specific to ``LIQUIDITY`` (``liquidity_type``, ``level_id``,
    ``lower``, ``upper``, ``source_family``, ``rank``) are ``None`` for
    ``DATA_WICK`` candidates, and vice versa (``event_ids``,
    ``wick_direction``, ``timestamp``) -- exactly one side is populated,
    matching ``source_type``.

    ``target_id`` is stable and deterministic: derived only from
    ``(entry_id, source_type, source_id)``, never from list position.
    """

    target_id: str
    entry_id: str
    source_type: TPSourceType
    source_id: str  # level_id (liquidity) or a stable per-candle id (data wick)
    direction: str  # the entry's own direction, copied for audit
    price: float
    distance: float  # abs(price - entry_price), always > 0
    known_at: datetime  # the source's own known_at (formed_at for liquidity, known_at for data wick)
    rank: int | None  # 1-based nearest-first rank among the (<=3) liquidity targets; None for data wick

    # LIQUIDITY-only audit:
    liquidity_type: LiquidityType | None
    level_id: str | None
    lower: float | None
    upper: float | None
    source_family: LiquiditySourceFamily | None

    # DATA_WICK-only audit:
    event_ids: tuple[str, ...] | None
    wick_direction: DataWickDirection | None
    timestamp: datetime | None


@dataclass(frozen=True)
class TPTargetCandidateResult:
    entry_id: str
    liquidity_targets: list[TPTargetCandidate]
    data_wick_targets: list[TPTargetCandidate]


def detect_tp_target_candidates(
    entry: MarketEntryContext,
    swing_equal_results: dict[str, LiquidityDetectionResult],
    session_results: dict[str, LiquidityDetectionResult],
    data_wick_result: DataWickDetectionResult,
    *,
    liquidity_index: LiquidityTargetIndex | None = None,
) -> TPTargetCandidateResult:
    """Build the ordered TP target candidates for ``entry``.

    ``swing_equal_results``/``session_results`` are keyed by timeframe,
    from ``tradelab.liquidity.detect_liquidity`` and
    ``tradelab.session_liquidity.detect_session_liquidity`` respectively
    -- the same convention ``tradelab.liquidity_entry`` uses. Unlike that
    module, no timeframe restriction beyond each detector's own native
    set is imposed here: a TP target is not an entry trigger, so an HTF
    swing high is just as valid a target as a 1m one.
    """
    liquidity_candidates = _liquidity_candidates(entry, swing_equal_results, session_results, liquidity_index=liquidity_index)
    data_wick_candidates = _data_wick_candidates(entry, data_wick_result)

    liquidity_candidates.sort(key=lambda c: (c.distance, c.known_at, c.target_id))
    liquidity_candidates = liquidity_candidates[:MAX_LIQUIDITY_TARGETS]
    liquidity_candidates = [
        TPTargetCandidate(
            target_id=c.target_id,
            entry_id=c.entry_id,
            source_type=c.source_type,
            source_id=c.source_id,
            direction=c.direction,
            price=c.price,
            distance=c.distance,
            known_at=c.known_at,
            rank=rank,
            liquidity_type=c.liquidity_type,
            level_id=c.level_id,
            lower=c.lower,
            upper=c.upper,
            source_family=c.source_family,
            event_ids=c.event_ids,
            wick_direction=c.wick_direction,
            timestamp=c.timestamp,
        )
        for rank, c in enumerate(liquidity_candidates, start=1)
    ]

    data_wick_candidates.sort(key=lambda c: (c.distance, c.known_at, c.target_id))

    all_ids = [c.target_id for c in liquidity_candidates] + [c.target_id for c in data_wick_candidates]
    if len(all_ids) != len(set(all_ids)):
        dupes = sorted({tid for tid in all_ids if all_ids.count(tid) > 1})
        raise ValueError(f"Duplicate target_id(s) produced: {dupes}")

    return TPTargetCandidateResult(
        entry_id=entry.entry_id,
        liquidity_targets=liquidity_candidates,
        data_wick_targets=data_wick_candidates,
    )


def _liquidity_candidates(
    entry: MarketEntryContext,
    swing_equal_results: dict[str, LiquidityDetectionResult],
    session_results: dict[str, LiquidityDetectionResult],
    *,
    liquidity_index: LiquidityTargetIndex | None = None,
) -> list[TPTargetCandidate]:
    if liquidity_index is not None:
        wanted_side = HIGH if entry.direction == LONG else LOW
        levels = liquidity_index.nearest(side=wanted_side, entry_price=entry.entry_price, entry_known_at=entry.entry_known_at)
        families = {LiquiditySourceFamily.SWING, LiquiditySourceFamily.EQUAL, LiquiditySourceFamily.SESSION}
        return [candidate for level in levels for candidate in _candidates_from_levels(
            entry, level.timeframe, LiquidityDetectionResult(levels=[level], takes=[], reclaims=[]), allowed_families=families
        )]
    candidates: list[TPTargetCandidate] = []
    for timeframe, result in swing_equal_results.items():
        if timeframe not in _SWING_EQUAL_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe {timeframe!r} in swing_equal_results; expected one of {sorted(_SWING_EQUAL_TIMEFRAMES)}")
        candidates.extend(_candidates_from_levels(entry, timeframe, result, allowed_families={LiquiditySourceFamily.SWING, LiquiditySourceFamily.EQUAL}))
    for timeframe, result in session_results.items():
        if timeframe not in _SESSION_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe {timeframe!r} in session_results; expected one of {sorted(_SESSION_TIMEFRAMES)}")
        candidates.extend(_candidates_from_levels(entry, timeframe, result, allowed_families={LiquiditySourceFamily.SESSION}))
    return candidates


def _candidates_from_levels(
    entry: MarketEntryContext,
    timeframe: str,
    result: LiquidityDetectionResult,
    allowed_families: set[LiquiditySourceFamily],
) -> list[TPTargetCandidate]:
    wanted_side = HIGH if entry.direction == LONG else LOW

    seen_level_ids: set[str] = set()
    out: list[TPTargetCandidate] = []
    for level in result.levels:
        if level.timeframe != timeframe:
            raise ValueError(f"LiquidityLevel {level.level_id!r} has timeframe {level.timeframe!r}, expected {timeframe!r}")
        family = _liquidity_source_family(level.liquidity_type)
        if family not in allowed_families:
            raise ValueError(
                f"LiquidityLevel {level.level_id!r} ({level.liquidity_type!r}) belongs to family {family.value!r}, "
                f"which is not among {sorted(f.value for f in allowed_families)} for the dict it was filed under"
            )
        if level.level_id in seen_level_ids:
            raise ValueError(f"Duplicate level_id {level.level_id!r} for timeframe {timeframe!r}")
        seen_level_ids.add(level.level_id)

        if level.status == LiquidityStatus.TAKEN:
            if level.taken_at is None or level.taken_at <= level.formed_at:
                raise ValueError(
                    f"LiquidityLevel {level.level_id!r} is TAKEN but taken_at ({level.taken_at!r}) is not strictly "
                    f"after formed_at ({level.formed_at!r})"
                )

        if level.side != wanted_side:
            continue  # wrong side of price for this entry's direction
        if level.status == LiquidityStatus.ABSORBED_IN_EQUAL_CLUSTER:
            continue  # not independently a target -- only the cluster it joined is
        if level.formed_at > entry.entry_known_at:
            continue  # not yet known at entry time
        if level.status == LiquidityStatus.TAKEN and level.taken_at <= entry.entry_known_at:
            continue  # already swept before entry -- stale, not a forward-looking target

        price = level.upper if wanted_side == HIGH else level.lower
        if price <= 0:
            raise ValueError(f"LiquidityLevel {level.level_id!r} has non-positive price {price!r}")
        if entry.direction == LONG and price <= entry.entry_price:
            continue  # wrong side of entry_price, or exactly at it
        if entry.direction == SHORT and price >= entry.entry_price:
            continue

        out.append(
            TPTargetCandidate(
                target_id=f"tp-liquidity-{entry.entry_id}-{level.level_id}",
                entry_id=entry.entry_id,
                source_type=TPSourceType.LIQUIDITY,
                source_id=level.level_id,
                direction=entry.direction,
                price=price,
                distance=abs(price - entry.entry_price),
                known_at=level.formed_at,
                rank=None,
                liquidity_type=level.liquidity_type,
                level_id=level.level_id,
                lower=level.lower,
                upper=level.upper,
                source_family=family,
                event_ids=None,
                wick_direction=None,
                timestamp=None,
            )
        )
    return out


def _data_wick_candidates(entry: MarketEntryContext, result: DataWickDetectionResult) -> list[TPTargetCandidate]:
    wanted_direction = DataWickDirection.UPPER if entry.direction == LONG else DataWickDirection.LOWER

    seen_source_ids: set[str] = set()
    out: list[TPTargetCandidate] = []
    for event in result.events:
        if event.direction != wanted_direction:
            continue  # wrong side of price for this entry's direction
        if event.known_at > entry.entry_known_at:
            continue  # not yet known at entry time
        if event.taken_at is not None and event.taken_at <= entry.entry_known_at:
            continue  # level was already collected before this entry

        if event.level <= 0:
            raise ValueError(f"DataWickEvent at {event.known_at!r} has non-positive level {event.level!r}")
        if entry.direction == LONG and event.level <= entry.entry_price:
            continue
        if entry.direction == SHORT and event.level >= entry.entry_price:
            continue
        if abs(event.level - entry.entry_price) > MAX_DATA_WICK_TARGET_DISTANCE_POINTS:
            continue

        source_id = f"{event.known_at.isoformat()}-{event.direction.value}"
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate DataWickEvent for {source_id!r}")
        seen_source_ids.add(source_id)

        out.append(
            TPTargetCandidate(
                target_id=f"tp-data_wick-{entry.entry_id}-{source_id}",
                entry_id=entry.entry_id,
                source_type=TPSourceType.DATA_WICK,
                source_id=source_id,
                direction=entry.direction,
                price=event.level,
                distance=abs(event.level - entry.entry_price),
                known_at=event.known_at,
                rank=None,
                liquidity_type=None,
                level_id=None,
                lower=None,
                upper=None,
                source_family=None,
                event_ids=event.event_ids,
                wick_direction=event.direction,
                timestamp=event.timestamp,
            )
        )
    return out
