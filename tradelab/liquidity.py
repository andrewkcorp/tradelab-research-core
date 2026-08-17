"""Liquidity levels: confirmed swings, equal highs/lows, take, and reclaim.

A standalone feature module, independent of ``tradelab.fvg``: it detects
liquidity levels, tracks their take (a wick breach), and separately
evaluates a multi-candle reclaim outcome after each take. It knows nothing
about FVG/iFVG, bias, entries, TP/SL, the backtester, session highs/lows,
or data wicks -- those are separate, later modules.

Liquidity is not an entry strategy here. This module emits neutral,
auditable features/confluence (levels, take events, reclaim outcomes) that
a later enumerator will combine with FVG, iFVG, or liquidity alone. Nothing
in this module decides whether, or how, to trade any of it.

Definitions implemented here match docs/strategy-spec-v0.1.md section 6:

Swing levels (6.1): a confirmed swing high/low, from closed candles only.
``pivot_strength`` candles are required on both sides; a candle at index
``p`` is a swing high when its high is strictly greater than every one of
the ``pivot_strength`` candles immediately to its left AND to its right
(swing low: symmetric, strictly lower). It becomes known only once the
last confirming candle on the right has closed.

Equal highs/lows (6.2): a subtype of liquidity, not a separate, parallel
confluence. Formed from 2+ confirmed same-side swings whose prices all
fall within a band of at most ``2 * tick_size``. A cluster is created the
moment a second qualifying swing appears (band = [min, max] of the two
prices, ``known_at`` = that second swing's confirmation time) and further
qualifying swings only grow ``member_count`` and the band -- never a new,
redundant level for the same zone. Swings may come from different days;
there is no time-based expiry on cluster eligibility.

The instant a swing becomes a cluster member (whether it forms the
cluster or joins an already-open one), its own standalone swing level
stops being independently active: its status becomes
``ABSORBED_IN_EQUAL_CLUSTER`` with ``absorbed_by_level_id`` pointing at
the cluster, it is removed from the active-level set, and from that
point on only the cluster itself can be taken or start a reclaim watch.
This keeps one physical price sweep from being reported as two events
(one for the cluster, one for a member swing) -- absorbed levels remain
in the full audit trail (``LiquidityDetectionResult.levels``) but never
appear in ``takes`` or ``reclaims`` again. A swing that is taken before
it ever pairs into a cluster is equally final: a taken level can never
later be absorbed into, or used to form, an equal-highs/lows cluster.

Take (6.1/6.2, mechanically confirmed here): a level is taken by the first
later candle whose wick strictly exceeds its outer boundary -- a wick that
only touches the level (high == upper, or low == lower) does not collect
it, and there is no close requirement of any kind. A swing high /
equal-highs level is taken when a candle's high is strictly greater than
the level's upper boundary; a swing low / equal-lows level when a
candle's low is strictly less than the lower boundary. The take is known
at that candle's close. The first take consumes the level (terminal; it
is never "taken" a second time).

Reclaim: a separate, multi-candle feature evaluated only on candles after
the take candle (never the take candle itself), within a
``reclaim_window_bars`` window. After a high-side take, reclaim looks for
price returning below the level's *inner* boundary (its lower bound --
for a plain swing this is the same single price as the outer boundary);
after a low-side take, above the inner (upper) boundary. ``reclaim_close``
(the candle's close crosses the inner boundary) is checked before
``reclaim_wick`` (only the wick reaches it) on each candle, since a
closing reclaim is the strictly stronger confirmation and implies the
weaker wick condition on the same candle. If neither happens within the
window, the outcome is the explicit ``NO_RECLAIM_IN_WINDOW`` state (never
silence); if the input data ends before the window is fully observed, the
outcome is ``PENDING`` rather than a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from tradelab.fvg import DEFAULT_TICK_SIZE, Candle, from_aggregated_bar, from_bar1m  # noqa: F401  (re-exported)

SUPPORTED_TIMEFRAMES = frozenset({"1m", "5m", "15m", "1H", "4H", "1D"})

HIGH = "high"
LOW = "low"

# Floating-point tolerance for the tick-band comparison, matching tradelab.fvg's
# treatment of tick-size thresholds.
_TICK_EPSILON = 1e-9


class LiquidityType(str, Enum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"


class LiquidityStatus(str, Enum):
    ACTIVE = "active"
    TAKEN = "taken"
    ABSORBED_IN_EQUAL_CLUSTER = "absorbed_in_equal_cluster"


class ReclaimOutcome(str, Enum):
    RECLAIM_WICK = "reclaim_wick"
    RECLAIM_CLOSE = "reclaim_close"
    NO_RECLAIM_IN_WINDOW = "no_reclaim_in_window"
    PENDING = "pending"  # window not yet fully observed given the available data


@dataclass(frozen=True)
class LiquidityLevel:
    """A single liquidity level's full audit record.

    ``lower`` == ``upper`` for a plain swing level (a single price); for an
    equal-highs/lows cluster they are the band's bounds. ``member_timestamps``
    holds the open time of each contributing swing candle, in join order --
    one entry for a plain swing, two or more for a cluster.

    ``absorbed_by_level_id`` is set only when ``status`` is
    ``ABSORBED_IN_EQUAL_CLUSTER``: it names the ``level_id`` of the
    equal-highs/lows cluster this swing became a member of. An absorbed
    level stays in the full audit trail but is no longer independently
    active -- it cannot be taken and cannot start its own reclaim watch.
    """

    level_id: str
    timeframe: str
    liquidity_type: LiquidityType
    side: str  # "high" or "low" -- derived from liquidity_type, stored explicitly
    lower: float
    upper: float
    member_count: int
    member_timestamps: tuple[datetime, ...]
    formed_at: datetime  # known_at of the candle that confirmed this level
    status: LiquidityStatus
    taken_at: datetime | None
    absorbed_by_level_id: str | None


@dataclass(frozen=True)
class LiquidityTakeEvent:
    level_id: str
    timeframe: str
    liquidity_type: LiquidityType
    side: str
    lower: float
    upper: float
    timestamp: datetime
    known_at: datetime


@dataclass(frozen=True)
class ReclaimResult:
    """Always present for every take -- never silently omitted. ``outcome``
    is one of the four ``ReclaimOutcome`` values; ``PENDING`` means the
    input data ended before the window could be fully observed."""

    level_id: str
    timeframe: str
    take_timestamp: datetime
    outcome: ReclaimOutcome
    resolved_at: datetime | None
    bars_after_take: int | None  # 1-indexed candle count after take; window size for NO_RECLAIM_IN_WINDOW


@dataclass(frozen=True)
class LiquidityDetectionResult:
    levels: list[LiquidityLevel]
    takes: list[LiquidityTakeEvent]
    reclaims: list[ReclaimResult]


@dataclass
class _OpenCluster:
    lower: float
    upper: float
    member_count: int
    member_timestamps: list[datetime]
    formed_at: datetime
    level: LiquidityLevel  # the currently-published (ACTIVE) LiquidityLevel for this cluster


@dataclass
class _ReclaimWatch:
    level: LiquidityLevel
    take_event: LiquidityTakeEvent
    bars_checked: int = 0


def detect_liquidity(
    candles: list[Candle],
    timeframe: str,
    *,
    pivot_strength: int,
    reclaim_window_bars: int,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> LiquidityDetectionResult:
    """Detect liquidity levels, take events, and reclaim outcomes on a
    chronological series of same-timeframe candles.

    ``pivot_strength`` and ``reclaim_window_bars`` have no default: both
    are meant to be swept by a future parameter grid, so callers must
    always state them explicitly rather than silently inheriting a
    default. Any positive integer is accepted for either.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {sorted(SUPPORTED_TIMEFRAMES)}")
    if not isinstance(pivot_strength, int) or pivot_strength < 1:
        raise ValueError(f"pivot_strength must be a positive integer, got {pivot_strength!r}")
    if not isinstance(reclaim_window_bars, int) or reclaim_window_bars < 1:
        raise ValueError(f"reclaim_window_bars must be a positive integer, got {reclaim_window_bars!r}")
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size!r}")

    _validate_chronology(candles)
    band_width = 2 * tick_size

    active_levels: dict[str, LiquidityLevel] = {}
    open_clusters: dict[str, list[_OpenCluster]] = {HIGH: [], LOW: []}
    unclustered: dict[str, list[LiquidityLevel]] = {HIGH: [], LOW: []}
    reclaim_watches: list[_ReclaimWatch] = []

    all_levels: dict[str, LiquidityLevel] = {}
    takes: list[LiquidityTakeEvent] = []
    reclaims: list[ReclaimResult] = []
    next_number = 1

    for index, candle in enumerate(candles):
        # 1. Advance every reclaim window already open *before* this candle
        #    (i.e. watches created on a strictly earlier candle). This runs
        #    before step 2 below creates any new watch, so a level taken by
        #    this very candle can never have this same candle counted as
        #    its own first reclaim candle.
        still_watching: list[_ReclaimWatch] = []
        for watch in reclaim_watches:
            watch.bars_checked += 1
            outcome_result = _check_reclaim(watch.level, candle, watch.bars_checked)
            if outcome_result is not None:
                reclaims.append(outcome_result)
                continue
            if watch.bars_checked >= reclaim_window_bars:
                reclaims.append(
                    ReclaimResult(
                        level_id=watch.level.level_id,
                        timeframe=watch.level.timeframe,
                        take_timestamp=watch.take_event.timestamp,
                        outcome=ReclaimOutcome.NO_RECLAIM_IN_WINDOW,
                        resolved_at=candle.known_at,
                        bars_after_take=reclaim_window_bars,
                    )
                )
                continue
            still_watching.append(watch)
        reclaim_watches = still_watching

        # 2. Take-check every currently active level against this candle
        #    (using each level's *current* band -- a cluster's band only
        #    reflects members confirmed strictly before this point). Any
        #    new watch created here is only appended to reclaim_watches
        #    after step 1 has already run for this candle, so it is first
        #    advanced starting with the *next* candle, never this one.
        newly_taken_watches: list[_ReclaimWatch] = []
        for level_id in list(active_levels):
            level = active_levels[level_id]
            if _is_taken(level, candle):
                take_event = LiquidityTakeEvent(
                    level_id=level.level_id,
                    timeframe=level.timeframe,
                    liquidity_type=level.liquidity_type,
                    side=level.side,
                    lower=level.lower,
                    upper=level.upper,
                    timestamp=candle.known_at,
                    known_at=candle.known_at,
                )
                takes.append(take_event)
                taken_level = replace(level, status=LiquidityStatus.TAKEN, taken_at=candle.known_at)
                all_levels[level_id] = taken_level
                del active_levels[level_id]
                newly_taken_watches.append(_ReclaimWatch(level=taken_level, take_event=take_event))
                # A taken cluster is consumed: it must stop absorbing new
                # members, exactly like a taken plain swing is consumed.
                if level.liquidity_type in (LiquidityType.EQUAL_HIGHS, LiquidityType.EQUAL_LOWS):
                    side_clusters = open_clusters[level.side]
                    open_clusters[level.side] = [c for c in side_clusters if c.level.level_id != level_id]
                else:
                    # A taken plain swing is equally terminal for clustering
                    # purposes: it must never be pulled off the unclustered
                    # pool to form an equal-highs/lows cluster after the
                    # fact.
                    side_unclustered = unclustered[level.side]
                    unclustered[level.side] = [lv for lv in side_unclustered if lv.level_id != level_id]
        reclaim_watches.extend(newly_taken_watches)

        # 3. Check whether a swing is confirmed as of this candle (the
        #    pivot candle at index - pivot_strength now has pivot_strength
        #    closed candles on its right, in addition to already having
        #    pivot_strength on its left).
        pivot_index = index - pivot_strength
        if pivot_index >= pivot_strength:
            for liquidity_type, side, matches in (
                (LiquidityType.SWING_HIGH, HIGH, _is_swing_high(candles, pivot_index, pivot_strength)),
                (LiquidityType.SWING_LOW, LOW, _is_swing_low(candles, pivot_index, pivot_strength)),
            ):
                if not matches:
                    continue
                pivot_candle = candles[pivot_index]
                price = pivot_candle.high if side == HIGH else pivot_candle.low
                level_id = f"{timeframe}-{liquidity_type.value}-{next_number}"
                next_number += 1
                swing_level = LiquidityLevel(
                    level_id=level_id,
                    timeframe=timeframe,
                    liquidity_type=liquidity_type,
                    side=side,
                    lower=price,
                    upper=price,
                    member_count=1,
                    member_timestamps=(pivot_candle.timestamp,),
                    formed_at=candle.known_at,
                    status=LiquidityStatus.ACTIVE,
                    taken_at=None,
                    absorbed_by_level_id=None,
                )
                active_levels[level_id] = swing_level
                all_levels[level_id] = swing_level

                # This swing may immediately become an equal-highs/lows
                # cluster member; if so, it stops being independently
                # active (see _join_or_form_cluster).
                next_number = _join_or_form_cluster(
                    side=side,
                    liquidity_type=LiquidityType.EQUAL_HIGHS if side == HIGH else LiquidityType.EQUAL_LOWS,
                    swing_level=swing_level,
                    band_width=band_width,
                    timeframe=timeframe,
                    open_clusters=open_clusters[side],
                    unclustered=unclustered[side],
                    active_levels=active_levels,
                    all_levels=all_levels,
                    next_number=next_number,
                )

    # Anything still open at the end of the data is PENDING, not a guess.
    for watch in reclaim_watches:
        reclaims.append(
            ReclaimResult(
                level_id=watch.level.level_id,
                timeframe=watch.level.timeframe,
                take_timestamp=watch.take_event.timestamp,
                outcome=ReclaimOutcome.PENDING,
                resolved_at=None,
                bars_after_take=None,
            )
        )

    ordered_levels = sorted(all_levels.values(), key=lambda lv: (lv.formed_at, lv.level_id))
    return LiquidityDetectionResult(levels=ordered_levels, takes=takes, reclaims=reclaims)


def _is_swing_high(candles: list[Candle], pivot_index: int, pivot_strength: int) -> bool:
    pivot_high = candles[pivot_index].high
    left = candles[pivot_index - pivot_strength : pivot_index]
    right = candles[pivot_index + 1 : pivot_index + 1 + pivot_strength]
    return all(pivot_high > c.high for c in left) and all(pivot_high > c.high for c in right)


def _is_swing_low(candles: list[Candle], pivot_index: int, pivot_strength: int) -> bool:
    pivot_low = candles[pivot_index].low
    left = candles[pivot_index - pivot_strength : pivot_index]
    right = candles[pivot_index + 1 : pivot_index + 1 + pivot_strength]
    return all(pivot_low < c.low for c in left) and all(pivot_low < c.low for c in right)


def _is_taken(level: LiquidityLevel, candle: Candle) -> bool:
    """A level is taken only when a wick strictly *exceeds* its outer
    boundary -- a wick that merely touches the level (high == upper /
    low == lower) does not collect it and does not consume it."""
    if level.side == HIGH:
        return candle.high > level.upper
    return candle.low < level.lower


def _check_reclaim(level: LiquidityLevel, candle: Candle, bars_after_take: int) -> ReclaimResult | None:
    inner = level.lower if level.side == HIGH else level.upper
    if level.side == HIGH:
        close_reclaim = candle.close < inner
        wick_reclaim = candle.low < inner
    else:
        close_reclaim = candle.close > inner
        wick_reclaim = candle.high > inner

    outcome = None
    if close_reclaim:
        outcome = ReclaimOutcome.RECLAIM_CLOSE
    elif wick_reclaim:
        outcome = ReclaimOutcome.RECLAIM_WICK
    if outcome is None:
        return None
    return ReclaimResult(
        level_id=level.level_id,
        timeframe=level.timeframe,
        take_timestamp=level.taken_at,
        outcome=outcome,
        resolved_at=candle.known_at,
        bars_after_take=bars_after_take,
    )


def _absorb_into_cluster(
    level_id: str,
    absorbed_by_level_id: str,
    active_levels: dict[str, LiquidityLevel],
    all_levels: dict[str, LiquidityLevel],
) -> None:
    """A swing that has just become an equal-highs/lows cluster member
    stops being independently active: it can no longer be taken, and it
    can no longer start its own reclaim watch. It stays visible in the
    full audit trail (``all_levels``) as ABSORBED_IN_EQUAL_CLUSTER, linked
    back to the cluster that absorbed it."""
    level = active_levels.pop(level_id)
    absorbed_level = replace(
        level,
        status=LiquidityStatus.ABSORBED_IN_EQUAL_CLUSTER,
        absorbed_by_level_id=absorbed_by_level_id,
    )
    all_levels[level_id] = absorbed_level


def _join_or_form_cluster(
    *,
    side: str,
    liquidity_type: LiquidityType,
    swing_level: LiquidityLevel,
    band_width: float,
    timeframe: str,
    open_clusters: list[_OpenCluster],
    unclustered: list[LiquidityLevel],
    active_levels: dict[str, LiquidityLevel],
    all_levels: dict[str, LiquidityLevel],
    next_number: int,
) -> int:
    """Try to join ``swing_level`` into an existing (still-ACTIVE) cluster,
    else try to pair it with a lone unclustered swing to form a new
    cluster, else record it as a new lone unclustered swing. Returns the
    possibly-incremented zone-numbering counter.

    Whenever ``swing_level`` (or, when a new cluster is formed, the lone
    swing it pairs with) becomes a cluster member, that swing is absorbed
    (see ``_absorb_into_cluster``) so the same physical level is never
    reported as both an independent swing and a cluster.
    """
    price = swing_level.upper  # == .lower for a plain swing
    pivot_timestamp = swing_level.member_timestamps[0]
    formed_at = swing_level.formed_at

    for cluster in open_clusters:
        new_lower = min(cluster.lower, price)
        new_upper = max(cluster.upper, price)
        if new_upper - new_lower <= band_width + _TICK_EPSILON:
            cluster.lower, cluster.upper = new_lower, new_upper
            cluster.member_count += 1
            cluster.member_timestamps.append(pivot_timestamp)
            updated_level = replace(
                cluster.level,
                lower=new_lower,
                upper=new_upper,
                member_count=cluster.member_count,
                member_timestamps=tuple(cluster.member_timestamps),
            )
            cluster.level = updated_level
            active_levels[updated_level.level_id] = updated_level
            all_levels[updated_level.level_id] = updated_level
            _absorb_into_cluster(swing_level.level_id, updated_level.level_id, active_levels, all_levels)
            return next_number

    for lone_index, lone in enumerate(unclustered):
        new_lower = min(lone.lower, price)
        new_upper = max(lone.upper, price)
        if new_upper - new_lower <= band_width + _TICK_EPSILON:
            unclustered.pop(lone_index)
            level_id = f"{timeframe}-{liquidity_type.value}-{next_number}"
            next_number += 1
            member_timestamps = [lone.member_timestamps[0], pivot_timestamp]
            cluster_level = LiquidityLevel(
                level_id=level_id,
                timeframe=timeframe,
                liquidity_type=liquidity_type,
                side=side,
                lower=new_lower,
                upper=new_upper,
                member_count=2,
                member_timestamps=tuple(member_timestamps),
                formed_at=formed_at,
                status=LiquidityStatus.ACTIVE,
                taken_at=None,
                absorbed_by_level_id=None,
            )
            active_levels[level_id] = cluster_level
            all_levels[level_id] = cluster_level
            open_clusters.append(
                _OpenCluster(
                    lower=new_lower,
                    upper=new_upper,
                    member_count=2,
                    member_timestamps=member_timestamps,
                    formed_at=formed_at,
                    level=cluster_level,
                )
            )
            _absorb_into_cluster(lone.level_id, level_id, active_levels, all_levels)
            _absorb_into_cluster(swing_level.level_id, level_id, active_levels, all_levels)
            return next_number

    # No match: this swing remains a standalone ACTIVE level and becomes
    # the new lone candidate for a future pairing.
    unclustered.append(swing_level)
    return next_number


def _validate_chronology(candles: list[Candle]) -> None:
    previous_timestamp: datetime | None = None
    previous_known_at: datetime | None = None
    for candle in candles:
        if previous_timestamp is not None and candle.timestamp <= previous_timestamp:
            raise ValueError(f"Candles must be strictly ordered by timestamp; got {candle.timestamp} after {previous_timestamp}")
        if previous_known_at is not None and candle.known_at <= previous_known_at:
            raise ValueError(
                f"Candles must have strictly increasing known_at; got {candle.known_at} "
                f"after {previous_known_at} -- this would let a later candle be evaluated "
                "before an earlier one is actually knowable (lookahead)"
            )
        previous_timestamp = candle.timestamp
        previous_known_at = candle.known_at
