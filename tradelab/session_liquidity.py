"""Session liquidity: high/low of fixed America/New_York time windows.

A neutral, auditable feature module, matching spec section 6.3. It knows
nothing about FVG/iFVG, bias, entries, TP/SL, the backtester, swing/equal
liquidity, or the strategy enumerator -- those are separate, later steps.
It operates on a single native timeframe at a time, exactly like
``tradelab.liquidity`` and ``tradelab.fvg``.

Unlike ``tradelab.fvg`` vs. ``tradelab.liquidity`` (deliberately fully
independent of each other), this module intentionally reuses
``tradelab.liquidity``'s public event model -- ``LiquidityLevel``,
``LiquidityTakeEvent``, ``ReclaimResult``, ``ReclaimOutcome``,
``LiquidityStatus``, ``LiquidityDetectionResult`` -- instead of defining a
parallel one, so a later enumerator sees one consistent liquidity event
shape regardless of source. It also reuses that module's take/reclaim
mechanics directly (``_is_taken``, ``_check_reclaim``) rather than
reimplementing them, because the spec requires session take/reclaim to be
*identical* to swing/equal-highs take/reclaim (strict wick-only take, the
same four-outcome reclaim) -- importing the tested implementation
guarantees that identity instead of merely aiming for it. Session
*detection* itself (the time-window state machine below) is independent
new logic; nothing here changes swing/equal-highs behavior.

Sessions (spec 6.3), all in ``America/New_York`` wall-clock time via
``zoneinfo.ZoneInfo`` (never a fixed UTC offset, so DST transitions are
handled correctly for free): start inclusive, end exclusive.

===========  =============
Session      NY window
===========  =============
Asia         20:00-00:00
London       02:00-05:00
Pre-NY       08:30-09:30
NY AM        09:30-12:00
NY lunch     12:00-13:00
NY PM        13:00-16:00
Overnight    18:00-09:30
===========  =============

Sessions may overlap in time (Asia is fully contained in Overnight) --
that is deliberate: each window produces its own high and its own low,
each with its own ``liquidity_type`` and ``level_id``, even when the
price happens to coincide.

A session instance (one occurrence of one window, e.g. "Monday's Asia
session") accumulates high/low from every candle whose *open* time falls
in ``[start, end)`` for that occurrence. It becomes known -- two
``LiquidityLevel`` records are created, ACTIVE, one per side -- only once
a later candle confirms the window has actually closed (its open time no
longer belongs to that occurrence). ``formed_at`` is the ``known_at`` of
the session's own last member candle, not the later confirming candle,
so it reflects exactly when the high/low was actually last updated. No
lookahead: the confirming candle is never assumed, only observed, and
the two new levels are never evaluated against the very candle that
confirmed them (evaluation starts on the next one) -- the same ordering
already used for swing/FVG confirmation.

A session with zero member candles never produces a level. A session
still open when the input data ends (no later candle ever confirmed its
close) is incomplete and also never produces a level -- this module
never fabricates a boundary or interpolates missing minutes to force a
result; see spec 6.3/6.4 and the module's own docstring conventions
already used by ``tradelab.liquidity``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from tradelab.fvg import Candle, from_aggregated_bar, from_bar1m  # noqa: F401  (re-exported)
from tradelab.liquidity import (
    HIGH,
    LOW,
    SUPPORTED_TIMEFRAMES,
    LiquidityDetectionResult,
    LiquidityLevel,
    LiquidityStatus,
    LiquidityTakeEvent,
    LiquidityType,
    ReclaimOutcome,
    ReclaimResult,
    _check_reclaim,
    _is_taken,
    _ReclaimWatch,
    _validate_chronology,
)

NY = ZoneInfo("America/New_York")

# (start_hour, start_minute, end_hour, end_minute) in NY wall-clock time,
# start inclusive / end exclusive. A window "wraps" past midnight when its
# end time-of-day is <= its start time-of-day (Asia, Overnight).
_SESSION_WINDOWS: dict[str, tuple[int, int, int, int]] = {
    "asia": (20, 0, 0, 0),
    "london": (2, 0, 5, 0),
    "pre_ny": (8, 30, 9, 30),
    "ny_am": (9, 30, 12, 0),
    "ny_lunch": (12, 0, 13, 0),
    "ny_pm": (13, 0, 16, 0),
    "overnight": (18, 0, 9, 30),
}

_SESSION_LIQUIDITY_TYPES: dict[str, tuple[LiquidityType, LiquidityType]] = {
    "asia": (LiquidityType.SESSION_ASIA_HIGH, LiquidityType.SESSION_ASIA_LOW),
    "london": (LiquidityType.SESSION_LONDON_HIGH, LiquidityType.SESSION_LONDON_LOW),
    "pre_ny": (LiquidityType.SESSION_PRE_NY_HIGH, LiquidityType.SESSION_PRE_NY_LOW),
    "ny_am": (LiquidityType.SESSION_NY_AM_HIGH, LiquidityType.SESSION_NY_AM_LOW),
    "ny_lunch": (LiquidityType.SESSION_NY_LUNCH_HIGH, LiquidityType.SESSION_NY_LUNCH_LOW),
    "ny_pm": (LiquidityType.SESSION_NY_PM_HIGH, LiquidityType.SESSION_NY_PM_LOW),
    "overnight": (LiquidityType.SESSION_OVERNIGHT_HIGH, LiquidityType.SESSION_OVERNIGHT_LOW),
}


@dataclass
class _OpenSessionInstance:
    anchor: date
    high: float
    low: float
    member_count: int
    member_timestamps: list[datetime]
    last_known_at: datetime


def _session_anchor(local_dt: datetime, window: tuple[int, int, int, int]) -> date | None:
    """Which occurrence (identified by its NY calendar start date)
    ``local_dt`` belongs to for this window, or ``None`` if it belongs to
    none. Non-wrapping windows anchor to their own date; wrapping windows
    (end-of-day <= start-of-day) anchor a post-midnight time to the
    *previous* date, since that's when the occurrence actually started."""
    start_h, start_m, end_h, end_m = window
    start_t = time(start_h, start_m)
    end_t = time(end_h, end_m)
    t = local_dt.time()
    wraps = end_t <= start_t
    if not wraps:
        return local_dt.date() if start_t <= t < end_t else None
    if t >= start_t:
        return local_dt.date()
    if t < end_t:
        return local_dt.date() - timedelta(days=1)
    return None


def detect_session_liquidity(
    candles: list[Candle],
    timeframe: str,
    *,
    reclaim_window_bars: int,
) -> LiquidityDetectionResult:
    """Detect session high/low liquidity levels, take events, and reclaim
    outcomes on a chronological series of same-timeframe candles.

    ``reclaim_window_bars`` has no default, matching
    ``tradelab.liquidity.detect_liquidity``: it is meant to be swept by a
    future parameter grid, so callers must always state it explicitly.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {sorted(SUPPORTED_TIMEFRAMES)}")
    if not isinstance(reclaim_window_bars, int) or reclaim_window_bars < 1:
        raise ValueError(f"reclaim_window_bars must be a positive integer, got {reclaim_window_bars!r}")

    _validate_chronology(candles)

    active_levels: dict[str, LiquidityLevel] = {}
    all_levels: dict[str, LiquidityLevel] = {}
    takes: list[LiquidityTakeEvent] = []
    reclaims: list[ReclaimResult] = []
    reclaim_watches: list[_ReclaimWatch] = []
    open_instances: dict[str, _OpenSessionInstance | None] = {key: None for key in _SESSION_WINDOWS}
    next_number = 1

    for candle in candles:
        # 1. Advance every reclaim window already open *before* this candle,
        #    same ordering as tradelab.liquidity: this always runs before
        #    step 2 below can add a new watch, so a level taken by this
        #    very candle can never count this same candle as its own first
        #    reclaim candle.
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

        # 2. Take-check every currently active session level against this
        #    candle, using the identical strict-take rule as swing/equal
        #    liquidity (reused, not reimplemented -- see module docstring).
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
        reclaim_watches.extend(newly_taken_watches)

        # 3. Update every session window's currently-open instance with
        #    this candle (checked by the candle's *open* time in NY, same
        #    "membership by bucket start" convention as
        #    tradelab.session_calendar). Windows are independent and may
        #    overlap (Asia inside Overnight): each is tracked separately.
        local_dt = candle.timestamp.astimezone(NY)
        for key, window in _SESSION_WINDOWS.items():
            anchor = _session_anchor(local_dt, window)
            instance = open_instances[key]
            if instance is not None and (anchor is None or anchor != instance.anchor):
                next_number = _finalize_session_instance(key, instance, timeframe, active_levels, all_levels, next_number)
                open_instances[key] = None
                instance = None
            if anchor is None:
                continue
            if instance is None:
                open_instances[key] = _OpenSessionInstance(
                    anchor=anchor,
                    high=candle.high,
                    low=candle.low,
                    member_count=1,
                    member_timestamps=[candle.timestamp],
                    last_known_at=candle.known_at,
                )
            else:
                instance.high = max(instance.high, candle.high)
                instance.low = min(instance.low, candle.low)
                instance.member_count += 1
                instance.member_timestamps.append(candle.timestamp)
                instance.last_known_at = candle.known_at

    # Any instance still open at the end of the data never had its close
    # confirmed by a later candle -- incomplete, so it produces no level
    # (spec 6.3/6.4: never fabricate a boundary from missing data).

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


def _finalize_session_instance(
    key: str,
    instance: _OpenSessionInstance,
    timeframe: str,
    active_levels: dict[str, LiquidityLevel],
    all_levels: dict[str, LiquidityLevel],
    next_number: int,
) -> int:
    high_type, low_type = _SESSION_LIQUIDITY_TYPES[key]
    for liquidity_type, side, price in (
        (high_type, HIGH, instance.high),
        (low_type, LOW, instance.low),
    ):
        level_id = f"{timeframe}-{liquidity_type.value}-{next_number}"
        next_number += 1
        level = LiquidityLevel(
            level_id=level_id,
            timeframe=timeframe,
            liquidity_type=liquidity_type,
            side=side,
            lower=price,
            upper=price,
            member_count=instance.member_count,
            member_timestamps=tuple(instance.member_timestamps),
            formed_at=instance.last_known_at,
            status=LiquidityStatus.ACTIVE,
            taken_at=None,
            absorbed_by_level_id=None,
        )
        active_levels[level_id] = level
        all_levels[level_id] = level
    return next_number
