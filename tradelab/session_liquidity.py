"""Session liquidity: high/low of fixed America/New_York time windows.

A neutral, auditable feature module, matching spec section 6.3. It knows
nothing about FVG/iFVG, bias, entries, TP/SL, the backtester, swing/equal
liquidity, or the strategy enumerator -- those are separate, later steps.

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
*detection* itself (the completeness/window logic below) is independent
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
price happens to coincide, and each has its own independent take/reclaim
(a wick breaking the narrower window is not automatically a break of the
wider one).

**Supported timeframes: only ``1m``, ``5m``, ``15m``** -- deliberately
narrower than ``tradelab.liquidity``'s six. Session boundaries (e.g.
Pre-NY 08:30-09:30) do not generally align with higher-timeframe bucket
edges (1H/4H/1D); computing a session high/low from a candle that is only
*partially* inside the window would silently mix in price action from
outside it and produce a wrong extreme. 1m/5m/15m candles evenly divide
every one of the windows above (all of them are whole multiples of 15
minutes, including across a DST shift, since the shift itself is always
a whole multiple of 60 minutes), so this constraint is never a real
limitation at those timeframes.

**Completeness.** A session occurrence's high/low is derived only from
the *complete*, contiguous set of candles the calendar grid actually
expects inside its window: every timeframe-sized slot from the window's
NY-local start (converted to UTC, so a DST shift changes the count of
expected slots exactly the way it changes the real elapsed time) up to
its end. If even one expected slot has no matching candle -- ``timestamp``
matches exactly nothing in the input -- the whole occurrence is
incomplete and produces **no level at all** (neither high nor low), never
a level computed from a partial subset. An occurrence with zero candles
is the same case (every expected slot missing) and needs no separate
handling.

**Availability (``formed_at`` / take eligibility).** A level's
``formed_at`` is the ``known_at`` of the occurrence's own last expected
member candle -- the instant its high/low was actually last updated.
Because completeness has already been fully determined once the window
closes, the level is genuinely knowable from that instant on, with no
extra waiting period: unlike swing/FVG confirmation (where the
*confirming* candle is deliberately exempt from evaluating what it just
confirmed, because it is structurally part of establishing the fact), a
session's confirming candle is an ordinary, unrelated candle that starts
strictly after the level was already fully known -- so it is itself
already eligible to take that level, not skipped. Concretely: this
module finalizes a just-closed occurrence *before* running that same
candle's take-check, so the first candle after the session is a normal
take candidate, exactly like every candle after it. Levels are still
never evaluated on any candle at or before ``formed_at`` (no lookahead in
either direction): completeness is only ever decided once a later candle
in the input has actually been observed to fall outside the window,
never assumed from the window's nominal boundary alone.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.fvg import Candle, from_aggregated_bar, from_bar1m  # noqa: F401  (re-exported)
from tradelab.liquidity import (
    HIGH,
    LOW,
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
UTC = timezone.utc

# Execution/setup-trigger timeframes only (spec 2. Interwały setupu/triggera:
# 15m, 5m, 1m) -- see module docstring for why 1H/4H/1D are excluded.
_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
}
SUPPORTED_TIMEFRAMES = frozenset(_TIMEFRAME_DELTAS)

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


def _expected_utc_timestamps(anchor: date, window: tuple[int, int, int, int], delta: timedelta) -> list[datetime]:
    """Every timeframe-sized slot start (in UTC) the calendar grid expects
    inside this occurrence's window, computed from the NY-local wall-clock
    boundaries so a DST shift changes the count exactly as it changes the
    real elapsed time -- never a fixed-offset approximation."""
    start_h, start_m, end_h, end_m = window
    wraps = time(end_h, end_m) <= time(start_h, start_m)
    end_anchor = anchor + timedelta(days=1) if wraps else anchor
    start_utc = datetime(anchor.year, anchor.month, anchor.day, start_h, start_m, tzinfo=NY).astimezone(UTC)
    end_utc = datetime(end_anchor.year, end_anchor.month, end_anchor.day, end_h, end_m, tzinfo=NY).astimezone(UTC)
    expected = []
    t = start_utc
    while t < end_utc:
        expected.append(t)
        t += delta
    return expected


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
    delta = _TIMEFRAME_DELTAS[timeframe]
    candles_by_timestamp = {c.timestamp: c for c in candles}

    active_levels: dict[str, LiquidityLevel] = {}
    all_levels: dict[str, LiquidityLevel] = {}
    takes: list[LiquidityTakeEvent] = []
    reclaims: list[ReclaimResult] = []
    reclaim_watches: list[_ReclaimWatch] = []
    open_anchors: dict[str, date | None] = {key: None for key in _SESSION_WINDOWS}
    next_number = 1

    for candle in candles:
        local_dt = candle.timestamp.astimezone(NY)

        # 0. Finalize any session window this candle has now exited. Unlike
        #    swing/FVG confirmation, the just-closed level(s) are NOT
        #    exempt from this same candle's take-check below: completeness
        #    was already fully determined at the session's own last member
        #    candle (formed_at, set inside _finalize_session_window), which
        #    is strictly earlier than this candle -- so this candle is a
        #    perfectly ordinary, already-eligible take candidate, and
        #    skipping it here would just be an artificial extra delay.
        for key, window in _SESSION_WINDOWS.items():
            anchor = _session_anchor(local_dt, window)
            open_anchor = open_anchors[key]
            if open_anchor is not None and anchor != open_anchor:
                next_number = _finalize_session_window(
                    key, open_anchor, timeframe, delta, candles_by_timestamp, active_levels, all_levels, next_number
                )
                open_anchors[key] = None

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

        # 3. Record which occurrence this candle opens or extends (checked
        #    by the candle's *open* time in NY, same "membership by bucket
        #    start" convention as tradelab.session_calendar). This only
        #    tracks *which anchor* is in progress per window -- the actual
        #    high/low/completeness is derived from the full candle set at
        #    finalization time (step 0 above), not accumulated here.
        for key, window in _SESSION_WINDOWS.items():
            anchor = _session_anchor(local_dt, window)
            if anchor is not None:
                open_anchors[key] = anchor

    # Any occurrence still open at the end of the data never had its close
    # confirmed by a later candle -- incomplete, so it produces no level
    # (spec 6.3/6.4/6.5: never fabricate a boundary from missing data).

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


def _finalize_session_window(
    key: str,
    anchor: date,
    timeframe: str,
    delta: timedelta,
    candles_by_timestamp: dict[datetime, Candle],
    active_levels: dict[str, LiquidityLevel],
    all_levels: dict[str, LiquidityLevel],
    next_number: int,
) -> int:
    """Build this occurrence's high/low levels only if every expected slot
    in its window has a matching candle; otherwise produce nothing at all
    (neither high nor low) for it."""
    window = _SESSION_WINDOWS[key]
    expected = _expected_utc_timestamps(anchor, window, delta)
    members: list[Candle] = []
    for ts in expected:
        member = candles_by_timestamp.get(ts)
        if member is None:
            return next_number  # incomplete: at least one expected candle is missing
        members.append(member)

    high = max(c.high for c in members)
    low = min(c.low for c in members)
    formed_at = members[-1].known_at
    member_timestamps = tuple(c.timestamp for c in members)
    member_count = len(members)

    high_type, low_type = _SESSION_LIQUIDITY_TYPES[key]
    for liquidity_type, side, price in ((high_type, HIGH, high), (low_type, LOW, low)):
        level_id = f"{timeframe}-{liquidity_type.value}-{next_number}"
        next_number += 1
        level = LiquidityLevel(
            level_id=level_id,
            timeframe=timeframe,
            liquidity_type=liquidity_type,
            side=side,
            lower=price,
            upper=price,
            member_count=member_count,
            member_timestamps=member_timestamps,
            formed_at=formed_at,
            status=LiquidityStatus.ACTIVE,
            taken_at=None,
            absorbed_by_level_id=None,
        )
        active_levels[level_id] = level
        all_levels[level_id] = level
    return next_number
