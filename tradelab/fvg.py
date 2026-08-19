"""Fair Value Gap (FVG) / inverted FVG (iFVG) detector.

A standalone feature module: it detects FVG zones, tracks each zone's
independent lifecycle, and emits one-shot iFVG events. It knows nothing
about bias, liquidity, entries, stops, targets, or the backtester -- those
are separate, later modules that will *consume* this one's output.

Definitions implemented here exactly match docs/strategy-spec-v0.1.md
section 5 (Fair Value Gap): this module does not redefine anything, it
only adds the implementation-level parameters the spec leaves open
(tick size as a parameter, the set of native timeframes, and explicit
``known_at`` bookkeeping).

FVG formation (spec 5.1): a three-closed-candle pattern on a single
timeframe.
- Bullish: ``candle_1.high < candle_3.low``; zone = ``[candle_1.high,
  candle_3.low]``.
- Bearish: ``candle_1.low > candle_3.high``; zone = ``[candle_3.high,
  candle_1.low]``.
- Qualifies only when the zone width is at least ``MIN_FVG_ZONE_TICKS``
  (5) times ``tick_size``, inclusive. Smaller gaps are ignored entirely
  -- no zone, no event, nothing recorded.
- A zone is only known once its forming (third) candle has closed
  (``formed_at = candle_3.known_at``); the candle that forms a zone is
  never itself evaluated against that new zone (evaluation of every
  already-active zone against the current candle happens *before* a new
  zone completing on that same candle is added to the active set).

Lifecycle (spec 5.2), evaluated only on candles strictly after a zone's
forming candle:
- ``ACTIVE``: fresh, or only partially entered by a wick (real entry into
  the zone's interior without reaching the far/opposite boundary).
  Partial entry does not consume the zone.
- ``MITIGATED``: a wick reaches or crosses the far boundary (full
  pass-through) but the candle does not close beyond the opposite
  boundary -- including closing back inside the zone. Terminal: a
  mitigated zone never returns to ``ACTIVE``. Exactly one
  ``FVGMitigationEvent`` is emitted for this transition, carrying the
  zone's own, unchanged direction -- whether it is the zone's first
  contact ever or a later one, on any candle strictly after the zone's
  own forming candle. A consumer that only needs a zone's own first
  contact (``tradelab.fvg_touch``) and one that needs to know whenever
  a zone's ``ACTIVE`` life ends without inversion, even long after an
  earlier partial touch (``tradelab.entry_signal_timeline``'s FVG
  suspend/restore mechanic), each use exactly the stream they need.
- iFVG event: full pass-through *and* a close beyond the opposite
  boundary (below the lower bound for a bullish zone, above the upper
  bound for a bearish zone). The zone becomes ``MITIGATED`` (its
  active/inactive lifecycle ends) and exactly one ``FVGEvent`` is
  emitted, with the *opposite* direction of the originating zone (spec
  5.3): a bullish zone inverts into a bearish iFVG, a bearish zone into
  a bullish one.

No lookahead: this module only ever reasons about a candle once it is
processed in the given chronological order, and every recorded timestamp
(``formed_at``, ``resolved_at``, ``FVGEvent``/``FVGMitigationEvent``'s
own ``timestamp``/``known_at``) is the ``known_at`` of the candle that
actually confirmed it -- never an open time, and never a value derived
from a candle not yet processed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum

# NQ/MNQ default; override per-instrument via the tick_size parameter.
# Never hardcoded into the qualification logic below.
DEFAULT_TICK_SIZE = 0.25

# Spec 5.1: zones narrower than this many ticks are ignored entirely.
MIN_FVG_ZONE_TICKS = 5

# Floating-point tolerance for the tick-count comparison, so that
# non-power-of-two tick sizes (e.g. 0.01, 0.1) don't get rejected by
# representation error right at the 5-tick boundary.
_TICK_EPSILON = 1e-9

SUPPORTED_TIMEFRAMES = frozenset({"1m", "5m", "15m", "1H", "4H", "1D"})

BULLISH = "bullish"
BEARISH = "bearish"
_OPPOSITE = {BULLISH: BEARISH, BEARISH: BULLISH}


class FVGStatus(str, Enum):
    """A zone's lifecycle state. There is no "inverted" status: an iFVG is
    a one-shot event (see FVGEvent), not a zone state -- a zone that
    produced one is simply MITIGATED, same as a plain mitigation."""

    ACTIVE = "active"
    MITIGATED = "mitigated"


@dataclass(frozen=True)
class Candle:
    """A single closed candle on one native timeframe.

    ``timestamp`` is the candle's open time; ``known_at`` is the instant
    its OHLC is fully known (its close). Both must be timezone-aware.
    Callers feed this module their own bars (e.g. ``tradelab.bar.Bar1m``
    or ``tradelab.aggregate.AggregatedBar``) converted to this type --
    see ``from_bar1m`` / ``from_aggregated_bar`` below.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    known_at: datetime

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(f"Candle.timestamp must be timezone-aware: {self.timestamp!r}")
        if self.known_at.tzinfo is None:
            raise ValueError(f"Candle.known_at must be timezone-aware: {self.known_at!r}")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        object.__setattr__(self, "known_at", self.known_at.astimezone(timezone.utc))
        if self.known_at <= self.timestamp:
            raise ValueError(
                f"Candle.known_at ({self.known_at!r}) must be strictly after timestamp "
                f"({self.timestamp!r}): a candle cannot be known before or at its own open"
            )


def from_bar1m(bar) -> Candle:
    """Convenience adapter from ``tradelab.bar.Bar1m`` (native 1m candle)."""
    return Candle(
        timestamp=bar.timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        known_at=bar.timestamp + timedelta(minutes=1),
    )


def from_aggregated_bar(bar) -> Candle:
    """Convenience adapter from ``tradelab.aggregate.AggregatedBar`` (5m/15m/1H/4H/1D)."""
    return Candle(
        timestamp=bar.timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        known_at=bar.known_at,
    )


@dataclass(frozen=True)
class FVGZone:
    """A single FVG's full audit record: geometry, provenance, and final
    (or current, if still ACTIVE) lifecycle state."""

    zone_id: str
    timeframe: str
    direction: str  # "bullish" or "bearish"
    lower: float
    upper: float
    candle_1_timestamp: datetime
    candle_2_timestamp: datetime
    candle_3_timestamp: datetime
    formed_at: datetime  # == candle_3.known_at
    status: FVGStatus
    resolved_at: datetime | None  # known_at of the candle that ended ACTIVE; None while active
    produced_ifvg: bool  # True if this zone's terminal transition emitted an FVGEvent


@dataclass(frozen=True)
class FVGEvent:
    """A one-shot iFVG event: no lifecycle, no retests, just an audit
    record of the inversion. ``timestamp`` and ``known_at`` are always
    equal, both set to the confirming candle's known_at -- kept as two
    explicit fields so "this is exactly when it became knowable" is a
    directly checkable, not merely implied, property."""

    zone_id: str
    timeframe: str
    direction: str  # opposite of the originating zone's direction
    original_zone_direction: str
    lower: float
    upper: float
    timestamp: datetime
    known_at: datetime


@dataclass(frozen=True)
class FVGMitigationEvent:
    """A one-shot event: a zone became ``MITIGATED`` *without* inverting
    (full pass-through, but the candle did not close beyond the opposite
    boundary) -- no lifecycle, no retests, just an audit record that
    this zone's ``ACTIVE`` life ended here, terminally, without ever
    producing an ``FVGEvent``. ``direction`` is the zone's own,
    unchanged direction (nothing inverted). ``timestamp`` and
    ``known_at`` are always equal, both set to the confirming candle's
    known_at, exactly like ``FVGEvent``. Emitted unconditionally
    whenever a zone resolves this way -- whether that resolution
    happens to be the zone's very first contact or a later one, on any
    candle strictly after the zone's own forming candle -- so a
    consumer that only cares about a zone's *first* contact (e.g.
    ``tradelab.fvg_touch``) and one that needs to know *whenever* a
    zone's ``ACTIVE`` life ends without inversion (e.g.
    ``tradelab.entry_signal_timeline``'s FVG suspend/restore mechanic)
    can each use exactly the stream they need, without either
    re-detecting anything."""

    zone_id: str
    timeframe: str
    direction: str
    lower: float
    upper: float
    timestamp: datetime
    known_at: datetime


@dataclass(frozen=True)
class FVGDetectionResult:
    zones: list[FVGZone]
    events: list[FVGEvent]
    mitigations: list[FVGMitigationEvent] = field(default_factory=list)


def _zone_width_qualifies(lower: float, upper: float, tick_size: float) -> bool:
    ticks = (upper - lower) / tick_size
    return ticks >= MIN_FVG_ZONE_TICKS - _TICK_EPSILON


def detect_fvgs(candles: list[Candle], timeframe: str, *, tick_size: float = DEFAULT_TICK_SIZE) -> FVGDetectionResult:
    """Detect FVGs and iFVG events on a chronological series of same-timeframe candles.

    ``candles`` must already be a single native timeframe's closed bars in
    strictly increasing order, both by ``timestamp`` and by ``known_at``
    (rejecting the latter is what actually guards against lookahead: two
    candles could have increasing open times yet an out-of-order or
    repeated known_at in malformed/synthetic input, which would let a
    later-confirmed candle be evaluated as if it were already knowable).
    This function never aggregates or mixes timeframes. Every candle is
    only evaluated against zones that already existed before it, and a
    zone formed on a given candle is only ever evaluated starting with
    the next candle.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {sorted(SUPPORTED_TIMEFRAMES)}")
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size!r}")

    zones_by_id: dict[str, FVGZone] = {}
    formation_order: list[str] = []
    active_ids: list[str] = []
    events: list[FVGEvent] = []
    mitigations: list[FVGMitigationEvent] = []
    next_zone_number = 1
    previous_timestamp: datetime | None = None
    previous_known_at: datetime | None = None

    for index, candle in enumerate(candles):
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

        # 1. Evaluate every zone that already existed before this candle.
        #    (A zone this same candle is about to form is appended in step
        #    2, below, so it is never evaluated against itself here.)
        still_active_ids: list[str] = []
        for zone_id in active_ids:
            zone = zones_by_id[zone_id]
            resolved, event, mitigation = _apply_lifecycle(zone, candle)
            if resolved is None:
                still_active_ids.append(zone_id)
            else:
                zones_by_id[zone_id] = resolved
                if event is not None:
                    events.append(event)
                if mitigation is not None:
                    mitigations.append(mitigation)
        active_ids = still_active_ids

        # 2. Check whether this candle completes a new 3-candle pattern.
        if index >= 2:
            new_zone = _try_form_zone(candles[index - 2], candles[index - 1], candle, timeframe, tick_size, next_zone_number)
            if new_zone is not None:
                next_zone_number += 1
                zones_by_id[new_zone.zone_id] = new_zone
                formation_order.append(new_zone.zone_id)
                active_ids.append(new_zone.zone_id)

    zones = [zones_by_id[zone_id] for zone_id in formation_order]
    return FVGDetectionResult(zones=zones, events=events, mitigations=mitigations)


def _try_form_zone(candle_1: Candle, candle_2: Candle, candle_3: Candle, timeframe: str, tick_size: float, zone_number: int) -> FVGZone | None:
    if candle_1.high < candle_3.low:
        lower, upper = candle_1.high, candle_3.low
        direction = BULLISH
    elif candle_1.low > candle_3.high:
        lower, upper = candle_3.high, candle_1.low
        direction = BEARISH
    else:
        return None

    if not _zone_width_qualifies(lower, upper, tick_size):
        return None

    return FVGZone(
        zone_id=f"{timeframe}-FVG-{zone_number}",
        timeframe=timeframe,
        direction=direction,
        lower=lower,
        upper=upper,
        candle_1_timestamp=candle_1.timestamp,
        candle_2_timestamp=candle_2.timestamp,
        candle_3_timestamp=candle_3.timestamp,
        formed_at=candle_3.known_at,
        status=FVGStatus.ACTIVE,
        resolved_at=None,
        produced_ifvg=False,
    )


def _apply_lifecycle(zone: FVGZone, candle: Candle) -> tuple[FVGZone | None, FVGEvent | None, FVGMitigationEvent | None]:
    """Returns (resolved_zone_or_None, ifvg_event_or_None,
    mitigation_event_or_None). resolved_zone is None if the zone stays
    ACTIVE (no state change, whether untouched or only partially
    touched); otherwise exactly one of the two events is set, matching
    which terminal outcome the zone reached."""
    if zone.direction == BULLISH:
        full_pass_through = candle.low <= zone.lower
        closed_beyond_opposite = candle.close < zone.lower
    else:
        full_pass_through = candle.high >= zone.upper
        closed_beyond_opposite = candle.close > zone.upper

    if not full_pass_through:
        return None, None, None  # untouched or only a partial wick entry: stays ACTIVE either way

    if closed_beyond_opposite:
        event = FVGEvent(
            zone_id=zone.zone_id,
            timeframe=zone.timeframe,
            direction=_OPPOSITE[zone.direction],
            original_zone_direction=zone.direction,
            lower=zone.lower,
            upper=zone.upper,
            timestamp=candle.known_at,
            known_at=candle.known_at,
        )
        resolved = replace(zone, status=FVGStatus.MITIGATED, resolved_at=candle.known_at, produced_ifvg=True)
        return resolved, event, None

    mitigation = FVGMitigationEvent(
        zone_id=zone.zone_id,
        timeframe=zone.timeframe,
        direction=zone.direction,
        lower=zone.lower,
        upper=zone.upper,
        timestamp=candle.known_at,
        known_at=candle.known_at,
    )
    resolved = replace(zone, status=FVGStatus.MITIGATED, resolved_at=candle.known_at, produced_ifvg=False)
    return resolved, None, mitigation
