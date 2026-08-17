"""FVG touch events: the first real wick contact with an active FVG zone,
on the zone's own native timeframe -- one input a future entry
enumerator will combine with iFVG events and liquidity triggers.

Not a strategy and not an executor: this module only prepares one kind
of audited trigger fact -- "this FVG zone was first touched here, and
here is what kind of touch it was." A later enumerator will test the FVG
touch trigger, iFVG, and the various liquidity types independently and
in directionally-agreeing combinations; nothing here decides entries,
TP/SL, bias, or backtesting. Data wicks are not an entry confluence and
do not affect HTF bias either -- they remain a separate, future TP/SL
level source (spec 4.3), unrelated to and untouched by this module.

Reuses ``tradelab.fvg.detect_fvgs``'s own output (``FVGDetectionResult``)
rather than re-detecting zones -- FVG/iFVG formation, lifecycle, and
inversion semantics are entirely unchanged and untouched here. What this
module adds is new: ``detect_fvgs`` only ever records a zone's
*terminal* resolution (the first candle whose wick fully passes through
it), treating an earlier *partial* touch exactly like no touch at all
("stays ACTIVE either way" -- see that module's ``_apply_lifecycle``).
This module re-walks the same candles to find each zone's genuinely
first touch of any kind -- partial or full -- since that first touch,
not the zone's eventual resolution, is the trigger fact a future entry
rule needs.

Touch classification, for the first candle (strictly after a zone's
forming candle) whose wick actually enters the zone's interior:
- Entering the interior requires the wick to cross **strictly past** the
  near boundary (the one price approaches from outside) -- an exact
  touch of that boundary alone is not enough. The far boundary's
  full-pass-through / close-beyond comparisons are exactly
  ``tradelab.fvg``'s own existing ones (inclusive of the boundary),
  unchanged.
- **partial_touch**: the wick enters but does not reach the zone's far
  boundary. The zone remains ACTIVE in ``tradelab.fvg``, unaffected by
  this module.
- **full_mitigation_no_inversion**: the wick fully passes through the
  far boundary, but the candle's close does not go beyond the opposite
  boundary. The zone becomes MITIGATED in ``tradelab.fvg`` -- but this
  first touch is still a valid FVG entry trigger.
- **inversion_ifvg**: full pass-through *and* a close beyond the
  opposite boundary -- exactly ``tradelab.fvg``'s own iFVG condition.
  This is *not* an FVG entry trigger; the existing, separate iFVG event
  already emitted by ``detect_fvgs`` is the trigger for that case.

At most one ``FVGTouchEvent`` per zone, ever -- the first touch,
whatever kind it is. Every later touch (even on a zone still ACTIVE
after only a partial touch) is silently ignored; a zone whose first
touch happens to be a full pass-through/inversion is never separately
"partially" touched beforehand by construction. ``is_entry_trigger`` is
true for ``partial_touch`` and ``full_mitigation_no_inversion``, false
for ``inversion_ifvg``.

No lookahead: a touch is known only at the touching candle's own
``known_at``, and the candle that forms a zone is never itself checked
against it -- the same "eligible starting the next candle" convention
already used by ``tradelab.fvg``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tradelab.fvg import (  # noqa: F401  (re-exported)
    BEARISH,
    BULLISH,
    SUPPORTED_TIMEFRAMES,
    Candle,
    FVGDetectionResult,
    FVGZone,
    from_aggregated_bar,
    from_bar1m,
)
from tradelab.liquidity import _validate_chronology


class FVGContactType(str, Enum):
    PARTIAL_TOUCH = "partial_touch"
    FULL_MITIGATION_NO_INVERSION = "full_mitigation_no_inversion"
    INVERSION_IFVG = "inversion_ifvg"


_ENTRY_TRIGGER_TYPES = frozenset({FVGContactType.PARTIAL_TOUCH, FVGContactType.FULL_MITIGATION_NO_INVERSION})


@dataclass(frozen=True)
class FVGTouchEvent:
    """The first (and only) recorded touch of one FVG zone -- full audit
    record: which zone, its geometry and direction, the touching
    candle's own timestamp/known_at, the touch's classification, and
    whether it is an FVG entry trigger."""

    zone_id: str
    timeframe: str
    direction: str  # the zone's own direction: "bullish" or "bearish"
    lower: float
    upper: float
    timestamp: datetime  # touching candle's open
    known_at: datetime  # touching candle's close -- when this event is actually known
    contact_type: FVGContactType
    is_entry_trigger: bool


@dataclass(frozen=True)
class FVGTouchDetectionResult:
    events: list[FVGTouchEvent]


def detect_fvg_touch_events(candles: list[Candle], fvg_result: FVGDetectionResult, timeframe: str) -> FVGTouchDetectionResult:
    """Detect each FVG zone's first touch on ``candles`` -- the exact same
    chronological series of native-timeframe candles ``fvg_result`` was
    computed from via ``tradelab.fvg.detect_fvgs(candles, timeframe)``.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {sorted(SUPPORTED_TIMEFRAMES)}")
    for zone in fvg_result.zones:
        if zone.timeframe != timeframe:
            raise ValueError(f"FVGZone {zone.zone_id!r} has timeframe {zone.timeframe!r}, expected {timeframe!r}")

    _validate_chronology(candles)

    zones_by_formation_ts: dict[datetime, list[FVGZone]] = {}
    for zone in fvg_result.zones:
        zones_by_formation_ts.setdefault(zone.candle_3_timestamp, []).append(zone)

    active_for_touch: list[FVGZone] = []
    activated_zone_ids: set[str] = set()
    events: list[FVGTouchEvent] = []

    for candle in candles:
        # 1. Check touch for zones that became eligible on a strictly
        #    earlier candle -- never the candle that forms them (added in
        #    step 2 below, after this check has already run for it).
        still_pending: list[FVGZone] = []
        for zone in active_for_touch:
            contact_type = _classify_touch(zone, candle)
            if contact_type is None:
                still_pending.append(zone)
                continue
            events.append(
                FVGTouchEvent(
                    zone_id=zone.zone_id,
                    timeframe=zone.timeframe,
                    direction=zone.direction,
                    lower=zone.lower,
                    upper=zone.upper,
                    timestamp=candle.timestamp,
                    known_at=candle.known_at,
                    contact_type=contact_type,
                    is_entry_trigger=contact_type in _ENTRY_TRIGGER_TYPES,
                )
            )
        active_for_touch = still_pending

        # 2. Any zone formed by this candle becomes eligible starting the
        #    next candle, never this one.
        for zone in zones_by_formation_ts.get(candle.timestamp, []):
            active_for_touch.append(zone)
            activated_zone_ids.add(zone.zone_id)

    missing = {zone.zone_id for zone in fvg_result.zones} - activated_zone_ids
    if missing:
        raise ValueError(
            f"fvg_result contains zone(s) whose forming candle is not present in candles: {sorted(missing)} -- "
            "candles must be the exact same series detect_fvgs(candles, timeframe) was computed from"
        )

    return FVGTouchDetectionResult(events=events)


def _classify_touch(zone: FVGZone, candle: Candle) -> FVGContactType | None:
    if zone.direction == BULLISH:
        entered = candle.low < zone.upper
        full_pass_through = candle.low <= zone.lower
        closed_beyond_opposite = candle.close < zone.lower
    else:
        entered = candle.high > zone.lower
        full_pass_through = candle.high >= zone.upper
        closed_beyond_opposite = candle.close > zone.upper

    if not entered:
        return None
    if not full_pass_through:
        return FVGContactType.PARTIAL_TOUCH
    if closed_beyond_opposite:
        return FVGContactType.INVERSION_IFVG
    return FVGContactType.FULL_MITIGATION_NO_INVERSION
