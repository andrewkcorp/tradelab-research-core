"""FVG/iFVG entry candidates: a neutral, audited adapter from already-
detected FVG touch and iFVG facts to the entry candidates a future
strategy enumerator will test.

Not a strategy, not a confluence combiner, and not a backtester: this
module only prepares candidate facts, exactly like
``tradelab.liquidity_entry`` does for liquidity. It does not filter by
HTF bias (2.1), red-folder news (4.2), or liquidity (6.x) -- those are
separate, later steps. Data wicks are never consumed here; they remain a
future TP/SL level source only (4.3), never an entry candidate.

Reuses ``tradelab.fvg.detect_fvgs`` (``FVGDetectionResult``: zones and
iFVG events) and ``tradelab.fvg_touch.detect_fvg_touch_events``
(``FVGTouchDetectionResult``) output directly -- it does not re-detect
zones, touches, or inversions, and does not change either module's
semantics.

**Timeframes: only ``1m``/``5m``/``15m``**, the same restriction already
used by ``tradelab.liquidity_entry`` (spec 2: the setup/trigger
timeframes) -- ``1H``/``4H``/``1D`` FVGs, though detected by
``tradelab.fvg`` on all six native timeframes, never produce an entry
candidate here.

**Two disjoint source types, never overlapping for the same fact**:

- ``FVG_TOUCH``: only a zone's first touch (``tradelab.fvg_touch``, spec
  5.5) whose ``is_entry_trigger`` is true -- ``partial_touch`` or
  ``full_mitigation_no_inversion``. A touch classified
  ``inversion_ifvg`` never becomes an ``FVG_TOUCH`` candidate (its
  ``is_entry_trigger`` is already false upstream; this module also
  rejects a malformed input that claims otherwise -- see Validation
  below). Direction follows the zone's own direction: a bullish zone's
  touch is long, a bearish zone's touch is short.
- ``IFVG``: every iFVG event (spec 5.3) as its own candidate, one per
  zone that inverted. Its direction is the event's own ``direction``
  used directly -- already the opposite of the originating zone's
  direction, so no re-derivation happens here: a bullish iFVG is long, a
  bearish iFVG is short.

A zone whose first touch happens to be the very candle that inverts it
therefore contributes only an ``IFVG`` candidate, never also an
``FVG_TOUCH`` one for that same event -- inversion is exclusively an
``IFVG`` fact. A zone that is first only *partially* touched and later,
on a different candle, goes on to genuinely invert still produces both:
an ``FVG_TOUCH`` candidate for the earlier partial touch and a separate
``IFVG`` candidate for the later inversion -- two different facts at two
different instants, not a duplicate of one.

No lookahead of its own: this module does not walk candles -- it only
ever copies ``timestamp``/``known_at`` already computed upstream, so a
candidate's ``known_at`` always equals its source event's ``known_at``
exactly. It does not, however, simply trust that its input came from a
real upstream detector call -- it re-checks the invariants that would
otherwise only be implicit:

- every iFVG/touch references a real zone on the declared timeframe,
  with matching direction/bounds;
- an iFVG's own ``known_at`` is strictly after its zone's ``formed_at``;
  a touch's ``timestamp`` is strictly after the zone's forming candle
  and its ``known_at`` is strictly after the zone's ``formed_at`` --
  malformed input that violates either is rejected outright rather than
  assumed impossible;
- a touch's ``contact_type``/``is_entry_trigger`` pair is internally
  consistent;
- an ``inversion_ifvg`` touch always has a genuinely corresponding iFVG
  event for the same zone and the same underlying candle (compared on
  ``known_at``, the one instant both event types actually share --
  ``FVGTouchEvent.timestamp`` is the candle's open while
  ``FVGEvent.timestamp`` equals its own ``known_at`` by contract, so
  they are never equal to each other for the same candle). Without this
  check, a mismatched input would silently drop that zone's only
  trigger: no ``FVG_TOUCH`` candidate (inversion is never one) and no
  ``IFVG`` candidate either (none was found). The reverse is not
  required -- a zone only partially touched and genuinely inverted later
  on a different candle has no ``inversion_ifvg``-typed touch at all
  (a zone's touch record is only ever its *first* contact), so that
  later iFVG event needs no matching touch.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tradelab.fvg import (  # noqa: F401  (re-exported)
    BEARISH,
    BULLISH,
    FVGDetectionResult,
    FVGEvent,
    FVGZone,
)
from tradelab.fvg_touch import FVGContactType, FVGTouchDetectionResult, FVGTouchEvent  # noqa: F401  (re-exported)
from tradelab.session_liquidity import SUPPORTED_TIMEFRAMES

LONG = "long"
SHORT = "short"

_DIRECTION = {BULLISH: LONG, BEARISH: SHORT}
_OPPOSITE = {BULLISH: BEARISH, BEARISH: BULLISH}

# Every touch's contact_type must agree with its own is_entry_trigger --
# this is tradelab.fvg_touch's own invariant, re-checked here rather than
# assumed, since this module cannot trust that its input actually came
# from a real detect_fvg_touch_events call.
_EXPECTED_ENTRY_TRIGGER = {
    FVGContactType.PARTIAL_TOUCH: True,
    FVGContactType.FULL_MITIGATION_NO_INVERSION: True,
    FVGContactType.INVERSION_IFVG: False,
}


class SourceType(str, Enum):
    FVG_TOUCH = "fvg_touch"
    IFVG = "ifvg"


@dataclass(frozen=True)
class FVGEntryCandidate:
    """One fully-audited FVG-based entry candidate.

    ``contact_type`` is the touch classification for ``FVG_TOUCH``
    candidates; it is ``None`` for ``IFVG`` candidates, where
    ``source_type`` itself is the identification.

    ``candidate_id`` is stable and deterministic: derived only from
    ``(timeframe, source_type, zone_id)``, never from list position or
    call order -- and never colliding between the two source types for
    the same zone, since ``source_type`` is part of it.
    """

    candidate_id: str
    source_type: SourceType
    direction: str  # "long" or "short"
    timeframe: str
    zone_id: str
    lower: float
    upper: float
    contact_type: FVGContactType | None
    timestamp: datetime
    known_at: datetime


@dataclass(frozen=True)
class FVGEntryCandidateResult:
    candidates: list[FVGEntryCandidate]


def detect_fvg_entry_candidates(
    fvg_results: dict[str, FVGDetectionResult],
    touch_results: dict[str, FVGTouchDetectionResult],
) -> FVGEntryCandidateResult:
    """Build every FVG-based entry candidate from already-computed FVG
    results (keyed by timeframe, from ``tradelab.fvg.detect_fvgs``) and
    touch results (keyed by timeframe, from
    ``tradelab.fvg_touch.detect_fvg_touch_events``).

    Only ``1m``/``5m``/``15m`` keys are accepted in either dict. Every
    key present in ``touch_results`` must also be present in
    ``fvg_results`` (touch events cannot be validated without their
    zones); a timeframe present only in ``fvg_results`` simply
    contributes no ``FVG_TOUCH`` candidates.
    """
    for timeframe in fvg_results:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported FVG entry timeframe {timeframe!r} in fvg_results; expected one of {sorted(SUPPORTED_TIMEFRAMES)}")
    for timeframe in touch_results:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported FVG entry timeframe {timeframe!r} in touch_results; expected one of {sorted(SUPPORTED_TIMEFRAMES)}")
        if timeframe not in fvg_results:
            raise ValueError(
                f"touch_results has timeframe {timeframe!r} with no matching fvg_results entry -- "
                "FVG touch events cannot be validated without their zones"
            )

    candidates: list[FVGEntryCandidate] = []
    for timeframe, fvg_result in fvg_results.items():
        candidates.extend(_candidates_for_timeframe(timeframe, fvg_result, touch_results.get(timeframe)))

    candidate_ids = [c.candidate_id for c in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        dupes = sorted({cid for cid in candidate_ids if candidate_ids.count(cid) > 1})
        raise ValueError(f"Duplicate candidate_id(s) produced: {dupes}")

    candidates.sort(key=lambda c: (c.known_at, c.candidate_id))
    return FVGEntryCandidateResult(candidates=candidates)


def _candidates_for_timeframe(
    timeframe: str,
    fvg_result: FVGDetectionResult,
    touch_result: FVGTouchDetectionResult | None,
) -> list[FVGEntryCandidate]:
    zones_by_id: dict[str, FVGZone] = {}
    for zone in fvg_result.zones:
        if zone.timeframe != timeframe:
            raise ValueError(f"FVGZone {zone.zone_id!r} has timeframe {zone.timeframe!r}, expected {timeframe!r}")
        if zone.zone_id in zones_by_id:
            raise ValueError(f"Duplicate zone_id {zone.zone_id!r} for timeframe {timeframe!r}")
        zones_by_id[zone.zone_id] = zone

    out: list[FVGEntryCandidate] = []

    ifvg_events_by_zone_id: dict[str, FVGEvent] = {}
    for event in fvg_result.events:
        if event.timeframe != timeframe:
            raise ValueError(f"FVGEvent for zone {event.zone_id!r} has timeframe {event.timeframe!r}, expected {timeframe!r}")
        zone = zones_by_id.get(event.zone_id)
        if zone is None:
            raise ValueError(f"FVGEvent references unknown zone_id {event.zone_id!r} on timeframe {timeframe!r}")
        if event.original_zone_direction != zone.direction:
            raise ValueError(
                f"FVGEvent for zone {event.zone_id!r} claims original_zone_direction {event.original_zone_direction!r}, "
                f"but the zone's actual direction is {zone.direction!r}"
            )
        if event.direction != _OPPOSITE[zone.direction]:
            raise ValueError(
                f"FVGEvent for zone {event.zone_id!r} has direction {event.direction!r}, expected the opposite of "
                f"the zone's direction {zone.direction!r} ({_OPPOSITE[zone.direction]!r})"
            )
        if event.lower != zone.lower or event.upper != zone.upper:
            raise ValueError(
                f"FVGEvent for zone {event.zone_id!r} has bounds ({event.lower!r}, {event.upper!r}), "
                f"expected ({zone.lower!r}, {zone.upper!r}) (from its FVGZone)"
            )
        if event.timestamp != event.known_at:
            raise ValueError(
                f"FVGEvent for zone {event.zone_id!r} has timestamp {event.timestamp!r} != known_at {event.known_at!r} "
                "-- violates the iFVG event contract (the two are always equal)"
            )
        if event.known_at <= zone.formed_at:
            raise ValueError(
                f"FVGEvent for zone {event.zone_id!r} has known_at {event.known_at!r} not strictly after its "
                f"zone's formed_at {zone.formed_at!r} -- an iFVG cannot be known before or at its own zone's formation"
            )
        if event.zone_id in ifvg_events_by_zone_id:
            raise ValueError(f"Duplicate iFVG event for zone_id {event.zone_id!r} on timeframe {timeframe!r}")
        ifvg_events_by_zone_id[event.zone_id] = event

        out.append(
            FVGEntryCandidate(
                candidate_id=f"{timeframe}-{SourceType.IFVG.value}-{event.zone_id}",
                source_type=SourceType.IFVG,
                direction=_DIRECTION[event.direction],
                timeframe=timeframe,
                zone_id=event.zone_id,
                lower=event.lower,
                upper=event.upper,
                contact_type=None,
                timestamp=event.timestamp,
                known_at=event.known_at,
            )
        )

    if touch_result is not None:
        seen_touch_zone_ids: set[str] = set()
        for touch in touch_result.events:
            if touch.timeframe != timeframe:
                raise ValueError(f"FVGTouchEvent for zone {touch.zone_id!r} has timeframe {touch.timeframe!r}, expected {timeframe!r}")
            zone = zones_by_id.get(touch.zone_id)
            if zone is None:
                raise ValueError(f"FVGTouchEvent references unknown zone_id {touch.zone_id!r} on timeframe {timeframe!r}")
            if touch.direction != zone.direction:
                raise ValueError(
                    f"FVGTouchEvent for zone {touch.zone_id!r} has direction {touch.direction!r}, "
                    f"expected the zone's own direction {zone.direction!r}"
                )
            if touch.lower != zone.lower or touch.upper != zone.upper:
                raise ValueError(
                    f"FVGTouchEvent for zone {touch.zone_id!r} has bounds ({touch.lower!r}, {touch.upper!r}), "
                    f"expected ({zone.lower!r}, {zone.upper!r}) (from its FVGZone)"
                )
            if touch.timestamp <= zone.candle_3_timestamp:
                raise ValueError(
                    f"FVGTouchEvent for zone {touch.zone_id!r} has timestamp {touch.timestamp!r} not strictly after "
                    f"its zone's forming candle ({zone.candle_3_timestamp!r}) -- a touch cannot belong to the "
                    "forming candle itself or an earlier one"
                )
            if touch.known_at <= zone.formed_at:
                raise ValueError(
                    f"FVGTouchEvent for zone {touch.zone_id!r} has known_at {touch.known_at!r} not strictly after "
                    f"its zone's formed_at {zone.formed_at!r} -- a touch cannot be known before or at its own "
                    "zone's formation"
                )
            if touch.is_entry_trigger != _EXPECTED_ENTRY_TRIGGER[touch.contact_type]:
                raise ValueError(
                    f"FVGTouchEvent for zone {touch.zone_id!r} has contact_type {touch.contact_type!r} with "
                    f"is_entry_trigger={touch.is_entry_trigger!r}, expected {_EXPECTED_ENTRY_TRIGGER[touch.contact_type]!r} "
                    "-- inversion_ifvg must never be an FVG-touch entry trigger"
                )
            if touch.contact_type == FVGContactType.INVERSION_IFVG:
                # An inversion touch must have a genuinely corresponding iFVG
                # event for the same candle -- otherwise the adapter would
                # silently drop this zone's only trigger (no FVG_TOUCH
                # candidate, since inversion is never one; no IFVG candidate
                # either, since none was found). The reverse is NOT required:
                # a zone first only partially touched, and only later
                # genuinely inverted on a different candle, has no
                # INVERSION_IFVG-typed touch at all (touch only ever records
                # the *first* contact) -- that FVGEvent is validated on its
                # own above and needs no matching touch.
                matching_event = ifvg_events_by_zone_id.get(touch.zone_id)
                if matching_event is None:
                    raise ValueError(
                        f"FVGTouchEvent for zone {touch.zone_id!r} has contact_type=inversion_ifvg but no matching "
                        f"FVGEvent was found on timeframe {timeframe!r} -- an inversion touch must have a "
                        "corresponding iFVG event, or the trigger it represents would be silently lost"
                    )
                # Compared on known_at, not timestamp: FVGTouchEvent.timestamp
                # is the candle's open, while FVGEvent.timestamp equals its
                # own known_at (the candle's close) by contract -- known_at
                # is the one instant both event types actually share for the
                # same underlying candle.
                if matching_event.known_at != touch.known_at:
                    raise ValueError(
                        f"FVGTouchEvent for zone {touch.zone_id!r} (inversion_ifvg) has known_at {touch.known_at!r}, "
                        f"but its matching FVGEvent has known_at {matching_event.known_at!r} -- they must refer to "
                        "the same candle"
                    )
            if touch.zone_id in seen_touch_zone_ids:
                raise ValueError(f"Duplicate FVGTouchEvent for zone_id {touch.zone_id!r} on timeframe {timeframe!r}")
            seen_touch_zone_ids.add(touch.zone_id)

            if not touch.is_entry_trigger:
                continue  # inversion_ifvg: the separate IFVG candidate above is the trigger for this zone

            out.append(
                FVGEntryCandidate(
                    candidate_id=f"{timeframe}-{SourceType.FVG_TOUCH.value}-{touch.zone_id}",
                    source_type=SourceType.FVG_TOUCH,
                    direction=_DIRECTION[touch.direction],
                    timeframe=timeframe,
                    zone_id=touch.zone_id,
                    lower=touch.lower,
                    upper=touch.upper,
                    contact_type=touch.contact_type,
                    timestamp=touch.timestamp,
                    known_at=touch.known_at,
                )
            )

    return out
