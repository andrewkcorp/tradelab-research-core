"""HTF (higher-timeframe) bias: an auditable, no-lookahead directional
state for ``1H``, ``4H``, and ``1D``, driven *exclusively* by FVG
formation and iFVG (spec section 2.1).

A neutral, auditable feature module. It does not detect FVGs itself --
it consumes the ``FVGDetectionResult`` already produced by
``tradelab.fvg.detect_fvgs`` for each higher timeframe, so FVG/iFVG
semantics are never duplicated or redefined here. It knows nothing about
liquidity (swing/equal-highs, session, or data wicks), market structure
(BOS/CHoCH), entries, TP/SL, or the backtester -- liquidity in particular
remains an independent confluence for later setup-level combination
(spec 3.4) and never influences any bias state computed here.

Two, and only two, kinds of directional confluence per timeframe:

- **FVG formation**: an ``FVGZone``'s own ``direction`` at its
  ``formed_at`` (spec 5.1) -- a bullish zone sets that timeframe's state
  to bullish, a bearish zone to bearish.
- **iFVG**: an ``FVGEvent``'s ``direction`` at its ``known_at`` (spec
  5.3) -- already the *opposite* of its originating zone's direction, so
  it is applied exactly like a formation event once retrieved (a bullish
  zone's iFVG is bearish confluence, and vice versa).

Plain mitigation (a zone reaching ``MITIGATED`` *without* producing an
``FVGEvent``) is not a directional confluence at all and is correctly
invisible here, since it never appears in ``FVGDetectionResult.events``.

**Persistence** (spec 2.1): each timeframe's state holds from one
confluence's ``known_at`` until the next *opposite* confluence on that
*same* timeframe -- never decays, never expires, never resets on its
own. Before a timeframe's first confluence, its state is ``neutral``.

**Same-instant conflict**: if two opposite-direction confluences on the
same timeframe become known at the exact same ``known_at`` (e.g. one
zone's iFVG and a brand new zone's formation both resolve from the same
candle), picking either one arbitrarily would be an unaudited coin
flip. Instead the timeframe's state becomes ``neutral`` at that instant,
exactly as if no confluence had fired -- symmetric with two *same*
direction confluences at once, which simply confirm that direction, no
conflict.

**Global bias** (spec 2.1): ``bullish`` iff at least one HTF is bullish
and none is bearish; ``bearish`` iff at least one HTF is bearish and none
is bullish; ``neutral`` otherwise (all neutral, or a bullish/bearish
conflict across HTFs). No weighting and no precedence of 1D over 4H/1H --
a single bullish 1D zone and a single bearish 4H zone is global
``neutral``, not "1D wins".

No lookahead: every ``TimeframeBiasChange`` is available starting exactly
at its ``known_at`` (an FVGZone/FVGEvent's own ``known_at``, produced by
``tradelab.fvg`` and never recomputed here) and never a moment earlier;
``bias_at`` only ever looks backward from the queried instant.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from tradelab.fvg import BEARISH, BULLISH, FVGDetectionResult  # noqa: F401  (re-exported)

SUPPORTED_TIMEFRAMES = frozenset({"1H", "4H", "1D"})
NEUTRAL = "neutral"
_OPPOSITE = {BULLISH: BEARISH, BEARISH: BULLISH}


class ConfluenceType(str, Enum):
    FVG_FORMATION = "fvg_formation"
    IFVG = "ifvg"


@dataclass(frozen=True)
class ConfluenceSource:
    """One underlying FVG/iFVG fact that contributed to a bias change.
    ``direction`` is the confluence's own direction -- always bullish or
    bearish, never neutral (neutral is only ever a *combined* state, on
    ``TimeframeBiasChange.state`` or ``BiasSnapshot``)."""

    confluence_type: ConfluenceType
    direction: str
    source_id: str  # FVGZone.zone_id (formation) or FVGEvent.zone_id (iFVG -- the zone it inverted)
    known_at: datetime


@dataclass(frozen=True)
class TimeframeBiasChange:
    """A single timeframe's state from ``known_at`` onward, until the
    next change on that same timeframe. ``sources`` holds every
    confluence that became known at this exact ``known_at`` -- normally
    one, but two or more when several resolve from the same candle
    (agreeing, or conflicting -- see module docstring)."""

    timeframe: str
    known_at: datetime
    state: str  # "bullish" / "bearish" / "neutral"
    sources: tuple[ConfluenceSource, ...]


@dataclass(frozen=True)
class HTFBiasResult:
    """Always exactly ``SUPPORTED_TIMEFRAMES`` as keys, each mapped to
    its own chronological (by ``known_at``) list of changes -- empty if
    that timeframe was never passed in, or had no FVG/iFVG at all."""

    timeframe_changes: dict[str, list[TimeframeBiasChange]]


@dataclass(frozen=True)
class BiasSnapshot:
    timeframe_states: dict[str, str]
    global_bias: str


def compute_htf_bias(fvg_results: dict[str, FVGDetectionResult]) -> HTFBiasResult:
    """Build the per-timeframe bias change timelines from already-computed
    FVG detection results -- one per higher timeframe (``1H``/``4H``/
    ``1D``). A timeframe absent from ``fvg_results`` is treated exactly
    like one with zero FVGs: always ``neutral``.
    """
    for timeframe in fvg_results:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported HTF bias timeframe {timeframe!r}; expected one of {sorted(SUPPORTED_TIMEFRAMES)}")

    timeframe_changes = {
        timeframe: _timeframe_timeline(timeframe, fvg_results[timeframe]) if timeframe in fvg_results else []
        for timeframe in SUPPORTED_TIMEFRAMES
    }
    return HTFBiasResult(timeframe_changes=timeframe_changes)


def _timeframe_timeline(timeframe: str, result: FVGDetectionResult) -> list[TimeframeBiasChange]:
    zones_by_id = {zone.zone_id: zone for zone in result.zones}
    for zone in result.zones:
        if zone.timeframe != timeframe:
            raise ValueError(f"FVGZone {zone.zone_id!r} has timeframe {zone.timeframe!r}, expected {timeframe!r}")
    for event in result.events:
        if event.timeframe != timeframe:
            raise ValueError(f"FVGEvent for zone {event.zone_id!r} has timeframe {event.timeframe!r}, expected {timeframe!r}")
        zone = zones_by_id.get(event.zone_id)
        if zone is None:
            raise ValueError(f"FVGEvent references unknown zone_id {event.zone_id!r} on timeframe {timeframe!r}")
        if event.known_at <= zone.formed_at:
            raise ValueError(
                f"FVGEvent for zone {event.zone_id!r} has known_at {event.known_at!r} not strictly after its "
                f"zone's formed_at {zone.formed_at!r} -- an iFVG cannot be known before or at its own zone's formation"
            )
        if event.original_zone_direction != zone.direction or event.direction != _OPPOSITE[zone.direction]:
            raise ValueError(
                f"FVGEvent for zone {event.zone_id!r} is inconsistent with its zone's actual direction {zone.direction!r}"
            )

    sources = [
        ConfluenceSource(ConfluenceType.FVG_FORMATION, zone.direction, zone.zone_id, zone.formed_at) for zone in result.zones
    ] + [ConfluenceSource(ConfluenceType.IFVG, event.direction, event.zone_id, event.known_at) for event in result.events]
    sources.sort(key=lambda s: (s.known_at, s.confluence_type.value, s.source_id))

    changes: list[TimeframeBiasChange] = []
    index = 0
    while index < len(sources):
        known_at = sources[index].known_at
        group = [sources[index]]
        index += 1
        while index < len(sources) and sources[index].known_at == known_at:
            group.append(sources[index])
            index += 1

        directions = {source.direction for source in group}
        if directions == {BULLISH}:
            state = BULLISH
        elif directions == {BEARISH}:
            state = BEARISH
        else:
            state = NEUTRAL  # both directions became known at this exact instant -- no arbitrary tie-break

        changes.append(TimeframeBiasChange(timeframe=timeframe, known_at=known_at, state=state, sources=tuple(group)))

    return changes


def bias_at(result: HTFBiasResult, ts_utc: datetime) -> BiasSnapshot:
    """The last known state of every HTF, and the combined global bias,
    as of ``ts_utc`` -- never a change whose ``known_at`` is strictly
    after ``ts_utc``."""
    if ts_utc.tzinfo is None:
        raise ValueError(f"bias_at: ts_utc must be timezone-aware: {ts_utc!r}")
    ts_utc = ts_utc.astimezone(timezone.utc)

    timeframe_states: dict[str, str] = {}
    for timeframe in SUPPORTED_TIMEFRAMES:
        state = NEUTRAL
        for change in result.timeframe_changes[timeframe]:
            if change.known_at > ts_utc:
                break
            state = change.state
        timeframe_states[timeframe] = state

    directions = set(timeframe_states.values())
    if BULLISH in directions and BEARISH not in directions:
        global_bias = BULLISH
    elif BEARISH in directions and BULLISH not in directions:
        global_bias = BEARISH
    else:
        global_bias = NEUTRAL

    return BiasSnapshot(timeframe_states=timeframe_states, global_bias=global_bias)
