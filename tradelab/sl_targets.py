"""SL target candidates: an audited, ordered stop-loss candidate list for
a given market entry, sourced from liquidity levels and the exact FVG
zones a strategy actually used as its own entry confluence.

Not exit simulation, not position management, not the news filter, not
bias, not the strategy enumerator -- those are separate, later steps.
This module only answers "given this one entry (and, if any, the FVG
confluence it was built from), what price levels could serve as SL?"

Reuses ``tradelab.tp_targets.MarketEntryContext`` directly (no
conflicting redefinition -- entry_id/direction/entry_price/
entry_known_at mean exactly the same thing for SL as for TP) and
``tradelab.tp_targets.LiquiditySourceFamily``/``_liquidity_source_family``
for the identical swing/equal/session categorization. Reuses
``tradelab.liquidity.detect_liquidity`` and
``tradelab.session_liquidity.detect_session_liquidity`` output directly
(the same two-dict convention ``tradelab.tp_targets``/
``tradelab.liquidity_entry`` already established) -- nothing is
re-detected, and neither upstream module's semantics are touched. Data
wicks are never an input here: they are a TP-only source type
(``tradelab.tp_targets``), never SL.

**Direction determines which side of price qualifies as SL -- the risk
side, opposite of TP** (spec):

===========  ===========================  =================================
Direction    Liquidity SL                  FVG boundary SL
===========  ===========================  =================================
long         liquidity LOW below           lower bound of a *linked* FVG
             entry_price                   zone, 5m or 15m
short        liquidity HIGH above           upper bound of a *linked* FVG
             entry_price                   zone, 5m or 15m
===========  ===========================  =================================

A target strictly on the wrong side of ``entry_price``, or exactly at
it, is never a candidate.

**FVG SL only from the strategy's own linked confluence.** Unlike
liquidity (scanned from the full detection universe), an FVG zone's
boundary is *never* pulled from an arbitrary, unrelated, possibly stale
FVG -- only from the exact ``FVGEntryCandidate`` object(s) the caller
explicitly says this entry actually used as its own ``FVG_TOUCH`` or
``IFVG`` confluence (``SelectedFVGConfluence``, one per entry, holding
those candidates directly -- a zone's own ``lower``/``upper`` are
already carried on that candidate, so no separate ``FVGDetectionResult``
lookup is needed or accepted here). Only ``5m``/``15m`` FVG candidates
qualify for SL; ``1m``/``1H``/``4H``/``1D`` are rejected outright
(``ValueError``) -- unlike liquidity SL, which has no timeframe
restriction (an HTF level is still a valid stop; an HTF *FVG* boundary
here is specifically excluded by this spec). ``lower`` is used for a
long SL and ``upper`` for a short one regardless of the zone's own
original bullish/bearish label or whether it was linked as ``FVG_TOUCH``
or ``IFVG``: an ``IFVG`` candidate's ``lower``/``upper`` describe the
same physical zone geometry either way, and picking the boundary on the
entry's own risk side is what makes it a sensible stop in both cases.

**Timing** (no lookahead), identical in spirit to
``tradelab.tp_targets``: a liquidity level must be known (``formed_at``)
no later than ``entry_known_at``, and if it was already ``TAKEN``, the
take must have happened strictly after ``entry_known_at`` -- a level
taken before entry is stale, already-consumed liquidity, never a stop; a
level taken after entry is a legitimate candidate (that a later
simulation may or may not actually hit first). A level
``ABSORBED_IN_EQUAL_CLUSTER`` is never its own SL candidate. A linked FVG
candidate must likewise have its own ``known_at`` no later than
``entry_known_at`` -- both kinds are silently excluded when this fails,
the same "known no later than entry" filtering convention
``tradelab.tp_targets`` already uses, not a hard error, since the
liquidity side of this same call routinely has to filter out many
not-yet-known levels as an entirely normal case.

This module does not trust its input came from a real detector call: it
re-validates timeframe/source-family consistency and internal
consistency for liquidity (mirroring ``tradelab.tp_targets`` exactly),
positive prices, timezone-aware timestamps, and
level_id/zone_id/sl_target_id uniqueness -- including that the
``SelectedFVGConfluence`` actually belongs to this same entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tradelab.fvg_entry import FVGEntryCandidate
from tradelab.fvg_entry import SourceType as FVGSourceType
from tradelab.fvg import FVGZone
from tradelab.liquidity import HIGH, LOW, LiquidityDetectionResult, LiquidityStatus, LiquidityType, liquidity_levels_at
from tradelab.liquidity import SUPPORTED_TIMEFRAMES as _SWING_EQUAL_TIMEFRAMES
from tradelab.session_liquidity import SUPPORTED_TIMEFRAMES as _SESSION_TIMEFRAMES
from tradelab.tp_targets import LONG, SHORT  # noqa: F401  (re-exported)
from tradelab.tp_targets import LiquiditySourceFamily  # noqa: F401  (re-exported)
from tradelab.tp_targets import MarketEntryContext  # noqa: F401  (re-exported)
from tradelab.tp_targets import _liquidity_source_family
from tradelab.liquidity_target_index import LiquidityTargetIndex

MAX_LIQUIDITY_SL_TARGETS = 3
FVG_SL_TIMEFRAMES = frozenset({"5m", "15m"})


class SLSourceType(str, Enum):
    LIQUIDITY = "liquidity"
    FVG_BOUNDARY = "fvg_boundary"


@dataclass(frozen=True)
class SelectedFVGConfluence:
    """The exact FVG entry confluence a strategy used for one entry --
    the only source FVG SL boundaries may ever be drawn from (spec: never
    an arbitrary, unrelated FVG zone)."""

    entry_id: str
    fvg_candidates: tuple[FVGEntryCandidate | FVGZone, ...]

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValueError("SelectedFVGConfluence.entry_id must be non-empty")
        zone_ids = [c.zone_id for c in self.fvg_candidates]
        if len(zone_ids) != len(set(zone_ids)):
            dupes = sorted({z for z in zone_ids if zone_ids.count(z) > 1})
            raise ValueError(f"Duplicate zone_id(s) in SelectedFVGConfluence: {dupes}")


@dataclass(frozen=True)
class SLTargetCandidate:
    """One fully-audited SL candidate for one entry.

    Fields specific to ``LIQUIDITY`` (``liquidity_type``, ``level_id``,
    ``source_family``, ``rank``) are ``None`` for ``FVG_BOUNDARY``
    candidates, and vice versa (``zone_id``, ``fvg_source_type``,
    ``fvg_candidate_id``, ``fvg_timeframe``).

    ``sl_target_id`` is stable and deterministic: derived only from
    ``(entry_id, source_type, source_id)``, never from list position.
    """

    sl_target_id: str
    entry_id: str
    source_type: SLSourceType
    source_id: str  # level_id (liquidity) or zone_id (fvg_boundary)
    direction: str
    price: float
    distance: float
    known_at: datetime
    rank: int | None  # 1-based, nearest-first, among the (<=3) liquidity targets; None for fvg_boundary
    lower: float
    upper: float

    # LIQUIDITY-only audit:
    liquidity_type: LiquidityType | None
    level_id: str | None
    source_family: LiquiditySourceFamily | None

    # FVG_BOUNDARY-only audit:
    zone_id: str | None
    fvg_source_type: FVGSourceType | None
    fvg_candidate_id: str | None
    fvg_timeframe: str | None


@dataclass(frozen=True)
class SLTargetCandidateResult:
    entry_id: str
    liquidity_targets: list[SLTargetCandidate]
    fvg_targets: list[SLTargetCandidate]


def detect_sl_target_candidates(
    entry: MarketEntryContext,
    selected_fvg: SelectedFVGConfluence,
    swing_equal_results: dict[str, LiquidityDetectionResult],
    session_results: dict[str, LiquidityDetectionResult],
    *,
    liquidity_index: LiquidityTargetIndex | None = None,
    max_fvg_targets: int | None = None,
) -> SLTargetCandidateResult:
    """Build the ordered SL target candidates for ``entry``.

    ``selected_fvg`` must belong to the same ``entry`` (its ``entry_id``
    must match ``entry.entry_id``); it may hold zero candidates if this
    entry had no FVG confluence at all. ``swing_equal_results``/
    ``session_results`` follow the same convention as
    ``tradelab.tp_targets``/``tradelab.liquidity_entry``.
    """
    if selected_fvg.entry_id != entry.entry_id:
        raise ValueError(
            f"SelectedFVGConfluence.entry_id {selected_fvg.entry_id!r} does not match entry.entry_id {entry.entry_id!r}"
        )

    if max_fvg_targets is not None and max_fvg_targets < 1:
        raise ValueError("max_fvg_targets must be positive or None")

    liquidity_candidates = _liquidity_sl_candidates(entry, swing_equal_results, session_results, liquidity_index=liquidity_index)
    fvg_candidates = _fvg_sl_candidates(entry, selected_fvg, max_targets=max_fvg_targets)

    liquidity_candidates.sort(key=lambda c: (c.distance, c.known_at, c.sl_target_id))
    liquidity_candidates = liquidity_candidates[:MAX_LIQUIDITY_SL_TARGETS]
    liquidity_candidates = [
        SLTargetCandidate(
            sl_target_id=c.sl_target_id,
            entry_id=c.entry_id,
            source_type=c.source_type,
            source_id=c.source_id,
            direction=c.direction,
            price=c.price,
            distance=c.distance,
            known_at=c.known_at,
            rank=rank,
            lower=c.lower,
            upper=c.upper,
            liquidity_type=c.liquidity_type,
            level_id=c.level_id,
            source_family=c.source_family,
            zone_id=c.zone_id,
            fvg_source_type=c.fvg_source_type,
            fvg_candidate_id=c.fvg_candidate_id,
            fvg_timeframe=c.fvg_timeframe,
        )
        for rank, c in enumerate(liquidity_candidates, start=1)
    ]

    fvg_candidates.sort(key=lambda c: (c.distance, c.known_at, c.sl_target_id))

    all_ids = [c.sl_target_id for c in liquidity_candidates] + [c.sl_target_id for c in fvg_candidates]
    if len(all_ids) != len(set(all_ids)):
        dupes = sorted({tid for tid in all_ids if all_ids.count(tid) > 1})
        raise ValueError(f"Duplicate sl_target_id(s) produced: {dupes}")

    return SLTargetCandidateResult(entry_id=entry.entry_id, liquidity_targets=liquidity_candidates, fvg_targets=fvg_candidates)


def _liquidity_sl_candidates(
    entry: MarketEntryContext,
    swing_equal_results: dict[str, LiquidityDetectionResult],
    session_results: dict[str, LiquidityDetectionResult],
    *,
    liquidity_index: LiquidityTargetIndex | None = None,
) -> list[SLTargetCandidate]:
    if liquidity_index is not None:
        wanted_side = LOW if entry.direction == LONG else HIGH
        levels = liquidity_index.nearest(side=wanted_side, entry_price=entry.entry_price, entry_known_at=entry.entry_known_at)
        families = {LiquiditySourceFamily.SWING, LiquiditySourceFamily.EQUAL, LiquiditySourceFamily.SESSION}
        return [candidate for level in levels for candidate in _candidates_from_levels(
            entry, level.timeframe, LiquidityDetectionResult(levels=[level], takes=[], reclaims=[]), allowed_families=families
        )]
    candidates: list[SLTargetCandidate] = []
    for timeframe, result in swing_equal_results.items():
        if timeframe not in _SWING_EQUAL_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe {timeframe!r} in swing_equal_results; expected one of {sorted(_SWING_EQUAL_TIMEFRAMES)}")
        candidates.extend(
            _candidates_from_levels(entry, timeframe, result, allowed_families={LiquiditySourceFamily.SWING, LiquiditySourceFamily.EQUAL})
        )
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
) -> list[SLTargetCandidate]:
    # Opposite side from TP: SL sits on the entry's own risk side.
    wanted_side = LOW if entry.direction == LONG else HIGH

    seen_level_ids: set[str] = set()
    out: list[SLTargetCandidate] = []
    for level in liquidity_levels_at(result, entry.entry_known_at):
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
            continue  # wrong side of price for this entry's risk side
        if level.status == LiquidityStatus.ABSORBED_IN_EQUAL_CLUSTER:
            continue  # not independently a target -- only the cluster it joined is
        if level.formed_at > entry.entry_known_at:
            continue  # not yet known at entry time
        if level.status == LiquidityStatus.TAKEN and level.taken_at <= entry.entry_known_at:
            continue  # already swept before entry -- stale, not a forward-looking SL

        price = level.upper if wanted_side == HIGH else level.lower
        if price <= 0:
            raise ValueError(f"LiquidityLevel {level.level_id!r} has non-positive price {price!r}")
        if entry.direction == LONG and price >= entry.entry_price:
            continue  # wrong side of entry_price, or exactly at it -- SL must be below for a long
        if entry.direction == SHORT and price <= entry.entry_price:
            continue  # SL must be above for a short

        out.append(
            SLTargetCandidate(
                sl_target_id=f"sl-liquidity-{entry.entry_id}-{level.level_id}",
                entry_id=entry.entry_id,
                source_type=SLSourceType.LIQUIDITY,
                source_id=level.level_id,
                direction=entry.direction,
                price=price,
                distance=abs(price - entry.entry_price),
                known_at=level.formed_at,
                rank=None,
                lower=level.lower,
                upper=level.upper,
                liquidity_type=level.liquidity_type,
                level_id=level.level_id,
                source_family=family,
                zone_id=None,
                fvg_source_type=None,
                fvg_candidate_id=None,
                fvg_timeframe=None,
            )
        )
    return out


def _fvg_sl_candidates(
    entry: MarketEntryContext,
    selected_fvg: SelectedFVGConfluence,
    *,
    max_targets: int | None = None,
) -> list[SLTargetCandidate]:
    """Return valid FVG-boundary stops, optionally retaining only nearest.

    Production always selects ``fvg_targets[0]``.  A dense 1m entry can be
    linked to thousands of active zones, so materialising every candidate only
    to retain index zero in a cache is a severe allocation bug.  With
    ``max_targets=1`` this maintains the exact same deterministic minimum as
    the historical ``sort(...)[0]`` path while retaining one object.
    """
    if max_targets is not None and max_targets < 1:
        raise ValueError("max_targets must be positive or None")
    out: list[SLTargetCandidate] = []
    for candidate in selected_fvg.fvg_candidates:
        if isinstance(candidate, FVGEntryCandidate):
            candidate_id = candidate.candidate_id
            candidate_known_at = candidate.known_at
            candidate_source_type = candidate.source_type
        else:
            candidate_id = f"zone-{candidate.zone_id}"
            candidate_known_at = candidate.formed_at
            candidate_source_type = FVGSourceType.FVG_TOUCH
        if candidate.timeframe not in FVG_SL_TIMEFRAMES:
            raise ValueError(
                f"FVG confluence {candidate_id!r} has timeframe {candidate.timeframe!r}; only "
                f"{sorted(FVG_SL_TIMEFRAMES)} qualify for FVG SL"
            )
        if candidate.lower >= candidate.upper:
            raise ValueError(
                f"FVG confluence {candidate_id!r} has non-increasing bounds "
                f"(lower={candidate.lower!r}, upper={candidate.upper!r})"
            )
        if candidate_known_at > entry.entry_known_at:
            continue  # not yet known at entry time

        price = candidate.lower if entry.direction == LONG else candidate.upper
        if price <= 0:
            raise ValueError(f"FVG confluence {candidate_id!r} has non-positive price {price!r}")
        if entry.direction == LONG and price >= entry.entry_price:
            continue
        if entry.direction == SHORT and price <= entry.entry_price:
            continue

        resolved = SLTargetCandidate(
                sl_target_id=f"sl-fvg_boundary-{entry.entry_id}-{candidate.zone_id}",
                entry_id=entry.entry_id,
                source_type=SLSourceType.FVG_BOUNDARY,
                source_id=candidate.zone_id,
                direction=entry.direction,
                price=price,
                distance=abs(price - entry.entry_price),
                known_at=candidate_known_at,
                rank=None,
                lower=candidate.lower,
                upper=candidate.upper,
                liquidity_type=None,
                level_id=None,
                source_family=None,
                zone_id=candidate.zone_id,
                fvg_source_type=candidate_source_type,
                fvg_candidate_id=candidate_id,
                fvg_timeframe=candidate.timeframe,
            )
        if max_targets == 1:
            if not out or (resolved.distance, resolved.known_at, resolved.sl_target_id) < (
                out[0].distance, out[0].known_at, out[0].sl_target_id
            ):
                out[:] = [resolved]
        else:
            out.append(resolved)
    return out
