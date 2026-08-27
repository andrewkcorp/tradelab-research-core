"""Fixed R-multiple take-profit targets.

A third TP family alongside liquidity ranks and data wicks, and the only one
with no market structure behind it: the target is pure arithmetic on the
trade's own entry and stop.

**R is measured from the prices the trade actually got**, never from a
planned distance and never from a dollar amount:

    R          = abs(entry_price - sl_price)     # both already tick-aligned
    long  TP   = entry_price + multiple * R
    short TP   = entry_price - multiple * R

``entry_price`` is the fill bar's open and ``sl_price`` is whatever the
variant's own SL rule resolved -- a liquidity level or an FVG boundary. Both
are already on the MNQ tick grid (bar prices are validated against it by
``tradelab.validation.validate_price_grid``, and every SL source is a price
read from those bars), so ``R`` is a whole number of ticks and the product is
too for the integral multiples this module ships. The final TP is snapped
anyway -- see below -- because relying on that invariant silently would make
a future fractional multiple produce off-grid targets with no warning.

**The $20 per-trade cost is not part of the geometry.** It is an accounting
fact applied in ``tradelab.accounting`` after the trade resolves, identical
for every TP family. Folding it into R would mean a "1R" target that is not
1R, and would make two variants with the same nominal multiple incomparable
across instruments. It stays out.

**Snapping direction is away from entry, not to-nearest.** When a multiple
does land off-grid, rounding toward entry would produce a target that is
reachable at *less* than the nominal R -- a silently optimistic result. This
module rounds away from entry instead, so the achieved R is always >= the
nominal R and never less. For the shipped 1.0 and 2.0 multiples on
tick-aligned inputs, the snap is a no-op and this rule never fires; it exists
so that it cannot fire *incorrectly* later.

**Everything else is unchanged.** Entry selection, SL selection, sizing, the
news blackout, the hard session-close policy, the same-bar SL-first
tie-break, the Apex audit and P&L accounting are exactly what every other
exit variant gets. Only the TP price differs.
"""
from __future__ import annotations

import math

from tradelab.tp_targets import MarketEntryContext, TPSourceType, TPTargetCandidate

LONG = "long"
SHORT = "short"

#: The shipped multiples, in enumeration order. Adding one changes the
#: variant universe and therefore requires a new run -- old results are not
#: comparable, because the set of variants itself differs.
FIXED_RR_MULTIPLES: tuple[float, ...] = (1.0, 2.0)


def fixed_rr_variant_label(multiple: float) -> str:
    """``fixed_rr_1`` / ``fixed_rr_2`` -- the id fragment used in
    ``ExitVariantSpec.variant_id`` and stored in the journal."""
    if multiple <= 0 or not math.isfinite(multiple):
        raise ValueError(f"fixed RR multiple must be finite and positive, got {multiple!r}")
    if float(multiple).is_integer():
        return f"fixed_rr_{int(multiple)}"
    return f"fixed_rr_{multiple!r}".replace(".", "p")


def _snap_away_from_entry(price: float, entry_price: float, tick_size: float) -> float:
    """``price`` moved onto the tick grid, never closer to ``entry_price``."""
    ticks = price / tick_size
    rounded = math.ceil(ticks) if price > entry_price else math.floor(ticks)
    # A value already on the grid must stay exactly where it is; ceil/floor of
    # a float that is one ulp off would otherwise move it a whole tick.
    if math.isclose(ticks, round(ticks), rel_tol=0.0, abs_tol=1e-8):
        rounded = round(ticks)
    return rounded * tick_size


def build_fixed_rr_tp_candidate(
    entry: MarketEntryContext,
    *,
    sl_price: float,
    multiple: float,
    tick_size: float,
    source_id: str,
) -> TPTargetCandidate | None:
    """The fixed-R TP for one entry, or ``None`` when it cannot exist.

    ``None`` is returned only when the stop distance is zero -- a degenerate
    trade the execution layer already refuses as ``NON_POSITIVE_RISK`` -- or
    when the resulting target would be at or through zero. It is never
    returned for lack of market structure, which is the entire point of this
    family: given a stop, the target always exists.

    ``source_id`` is the SL candidate's own ``source_id``, so the audit trail
    records which stop this target was measured against, not merely that a
    multiple was applied.
    """
    if tick_size <= 0 or not math.isfinite(tick_size):
        raise ValueError(f"tick_size must be finite and positive, got {tick_size!r}")
    if multiple <= 0 or not math.isfinite(multiple):
        raise ValueError(f"fixed RR multiple must be finite and positive, got {multiple!r}")
    if entry.direction not in (LONG, SHORT):
        raise ValueError(f"entry.direction must be 'long' or 'short', got {entry.direction!r}")

    entry_price = entry.entry_price
    risk = abs(entry_price - sl_price)
    if risk <= 0:
        return None
    # The stop must be on the side the direction implies. A long whose stop is
    # above its entry is not a wide stop, it is a broken one, and mirroring the
    # arithmetic would silently invent a target on the wrong side of price.
    if entry.direction == LONG and sl_price >= entry_price:
        return None
    if entry.direction == SHORT and sl_price <= entry_price:
        return None

    raw = entry_price + multiple * risk if entry.direction == LONG else entry_price - multiple * risk
    price = _snap_away_from_entry(raw, entry_price, tick_size)
    if price <= 0:
        return None
    distance = abs(price - entry_price)
    if distance <= 0:
        return None

    label = fixed_rr_variant_label(multiple)
    return TPTargetCandidate(
        target_id=f"{entry.entry_id}:{TPSourceType.FIXED_RR.value}:{label}",
        entry_id=entry.entry_id,
        source_type=TPSourceType.FIXED_RR,
        source_id=source_id,
        direction=entry.direction,
        price=price,
        distance=distance,
        # There is no external source to be "known" at some earlier instant:
        # the target is a function of this trade's own entry and stop, both of
        # which are facts at exactly this moment. Using the entry instant is
        # the literal truth here, not a placeholder -- and it makes any
        # lookahead check applied to it trivially pass, correctly.
        known_at=entry.entry_known_at,
        rank=None,
        liquidity_type=None,
        level_id=None,
        lower=None,
        upper=None,
        source_family=None,
        event_ids=None,
        wick_direction=None,
        timestamp=None,
        rr_multiple=float(multiple),
        rr_sl_price=sl_price,
    )
