"""Canonical entry selector catalog: the exhaustive, deterministically
ordered set of every selector string ``tradelab.entry_signal_timeline``
(equivalently, ``tradelab.fvg_entry``/``tradelab.liquidity_entry``, which
it normalizes) can ever actually emit -- and the single source of truth
every other module validates a selector string against.

**Why this module exists.** Before it, ``tradelab.strategy_spec``
validated a selector by checking each ``-``-separated component against
its own valid set independently (timeframe is one of three, source_family
is one of two, liquidity_type is any real ``LiquidityType``, ...). That
missed *cross-field* impossibilities: ``liquidity-1m-session-swing_high-
take-reversal`` has a perfectly valid timeframe, a perfectly valid
``source_family`` value, and a perfectly valid ``liquidity_type`` value --
but ``swing_high`` is never a session type, so ``tradelab.liquidity_entry``
can never actually produce a candidate whose selector looks like that
(``session`` only ever pairs with a ``session_*`` type, see
``tradelab.liquidity_entry.SourceFamily``/``_candidates_for_source``'s
own cross-validation). A syntactically-valid-looking but semantically
impossible selector used to pass validation anyway. This module fixes
that by validating against **exact membership in the fully enumerated
catalog** instead of independently checking each component -- a
combination that was never actually produced by construction simply
isn't in the set, no separate cross-validation rule needed.

**The catalog, by construction** (spec, this task):

- **FVG**: 3 timeframes (``1m``/``5m``/``15m``, ``tradelab.fvg_entry.
  SUPPORTED_TIMEFRAMES``) x 2 source types (``fvg_touch``/``ifvg``,
  ``tradelab.fvg_entry.SourceType``) = **6** ``CANONICAL_FVG_SELECTORS``.
- **Liquidity**: 3 timeframes x 18 valid ``(source_family, liquidity_type)``
  pairs x 4 trigger phases x 2 entry modes = **432**
  ``CANONICAL_LIQUIDITY_SELECTORS``. The 18 pairs are never a free
  cross product of ``SourceFamily`` x ``LiquidityType`` -- exactly the
  bug this module fixes -- they are the same split
  ``tradelab.liquidity_entry`` itself enforces: ``swing_equal`` pairs
  only with ``swing_high``/``swing_low``/``equal_highs``/``equal_lows``
  (4 types); ``session`` pairs only with the 14 ``session_*`` types
  (every ``LiquidityType`` whose value starts with ``"session_"``,
  derived from the enum itself rather than hand-listed, so this catalog
  can never drift from ``tradelab.liquidity.LiquidityType`` if it ever
  gains or loses a session type).
- **Total**: 6 + 432 = **438** ``ALL_CANONICAL_ENTRY_SELECTORS``.

**These are selector *families*, not strategies, and not timeframe-
locked strategies.** 438 is the count of distinct signal categories this
package's detectors can ever emit -- a future strategy enumerator is
free to combine selectors *across* timeframes (a 1m primary entry with
5m and 15m required confluence, say) exactly as ``tradelab.strategy_
spec.StrategySpec`` already allows; nothing here restricts a strategy to
selectors from a single timeframe, and 438 is never a claim about how
many strategies, or how many single-timeframe strategies, could exist.

**Direction table reused, not duplicated.** A liquidity selector's fixed
direction comes from ``tradelab.liquidity_entry``'s own ``_DIRECTION``
table (side + entry_mode -> direction) -- imported directly rather than
re-declared here, the same cross-module private-helper reuse this
package already uses elsewhere (e.g. ``tradelab.fvg_touch`` importing
``tradelab.liquidity._validate_chronology``), so there is exactly one
place this table can ever drift from the module that actually builds
real candidates with it.

``tradelab.strategy_spec`` no longer defines its own selector-format
parsing or direction table at all -- it imports ``selector_timeframe``/
``selector_fixed_direction`` from here directly, making this module the
single source of truth requirement 3 of this task asks for.
"""
from __future__ import annotations

from tradelab.fvg_entry import SourceType as FVGSourceType
from tradelab.fvg_entry import SUPPORTED_TIMEFRAMES as _FVG_ENTRY_TIMEFRAMES
from tradelab.liquidity import HIGH, LOW, LiquidityType
from tradelab.liquidity_entry import CandidatePhase, EntryMode, SourceFamily
from tradelab.liquidity_entry import SUPPORTED_TIMEFRAMES as _LIQUIDITY_ENTRY_TIMEFRAMES
from tradelab.liquidity_entry import _DIRECTION as _LIQUIDITY_DIRECTION

# Canonical, human-readable timeframe order (ascending duration) -- the
# same three timeframes as _FVG_ENTRY_TIMEFRAMES/_LIQUIDITY_ENTRY_TIMEFRAMES
# (both frozenset({"1m", "5m", "15m"})), just deterministically ordered
# rather than relying on a set's/lexicographic sort's incidental order.
_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m")
if set(_TIMEFRAMES) != set(_FVG_ENTRY_TIMEFRAMES) or set(_TIMEFRAMES) != set(_LIQUIDITY_ENTRY_TIMEFRAMES):
    raise AssertionError(  # pragma: no cover -- would only fire if fvg_entry/liquidity_entry's own supported timeframes ever changed
        f"_TIMEFRAMES {_TIMEFRAMES!r} no longer matches fvg_entry/liquidity_entry's own SUPPORTED_TIMEFRAMES "
        f"({sorted(_FVG_ENTRY_TIMEFRAMES)!r} / {sorted(_LIQUIDITY_ENTRY_TIMEFRAMES)!r}) -- update this catalog"
    )

# The 4 swing/equal liquidity types and the 14 session_* liquidity types --
# both derived from LiquidityType itself (never hand-listed), split the
# same way tradelab.liquidity_entry.SourceFamily/_candidates_for_source
# already enforces: a session type is exactly "value starts with
# 'session_'"; everything else is swing/equal.
_SWING_EQUAL_TYPES: tuple[LiquidityType, ...] = tuple(lt for lt in LiquidityType if not lt.value.startswith("session_"))
_SESSION_TYPES: tuple[LiquidityType, ...] = tuple(lt for lt in LiquidityType if lt.value.startswith("session_"))

# The 18 valid (source_family, liquidity_type) pairs -- never a free
# cross product of SourceFamily x LiquidityType.
_FAMILY_TYPE_PAIRS: tuple[tuple[SourceFamily, LiquidityType], ...] = tuple(
    (SourceFamily.SWING_EQUAL, lt) for lt in _SWING_EQUAL_TYPES
) + tuple((SourceFamily.SESSION, lt) for lt in _SESSION_TYPES)


# Explicit per-member side, matching exactly what tradelab.liquidity/
# tradelab.session_liquidity actually assign at construction time --
# deliberately *not* derived by string-suffix matching (e.g.
# ``.value.endswith("_high")``): LiquidityType.EQUAL_HIGHS is
# "equal_highs" (plural) and LiquidityType.EQUAL_LOWS is "equal_lows",
# neither of which ends in the singular "_high"/"_low", so a naive
# suffix check silently mis-derives both. An explicit mapping has no
# such edge case to get wrong.
_TYPE_SIDE: dict[LiquidityType, str] = {
    LiquidityType.SWING_HIGH: HIGH,
    LiquidityType.SWING_LOW: LOW,
    LiquidityType.EQUAL_HIGHS: HIGH,
    LiquidityType.EQUAL_LOWS: LOW,
    LiquidityType.SESSION_ASIA_HIGH: HIGH,
    LiquidityType.SESSION_ASIA_LOW: LOW,
    LiquidityType.SESSION_LONDON_HIGH: HIGH,
    LiquidityType.SESSION_LONDON_LOW: LOW,
    LiquidityType.SESSION_PRE_NY_HIGH: HIGH,
    LiquidityType.SESSION_PRE_NY_LOW: LOW,
    LiquidityType.SESSION_NY_AM_HIGH: HIGH,
    LiquidityType.SESSION_NY_AM_LOW: LOW,
    LiquidityType.SESSION_NY_LUNCH_HIGH: HIGH,
    LiquidityType.SESSION_NY_LUNCH_LOW: LOW,
    LiquidityType.SESSION_NY_PM_HIGH: HIGH,
    LiquidityType.SESSION_NY_PM_LOW: LOW,
    LiquidityType.SESSION_OVERNIGHT_HIGH: HIGH,
    LiquidityType.SESSION_OVERNIGHT_LOW: LOW,
}


def _side_of(liquidity_type: LiquidityType) -> str:
    try:
        return _TYPE_SIDE[liquidity_type]
    except KeyError:  # pragma: no cover -- would only fire if LiquidityType gained a member this catalog doesn't know about
        raise AssertionError(f"no known side for {liquidity_type!r} -- update _TYPE_SIDE in tradelab.selector_catalog") from None


def _fvg_selector(timeframe: str, source_type: FVGSourceType) -> str:
    return f"fvg-{timeframe}-{source_type.value}"


def _liquidity_selector(
    timeframe: str, source_family: SourceFamily, liquidity_type: LiquidityType, trigger_phase: CandidatePhase, entry_mode: EntryMode
) -> str:
    return f"liquidity-{timeframe}-{source_family.value}-{liquidity_type.value}-{trigger_phase.value}-{entry_mode.value}"


CANONICAL_FVG_SELECTORS: tuple[str, ...] = tuple(
    _fvg_selector(timeframe, source_type) for timeframe in _TIMEFRAMES for source_type in FVGSourceType
)

CANONICAL_LIQUIDITY_SELECTORS: tuple[str, ...] = tuple(
    _liquidity_selector(timeframe, source_family, liquidity_type, trigger_phase, entry_mode)
    for timeframe in _TIMEFRAMES
    for source_family, liquidity_type in _FAMILY_TYPE_PAIRS
    for trigger_phase in CandidatePhase
    for entry_mode in EntryMode
)

ALL_CANONICAL_ENTRY_SELECTORS: tuple[str, ...] = CANONICAL_FVG_SELECTORS + CANONICAL_LIQUIDITY_SELECTORS

_CANONICAL_SELECTOR_SET = frozenset(ALL_CANONICAL_ENTRY_SELECTORS)

# Precomputed once, alongside the selector strings themselves, rather
# than re-parsed on every selector_fixed_direction() call.
_LIQUIDITY_SELECTOR_DIRECTION: dict[str, str] = {
    _liquidity_selector(timeframe, source_family, liquidity_type, trigger_phase, entry_mode): _LIQUIDITY_DIRECTION[
        (_side_of(liquidity_type), entry_mode)
    ]
    for timeframe in _TIMEFRAMES
    for source_family, liquidity_type in _FAMILY_TYPE_PAIRS
    for trigger_phase in CandidatePhase
    for entry_mode in EntryMode
}


def is_canonical_selector(selector: str) -> bool:
    """True iff ``selector`` is exactly one of the 438 entries in
    ``ALL_CANONICAL_ENTRY_SELECTORS`` -- a real selector string
    ``tradelab.entry_signal_timeline`` could actually produce, never a
    merely syntactically-plausible one."""
    return isinstance(selector, str) and selector in _CANONICAL_SELECTOR_SET


def _require_canonical(selector: str) -> None:
    if not is_canonical_selector(selector):
        raise ValueError(
            f"{selector!r} is not a canonical entry selector -- it must be exactly one of the "
            f"{len(ALL_CANONICAL_ENTRY_SELECTORS)} entries tradelab.entry_signal_timeline can actually "
            "produce (see tradelab.selector_catalog.ALL_CANONICAL_ENTRY_SELECTORS)"
        )


def selector_timeframe(selector: str) -> str:
    """The timeframe encoded in a canonical ``fvg-*``/``liquidity-*``
    selector; raises ``ValueError`` if ``selector`` is not canonical."""
    _require_canonical(selector)
    return selector.split("-")[1]


def selector_fixed_direction(selector: str) -> str | None:
    """The single direction a canonical selector is pinned to, or
    ``None`` if it is direction-agnostic. Every ``fvg-*`` selector
    (``fvg_touch``/``ifvg``) spans both directions -- direction lives on
    the individual ``EntrySignal``, not the selector. Every
    ``liquidity-*`` selector is pinned to exactly one direction by its
    ``entry_mode``/implied side (``tradelab.liquidity_entry._DIRECTION``).
    Raises ``ValueError`` if ``selector`` is not canonical."""
    _require_canonical(selector)
    if selector in CANONICAL_FVG_SELECTORS:
        return None
    return _LIQUIDITY_SELECTOR_DIRECTION[selector]
