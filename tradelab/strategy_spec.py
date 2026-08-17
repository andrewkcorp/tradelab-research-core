"""Strategy spec: a declarative, immutable description of one strategy
variant, plus a lazy, deterministic generator over a caller-supplied
parameter grid.

Not a trade simulator, not P&L, not position management, not market
data, not order execution -- those are separate, later steps. This
module only answers two questions: "what IS a strategy" (exactly one
primary entry selector, zero or more required additional confluence
selectors, a bias policy, exactly one TP choice and one SL choice), and
"how do I enumerate every combination of a parameter space without ever
materializing it as a list".

**Reuses the exact, canonical selector strings ``tradelab.entry_signal_
timeline`` already builds and queries** (``fvg-{timeframe}-{source_type}``
/ ``liquidity-{timeframe}-{source_family}-{liquidity_type}-{trigger_
phase}-{entry_mode}``) -- this module does not invent a second signal-
activity semantics. At simulation time a future module simply calls
``query_active_signals(timeline, {primary_entry_selector, *required_
confluence_selectors}, direction, ts_utc)`` with these same plain
strings. Selector-scoped invalidation (an unselected variant never
invalidates the strategy; a later, opposite-direction signal of the
*same* selector can) is entirely ``tradelab.entry_signal_timeline``'s
existing rule -- this module adds no new invalidation mechanism, it only
chooses which selector strings a strategy asks that mechanism about.

**Data wicks are never usable as SL, never an entry condition, never
bias.** ``tradelab.sl_targets.SLSourceType`` has no ``DATA_WICK`` member
at all, so an ``SLChoice`` can never reference one (``SLChoice.__post_
init__`` rejects any ``source_type`` that is not an ``SLSourceType``
instance). Bias is exclusively ``tradelab.bias``'s FVG/iFVG 1H/4H/1D
computation -- ``BiasPolicy`` here is a purely declarative filter
(``IGNORE``/``REQUIRE_ALIGNED``/``REQUIRE_NEUTRAL``) over that module's
existing ``global_bias`` values, never a second bias computation.

**This module does not compute TP/SL prices, entry fills, or anything
data-driven** -- ``TPChoice``/``SLChoice`` only *declare* which
``tp_targets``/``sl_targets`` candidate kind and rank a future backtest
step will pull (``LIQUIDITY`` ranks 1..3, nearest-first, matching
``TPTargetCandidate.rank``/``SLTargetCandidate.rank`` exactly;
``DATA_WICK`` TP and ``FVG_BOUNDARY`` SL are never ranked upstream
either, so their ``rank`` must be ``None`` here too). An ``FVG_BOUNDARY``
SL additionally names, via ``fvg_link_selector``, which of *this
strategy's own* fvg-* selectors will supply the linked confluence at
simulation time (``tradelab.sl_targets.SelectedFVGConfluence``) --
never an arbitrary, unrelated FVG zone: ``StrategySpec.__post_init__``
requires that selector to be the strategy's own ``primary_entry_
selector`` or one of its own ``required_confluence_selectors``, and
``SLChoice.__post_init__`` requires its timeframe to be ``5m``/``15m``
(``tradelab.sl_targets.FVG_SL_TIMEFRAMES``), exactly mirroring that
module's own restriction.

**Direction consistency.** A liquidity selector's ``entry_mode`` plus
the ``high``/``low`` side implied by its ``liquidity_type`` pin it to
exactly one fixed direction (the same table ``tradelab.liquidity_entry``
uses to build the candidate in the first place); an fvg-* selector is
direction-agnostic at the spec level (a single selector spans both a
bullish and a bearish zone -- direction lives on the individual
``EntrySignal``, not the selector). A ``long`` strategy therefore cannot
declare a ``primary_entry_selector`` pinned to ``short``, and vice versa;
an fvg-* primary entry selector is compatible with either strategy
direction.

**The generator never filters by historical results and never
simulates a trade** -- ``iter_strategy_specs`` only skips combinations
that are structurally invalid per the rules above (e.g. the primary
selector reappearing in the confluence set, or a direction/selector
mismatch); every combination it does *not* skip is a fully valid
``StrategySpec``, deterministically ordered and never materialized as a
list (it is a generator function -- calling it returns a generator
object without running any of its body).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
from typing import Iterable, Iterator

from tradelab.fvg_entry import SourceType as FVGSourceType
from tradelab.fvg_entry import SUPPORTED_TIMEFRAMES as _FVG_ENTRY_TIMEFRAMES
from tradelab.liquidity import LiquidityType
from tradelab.liquidity_entry import CandidatePhase, EntryMode, SourceFamily
from tradelab.liquidity_entry import SUPPORTED_TIMEFRAMES as _LIQUIDITY_ENTRY_TIMEFRAMES
from tradelab.sl_targets import FVG_SL_TIMEFRAMES, MAX_LIQUIDITY_SL_TARGETS, SLSourceType
from tradelab.tp_targets import MAX_LIQUIDITY_TARGETS, TPSourceType

LONG = "long"
SHORT = "short"
_DIRECTIONS = frozenset({LONG, SHORT})

_LIQUIDITY_RANKS = frozenset(range(1, MAX_LIQUIDITY_TARGETS + 1))
_LIQUIDITY_SL_RANKS = frozenset(range(1, MAX_LIQUIDITY_SL_TARGETS + 1))

_LIQUIDITY_DIRECTION = {
    ("high", EntryMode.REVERSAL): SHORT,
    ("high", EntryMode.CONTINUATION): LONG,
    ("low", EntryMode.REVERSAL): LONG,
    ("low", EntryMode.CONTINUATION): SHORT,
}


class BiasPolicy(str, Enum):
    IGNORE = "ignore"
    REQUIRE_ALIGNED = "require_aligned"
    REQUIRE_NEUTRAL = "require_neutral"


def _split_selector(selector: str) -> list[str]:
    if not isinstance(selector, str) or not selector:
        raise ValueError(f"selector must be a non-empty string, got {selector!r}")
    return selector.split("-")


def _validate_fvg_selector(selector: str) -> str:
    """Validate `selector` is a canonical ``fvg-{timeframe}-{source_type}``
    selector (``tradelab.entry_signal_timeline``'s own format); return its
    timeframe."""
    parts = _split_selector(selector)
    if len(parts) != 3 or parts[0] != "fvg":
        raise ValueError(f"Malformed fvg selector {selector!r}; expected 'fvg-{{timeframe}}-{{source_type}}'")
    _, timeframe, source_type = parts
    if timeframe not in _FVG_ENTRY_TIMEFRAMES:
        raise ValueError(
            f"Unsupported FVG entry timeframe {timeframe!r} in selector {selector!r}; expected one of {sorted(_FVG_ENTRY_TIMEFRAMES)}"
        )
    valid_source_types = {t.value for t in FVGSourceType}
    if source_type not in valid_source_types:
        raise ValueError(f"Unknown FVG source_type {source_type!r} in selector {selector!r}; expected one of {sorted(valid_source_types)}")
    return timeframe


def _validate_liquidity_selector(selector: str) -> tuple[str, str]:
    """Validate `selector` is a canonical ``liquidity-{timeframe}-
    {source_family}-{liquidity_type}-{trigger_phase}-{entry_mode}``
    selector; return ``(timeframe, fixed_direction)`` -- a liquidity
    selector's entry_mode and implied side always pin it to exactly one
    direction (the same table ``tradelab.liquidity_entry`` uses)."""
    parts = _split_selector(selector)
    if len(parts) != 6 or parts[0] != "liquidity":
        raise ValueError(
            f"Malformed liquidity selector {selector!r}; expected "
            "'liquidity-{timeframe}-{source_family}-{liquidity_type}-{trigger_phase}-{entry_mode}'"
        )
    _, timeframe, source_family, liquidity_type, trigger_phase, entry_mode = parts
    if timeframe not in _LIQUIDITY_ENTRY_TIMEFRAMES:
        raise ValueError(
            f"Unsupported liquidity entry timeframe {timeframe!r} in selector {selector!r}; "
            f"expected one of {sorted(_LIQUIDITY_ENTRY_TIMEFRAMES)}"
        )
    if source_family not in {f.value for f in SourceFamily}:
        raise ValueError(f"Unknown liquidity source_family {source_family!r} in selector {selector!r}")
    if liquidity_type not in {t.value for t in LiquidityType}:
        raise ValueError(f"Unknown liquidity_type {liquidity_type!r} in selector {selector!r}")
    if trigger_phase not in {p.value for p in CandidatePhase}:
        raise ValueError(f"Unknown trigger_phase {trigger_phase!r} in selector {selector!r}")
    if entry_mode not in {m.value for m in EntryMode}:
        raise ValueError(f"Unknown entry_mode {entry_mode!r} in selector {selector!r}")

    if liquidity_type.endswith("_high"):
        side = "high"
    elif liquidity_type.endswith("_low"):
        side = "low"
    else:
        raise ValueError(f"liquidity_type {liquidity_type!r} in selector {selector!r} has no recognizable high/low side")

    direction = _LIQUIDITY_DIRECTION[(side, EntryMode(entry_mode))]
    return timeframe, direction


def selector_timeframe(selector: str) -> str:
    """The timeframe encoded in a canonical ``fvg-*``/``liquidity-*``
    selector; raises ``ValueError`` for anything else."""
    parts = _split_selector(selector)
    if parts[0] == "fvg":
        return _validate_fvg_selector(selector)
    if parts[0] == "liquidity":
        return _validate_liquidity_selector(selector)[0]
    raise ValueError(f"selector {selector!r} does not start with 'fvg-' or 'liquidity-'")


def selector_fixed_direction(selector: str) -> str | None:
    """The single direction a selector is pinned to, or ``None`` if it is
    direction-agnostic. Every ``fvg-*`` selector (``fvg_touch``/``ifvg``)
    spans both directions -- direction lives on the individual
    ``EntrySignal``, not the selector. Every ``liquidity-*`` selector is
    pinned to exactly one direction by its ``entry_mode``/implied side."""
    parts = _split_selector(selector)
    if parts[0] == "fvg":
        _validate_fvg_selector(selector)
        return None
    if parts[0] == "liquidity":
        return _validate_liquidity_selector(selector)[1]
    raise ValueError(f"selector {selector!r} does not start with 'fvg-' or 'liquidity-'")


@dataclass(frozen=True)
class TPChoice:
    """Which ``tp_targets`` candidate this strategy will pull: ``LIQUIDITY``
    ranks 1..3 (nearest-first, mirrors ``TPTargetCandidate.rank``
    exactly); ``DATA_WICK`` is uncapped/unranked upstream (``tp_targets``
    never sets a rank for it), so ``rank`` must be ``None`` there too."""

    source_type: TPSourceType
    rank: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, TPSourceType):
            raise ValueError(f"TPChoice.source_type must be a TPSourceType, got {self.source_type!r}")
        if self.source_type == TPSourceType.LIQUIDITY:
            if self.rank not in _LIQUIDITY_RANKS:
                raise ValueError(f"TPChoice.rank must be one of {sorted(_LIQUIDITY_RANKS)} for LIQUIDITY, got {self.rank!r}")
        else:
            if self.rank is not None:
                raise ValueError(f"TPChoice.rank must be None for {self.source_type.value!r}, got {self.rank!r}")


@dataclass(frozen=True)
class SLChoice:
    """Which ``sl_targets`` candidate this strategy will pull. ``LIQUIDITY``
    ranks 1..3, exactly like ``TPChoice``. ``FVG_BOUNDARY`` is never
    ranked (``sl_targets`` never sets a rank for it either) and instead
    names, via ``fvg_link_selector``, which of the strategy's own
    ``fvg-*`` selectors supplies the linked confluence -- never an
    arbitrary FVG zone (``StrategySpec.__post_init__`` additionally
    requires that selector to actually belong to the strategy). Data wick
    can never be an SL source at all: ``SLSourceType`` has no such
    member, so passing one (e.g. ``tradelab.tp_targets.TPSourceType.
    DATA_WICK``) fails the ``isinstance`` check below."""

    source_type: SLSourceType
    rank: int | None = None
    fvg_link_selector: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, SLSourceType):
            raise ValueError(f"SLChoice.source_type must be an SLSourceType, got {self.source_type!r}")
        if self.source_type == SLSourceType.LIQUIDITY:
            if self.rank not in _LIQUIDITY_SL_RANKS:
                raise ValueError(f"SLChoice.rank must be one of {sorted(_LIQUIDITY_SL_RANKS)} for LIQUIDITY, got {self.rank!r}")
            if self.fvg_link_selector is not None:
                raise ValueError("SLChoice.fvg_link_selector must be None for LIQUIDITY")
        else:  # FVG_BOUNDARY
            if self.rank is not None:
                raise ValueError(f"SLChoice.rank must be None for FVG_BOUNDARY, got {self.rank!r}")
            if not self.fvg_link_selector:
                raise ValueError("SLChoice.fvg_link_selector is required for FVG_BOUNDARY")
            timeframe = _validate_fvg_selector(self.fvg_link_selector)
            if timeframe not in FVG_SL_TIMEFRAMES:
                raise ValueError(
                    f"SLChoice.fvg_link_selector {self.fvg_link_selector!r} has timeframe {timeframe!r}; "
                    f"only {sorted(FVG_SL_TIMEFRAMES)} qualify for FVG boundary SL"
                )


@dataclass(frozen=True)
class StrategySpec:
    """One fully-validated, immutable strategy declaration. Not a trade,
    not a backtest result -- only a description of which signals and
    which TP/SL choice a future backtest step should use.

    ``strategy_id`` is a stable, deterministic string, ordinarily built
    by :func:`iter_strategy_specs`; it is a plain field here (not
    computed in ``__post_init__``) so a caller constructing a
    ``StrategySpec`` by hand -- e.g. in a test -- controls it directly,
    the same convention every other ``*_id``/``*_target_id`` field in
    this package already follows.
    """

    strategy_id: str
    direction: str
    primary_entry_selector: str
    required_confluence_selectors: tuple[str, ...]
    bias_policy: BiasPolicy
    tp_choice: TPChoice
    sl_choice: SLChoice

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("StrategySpec.strategy_id must be non-empty")
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"StrategySpec.direction must be one of {sorted(_DIRECTIONS)}, got {self.direction!r}")
        if not self.primary_entry_selector:
            raise ValueError("StrategySpec.primary_entry_selector is mandatory")
        selector_timeframe(self.primary_entry_selector)  # raises on malformed/unknown selector

        required = self.required_confluence_selectors
        if not isinstance(required, tuple):
            raise ValueError(f"StrategySpec.required_confluence_selectors must be a tuple, got {type(required).__name__}")
        if len(required) != len(set(required)):
            dupes = sorted({s for s in required if required.count(s) > 1})
            raise ValueError(f"Duplicate required_confluence_selectors: {dupes}")
        if self.primary_entry_selector in required:
            raise ValueError(
                f"primary_entry_selector {self.primary_entry_selector!r} must not also appear in required_confluence_selectors"
            )
        for s in required:
            selector_timeframe(s)  # raises on malformed/unknown selector

        fixed_direction = selector_fixed_direction(self.primary_entry_selector)
        if fixed_direction is not None and fixed_direction != self.direction:
            raise ValueError(
                f"{self.direction!r} strategy cannot use primary_entry_selector {self.primary_entry_selector!r}, "
                f"which is pinned to {fixed_direction!r}"
            )

        if not isinstance(self.bias_policy, BiasPolicy):
            raise ValueError(f"StrategySpec.bias_policy must be a BiasPolicy, got {self.bias_policy!r}")
        if not isinstance(self.tp_choice, TPChoice):
            raise ValueError(f"StrategySpec.tp_choice must be a TPChoice, got {self.tp_choice!r}")
        if not isinstance(self.sl_choice, SLChoice):
            raise ValueError(f"StrategySpec.sl_choice must be an SLChoice, got {self.sl_choice!r}")

        if self.sl_choice.source_type == SLSourceType.FVG_BOUNDARY:
            link = self.sl_choice.fvg_link_selector
            if link != self.primary_entry_selector and link not in required:
                raise ValueError(
                    f"SLChoice.fvg_link_selector {link!r} must be either this strategy's primary_entry_selector "
                    "or one of its required_confluence_selectors -- FVG boundary SL may only come from the "
                    "strategy's own linked FVG confluence, never an arbitrary FVG zone"
                )

    def to_dict(self) -> dict:
        """A plain, JSON-compatible dict -- for pairing with later
        backtest results (enums flattened to their ``.value``, tuples to
        lists)."""
        return {
            "strategy_id": self.strategy_id,
            "direction": self.direction,
            "primary_entry_selector": self.primary_entry_selector,
            "required_confluence_selectors": list(self.required_confluence_selectors),
            "bias_policy": self.bias_policy.value,
            "tp_choice": {
                "source_type": self.tp_choice.source_type.value,
                "rank": self.tp_choice.rank,
            },
            "sl_choice": {
                "source_type": self.sl_choice.source_type.value,
                "rank": self.sl_choice.rank,
                "fvg_link_selector": self.sl_choice.fvg_link_selector,
            },
        }


def _build_strategy_id(
    direction: str,
    primary_entry_selector: str,
    required_confluence_selectors: tuple[str, ...],
    bias_policy: BiasPolicy,
    tp_choice: TPChoice,
    sl_choice: SLChoice,
) -> str:
    """Stable, readable, deterministic -- derived only from the
    configuration itself, never from a position in some list."""
    confluence_part = "+".join(sorted(required_confluence_selectors)) if required_confluence_selectors else "none"
    tp_part = tp_choice.source_type.value if tp_choice.rank is None else f"{tp_choice.source_type.value}{tp_choice.rank}"
    if sl_choice.source_type == SLSourceType.FVG_BOUNDARY:
        sl_part = f"{sl_choice.source_type.value}[{sl_choice.fvg_link_selector}]"
    else:
        sl_part = sl_choice.source_type.value if sl_choice.rank is None else f"{sl_choice.source_type.value}{sl_choice.rank}"
    return f"strategy-{direction}-{primary_entry_selector}-{confluence_part}-{bias_policy.value}-tp_{tp_part}-sl_{sl_part}"


def iter_strategy_specs(
    *,
    directions: Iterable[str],
    primary_entry_selectors: Iterable[str],
    bias_policies: Iterable[BiasPolicy],
    tp_choices: Iterable[TPChoice],
    sl_choices: Iterable[SLChoice],
    confluence_selector_pool: Iterable[str] = (),
    max_required_confluences: int = 0,
) -> Iterator[StrategySpec]:
    """Lazily yield every valid ``StrategySpec`` across the given
    parameter spaces, in a fixed, deterministic order -- a generator
    function, so calling this never runs any of its body up front and
    never materializes the combination space as a list; each
    ``StrategySpec`` is produced only when the caller asks for the next
    one.

    ``confluence_selector_pool`` is the set of selectors eligible to
    become *required additional* confluence; ``max_required_confluences``
    caps how many of them (0..N, every subset size in between) a single
    generated strategy may require at once. Combinations that violate a
    ``StrategySpec`` validation rule (the primary selector reappearing in
    its own confluence set, a direction/selector mismatch, an invalid
    TP/SL link) are silently skipped -- this is a structural filter, not
    a historical-results filter: nothing here ever inspects backtest
    output or simulates a trade.
    """
    if max_required_confluences < 0:
        raise ValueError(f"max_required_confluences must be >= 0, got {max_required_confluences!r}")

    directions = tuple(directions)
    primary_entry_selectors = tuple(primary_entry_selectors)
    pool = tuple(sorted(set(confluence_selector_pool)))
    bias_policies = tuple(bias_policies)
    tp_choices = tuple(tp_choices)
    sl_choices = tuple(sl_choices)

    confluence_combos: list[tuple[str, ...]] = []
    for k in range(0, max_required_confluences + 1):
        confluence_combos.extend(combinations(pool, k))

    for direction, primary, confluence, bias_policy, tp_choice, sl_choice in product(
        directions, primary_entry_selectors, confluence_combos, bias_policies, tp_choices, sl_choices
    ):
        if primary in confluence:
            continue
        try:
            strategy_id = _build_strategy_id(direction, primary, confluence, bias_policy, tp_choice, sl_choice)
            spec = StrategySpec(
                strategy_id=strategy_id,
                direction=direction,
                primary_entry_selector=primary,
                required_confluence_selectors=confluence,
                bias_policy=bias_policy,
                tp_choice=tp_choice,
                sl_choice=sl_choice,
            )
        except ValueError:
            continue
        yield spec
