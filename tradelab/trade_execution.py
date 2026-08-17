"""Trade execution: a deterministic, single-trade fill simulator on
ordered tick data -- the foundation a future full backtester will build
on.

Not a backtester, not P&L in USD, not position sizing, not the $20 per-
trade cost, not daily loss limits, not concurrent positions, not the
news filter, and not market data loading -- those are all separate,
later steps. This module answers exactly one question: "given one
already-decided trade request (direction, the instant its signal became
known, and one already-chosen TP price and one already-chosen SL
price), and an ordered stream of ticks, what actually happens to that
one trade?" It does not choose *which* TP/SL candidate to use -- that
remains a future backtest layer's job, consuming ``tradelab.tp_targets``/
``tradelab.sl_targets`` output and picking exactly one of each before
ever constructing a ``MarketTradeRequest`` here.

**Deliberately operates on the tick's own transaction price.** Spread,
queue position/partial fills, contract size, the flat $20 round-trip
cost (spec 3), and prop-firm risk management are all missing on
purpose -- they belong in the backtester layer above this one, which
will call this module once per trade and then apply those account-level
concerns to its result. This module's own fills are exactly two, no
more: the entry fill and the exit fill.

**Market entry** (spec): the very first tick strictly after
``MarketTradeRequest.signal_known_at`` -- never a tick at or before it
(the tick that made the signal known can never also be its own entry
fill; a market that never trades another tick after the signal never
enters at all, deterministically becoming ``TradeStatus.NO_ENTRY``, not
a guess or a silent no-op).

**Exit scan** starts on the tick *after* the entry tick, never the entry
tick itself -- one tick can only ever be one fill (this is exactly why
a tick stream, not a candle stream, is used here: a single tick has one
unambiguous price, so "TP and SL in the same tick" can never be a real
event; it only becomes structurally impossible once entry validation
below has run, see ``_validate_entry_price_against_targets``).

**Exit rule and fill pricing** (spec, never mix the two):

===========  =============================  ===============================
Direction    Hits TP when                   Hits SL when
===========  =============================  ===============================
long         tick price >= tp_price          tick price <= sl_price
short        tick price <= tp_price          tick price >= sl_price
===========  =============================  ===============================

- **TP is a limit**: the fill price is exactly ``tp_price``, regardless
  of how far the triggering tick's own price actually printed past it.
- **SL is a stop-market**: the fill price is the triggering tick's own
  price, not ``sl_price`` -- this is a deliberately conservative choice
  that lets a real price gap/jump through the stop level show up as
  worse-than-expected slippage in the result, exactly like a real
  stop-market order would fill.

Whichever condition a tick satisfies first (walking ticks in strict
``(timestamp, sequence)`` order) ends the trade; nothing after that is
examined. If the tick stream runs out after entry with neither
condition ever met, the result is ``TradeStatus.OPEN_UNRESOLVED`` --
this module never invents a synthetic close.

**No lookahead**: entry is decided by a single forward walk through the
ticks that only ever looks at the *current* tick against
``signal_known_at``/the running exit conditions -- it never peeks ahead
to a later tick to decide the current one, and the exit scan can never
begin before the entry tick has actually been consumed.

**Validation.** ``TickPriceEvent``/``MarketTradeRequest`` reject naive
timestamps and non-positive prices at construction, the same convention
every other dataclass in this package already follows. The tick stream
itself must already be strictly ordered by ``(timestamp, sequence)`` --
malformed/reordered input is rejected outright (``ValueError``) rather
than silently sorted, since silently re-ordering ticks is exactly the
kind of look-ahead bug this module exists to prevent. The directional
price relationship spec requires -- long: ``sl_price < entry_price <
tp_price``; short: ``tp_price < entry_price < sl_price`` -- can only be
checked once the market entry fill is actually known, so it is checked
there, immediately after entry, before any exit scanning begins.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

LONG = "long"
SHORT = "short"
_DIRECTIONS = frozenset({LONG, SHORT})


@dataclass(frozen=True)
class TickPriceEvent:
    """One priced instant on the tape. ``sequence`` is the deterministic
    tie-breaker among ticks that share the exact same ``timestamp`` --
    ordering within this module is always by ``(timestamp, sequence)``,
    never ``timestamp`` alone."""

    timestamp: datetime
    sequence: int
    price: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(f"TickPriceEvent.timestamp must be timezone-aware: {self.timestamp!r}")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        if self.price <= 0:
            raise ValueError(f"TickPriceEvent.price must be positive, got {self.price!r}")


@dataclass(frozen=True)
class MarketTradeRequest:
    """One already-decided trade to simulate: a direction, the instant
    its signal became known, and exactly one already-chosen TP price and
    one already-chosen SL price -- this module does not choose between
    candidates, a future backtest layer already did that before building
    this request."""

    trade_id: str
    direction: str  # "long" or "short"
    signal_known_at: datetime
    tp_price: float
    sl_price: float

    def __post_init__(self) -> None:
        if not self.trade_id:
            raise ValueError("MarketTradeRequest.trade_id must be non-empty")
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"MarketTradeRequest.direction must be one of {sorted(_DIRECTIONS)}, got {self.direction!r}")
        if self.signal_known_at.tzinfo is None:
            raise ValueError(f"MarketTradeRequest.signal_known_at must be timezone-aware: {self.signal_known_at!r}")
        object.__setattr__(self, "signal_known_at", self.signal_known_at.astimezone(timezone.utc))
        if self.tp_price <= 0:
            raise ValueError(f"MarketTradeRequest.tp_price must be positive, got {self.tp_price!r}")
        if self.sl_price <= 0:
            raise ValueError(f"MarketTradeRequest.sl_price must be positive, got {self.sl_price!r}")


class TradeStatus(str, Enum):
    """``NO_ENTRY``: the tick stream never produced a tick strictly after
    ``signal_known_at``, so the trade never opened at all -- distinct
    from ``OPEN_UNRESOLVED``, which means it *did* open but the stream
    ran out before TP or SL was hit."""

    NO_ENTRY = "no_entry"
    OPEN_UNRESOLVED = "open_unresolved"
    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"


@dataclass(frozen=True)
class TradeExecutionResult:
    """The fully-audited outcome of simulating one ``MarketTradeRequest``
    against one tick stream. ``entry_timestamp``/``entry_price`` are
    ``None`` only for ``NO_ENTRY``; ``exit_timestamp``/``exit_price``/
    ``gross_pnl_points`` are ``None`` for ``NO_ENTRY`` and
    ``OPEN_UNRESOLVED`` alike -- an unresolved or never-opened trade has
    no gross P&L to report, not a zero one."""

    trade_id: str
    direction: str
    status: TradeStatus
    entry_timestamp: datetime | None
    entry_price: float | None
    exit_timestamp: datetime | None
    exit_price: float | None
    gross_pnl_points: float | None


def _validate_entry_price_against_targets(request: MarketTradeRequest, entry_price: float) -> None:
    if request.direction == LONG:
        if not (request.sl_price < entry_price < request.tp_price):
            raise ValueError(
                f"long trade {request.trade_id!r} requires sl_price < entry_price < tp_price, got "
                f"sl_price={request.sl_price!r}, entry_price={entry_price!r}, tp_price={request.tp_price!r}"
            )
    else:
        if not (request.tp_price < entry_price < request.sl_price):
            raise ValueError(
                f"short trade {request.trade_id!r} requires tp_price < entry_price < sl_price, got "
                f"tp_price={request.tp_price!r}, entry_price={entry_price!r}, sl_price={request.sl_price!r}"
            )


def simulate_trade(request: MarketTradeRequest, ticks: Iterable[TickPriceEvent]) -> TradeExecutionResult:
    """Walk ``ticks`` once, in the caller's own order, and simulate
    exactly one trade for ``request`` -- see the module docstring for the
    full entry/exit/fill-pricing contract."""
    entry_tick: TickPriceEvent | None = None
    exit_status: TradeStatus | None = None
    exit_timestamp: datetime | None = None
    exit_price: float | None = None

    prev_key: tuple[datetime, int] | None = None
    for tick in ticks:
        key = (tick.timestamp, tick.sequence)
        if prev_key is not None and key <= prev_key:
            raise ValueError(
                f"Ticks must be strictly ordered by (timestamp, sequence); {key!r} does not come strictly after {prev_key!r}"
            )
        prev_key = key

        if entry_tick is None:
            if tick.timestamp > request.signal_known_at:
                entry_tick = tick
                _validate_entry_price_against_targets(request, entry_tick.price)
            continue  # the entry tick itself is never also evaluated for exit

        if request.direction == LONG:
            tp_hit = tick.price >= request.tp_price
            sl_hit = tick.price <= request.sl_price
        else:
            tp_hit = tick.price <= request.tp_price
            sl_hit = tick.price >= request.sl_price

        # tp_hit and sl_hit can never both be true for the same tick: the
        # entry validation above already forces tp_price and sl_price onto
        # opposite sides of entry_price with no overlap between them, so a
        # single tick price can never satisfy both conditions at once.
        if sl_hit:
            exit_status = TradeStatus.SL_HIT
            exit_timestamp = tick.timestamp
            exit_price = tick.price  # stop-market: the tick's own price, conservative on gaps
            break
        if tp_hit:
            exit_status = TradeStatus.TP_HIT
            exit_timestamp = tick.timestamp
            exit_price = request.tp_price  # limit: fills exactly at the TP price
            break

    if entry_tick is None:
        return TradeExecutionResult(
            trade_id=request.trade_id,
            direction=request.direction,
            status=TradeStatus.NO_ENTRY,
            entry_timestamp=None,
            entry_price=None,
            exit_timestamp=None,
            exit_price=None,
            gross_pnl_points=None,
        )

    if exit_status is None:
        return TradeExecutionResult(
            trade_id=request.trade_id,
            direction=request.direction,
            status=TradeStatus.OPEN_UNRESOLVED,
            entry_timestamp=entry_tick.timestamp,
            entry_price=entry_tick.price,
            exit_timestamp=None,
            exit_price=None,
            gross_pnl_points=None,
        )

    gross_pnl_points = exit_price - entry_tick.price if request.direction == LONG else entry_tick.price - exit_price
    return TradeExecutionResult(
        trade_id=request.trade_id,
        direction=request.direction,
        status=exit_status,
        entry_timestamp=entry_tick.timestamp,
        entry_price=entry_tick.price,
        exit_timestamp=exit_timestamp,
        exit_price=exit_price,
        gross_pnl_points=gross_pnl_points,
    )
