"""Accounting/risk layer: converts a gross-points ``BacktestResult``
(``tradelab.backtest``) into contract sizing, dollar risk, and USD P&L
for one instrument (NQ or MNQ) -- the first module in this package that
touches real money units at all.

Not a new backtester: this module never re-runs, re-simulates, or
second-guesses a single confluence/entry/TP/SL/execution decision
``tradelab.backtest.backtest_strategy`` already made. It only takes that
module's already-decided ``ExecutedTrade`` records (entry/SL/TP prices,
exit outcome, gross P&L in points) and answers a purely financial
question for each one: how many contracts, how much dollar risk, how
much dollar P&L, and how much did opening it cost?

Also explicitly out of scope, on purpose, for this first version:
equity curve, drawdown, the $2,000 drawdown-episode counter (spec
section 3), strategy ranking, and multi-strategy enumeration -- this
module accounts for exactly one already-computed ``BacktestResult``
against one instrument and one risk configuration, nothing more.

**Sizing** (this task's own spec, layered on top of the existing "$500
target / $550 max" risk model, spec section 3):

1. ``risk_per_contract_usd = abs(entry_price - sl_price) * point_value_usd``.
2. Every positive integer contract count ``n`` with ``n *
   risk_per_contract_usd <= max_stop_risk_usd`` (550 by default) is a
   candidate -- never rounded up past this ceiling, not even by a
   floating-point hair (see ``_RISK_EPSILON`` below).
3. Among the candidates, the one whose planned risk (``n *
   risk_per_contract_usd``) is closest to ``target_risk_usd`` (500 by
   default) wins.
4. A tie (two candidate counts equidistant from the target) is broken
   toward the *smaller* contract count -- deliberately the more
   conservative of the two, never the larger.
5. If even a single contract's own risk already exceeds
   ``max_stop_risk_usd``, there is no valid contract count at all: the
   trade is **untradeable**, recorded as an auditable
   ``UntradeableTrade`` (reason ``STOP_RISK_EXCEEDS_MAX``) rather than
   silently skipped or forced through with an oversized stop.

Since planned risk (``n * risk_per_contract_usd``) is strictly
increasing in ``n``, its distance from ``target_risk_usd`` is a single
V-shaped (unimodal) function of ``n`` -- so a plain ascending scan from
``n = 1``, keeping the *first* count that achieves the lowest distance
seen so far (a later, merely-equal count never overwrites it), finds
the closest-to-target, smallest-on-tie answer directly and correctly;
see ``_choose_quantity``.

**The $20 entry cost is charged the instant a trade opens, independent
of how it later resolves.** A trade that closes on TP, closes on SL, or
is still open when the data runs out (``OPEN_UNRESOLVED``) all incur
exactly ``entry_cost_usd`` (20 by default) the moment the position
opened -- this is why ``entry_costs_charged_usd_total`` sums it across
*every* ``AccountedTrade`` including unresolved ones, while
``realized_gross_pnl_usd_total``/``realized_net_pnl_usd_total`` only
ever sum over trades that actually have a realized exit (``gross_pnl_
usd``/``net_pnl_usd`` are ``None`` for ``OPEN_UNRESOLVED``, never an
invented number). The $20 cost is **not** part of the $550 stop-risk
ceiling -- sizing is decided purely from ``risk_per_contract_usd``
against ``max_stop_risk_usd`` before the entry cost is even considered,
so the worst realistic dollar loss on a single trade is the sized SL
risk (up to $550) *plus* the $20 entry cost, i.e. up to $570, never
capped at $550 itself.

An **untradeable trade is never charged the entry cost**: sizing
rejects it before any position (and therefore any cost) is ever
incurred -- it is recorded purely as an audit fact ("this signal fired,
but no valid contract count existed for it"), never as a position that
opened and then somehow avoided its own entry fee.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tradelab.backtest import BacktestResult, ExecutedTrade, TradeExitReason

_RISK_EPSILON = 1e-9


class Instrument(str, Enum):
    NQ = "NQ"
    MNQ = "MNQ"


POINT_VALUE_USD: dict[Instrument, float] = {
    Instrument.NQ: 20.0,
    Instrument.MNQ: 2.0,
}

INSTRUMENT_TICK_SIZE: dict[Instrument, float] = {
    Instrument.NQ: 0.25,
    Instrument.MNQ: 0.25,
}


def point_value_usd(instrument: Instrument) -> float:
    return POINT_VALUE_USD[instrument]


def instrument_tick_size(instrument: Instrument) -> float:
    return INSTRUMENT_TICK_SIZE[instrument]


@dataclass(frozen=True)
class RiskConfig:
    target_risk_usd: float = 500.0
    max_stop_risk_usd: float = 550.0
    entry_cost_usd: float = 20.0
    # Apex 50K EOD allows up to 60 micros, but TradeLab deliberately uses
    # a tighter research guardrail.  A setup needing more than this is still
    # traded at the cap, provided one MNQ contract fits the dollar-risk limit.
    max_contracts: int = 30

    def __post_init__(self) -> None:
        if self.target_risk_usd <= 0:
            raise ValueError(f"RiskConfig.target_risk_usd must be positive, got {self.target_risk_usd!r}")
        if self.max_stop_risk_usd <= 0:
            raise ValueError(f"RiskConfig.max_stop_risk_usd must be positive, got {self.max_stop_risk_usd!r}")
        if self.entry_cost_usd <= 0:
            raise ValueError(f"RiskConfig.entry_cost_usd must be positive, got {self.entry_cost_usd!r}")
        if not isinstance(self.max_contracts, int) or isinstance(self.max_contracts, bool) or self.max_contracts <= 0:
            raise ValueError(f"RiskConfig.max_contracts must be a positive integer, got {self.max_contracts!r}")
        if self.target_risk_usd > self.max_stop_risk_usd:
            raise ValueError(
                f"RiskConfig.target_risk_usd ({self.target_risk_usd!r}) must be <= max_stop_risk_usd "
                f"({self.max_stop_risk_usd!r})"
            )


class UntradeableReason(str, Enum):
    STOP_RISK_EXCEEDS_MAX = "stop_risk_exceeds_max"


@dataclass(frozen=True)
class AccountedTrade:
    """One ``ExecutedTrade`` (joinable back via ``trade_id``) with
    contract sizing and USD P&L applied. ``gross_pnl_usd``/``net_pnl_
    usd`` are ``None`` only when the underlying trade's own ``exit_
    reason`` is ``OPEN_UNRESOLVED`` -- ``entry_cost_usd`` is still
    charged either way (see module docstring).

    ``entry_timestamp``/``exit_timestamp`` are copied straight from the
    underlying ``ExecutedTrade`` (``exit_timestamp`` is ``None`` for the
    same ``OPEN_UNRESOLVED`` case) -- audit fields only, added so
    ``tradelab.equity`` can build a cash-flow timeline without any
    further lookup back into the original ``BacktestResult``. Nothing
    about the sizing/P&L computation above changes because of them.
    """

    trade_id: str
    strategy_id: str
    direction: str
    instrument: Instrument
    entry_timestamp: datetime
    exit_timestamp: datetime | None
    entry_price: float
    sl_price: float
    tp_price: float
    quantity: int
    risk_per_contract_usd: float
    planned_risk_usd: float
    entry_cost_usd: float
    exit_reason: TradeExitReason
    gross_pnl_usd: float | None
    net_pnl_usd: float | None


@dataclass(frozen=True)
class UntradeableTrade:
    """A trade that ``tradelab.backtest`` executed but whose stop
    distance is too wide for even one contract to stay within ``max_
    stop_risk_usd`` -- never opened, never charged the entry cost."""

    trade_id: str
    strategy_id: str
    direction: str
    instrument: Instrument
    entry_price: float
    sl_price: float
    risk_per_contract_usd: float
    reason: UntradeableReason


@dataclass(frozen=True)
class AccountBacktestResult:
    strategy_id: str
    instrument: Instrument
    risk_config: RiskConfig
    accounted_trades: list[AccountedTrade]
    untradeable_trades: list[UntradeableTrade]
    trade_count: int
    untradeable_count: int
    open_unresolved_count: int
    realized_gross_pnl_usd_total: float
    realized_net_pnl_usd_total: float
    entry_costs_charged_usd_total: float


def apply_accounting(backtest_result: BacktestResult, instrument: Instrument, risk_config: RiskConfig) -> AccountBacktestResult:
    """Apply contract sizing, dollar risk, and USD P&L to every trade in
    ``backtest_result`` for one ``instrument``/``risk_config`` -- never
    mutates ``backtest_result`` itself (its own dataclasses are already
    frozen; this function only ever reads from it)."""
    pv = point_value_usd(instrument)

    accounted: list[AccountedTrade] = []
    untradeable: list[UntradeableTrade] = []

    for trade in backtest_result.trades:
        risk_per_contract_usd = _risk_per_contract(trade, pv)
        quantity = _choose_quantity(
            risk_per_contract_usd,
            risk_config.target_risk_usd,
            risk_config.max_stop_risk_usd,
            risk_config.max_contracts,
        )

        if quantity is None:
            untradeable.append(
                UntradeableTrade(
                    trade_id=trade.trade_id,
                    strategy_id=trade.strategy_id,
                    direction=trade.direction,
                    instrument=instrument,
                    entry_price=trade.entry_price,
                    sl_price=trade.sl_price,
                    risk_per_contract_usd=risk_per_contract_usd,
                    reason=UntradeableReason.STOP_RISK_EXCEEDS_MAX,
                )
            )
            continue

        gross_pnl_usd = None
        net_pnl_usd = None
        if trade.gross_pnl_points is not None:
            gross_pnl_usd = trade.gross_pnl_points * pv * quantity
            net_pnl_usd = gross_pnl_usd - risk_config.entry_cost_usd

        accounted.append(
            AccountedTrade(
                trade_id=trade.trade_id,
                strategy_id=trade.strategy_id,
                direction=trade.direction,
                instrument=instrument,
                entry_timestamp=trade.entry_timestamp,
                exit_timestamp=trade.exit_timestamp,
                entry_price=trade.entry_price,
                sl_price=trade.sl_price,
                tp_price=trade.tp_price,
                quantity=quantity,
                risk_per_contract_usd=risk_per_contract_usd,
                planned_risk_usd=quantity * risk_per_contract_usd,
                entry_cost_usd=risk_config.entry_cost_usd,
                exit_reason=trade.exit_reason,
                gross_pnl_usd=gross_pnl_usd,
                net_pnl_usd=net_pnl_usd,
            )
        )

    open_unresolved_count = sum(1 for t in accounted if t.exit_reason == TradeExitReason.OPEN_UNRESOLVED)
    realized_gross_pnl_usd_total = sum(t.gross_pnl_usd for t in accounted if t.gross_pnl_usd is not None)
    realized_net_pnl_usd_total = sum(t.net_pnl_usd for t in accounted if t.net_pnl_usd is not None)
    # Charged at entry regardless of outcome -- unresolved trades are included.
    entry_costs_charged_usd_total = sum(t.entry_cost_usd for t in accounted)

    return AccountBacktestResult(
        strategy_id=backtest_result.strategy_id,
        instrument=instrument,
        risk_config=risk_config,
        accounted_trades=accounted,
        untradeable_trades=untradeable,
        trade_count=len(accounted),
        untradeable_count=len(untradeable),
        open_unresolved_count=open_unresolved_count,
        realized_gross_pnl_usd_total=realized_gross_pnl_usd_total,
        realized_net_pnl_usd_total=realized_net_pnl_usd_total,
        entry_costs_charged_usd_total=entry_costs_charged_usd_total,
    )


def _risk_per_contract(trade: ExecutedTrade, pv: float) -> float:
    risk = abs(trade.entry_price - trade.sl_price) * pv
    if risk <= 0:
        raise ValueError(
            f"ExecutedTrade {trade.trade_id!r} has entry_price == sl_price ({trade.entry_price!r}); "
            "risk_per_contract_usd must be positive to size a position"
        )
    return risk


def _choose_quantity(
    risk_per_contract_usd: float,
    target_risk_usd: float,
    max_stop_risk_usd: float,
    max_contracts: int,
) -> int | None:
    """The contract count whose planned risk is closest to ``target_risk_
    usd`` among every positive integer count up to ``max_contracts`` that
    keeps planned risk at or below ``max_stop_risk_usd`` -- ``None`` if not
    even one contract qualifies (untradeable). Ties favor the smaller count;
    see the module docstring for why an ascending scan alone already
    guarantees that without extra tie-breaking logic."""
    best_quantity: int | None = None
    best_distance: float | None = None
    quantity = 1
    while quantity <= max_contracts and quantity * risk_per_contract_usd <= max_stop_risk_usd + _RISK_EPSILON:
        distance = abs(quantity * risk_per_contract_usd - target_risk_usd)
        if best_distance is None or distance < best_distance - _RISK_EPSILON:
            best_distance = distance
            best_quantity = quantity
        quantity += 1
    return best_quantity
