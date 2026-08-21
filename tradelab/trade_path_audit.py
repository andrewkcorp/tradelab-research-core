"""Intratrade MAE/MFE audit over the 1-minute path actually available.

These figures are diagnostic only.  They do not revise fills, exits, cash
equity or ranking; they expose how much a closed/open trade moved against or
in favour of the position between its entry and its known exit.  This is the
first mark-to-market risk layer needed to distinguish a cash-only drawdown
from the risk that was actually carried while a trade was open.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tradelab.accounting import AccountBacktestResult, Instrument, point_value_usd
from tradelab.backtest import ExecutedTrade
from tradelab.bar import Bar1m


@dataclass(frozen=True)
class TradePathAudit:
    trade_id: str
    bars_observed: int
    mae_points: float
    mfe_points: float
    mae_price_usd: float
    mfe_price_usd: float


@dataclass(frozen=True)
class AccountMarkToMarketAudit:
    """Conservative account-level intrabar risk observation.

    ``worst_equity_usd`` marks a long to each bar's low and a short to each
    bar's high. It is deliberately an audit lower bound, not an additional
    fill model: it cannot alter entries, exits, cash equity or ranking.
    """

    worst_equity_usd: float
    max_mark_to_market_drawdown_usd: float
    timestamp_of_worst_equity: datetime | None


def audit_trade_paths(
    trades: tuple[ExecutedTrade, ...] | list[ExecutedTrade],
    account_result: AccountBacktestResult,
    bars: tuple[Bar1m, ...] | list[Bar1m],
) -> tuple[TradePathAudit, ...]:
    """Compute per-trade MAE/MFE from entry through exit inclusively.

    An unresolved trade is audited through the final available bar.  Missing
    bars produce a loud ``ValueError`` rather than a silently understated
    adverse excursion.
    """
    executions = {trade.trade_id: trade for trade in trades}
    ordered_bars = tuple(bars)
    timestamps = [bar.timestamp for bar in ordered_bars]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError("bars must be strictly timestamp-ordered with no duplicates")
    by_timestamp = {bar.timestamp: i for i, bar in enumerate(ordered_bars)}
    audits: list[TradePathAudit] = []
    for accounted in account_result.accounted_trades:
        execution = executions.get(accounted.trade_id)
        if execution is None:
            raise ValueError(f"accounted trade {accounted.trade_id!r} has no matching executed trade")
        start = by_timestamp.get(accounted.entry_timestamp)
        if start is None:
            raise ValueError(f"entry timestamp {accounted.entry_timestamp!r} is absent from bars")
        end_timestamp = accounted.exit_timestamp or ordered_bars[-1].timestamp
        end = by_timestamp.get(end_timestamp)
        if end is None:
            raise ValueError(f"exit timestamp {end_timestamp!r} is absent from bars")
        if end < start:
            raise ValueError(f"trade {accounted.trade_id!r} exits before entry")
        segment = ordered_bars[start:end + 1]
        if accounted.direction == "long":
            mae_points = max(0.0, accounted.entry_price - min(bar.low for bar in segment))
            mfe_points = max(0.0, max(bar.high for bar in segment) - accounted.entry_price)
        elif accounted.direction == "short":
            mae_points = max(0.0, max(bar.high for bar in segment) - accounted.entry_price)
            mfe_points = max(0.0, accounted.entry_price - min(bar.low for bar in segment))
        else:
            raise ValueError(f"unknown trade direction {accounted.direction!r}")
        pv = point_value_usd(accounted.instrument)
        audits.append(TradePathAudit(
            trade_id=accounted.trade_id, bars_observed=len(segment), mae_points=mae_points, mfe_points=mfe_points,
            mae_price_usd=mae_points * pv * accounted.quantity,
            mfe_price_usd=mfe_points * pv * accounted.quantity,
        ))
    return tuple(audits)


def audit_account_mark_to_market(
    trades: tuple[ExecutedTrade, ...] | list[ExecutedTrade],
    account_result: AccountBacktestResult,
    bars: tuple[Bar1m, ...] | list[Bar1m],
    *,
    initial_balance_usd: float = 50_000.0,
) -> AccountMarkToMarketAudit:
    """Audit worst intrabar account equity without changing execution.

    TradeLab strategies maintain one open position at a time. Therefore a
    chronological replay of entry fee, realised exit P&L, and that position's
    adverse bar extreme yields an exact conservative MTM path for the
    simulated one-position system. A missing execution/bar is a loud error,
    never a silently optimistic risk result.
    """
    executions = {trade.trade_id: trade for trade in trades}
    accounted = {trade.trade_id: trade for trade in account_result.accounted_trades}
    ordered_bars = tuple(bars)
    if not ordered_bars:
        return AccountMarkToMarketAudit(initial_balance_usd, 0.0, None)
    by_timestamp = {bar.timestamp: bar for bar in ordered_bars}
    if len(by_timestamp) != len(ordered_bars):
        raise ValueError("bars must have unique timestamps")
    starts: dict[datetime, object] = {}
    ends: dict[datetime, object] = {}
    for trade_id, value in accounted.items():
        if trade_id not in executions:
            raise ValueError(f"accounted trade {trade_id!r} has no matching executed trade")
        if value.entry_timestamp not in by_timestamp:
            raise ValueError(f"entry timestamp {value.entry_timestamp!r} is absent from bars")
        starts[value.entry_timestamp] = value
        if value.exit_timestamp is not None:
            if value.exit_timestamp not in by_timestamp:
                raise ValueError(f"exit timestamp {value.exit_timestamp!r} is absent from bars")
            ends[value.exit_timestamp] = value
    balance = initial_balance_usd
    peak = balance
    max_drawdown = 0.0
    worst = balance
    worst_at: datetime | None = None
    open_trade = None
    for bar in ordered_bars:
        if bar.timestamp in starts:
            open_trade = starts[bar.timestamp]
            balance -= open_trade.entry_cost_usd
        marked = balance
        if open_trade is not None:
            pv = point_value_usd(open_trade.instrument)
            adverse_price = bar.low if open_trade.direction == "long" else bar.high
            points = (adverse_price - open_trade.entry_price) if open_trade.direction == "long" else (open_trade.entry_price - adverse_price)
            marked += points * pv * open_trade.quantity
        peak = max(peak, marked)
        max_drawdown = max(max_drawdown, peak - marked)
        if marked < worst:
            worst, worst_at = marked, bar.timestamp
        if bar.timestamp in ends:
            closing = ends[bar.timestamp]
            assert closing.gross_pnl_usd is not None
            balance += closing.gross_pnl_usd
            open_trade = None
            peak = max(peak, balance)
            max_drawdown = max(max_drawdown, peak - balance)
            if balance < worst:
                worst, worst_at = balance, bar.timestamp
    return AccountMarkToMarketAudit(worst, max_drawdown, worst_at)
