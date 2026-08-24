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
from tradelab.bar_range_index import BarRangeIndex


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
    bar's high. ``max_mark_to_market_drawdown_usd`` is conservative when the
    intrabar order is unknown: each bar's favourable extreme may establish a
    peak before that same bar's adverse extreme measures drawdown.  This is an
    audit bound, not an additional fill model: it cannot alter entries, exits,
    cash equity or ranking.
    """

    worst_equity_usd: float
    max_mark_to_market_drawdown_usd: float
    timestamp_of_worst_equity: datetime | None



def _as_sequence(bars):
    """Bind ``bars`` without copying when it is already an indexable sequence.

    ``tuple(bars)`` is free for a tuple but catastrophic for a memory-mapped
    bar series: it materialises every ``Bar1m``, which on the real artifact is
    1,057,182 objects built and thrown away *per audited variant* -- the exact
    cost the mapping exists to avoid, and measured at a 2.9x throughput
    regression before this was fixed. Everything below needs only ``len()``,
    indexing and slicing, all of which a mapped series provides directly.

    A one-shot iterator still gets materialised, because it must be traversed
    more than once.
    """
    if hasattr(bars, "__len__") and hasattr(bars, "__getitem__"):
        return bars
    return tuple(bars)



def _low_high_time_range(bars, start: int, stop: int):
    """``(low, high, mark)`` for each bar in ``[start, stop)``.

    The production mark-to-market loop reads exactly these three fields. A
    memory-mapped series exposes each as a column, so iterating the columns
    avoids materialising a ``Bar1m`` per minute of every audited trade's span --
    thousands of objects for a multi-day trade, rebuilt for every exit variant.
    A plain tuple or list falls through to attribute access and behaves exactly
    as before.

    ``mark`` is a bar *position* for a mapped series and a ``datetime`` for a
    plain one; the caller decodes it only when a bar actually sets a new worst
    mark, which is rare, so no timestamp is built for the common case.
    """
    low_at = getattr(bars, "low_at", None)
    high_at = getattr(bars, "high_at", None)
    timestamp_at = getattr(bars, "timestamp_at", None)
    if low_at is not None and high_at is not None and timestamp_at is not None:
        for position in range(start, stop):
            yield low_at(position), high_at(position), position
        return
    for position in range(start, stop):
        bar = bars[position]
        yield bar.low, bar.high, bar.timestamp


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
    ordered_bars = _as_sequence(bars)
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


def _audit_account_mark_to_market_reference(
    trades: tuple[ExecutedTrade, ...] | list[ExecutedTrade],
    account_result: AccountBacktestResult,
    bars: tuple[Bar1m, ...] | list[Bar1m],
    *,
    initial_balance_usd: float = 50_000.0,
) -> AccountMarkToMarketAudit:
    """Original full-bar replay retained as the equivalence oracle."""
    executions = {trade.trade_id: trade for trade in trades}
    ordered_bars = _as_sequence(bars)
    if not ordered_bars:
        return AccountMarkToMarketAudit(initial_balance_usd, 0.0, None)
    by_timestamp = {bar.timestamp: bar for bar in ordered_bars}
    if len(by_timestamp) != len(ordered_bars):
        raise ValueError("bars must have unique timestamps")
    ordered_trades: list[tuple[int, object, int, int]] = []
    by_timestamp_position = {bar.timestamp: position for position, bar in enumerate(ordered_bars)}
    for sequence, value in enumerate(account_result.accounted_trades):
        if value.trade_id not in executions:
            raise ValueError(f"accounted trade {value.trade_id!r} has no matching executed trade")
        if value.entry_timestamp not in by_timestamp:
            raise ValueError(f"entry timestamp {value.entry_timestamp!r} is absent from bars")
        start = by_timestamp_position[value.entry_timestamp]
        if value.exit_timestamp is not None:
            if value.exit_timestamp not in by_timestamp:
                raise ValueError(f"exit timestamp {value.exit_timestamp!r} is absent from bars")
            end = by_timestamp_position[value.exit_timestamp]
        else:
            end = len(ordered_bars) - 1
        ordered_trades.append((sequence, value, start, end))
    balance = initial_balance_usd
    peak = balance
    max_drawdown = 0.0
    worst = balance
    worst_at: datetime | None = None
    previous_end = -1
    for _, open_trade, start, end in sorted(ordered_trades, key=lambda item: (item[2], item[0])):
        # Equality is valid: execution can close one position and enter the
        # next on the same 1m candle.  A dict keyed by timestamp used to lose
        # all but one such trade here.
        if start < previous_end:
            raise ValueError("accounted trades overlap; one-position MTM audit requires non-overlapping trades")
        previous_end = end
        balance -= open_trade.entry_cost_usd
        for bar in ordered_bars[start:end + 1]:
            pv = point_value_usd(open_trade.instrument)
            adverse_price = bar.low if open_trade.direction == "long" else bar.high
            favourable_price = bar.high if open_trade.direction == "long" else bar.low
            adverse_points = (adverse_price - open_trade.entry_price) if open_trade.direction == "long" else (open_trade.entry_price - adverse_price)
            favourable_points = (favourable_price - open_trade.entry_price) if open_trade.direction == "long" else (open_trade.entry_price - favourable_price)
            adverse_marked = balance + adverse_points * pv * open_trade.quantity
            favourable_marked = balance + favourable_points * pv * open_trade.quantity
            peak = max(peak, favourable_marked)
            max_drawdown = max(max_drawdown, peak - adverse_marked)
            if adverse_marked < worst:
                worst, worst_at = adverse_marked, bar.timestamp
        if open_trade.exit_timestamp is not None:
            assert open_trade.gross_pnl_usd is not None
            balance += open_trade.gross_pnl_usd
            peak = max(peak, balance)
            max_drawdown = max(max_drawdown, peak - balance)
            if balance < worst:
                worst, worst_at = balance, open_trade.exit_timestamp
    return AccountMarkToMarketAudit(worst, max_drawdown, worst_at)


def audit_account_mark_to_market(
    trades: tuple[ExecutedTrade, ...] | list[ExecutedTrade],
    account_result: AccountBacktestResult,
    bars: tuple[Bar1m, ...] | list[Bar1m],
    *,
    initial_balance_usd: float = 50_000.0,
    bar_range_index: BarRangeIndex | None = None,
) -> AccountMarkToMarketAudit:
    """Audit worst intrabar account equity without changing execution.

    TradeLab strategies maintain one open position at a time. Therefore a
    chronological replay of entry fee, realised exit P&L, and that position's
    adverse bar extreme yields an exact conservative MTM path for the
    simulated one-position system. A missing execution/bar is a loud error,
    never a silently optimistic risk result.
    """
    executions = {trade.trade_id: trade for trade in trades}
    ordered_bars = _as_sequence(bars)
    if not ordered_bars:
        return AccountMarkToMarketAudit(initial_balance_usd, 0.0, None)
    index = bar_range_index if bar_range_index is not None else BarRangeIndex.build(ordered_bars)

    def exact_position(timestamp: datetime) -> int:
        position = index.find_bar_at_or_after(timestamp)
        if position is None or ordered_bars[position].timestamp != timestamp:
            raise ValueError(f"timestamp {timestamp!r} is absent from bars")
        return position

    ordered_trades: list[tuple[int, object, int, int]] = []
    for sequence, value in enumerate(account_result.accounted_trades):
        if value.trade_id not in executions:
            raise ValueError(f"accounted trade {value.trade_id!r} has no matching executed trade")
        start = exact_position(value.entry_timestamp)
        if value.exit_timestamp is not None:
            end = exact_position(value.exit_timestamp)
        else:
            end = len(ordered_bars) - 1
        ordered_trades.append((sequence, value, start, end))

    balance = initial_balance_usd
    peak = balance
    max_drawdown = 0.0
    worst = balance
    worst_at: datetime | None = None
    previous_end = -1
    for _, open_trade, start, end in sorted(ordered_trades, key=lambda item: (item[2], item[0])):
        # Multiple signals sharing a timestamp may each enter and resolve
        # immediately on that same 1m bar. They are sequential in execution
        # order and therefore valid when start == previous_end.
        if start < previous_end:
            raise ValueError("accounted trades overlap; one-position MTM audit requires non-overlapping trades")
        previous_end = end
        balance -= open_trade.entry_cost_usd
        pv = point_value_usd(open_trade.instrument)
        long_side = open_trade.direction == "long"
        timestamp_at = getattr(ordered_bars, "timestamp_at", None)
        for bar_low, bar_high, bar_mark in _low_high_time_range(ordered_bars, start, end + 1):
            adverse_price = bar_low if long_side else bar_high
            favourable_price = bar_high if long_side else bar_low
            adverse_points = (
                adverse_price - open_trade.entry_price
                if open_trade.direction == "long"
                else open_trade.entry_price - adverse_price
            )
            favourable_points = (
                favourable_price - open_trade.entry_price
                if open_trade.direction == "long"
                else open_trade.entry_price - favourable_price
            )
            adverse_marked = balance + adverse_points * pv * open_trade.quantity
            favourable_marked = balance + favourable_points * pv * open_trade.quantity
            # We do not know whether high or low printed first inside the 1m
            # candle.  For a risk audit, assume the favourable extreme could
            # establish the peak first and then measure the adverse extreme.
            peak = max(peak, favourable_marked)
            max_drawdown = max(max_drawdown, peak - adverse_marked)
            if adverse_marked < worst:
                worst = adverse_marked
                worst_at = timestamp_at(bar_mark) if timestamp_at is not None else bar_mark
        if open_trade.exit_timestamp is not None:
            assert open_trade.gross_pnl_usd is not None
            balance += open_trade.gross_pnl_usd
            peak = max(peak, balance)
            max_drawdown = max(max_drawdown, peak - balance)
            if balance < worst:
                worst, worst_at = balance, open_trade.exit_timestamp
    return AccountMarkToMarketAudit(worst, max_drawdown, worst_at)
