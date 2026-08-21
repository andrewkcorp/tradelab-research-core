"""Trade-level outlier audit for a completed strategy result.

The backtest rank remains based on full realised P&L.  This module exposes
the exact trades responsible for the tails and the counterfactual P&L after
removing the largest winner, so a high rank cannot hide a single-data-point
dependency.  It is read-only and never changes accounting or ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

from tradelab.accounting import AccountBacktestResult
from tradelab.backtest import ExecutedTrade


@dataclass(frozen=True)
class AuditedTrade:
    trade_id: str
    direction: str
    quantity: int
    entry_timestamp: str
    exit_timestamp: str | None
    entry_price: float
    sl_price: float
    tp_price: float
    stop_distance_points: float
    target_distance_points: float
    gross_pnl_usd: float | None
    net_pnl_usd: float | None
    tp_source_id: str
    sl_source_id: str


@dataclass(frozen=True)
class OutlierAudit:
    largest_winners: tuple[AuditedTrade, ...]
    largest_losers: tuple[AuditedTrade, ...]
    realised_net_pnl_usd: float
    realised_net_pnl_without_largest_winner_usd: float
    largest_winner_share_of_positive_pnl: float | None


def audit_trade_outliers(
    trades: tuple[ExecutedTrade, ...] | list[ExecutedTrade], account_result: AccountBacktestResult, *, limit: int = 20,
) -> OutlierAudit:
    """Return deterministic realised-P&L tails joined to execution facts."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit!r}")
    execution_by_id = {trade.trade_id: trade for trade in trades}
    audited = []
    for accounted in account_result.accounted_trades:
        if accounted.net_pnl_usd is None:
            continue
        execution = execution_by_id.get(accounted.trade_id)
        if execution is None:
            raise ValueError(f"accounted trade {accounted.trade_id!r} has no matching executed trade")
        audited.append(AuditedTrade(
            trade_id=accounted.trade_id, direction=accounted.direction, quantity=accounted.quantity,
            entry_timestamp=accounted.entry_timestamp.isoformat(),
            exit_timestamp=None if accounted.exit_timestamp is None else accounted.exit_timestamp.isoformat(),
            entry_price=accounted.entry_price, sl_price=accounted.sl_price, tp_price=accounted.tp_price,
            stop_distance_points=abs(accounted.entry_price - accounted.sl_price),
            target_distance_points=abs(accounted.tp_price - accounted.entry_price),
            gross_pnl_usd=accounted.gross_pnl_usd, net_pnl_usd=accounted.net_pnl_usd,
            tp_source_id=execution.tp_source_id, sl_source_id=execution.sl_source_id,
        ))
    ranked_winners = sorted((trade for trade in audited if trade.net_pnl_usd > 0), key=lambda trade: (-trade.net_pnl_usd, trade.trade_id))
    ranked_losers = sorted((trade for trade in audited if trade.net_pnl_usd < 0), key=lambda trade: (trade.net_pnl_usd, trade.trade_id))
    realised_net = sum(trade.net_pnl_usd for trade in audited)
    largest_winner = ranked_winners[0] if ranked_winners else None
    positive_pnl = sum(trade.net_pnl_usd for trade in audited if trade.net_pnl_usd > 0)
    return OutlierAudit(
        largest_winners=tuple(ranked_winners[:limit]), largest_losers=tuple(ranked_losers[:limit]),
        realised_net_pnl_usd=realised_net,
        realised_net_pnl_without_largest_winner_usd=realised_net - (largest_winner.net_pnl_usd if largest_winner else 0.0),
        largest_winner_share_of_positive_pnl=(largest_winner.net_pnl_usd / positive_pnl if largest_winner and positive_pnl else None),
    )
