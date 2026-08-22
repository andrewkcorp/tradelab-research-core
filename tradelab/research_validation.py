"""Post-backtest validation metrics which never alter execution.

All functions consume already-accounted trades.  They are intentionally
separate from signal generation, fills and ranking so sensitivity/statistical
analysis cannot feed information back into the strategy path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import random
from statistics import NormalDist
from typing import Iterable, Sequence

from tradelab.accounting import AccountBacktestResult, point_value_usd
from tradelab.apex_eod import trading_day_id
from tradelab.backtest import TradeExitReason


@dataclass(frozen=True)
class PerformanceValidation:
    resolved_trade_count: int
    gross_profit_usd: float
    gross_loss_usd: float
    profit_factor: float | None
    average_win_usd: float | None
    average_loss_usd: float | None
    expectancy_usd: float | None
    average_r: float | None
    total_r: float
    annualized_daily_sharpe: float | None
    largest_winner_usd: float | None
    largest_loser_usd: float | None
    top_winner_share_of_positive_pnl: float | None
    profitable_year_count: int
    evaluated_year_count: int
    minimum_year_net_pnl_usd: float | None
    maximum_year_net_pnl_usd: float | None
    yearly_net_pnl_usd: dict[str, float]


def _weekday_dates(start: date, end: date) -> list[date]:
    out: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            out.append(current)
        current += timedelta(days=1)
    return out


def _sample_sharpe(values: Sequence[float], annualization: int = 252) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance <= 0:
        return None
    return mean / math.sqrt(variance) * math.sqrt(annualization)


def summarize_account_result(account: AccountBacktestResult) -> PerformanceValidation:
    """Calculate net-trade and daily metrics from an immutable account result."""
    resolved = [trade for trade in account.accounted_trades if trade.net_pnl_usd is not None]
    pnl = [float(trade.net_pnl_usd) for trade in resolved]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    r_values = [
        float(trade.net_pnl_usd) / trade.planned_risk_usd
        for trade in resolved
        if trade.planned_risk_usd > 0
    ]

    daily: dict[str, float] = {}
    yearly: dict[str, float] = {}
    for trade in resolved:
        day = trading_day_id(trade.exit_timestamp or trade.entry_timestamp)
        daily[day] = daily.get(day, 0.0) + float(trade.net_pnl_usd)
        year = day[:4]
        yearly[year] = yearly.get(year, 0.0) + float(trade.net_pnl_usd)
    daily_values: list[float] = []
    if daily:
        first, last = date.fromisoformat(min(daily)), date.fromisoformat(max(daily))
        daily_values = [daily.get(day.isoformat(), 0.0) for day in _weekday_dates(first, last)]

    return PerformanceValidation(
        resolved_trade_count=len(resolved),
        gross_profit_usd=gross_profit,
        gross_loss_usd=gross_loss,
        profit_factor=(gross_profit / gross_loss if gross_loss > 0 else None),
        average_win_usd=(sum(wins) / len(wins) if wins else None),
        average_loss_usd=(sum(losses) / len(losses) if losses else None),
        expectancy_usd=(sum(pnl) / len(pnl) if pnl else None),
        average_r=(sum(r_values) / len(r_values) if r_values else None),
        total_r=sum(r_values),
        annualized_daily_sharpe=_sample_sharpe(daily_values),
        largest_winner_usd=max(wins) if wins else None,
        largest_loser_usd=min(losses) if losses else None,
        top_winner_share_of_positive_pnl=(max(wins) / gross_profit if gross_profit > 0 else None),
        profitable_year_count=sum(value > 0 for value in yearly.values()),
        evaluated_year_count=len(yearly),
        minimum_year_net_pnl_usd=(min(yearly.values()) if yearly else None),
        maximum_year_net_pnl_usd=(max(yearly.values()) if yearly else None),
        yearly_net_pnl_usd=dict(sorted(yearly.items())),
    )


def deflated_sharpe_probability(
    *,
    observed_sharpe: float,
    observations: int,
    independent_trials: int,
    trial_sharpe_variance: float,
    return_skewness: float = 0.0,
    return_kurtosis: float = 3.0,
) -> float:
    """Probability that Sharpe exceeds selection-inflated luck.

    Implements the Bailey/Lopez de Prado Deflated Sharpe Ratio probability.
    Inputs must use one consistent, *non-annualized* observation frequency;
    for TradeLab that means daily Sharpe and daily return moments.  The
    caller supplies the effective number of independent strategy trials --
    using the raw number of highly correlated variants would be deliberately
    conservative and must be reported as such.

    This is post-selection validation only.  It never changes fills, P&L or
    leaderboard membership.
    """
    if observations < 2:
        raise ValueError("at least two return observations are required")
    if independent_trials < 1:
        raise ValueError("independent_trials must be positive")
    if trial_sharpe_variance < 0:
        raise ValueError("trial_sharpe_variance cannot be negative")
    if return_kurtosis < 1:
        raise ValueError("return_kurtosis must be at least 1")

    expected_maximum = 0.0
    if independent_trials > 1 and trial_sharpe_variance > 0:
        normal = NormalDist()
        euler_gamma = 0.5772156649015329
        n = float(independent_trials)
        expected_maximum = math.sqrt(trial_sharpe_variance) * (
            (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / n)
            + euler_gamma * normal.inv_cdf(1.0 - 1.0 / (n * math.e))
        )
    denominator_sq = (
        1.0
        - return_skewness * observed_sharpe
        + ((return_kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe
    )
    if denominator_sq <= 0:
        raise ValueError("return moments produce a non-positive DSR denominator")
    statistic = (observed_sharpe - expected_maximum) * math.sqrt(observations - 1) / math.sqrt(denominator_sq)
    return NormalDist().cdf(statistic)


def stop_slippage_sensitivity(
    account: AccountBacktestResult,
    *,
    tick_size: float,
    extra_stop_slippage_ticks: Iterable[int] = (0, 1, 2, 3),
) -> dict[int, float]:
    """Net P&L under extra adverse ticks on stop exits only.

    The base fill already includes gap-through-stop behavior.  This analytic
    sensitivity subtracts an additional deterministic penalty and never
    re-simulates signals or changes the stored result.
    """
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    base = account.realized_net_pnl_usd_total
    stop_notional_per_tick = sum(
        tick_size * point_value_usd(trade.instrument) * trade.quantity
        for trade in account.accounted_trades
        if trade.exit_reason == TradeExitReason.SL_HIT and trade.net_pnl_usd is not None
    )
    output: dict[int, float] = {}
    for ticks in extra_stop_slippage_ticks:
        if not isinstance(ticks, int) or ticks < 0:
            raise ValueError("extra stop slippage ticks must be non-negative integers")
        output[ticks] = base - ticks * stop_notional_per_tick
    return output


def benjamini_hochberg(p_values: Sequence[float], *, false_discovery_rate: float = 0.10) -> tuple[bool, ...]:
    """Return which hypotheses pass Benjamini-Hochberg FDR control."""
    if not 0 < false_discovery_rate < 1:
        raise ValueError("false_discovery_rate must lie strictly between 0 and 1")
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    count = len(p_values)
    if count == 0:
        return ()
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    largest_rank = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= false_discovery_rate * rank / count:
            largest_rank = rank
    accepted = {index for index, _ in ordered[:largest_rank]}
    return tuple(index in accepted for index in range(count))


def stationary_block_bootstrap_mean_p_value(
    values: Sequence[float],
    *,
    resamples: int = 1000,
    average_block_length: float = 10.0,
    seed: int = 0,
) -> float:
    """One-sided p-value for positive mean, preserving local dependence.

    The null distribution resamples the mean-centered sequence with geometric
    block restarts (Politis-Romano stationary bootstrap).  A fixed seed makes
    the research artifact exactly reproducible.
    """
    if len(values) < 2:
        raise ValueError("at least two observations are required")
    if resamples < 1 or average_block_length < 1:
        raise ValueError("resamples and average_block_length must be positive")
    observed = sum(values) / len(values)
    centered = [value - observed for value in values]
    rng = random.Random(seed)
    restart_probability = 1.0 / average_block_length
    exceed = 0
    n = len(centered)
    for _ in range(resamples):
        index = rng.randrange(n)
        total = 0.0
        for position in range(n):
            if position and rng.random() < restart_probability:
                index = rng.randrange(n)
            total += centered[index]
            index = (index + 1) % n
        exceed += total / n >= observed
    return (exceed + 1) / (resamples + 1)
