"""Read-only audit of an Apex 50K EOD Evaluation account.

This is deliberately an *audit*, never an execution gate: all supplied
trades remain in the result even after a limit is crossed.  It answers
whether a fully simulated strategy would have breached the configured
prop-account limits, along with when and by how much.  The strategy's
signals, fills, exits and subsequent entries are not changed.

The audit replays the same realised cash events as :mod:`tradelab.equity`:
the entry fee at entry and gross P&L at exit.  It uses CME trading days
([18:00 NY, 18:00 NY)) so DST is handled by ``America/New_York`` rather
than a fixed UTC offset.  EOD drawdown is frozen intraday at the floor
established by the prior completed trading day, and can only tighten at a
later EOD close.  Daily-loss-limit measurements reset per CME trading day.

Mark-to-market/intrabar adverse excursion is intentionally not invented in
this layer; it belongs in the later trade-path audit.  Therefore an audit
breach here is a confirmed realised-cash breach, while absence of one is not
yet proof that intratrade risk never crossed a limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from tradelab.accounting import AccountBacktestResult, AccountedTrade
from tradelab.backtest import TradeExitReason


NY = ZoneInfo("America/New_York")


class ApexLimit(str, Enum):
    EOD_DRAWDOWN = "eod_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"


@dataclass(frozen=True)
class ApexEODConfig:
    """Apex 50K EOD Evaluation limits used by TradeLab research."""

    initial_balance_usd: float = 50_000.0
    eod_drawdown_usd: float = 2_000.0
    daily_loss_limit_usd: float = 1_000.0
    # Apex 50K EOD Evaluation permits 6 standard = 60 micro contracts.
    # TradeLab's stricter 30-MNQ research sizing cap lives in RiskConfig.
    max_mnq_contracts: int = 60

    def __post_init__(self) -> None:
        for name in ("initial_balance_usd", "eod_drawdown_usd", "daily_loss_limit_usd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"ApexEODConfig.{name} must be positive, got {getattr(self, name)!r}")
        if not isinstance(self.max_mnq_contracts, int) or isinstance(self.max_mnq_contracts, bool) or self.max_mnq_contracts <= 0:
            raise ValueError(f"ApexEODConfig.max_mnq_contracts must be a positive integer, got {self.max_mnq_contracts!r}")


@dataclass(frozen=True)
class ApexBreach:
    limit: ApexLimit
    trading_day_id: str
    timestamp: datetime
    trade_id: str
    balance_usd: float
    limit_floor_usd: float
    breach_amount_usd: float


@dataclass(frozen=True)
class ApexTradingDayAudit:
    trading_day_id: str
    opening_balance_usd: float
    closing_balance_usd: float
    daily_pnl_usd: float
    eod_drawdown_floor_usd: float
    daily_loss_limit_breached: bool
    eod_drawdown_breached: bool


@dataclass(frozen=True)
class ApexEODAudit:
    config: ApexEODConfig
    breaches: tuple[ApexBreach, ...]
    trading_days: tuple[ApexTradingDayAudit, ...]
    max_daily_loss_usd: float
    largest_eod_drawdown_breach_usd: float

    @property
    def has_breach(self) -> bool:
        return bool(self.breaches)

    @property
    def breached_limits(self) -> tuple[ApexLimit, ...]:
        return tuple(sorted({breach.limit for breach in self.breaches}, key=lambda value: value.value))


def trading_day_id(timestamp: datetime) -> str:
    """CME trading-day identifier for an aware timestamp.

    The id is the NY calendar date on which that 18:00-started trading day
    began.  E.g. 10:00 NY on 2026-01-06 belongs to ``2026-01-05``.
    """
    if timestamp.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware, got {timestamp!r}")
    local = timestamp.astimezone(NY)
    date = local.date() if local.hour >= 18 else (local - timedelta(days=1)).date()
    return date.isoformat()


def audit_apex_eod(account_result: AccountBacktestResult, config: ApexEODConfig = ApexEODConfig()) -> ApexEODAudit:
    """Audit realised cash events without mutating or filtering any trades."""
    events = _events(account_result.accounted_trades)
    balance = config.initial_balance_usd
    eod_floor = balance - config.eod_drawdown_usd
    current_day: str | None = None
    day_open_balance = balance
    day_daily_breach = False
    day_eod_breach = False
    days: list[ApexTradingDayAudit] = []
    breaches: list[ApexBreach] = []
    breached_this_day: set[ApexLimit] = set()
    max_daily_loss = 0.0
    max_eod_breach = 0.0

    def close_day(day_id: str) -> None:
        nonlocal eod_floor
        days.append(
            ApexTradingDayAudit(
                trading_day_id=day_id,
                opening_balance_usd=day_open_balance,
                closing_balance_usd=balance,
                daily_pnl_usd=balance - day_open_balance,
                eod_drawdown_floor_usd=eod_floor,
                daily_loss_limit_breached=day_daily_breach,
                eod_drawdown_breached=day_eod_breach,
            )
        )
        # Apex EOD floor is only allowed to rise after a completed day.
        eod_floor = max(eod_floor, balance - config.eod_drawdown_usd)

    for timestamp, trade_id, cash_delta in events:
        event_day = trading_day_id(timestamp)
        if current_day is None:
            current_day = event_day
            day_open_balance = balance
        elif event_day != current_day:
            close_day(current_day)
            current_day = event_day
            day_open_balance = balance
            day_daily_breach = False
            day_eod_breach = False
            breached_this_day = set()

        balance += cash_delta
        daily_loss = max(0.0, day_open_balance - balance)
        max_daily_loss = max(max_daily_loss, daily_loss)
        if daily_loss >= config.daily_loss_limit_usd and ApexLimit.DAILY_LOSS_LIMIT not in breached_this_day:
            breached_this_day.add(ApexLimit.DAILY_LOSS_LIMIT)
            day_daily_breach = True
            breaches.append(ApexBreach(
                limit=ApexLimit.DAILY_LOSS_LIMIT, trading_day_id=current_day,
                timestamp=timestamp, trade_id=trade_id, balance_usd=balance,
                limit_floor_usd=day_open_balance - config.daily_loss_limit_usd,
                breach_amount_usd=daily_loss - config.daily_loss_limit_usd,
            ))
        eod_breach = max(0.0, eod_floor - balance)
        max_eod_breach = max(max_eod_breach, eod_breach)
        if balance <= eod_floor and ApexLimit.EOD_DRAWDOWN not in breached_this_day:
            breached_this_day.add(ApexLimit.EOD_DRAWDOWN)
            day_eod_breach = True
            breaches.append(ApexBreach(
                limit=ApexLimit.EOD_DRAWDOWN, trading_day_id=current_day,
                timestamp=timestamp, trade_id=trade_id, balance_usd=balance,
                limit_floor_usd=eod_floor, breach_amount_usd=eod_breach,
            ))

    if current_day is not None:
        close_day(current_day)
    return ApexEODAudit(
        config=config, breaches=tuple(breaches), trading_days=tuple(days),
        max_daily_loss_usd=max_daily_loss, largest_eod_drawdown_breach_usd=max_eod_breach,
    )


def _events(trades: list[AccountedTrade]) -> list[tuple[datetime, str, float]]:
    events: list[tuple[datetime, int, str, float]] = []
    for trade in trades:
        events.append((trade.entry_timestamp, 0, trade.trade_id, -trade.entry_cost_usd))
        if trade.exit_reason != TradeExitReason.OPEN_UNRESOLVED:
            assert trade.exit_timestamp is not None
            assert trade.gross_pnl_usd is not None
            events.append((trade.exit_timestamp, 1, trade.trade_id, trade.gross_pnl_usd))
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(timestamp, trade_id, cash_delta) for timestamp, _kind, trade_id, cash_delta in events]
