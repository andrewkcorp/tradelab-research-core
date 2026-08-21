"""Read-only audit of an Apex 50K EOD Evaluation account.

This is deliberately an *audit*, never an execution gate: all supplied
trades remain in the result even after a limit is crossed.  It answers
whether a fully simulated strategy would have breached the configured
prop-account limits, along with when and by how much.  The strategy's
signals, fills, exits and subsequent entries are not changed.

Production calls replay realised cash events plus the conservative adverse
1-minute mark of every held position, because Apex applies DLL to total
equity including unrealised P&L. It uses CME trading days ([18:00 NY,
18:00 NY)) so DST is handled by ``America/New_York`` rather than a fixed UTC
offset. EOD drawdown is frozen intraday at the floor established by the prior
completed trading day and can only tighten at a later EOD close. DLL resets
per CME trading day. A bars-omitted realised-only mode is retained explicitly
for legacy/reference tests and marks its result ``uses_mark_to_market=False``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from tradelab.accounting import AccountBacktestResult, AccountedTrade, point_value_usd
from tradelab.bar import Bar1m
from tradelab.bar_range_index import BarRangeIndex
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
    uses_mark_to_market: bool

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


def audit_apex_eod(
    account_result: AccountBacktestResult,
    config: ApexEODConfig = ApexEODConfig(),
    *,
    bars: tuple[Bar1m, ...] | list[Bar1m] | None = None,
    bar_range_index: BarRangeIndex | None = None,
) -> ApexEODAudit:
    """Audit Apex limits without mutating or filtering any trades.

    Production callers pass ``bars`` so the EOD threshold and DLL are checked
    against conservative intrabar account equity, including unrealised P&L as
    required by Apex.  The bars-omitted cash-only path remains available as a
    deliberately weaker reference for legacy callers; its result says so via
    ``uses_mark_to_market=False``.
    """
    if bars is not None:
        return _audit_apex_eod_mark_to_market(
            account_result, config, tuple(bars), bar_range_index=bar_range_index,
        )
    return _audit_apex_eod_realized(account_result, config)


def _audit_apex_eod_realized(account_result: AccountBacktestResult, config: ApexEODConfig) -> ApexEODAudit:
    """Legacy/diagnostic cash-event audit; production passes market bars."""
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
        uses_mark_to_market=False,
    )


def _audit_apex_eod_mark_to_market(
    account_result: AccountBacktestResult,
    config: ApexEODConfig,
    bars: tuple[Bar1m, ...],
    *,
    bar_range_index: BarRangeIndex | None,
) -> ApexEODAudit:
    """Conservative real-time Apex audit over the actually held 1m ranges.

    A long is checked at each bar low and a short at each bar high.  Day-end
    threshold updates use the last held bar's close equity (or realised cash
    while flat). This is deliberately audit-only: a breach is recorded but the
    replay continues through every later trade exactly as simulated.
    """
    if not bars:
        if account_result.accounted_trades:
            raise ValueError("market bars are required to MTM-audit accounted trades")
        return ApexEODAudit(config, (), (), 0.0, 0.0, True)
    index = bar_range_index if bar_range_index is not None else BarRangeIndex.build(bars)

    def exact_position(timestamp: datetime) -> int:
        position = index.find_bar_at_or_after(timestamp)
        if position is None or bars[position].timestamp != timestamp:
            raise ValueError(f"timestamp {timestamp!r} is absent from bars")
        return position

    ordered_trades = sorted(account_result.accounted_trades, key=lambda trade: (trade.entry_timestamp, trade.trade_id))
    balance = config.initial_balance_usd
    eod_floor = balance - config.eod_drawdown_usd
    current_day: str | None = None
    day_open_equity = balance
    day_daily_breach = False
    day_eod_breach = False
    breached_this_day: set[ApexLimit] = set()
    days: list[ApexTradingDayAudit] = []
    breaches: list[ApexBreach] = []
    max_daily_loss = 0.0
    max_eod_breach = 0.0
    last_close_equity = balance
    previous_end = -1

    def close_day(day_id: str, closing_equity: float) -> None:
        nonlocal eod_floor
        days.append(ApexTradingDayAudit(
            trading_day_id=day_id,
            opening_balance_usd=day_open_equity,
            closing_balance_usd=closing_equity,
            daily_pnl_usd=closing_equity - day_open_equity,
            eod_drawdown_floor_usd=eod_floor,
            daily_loss_limit_breached=day_daily_breach,
            eod_drawdown_breached=day_eod_breach,
        ))
        eod_floor = max(eod_floor, closing_equity - config.eod_drawdown_usd)

    def start_day(day_id: str, opening_equity: float) -> None:
        nonlocal current_day, day_open_equity, day_daily_breach, day_eod_breach, breached_this_day
        current_day = day_id
        day_open_equity = opening_equity
        day_daily_breach = False
        day_eod_breach = False
        breached_this_day = set()

    def advance_to_day(day_id: str, opening_equity: float) -> None:
        if current_day is None:
            start_day(day_id, opening_equity)
        elif day_id != current_day:
            close_day(current_day, opening_equity)
            start_day(day_id, opening_equity)

    def check_equity(equity: float, timestamp: datetime, trade_id: str) -> None:
        nonlocal day_daily_breach, day_eod_breach, max_daily_loss, max_eod_breach
        assert current_day is not None
        daily_loss = max(0.0, day_open_equity - equity)
        max_daily_loss = max(max_daily_loss, daily_loss)
        if daily_loss >= config.daily_loss_limit_usd and ApexLimit.DAILY_LOSS_LIMIT not in breached_this_day:
            breached_this_day.add(ApexLimit.DAILY_LOSS_LIMIT)
            day_daily_breach = True
            breaches.append(ApexBreach(
                ApexLimit.DAILY_LOSS_LIMIT, current_day, timestamp, trade_id,
                equity, day_open_equity - config.daily_loss_limit_usd,
                daily_loss - config.daily_loss_limit_usd,
            ))
        eod_breach = max(0.0, eod_floor - equity)
        max_eod_breach = max(max_eod_breach, eod_breach)
        if equity <= eod_floor and ApexLimit.EOD_DRAWDOWN not in breached_this_day:
            breached_this_day.add(ApexLimit.EOD_DRAWDOWN)
            day_eod_breach = True
            breaches.append(ApexBreach(
                ApexLimit.EOD_DRAWDOWN, current_day, timestamp, trade_id,
                equity, eod_floor, eod_breach,
            ))

    for trade in ordered_trades:
        start = exact_position(trade.entry_timestamp)
        end = exact_position(trade.exit_timestamp) if trade.exit_timestamp is not None else len(bars) - 1
        # The execution engine can close immediately on the entry bar and
        # process a later signal from that same timestamp. Those trades are
        # sequential, not overlapping, despite sharing one 1m bar index.
        if start < previous_end:
            raise ValueError("accounted trades overlap; Apex audit requires the one-position execution model")
        previous_end = end

        advance_to_day(trading_day_id(trade.entry_timestamp), last_close_equity)
        balance -= trade.entry_cost_usd
        last_close_equity = balance
        check_equity(balance, trade.entry_timestamp, trade.trade_id)
        pv = point_value_usd(trade.instrument)

        for bar_position in range(start, end + 1):
            bar = bars[bar_position]
            bar_day = trading_day_id(bar.timestamp)
            advance_to_day(bar_day, last_close_equity)
            adverse_price = bar.low if trade.direction == "long" else bar.high
            close_points = (
                bar.close - trade.entry_price
                if trade.direction == "long"
                else trade.entry_price - bar.close
            )
            adverse_points = (
                adverse_price - trade.entry_price
                if trade.direction == "long"
                else trade.entry_price - adverse_price
            )
            adverse_equity = balance + adverse_points * pv * trade.quantity
            last_close_equity = balance + close_points * pv * trade.quantity
            check_equity(adverse_equity, bar.timestamp, trade.trade_id)

        if trade.exit_timestamp is not None:
            assert trade.gross_pnl_usd is not None
            balance += trade.gross_pnl_usd
            last_close_equity = balance
            check_equity(balance, trade.exit_timestamp, trade.trade_id)

    if current_day is not None:
        close_day(current_day, last_close_equity)
    return ApexEODAudit(
        config=config, breaches=tuple(breaches), trading_days=tuple(days),
        max_daily_loss_usd=max_daily_loss,
        largest_eod_drawdown_breach_usd=max_eod_breach,
        uses_mark_to_market=True,
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
