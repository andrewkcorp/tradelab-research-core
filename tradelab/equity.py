"""Equity curve and drawdown: a realized, cash-based balance timeline
built on top of ``tradelab.accounting``'s already-decided ``AccountBack
testResult`` -- the first module in this package that tracks account
balance over time at all.

**Deliberately cash-based, not mark-to-market.** This module only ever
moves the account balance at two instants per trade: the moment its
entry cost is charged, and (for a trade that actually closed) the
moment its gross P&L is realized at exit. It never marks an open
position to market against a candle's high/low while the position is
still running -- there is no "unrealized drawdown while a trade is
open" concept here at all. A future module may add that; this one
intentionally does not, so every number here is a plain fact about cash
that has actually moved, never a projection of what a still-open
position is *currently* worth.

Not a new backtester and not a new accounting pass: this module never
re-runs ``tradelab.backtest.backtest_strategy`` or ``tradelab.
accounting.apply_accounting`` and never second-guesses a single sizing
or P&L decision either of them already made. It only takes ``Account
BacktestResult``'s already-computed ``AccountedTrade`` records and
replays their cash effects, in time order, against a starting account
balance.

**Cash-flow order per trade** (this task's own spec):

- **Untradeable trades create no equity event at all** -- sizing
  already rejected them before any position, and therefore any cash
  effect, was ever possible (``tradelab.accounting``'s own contract:
  an untradeable trade is never charged the entry cost either).
- **Every accounted trade's entry charges exactly ``-entry_cost_usd``**
  at its own ``entry_timestamp`` -- the cost is taken the instant the
  position opens, regardless of how (or whether) it later closes.
- **A trade that actually closed** (``exit_reason`` is not
  ``OPEN_UNRESOLVED``) additionally credits exactly ``+gross_pnl_usd``
  at its own ``exit_timestamp``. The entry cost is **never** subtracted
  a second time here -- ``AccountedTrade.net_pnl_usd`` (gross minus the
  entry cost, from ``tradelab.accounting``) already exists for anyone
  who wants the net figure for a single trade; this module's cash flow
  keeps the cost and the gross P&L as two separate ledger entries
  instead, since that is what actually happened, in that order, at two
  different instants.
- **An unresolved trade contributes only its entry-cost event** -- the
  cost already left the account; there is no invented exit P&L to add,
  ever.
- **When an entry and an exit land on the exact same timestamp** (a
  same-instant entry/exit from two different trades, or -- in a future
  multi-strategy context -- overlapping trades), the entry cost is
  always processed first, deterministically, never left to depend on
  input order or an unstable sort.

**``running_peak``/``drawdown``**: ``running_peak`` is the highest
balance observed so far, *including* ``initial_balance_usd`` itself as
the peak's own starting value (so a losing first trade immediately
shows a drawdown off the $50,000 starting balance, not off some lower
number). ``drawdown = running_peak - balance`` at every single event,
never only at trade boundaries.

**Drawdown-over-threshold episodes**: an episode is counted on the
*transition* from ``drawdown <= drawdown_threshold_usd`` to ``drawdown
> drawdown_threshold_usd`` -- exactly one count per rising edge across
that boundary. Every subsequent point that stays above the threshold
(however many events that takes) is still the same episode; a later
recovery back to ``drawdown <= drawdown_threshold_usd`` followed by a
fresh breach starts a genuinely new episode. The state before the first
event is treated as ``drawdown = 0`` (i.e. not over threshold), so a
breach on the very first event still counts as one episode. A drawdown
of *exactly* ``drawdown_threshold_usd`` is not "over" it -- the
comparison is strictly greater-than, matching "$2000 to nie jest
większe niż $2000" literally.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tradelab.accounting import AccountBacktestResult, AccountedTrade
from tradelab.backtest import TradeExitReason


class EquityEventType(str, Enum):
    ENTRY_COST = "entry_cost"
    EXIT_GROSS_PNL = "exit_gross_pnl"


_EVENT_TYPE_ORDER = {EquityEventType.ENTRY_COST: 0, EquityEventType.EXIT_GROSS_PNL: 1}


@dataclass(frozen=True)
class EquityConfig:
    initial_balance_usd: float = 50_000.0
    drawdown_threshold_usd: float = 2_000.0

    def __post_init__(self) -> None:
        if self.initial_balance_usd <= 0:
            raise ValueError(f"EquityConfig.initial_balance_usd must be positive, got {self.initial_balance_usd!r}")
        if self.drawdown_threshold_usd <= 0:
            raise ValueError(f"EquityConfig.drawdown_threshold_usd must be positive, got {self.drawdown_threshold_usd!r}")


@dataclass(frozen=True)
class EquityPoint:
    """One cash-flow event on the equity timeline. ``sequence`` is this
    event's own 0-based position in the deterministic processed order
    (time, then ``ENTRY_COST`` before ``EXIT_GROSS_PNL`` on a tie, then
    ``trade_id``) -- an audit field, not an input a caller supplies."""

    timestamp: datetime
    sequence: int
    trade_id: str
    event_type: EquityEventType
    cash_delta_usd: float
    balance_usd: float
    running_peak_usd: float
    drawdown_usd: float


@dataclass(frozen=True)
class EquityResult:
    equity_points: list[EquityPoint]
    final_balance_usd: float
    net_cash_pnl_usd: float
    max_drawdown_usd: float
    drawdown_over_threshold_episode_count: int


def build_equity_curve(account_result: AccountBacktestResult, equity_config: EquityConfig) -> EquityResult:
    """Replay ``account_result``'s already-decided trades as a realized,
    cash-based equity timeline -- never mutates ``account_result`` (its
    own dataclasses are already frozen; this function only ever reads
    from it)."""
    raw_events = _collect_events(account_result.accounted_trades)
    raw_events.sort(key=lambda e: (e[0], _EVENT_TYPE_ORDER[e[1]], e[2]))

    balance = equity_config.initial_balance_usd
    running_peak = equity_config.initial_balance_usd
    max_drawdown = 0.0
    equity_points: list[EquityPoint] = []
    episode_count = 0
    was_over_threshold = False

    for sequence, (timestamp, event_type, trade_id, cash_delta) in enumerate(raw_events):
        balance += cash_delta
        running_peak = max(running_peak, balance)
        drawdown = running_peak - balance
        max_drawdown = max(max_drawdown, drawdown)

        is_over_threshold = drawdown > equity_config.drawdown_threshold_usd
        if is_over_threshold and not was_over_threshold:
            episode_count += 1
        was_over_threshold = is_over_threshold

        equity_points.append(
            EquityPoint(
                timestamp=timestamp,
                sequence=sequence,
                trade_id=trade_id,
                event_type=event_type,
                cash_delta_usd=cash_delta,
                balance_usd=balance,
                running_peak_usd=running_peak,
                drawdown_usd=drawdown,
            )
        )

    return EquityResult(
        equity_points=equity_points,
        final_balance_usd=balance,
        net_cash_pnl_usd=balance - equity_config.initial_balance_usd,
        max_drawdown_usd=max_drawdown,
        drawdown_over_threshold_episode_count=episode_count,
    )


def _collect_events(accounted_trades: list[AccountedTrade]) -> list[tuple[datetime, EquityEventType, str, float]]:
    events: list[tuple[datetime, EquityEventType, str, float]] = []
    for trade in accounted_trades:
        events.append((trade.entry_timestamp, EquityEventType.ENTRY_COST, trade.trade_id, -trade.entry_cost_usd))
        if trade.exit_reason != TradeExitReason.OPEN_UNRESOLVED:
            events.append((trade.exit_timestamp, EquityEventType.EXIT_GROSS_PNL, trade.trade_id, trade.gross_pnl_usd))
    return events
