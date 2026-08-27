"""Post-hoc session and hour-of-day analysis of an already-executed run.

**This is a view, not a strategy, and not a backtest.** Nothing here re-runs
a simulation, re-detects an FVG, a liquidity level, a sweep or a bias, or
changes a single entry rule. It reads the trade-level rows a finished run
persisted (``variant_trades`` in ``tradelab.backtest_v2_journal``) and
recomputes the same metrics over a chosen subset of them.

**What that means for interpretation, stated where the code is rather than
only in the docs.** Selecting the session that performed best *in this
result* and calling it a filter is selection on the same data that produced
the result. It is a hypothesis. It has to be confirmed on a separate, unused
holdout before it may become a production rule, and the UI says so on every
render of this view.

**The equity model is the engine's own, not a simplification.** A filtered
curve is rebuilt with exactly ``tradelab.equity``'s two-event model -- the
$20 entry cost booked at the entry instant, gross P&L booked at the exit
instant, ordered by ``(timestamp, ENTRY_COST before EXIT_GROSS_PNL,
trade_id)`` -- so that selecting *every* session reproduces the run's own
headline P&L, max drawdown and ROI exactly, and any difference under a filter
is the filter, never a different accounting convention. The equivalence is
asserted directly in the tests, not assumed.

An unresolved (still-open) trade contributes its entry cost and no P&L, which
is again what the engine does.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from tradelab.trading_sessions import MAINTENANCE_BREAK_KEY, SESSION_KEYS, ny_hour, session_for

OPEN_UNRESOLVED = "open_unresolved"

# Mirrors tradelab.equity._EVENT_TYPE_ORDER: on an identical timestamp the
# entry cost is booked before the exit P&L.
_ENTRY_COST = 0
_EXIT_GROSS = 1


@dataclass(frozen=True)
class SessionFilter:
    """What subset of a variant's trades to analyse.

    ``sessions`` is a set of session keys; empty means "every session", which
    is how the unfiltered baseline is expressed. ``hour_from``/``hour_to`` are
    New York local hours, inclusive on both ends, and may wrap past midnight
    (``hour_from=22, hour_to=2`` means 22, 23, 0, 1, 2) -- the Asian session
    is not expressible otherwise.

    A trade is assigned by its **entry** instant. A trade that opens in London
    and closes in New York belongs to London: the decision to be in the market
    was made in London, and that is what a session filter is about.
    """

    sessions: frozenset[str] = frozenset()
    hour_from: int | None = None
    hour_to: int | None = None

    def __post_init__(self) -> None:
        unknown = self.sessions - set(SESSION_KEYS) - {MAINTENANCE_BREAK_KEY}
        if unknown:
            raise ValueError(f"unknown session key(s): {sorted(unknown)}")
        for name in ("hour_from", "hour_to"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 23:
                raise ValueError(f"SessionFilter.{name} must be a New York hour 0-23, got {value!r}")
        if (self.hour_from is None) != (self.hour_to is None):
            raise ValueError("SessionFilter hour_from and hour_to must be supplied together")

    @property
    def is_identity(self) -> bool:
        return not self.sessions and self.hour_from is None

    def _hour_matches(self, hour: int) -> bool:
        if self.hour_from is None:
            return True
        if self.hour_from <= self.hour_to:
            return self.hour_from <= hour <= self.hour_to
        return hour >= self.hour_from or hour <= self.hour_to

    def matches(self, trade: Mapping[str, Any]) -> bool:
        session = trade.get("entry_session") or session_for(_as_datetime(trade["entry_at"]))
        if self.sessions and session not in self.sessions:
            return False
        hour = trade.get("entry_hour_ny")
        if hour is None:
            hour = ny_hour(_as_datetime(trade["entry_at"]))
        return self._hour_matches(int(hour))


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def filter_trades(trades: Iterable[Mapping[str, Any]], selection: SessionFilter) -> list[Mapping[str, Any]]:
    return [trade for trade in trades if selection.matches(trade)]


def analyse(
    trades: Sequence[Mapping[str, Any]],
    *,
    initial_balance_usd: float = 50_000.0,
    drawdown_threshold_usd: float = 2_000.0,
) -> dict[str, Any]:
    """Recompute the run's metrics over exactly ``trades``.

    Returns trade count, net P&L, win rate, ROI, max drawdown, the count of
    drawdown episodes past the threshold, the equity curve, and a per-session
    and per-NY-hour breakdown of the same subset.
    """
    events: list[tuple[datetime, int, str, float]] = []
    for trade in trades:
        trade_id = str(trade.get("trade_id") or trade.get("trade_seq") or "")
        events.append((_as_datetime(trade["entry_at"]), _ENTRY_COST, trade_id, -float(trade.get("entry_cost_usd") or 0.0)))
        if trade.get("exit_reason") != OPEN_UNRESOLVED and trade.get("exit_at") is not None:
            events.append((_as_datetime(trade["exit_at"]), _EXIT_GROSS, trade_id, float(trade.get("gross_pnl_usd") or 0.0)))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    balance = initial_balance_usd
    peak = initial_balance_usd
    max_drawdown = 0.0
    episode_count = 0
    in_episode = False
    curve: list[dict[str, Any]] = []
    for timestamp, _order, _trade_id, delta in events:
        balance += delta
        if balance > peak:
            peak = balance
        drawdown = peak - balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if drawdown >= drawdown_threshold_usd:
            if not in_episode:
                episode_count += 1
                in_episode = True
        else:
            in_episode = False
        curve.append({"ts": timestamp.isoformat(), "equity": balance})

    resolved = [t for t in trades if t.get("net_pnl_usd") is not None]
    wins = sum(1 for t in resolved if float(t["net_pnl_usd"]) > 0)
    net_pnl = balance - initial_balance_usd

    return {
        "trade_count": len(trades),
        "resolved_trade_count": len(resolved),
        "net_pnl_usd": net_pnl,
        "roi": net_pnl / initial_balance_usd,
        "win_rate": (wins / len(resolved)) if resolved else None,
        "max_drawdown_usd": max_drawdown,
        "drawdown_over_2000_episode_count": episode_count,
        "final_balance_usd": balance,
        "equity_curve": curve,
        "by_session": _breakdown(trades, lambda t: t.get("entry_session") or session_for(_as_datetime(t["entry_at"]))),
        "by_hour_ny": _breakdown(
            trades,
            lambda t: str(t["entry_hour_ny"] if t.get("entry_hour_ny") is not None else ny_hour(_as_datetime(t["entry_at"]))),
        ),
    }


def _breakdown(trades: Sequence[Mapping[str, Any]], key) -> dict[str, dict[str, Any]]:
    """Trade count, net P&L and win rate per bucket.

    Deliberately *not* an equity curve per bucket: a per-bucket drawdown is
    only meaningful for a portfolio that traded only that bucket, which is
    what re-running ``analyse`` with that bucket selected gives. Reporting one
    here would invite reading it as such.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for trade in trades:
        bucket = buckets.setdefault(key(trade), {"trade_count": 0, "net_pnl_usd": 0.0, "wins": 0, "resolved": 0})
        bucket["trade_count"] += 1
        pnl = trade.get("net_pnl_usd")
        if pnl is not None:
            bucket["net_pnl_usd"] += float(pnl)
            bucket["resolved"] += 1
            if float(pnl) > 0:
                bucket["wins"] += 1
    for bucket in buckets.values():
        bucket["win_rate"] = (bucket["wins"] / bucket["resolved"]) if bucket["resolved"] else None
    return dict(sorted(buckets.items()))


#: Package-level aliases. ``analyse``/``filter_trades`` are the right names
#: inside this module and far too generic at the package level, where they sit
#: beside two dozen other detectors.
analyse_session_subset = analyse
filter_session_trades = filter_trades
