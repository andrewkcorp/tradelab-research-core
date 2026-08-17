#!/usr/bin/env python3
"""Equity/drawdown tests: winning and losing trades ($20 entry cost,
then gross P&L at exit, never subtracting the cost twice), unresolved
trades (entry cost only, no invented exit P&L), untradeable trades
(zero equity events), same-timestamp entry/exit ordering (cost always
first), running peak/drawdown bookkeeping, a single long drawdown
episode counted once, two disjoint episodes counted twice, an exactly-
$2000 drawdown never counting as "over" the threshold, EquityConfig
validation, and that build_equity_curve never mutates its input
AccountBacktestResult."""
from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.backtest import BacktestResult, ExecutedTrade, TradeExitReason
from tradelab.accounting import Instrument, RiskConfig, UntradeableReason, apply_accounting
from tradelab.equity import EquityConfig, EquityEventType, build_equity_curve

UTC = timezone.utc
T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
DEFAULT_RISK_CONFIG = RiskConfig()
DEFAULT_EQUITY_CONFIG = EquityConfig()


def executed_trade(
    trade_id: str = "t1",
    entry_price: float = 100.0,
    sl_price: float = 95.0,
    tp_price: float = 110.0,
    exit_reason: TradeExitReason = TradeExitReason.TP_HIT,
    exit_price: float | None = 110.0,
    gross_pnl_points: float | None = 10.0,
    entry_timestamp: datetime = T0,
    exit_timestamp: datetime | None = None,
) -> ExecutedTrade:
    if exit_timestamp is None and exit_price is not None:
        exit_timestamp = entry_timestamp + timedelta(minutes=5)
    return ExecutedTrade(
        trade_id=trade_id,
        strategy_id="s1",
        direction="long",
        signal_known_at=entry_timestamp,
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        tp_price=tp_price,
        tp_source_type="liquidity",
        tp_source_id="tp-1",
        sl_price=sl_price,
        sl_source_type="liquidity",
        sl_source_id="sl-1",
        exit_timestamp=exit_timestamp,
        exit_price=exit_price,
        exit_reason=exit_reason,
        gross_pnl_points=gross_pnl_points,
    )


def bt_result(*trades: ExecutedTrade) -> BacktestResult:
    return BacktestResult(
        strategy_id="s1",
        trades=list(trades),
        skipped_signals=[],
        trade_count=len(trades),
        tp_hit_count=sum(1 for t in trades if t.exit_reason == TradeExitReason.TP_HIT),
        sl_hit_count=sum(1 for t in trades if t.exit_reason == TradeExitReason.SL_HIT),
        open_unresolved_count=sum(1 for t in trades if t.exit_reason == TradeExitReason.OPEN_UNRESOLVED),
        gross_pnl_points_total=sum(t.gross_pnl_points for t in trades if t.gross_pnl_points is not None),
    )


def account(*trades: ExecutedTrade, instrument: Instrument = Instrument.NQ, risk_config: RiskConfig = DEFAULT_RISK_CONFIG):
    return apply_accounting(bt_result(*trades), instrument, risk_config)


class EquityConfigTests(unittest.TestCase):
    def test_defaults_are_50000_and_2000(self):
        config = EquityConfig()
        self.assertEqual(config.initial_balance_usd, 50_000.0)
        self.assertEqual(config.drawdown_threshold_usd, 2_000.0)

    def test_non_positive_initial_balance_rejected(self):
        with self.assertRaises(ValueError):
            EquityConfig(initial_balance_usd=0)
        with self.assertRaises(ValueError):
            EquityConfig(initial_balance_usd=-50_000)

    def test_non_positive_drawdown_threshold_rejected(self):
        with self.assertRaises(ValueError):
            EquityConfig(drawdown_threshold_usd=0)
        with self.assertRaises(ValueError):
            EquityConfig(drawdown_threshold_usd=-2_000)


class WinningTradeTests(unittest.TestCase):
    def test_entry_cost_then_gross_pnl_and_correct_final_balance(self):
        # 5pt stop, NQ -> risk_per_contract=100 -> quantity=5 (closest to $500).
        # 10pt gain -> gross_pnl_usd = 10 * 20 * 5 = 1000.
        trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.TP_HIT, exit_price=110.0, gross_pnl_points=10.0)
        result = build_equity_curve(account(trade), DEFAULT_EQUITY_CONFIG)

        self.assertEqual(len(result.equity_points), 2)
        entry_point, exit_point = result.equity_points
        self.assertEqual(entry_point.event_type, EquityEventType.ENTRY_COST)
        self.assertEqual(entry_point.cash_delta_usd, -20.0)
        self.assertEqual(entry_point.balance_usd, 49_980.0)
        self.assertEqual(exit_point.event_type, EquityEventType.EXIT_GROSS_PNL)
        self.assertEqual(exit_point.cash_delta_usd, 1000.0)
        self.assertEqual(exit_point.balance_usd, 50_980.0)

        self.assertEqual(result.final_balance_usd, 50_980.0)
        self.assertEqual(result.net_cash_pnl_usd, 980.0)  # +1000 gross - 20 cost, never subtracted twice


class LosingTradeTests(unittest.TestCase):
    def test_entry_cost_then_negative_gross_pnl(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.SL_HIT, exit_price=95.0, gross_pnl_points=-5.0)
        result = build_equity_curve(account(trade), DEFAULT_EQUITY_CONFIG)

        entry_point, exit_point = result.equity_points
        self.assertEqual(entry_point.cash_delta_usd, -20.0)
        self.assertEqual(exit_point.cash_delta_usd, -500.0)  # -5 * 20 * 5
        self.assertEqual(result.final_balance_usd, 50_000.0 - 20.0 - 500.0)
        self.assertEqual(result.net_cash_pnl_usd, -520.0)


class UnresolvedTradeTests(unittest.TestCase):
    def test_only_entry_cost_event_no_invented_exit_pnl(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.OPEN_UNRESOLVED, exit_price=None, gross_pnl_points=None)
        result = build_equity_curve(account(trade), DEFAULT_EQUITY_CONFIG)

        self.assertEqual(len(result.equity_points), 1)
        self.assertEqual(result.equity_points[0].event_type, EquityEventType.ENTRY_COST)
        self.assertEqual(result.equity_points[0].cash_delta_usd, -20.0)
        self.assertEqual(result.final_balance_usd, 49_980.0)
        self.assertEqual(result.net_cash_pnl_usd, -20.0)


class UntradeableTradeTests(unittest.TestCase):
    def test_no_equity_events_at_all(self):
        trade = executed_trade(entry_price=100.0, sl_price=70.0)  # 30pt stop, NQ -> $600/contract > $550
        acc = account(trade)
        self.assertEqual(acc.untradeable_trades[0].reason, UntradeableReason.STOP_RISK_EXCEEDS_MAX)
        result = build_equity_curve(acc, DEFAULT_EQUITY_CONFIG)
        self.assertEqual(result.equity_points, [])
        self.assertEqual(result.final_balance_usd, 50_000.0)
        self.assertEqual(result.net_cash_pnl_usd, 0.0)


class SameTimestampOrderingTests(unittest.TestCase):
    def test_entry_cost_always_processed_before_exit_gross_pnl_at_the_same_instant(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.TP_HIT, exit_price=110.0, gross_pnl_points=10.0, entry_timestamp=T0, exit_timestamp=T0)
        result = build_equity_curve(account(trade), DEFAULT_EQUITY_CONFIG)
        self.assertEqual([p.event_type for p in result.equity_points], [EquityEventType.ENTRY_COST, EquityEventType.EXIT_GROSS_PNL])
        self.assertEqual(result.equity_points[0].sequence, 0)
        self.assertEqual(result.equity_points[1].sequence, 1)

    def test_ordering_holds_across_two_different_trades_sharing_an_instant(self):
        # trade A's exit lands on the exact same instant trade B's entry does.
        trade_a = executed_trade(trade_id="a", entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.TP_HIT, exit_price=110.0, gross_pnl_points=10.0, entry_timestamp=T0, exit_timestamp=T0 + timedelta(minutes=5))
        trade_b = executed_trade(trade_id="b", entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.TP_HIT, exit_price=110.0, gross_pnl_points=5.0, entry_timestamp=T0 + timedelta(minutes=5), exit_timestamp=T0 + timedelta(minutes=10))
        result = build_equity_curve(account(trade_a, trade_b), DEFAULT_EQUITY_CONFIG)
        shared_instant_events = [p for p in result.equity_points if p.timestamp == T0 + timedelta(minutes=5)]
        self.assertEqual(len(shared_instant_events), 2)
        self.assertEqual(shared_instant_events[0].event_type, EquityEventType.ENTRY_COST)
        self.assertEqual(shared_instant_events[0].trade_id, "b")
        self.assertEqual(shared_instant_events[1].event_type, EquityEventType.EXIT_GROSS_PNL)
        self.assertEqual(shared_instant_events[1].trade_id, "a")


class RunningPeakAndDrawdownTests(unittest.TestCase):
    def test_running_peak_starts_at_initial_balance_and_never_decreases(self):
        losing = executed_trade(trade_id="t1", entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.SL_HIT, exit_price=95.0, gross_pnl_points=-5.0, entry_timestamp=T0, exit_timestamp=T0 + timedelta(minutes=1))
        result = build_equity_curve(account(losing), DEFAULT_EQUITY_CONFIG)
        for point in result.equity_points:
            self.assertEqual(point.running_peak_usd, 50_000.0)  # never below the starting balance
            self.assertGreaterEqual(point.running_peak_usd, point.balance_usd)

    def test_drawdown_is_peak_minus_balance_at_every_point(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.SL_HIT, exit_price=95.0, gross_pnl_points=-5.0)
        result = build_equity_curve(account(trade), DEFAULT_EQUITY_CONFIG)
        for point in result.equity_points:
            self.assertEqual(point.drawdown_usd, point.running_peak_usd - point.balance_usd)

    def test_max_drawdown_is_the_largest_drawdown_seen(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.SL_HIT, exit_price=95.0, gross_pnl_points=-5.0)
        result = build_equity_curve(account(trade), DEFAULT_EQUITY_CONFIG)
        self.assertEqual(result.max_drawdown_usd, max(p.drawdown_usd for p in result.equity_points))


class DrawdownEpisodeTests(unittest.TestCase):
    def _trade(self, trade_id, minute_offset, gross_pnl_points):
        entry_ts = T0 + timedelta(minutes=minute_offset)
        exit_ts = entry_ts + timedelta(minutes=1)
        exit_reason = TradeExitReason.TP_HIT if gross_pnl_points > 0 else TradeExitReason.SL_HIT
        return executed_trade(
            trade_id=trade_id, entry_price=100.0, sl_price=95.0, exit_reason=exit_reason,
            exit_price=100.0 + gross_pnl_points, gross_pnl_points=gross_pnl_points,
            entry_timestamp=entry_ts, exit_timestamp=exit_ts,
        )

    def test_a_single_sustained_episode_counts_once(self):
        # NQ, 5 contracts (from the fixed 5pt stop): -25pt then -5pt both
        # keep drawdown above $2000 throughout -- one continuous episode.
        t1 = self._trade("t1", 0, -25.0)  # -2500 usd
        t2 = self._trade("t2", 2, -5.0)  # additional -500 usd, still over threshold
        result = build_equity_curve(account(t1, t2), DEFAULT_EQUITY_CONFIG)
        self.assertEqual(result.drawdown_over_threshold_episode_count, 1)

    def test_two_disjoint_episodes_count_twice(self):
        t1 = self._trade("t1", 0, -30.0)  # -3000 usd -> breach #1
        t2 = self._trade("t2", 2, 40.0)  # +4000 usd -> recovers back under threshold
        t3 = self._trade("t3", 4, -30.0)  # -3000 usd -> breach #2, a fresh episode
        result = build_equity_curve(account(t1, t2, t3), DEFAULT_EQUITY_CONFIG)
        self.assertEqual(result.drawdown_over_threshold_episode_count, 2)

    def test_drawdown_of_exactly_2000_is_not_over_threshold(self):
        # entry cost (-20) + exit (-1980) = exactly -2000 drawdown.
        trade = self._trade("t1", 0, -19.8)  # -19.8 * 20 * 5 = -1980
        acc = account(trade)
        result = build_equity_curve(acc, DEFAULT_EQUITY_CONFIG)
        self.assertEqual(result.equity_points[-1].drawdown_usd, 2_000.0)
        self.assertEqual(result.drawdown_over_threshold_episode_count, 0)

    def test_drawdown_one_cent_over_2000_counts_as_an_episode(self):
        trade = self._trade("t1", 0, -19.801)  # drawdown fractionally over 2000
        result = build_equity_curve(account(trade), DEFAULT_EQUITY_CONFIG)
        self.assertGreater(result.equity_points[-1].drawdown_usd, 2_000.0)
        self.assertEqual(result.drawdown_over_threshold_episode_count, 1)


class ConfigValidationRejectionTests(unittest.TestCase):
    def test_invalid_equity_config_never_reaches_build_equity_curve(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0)
        with self.assertRaises(ValueError):
            build_equity_curve(account(trade), EquityConfig(initial_balance_usd=-1))


class ImmutabilityTests(unittest.TestCase):
    def test_build_equity_curve_does_not_modify_the_input_account_result(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0)
        original = account(trade)
        snapshot = copy.deepcopy(original)

        build_equity_curve(original, DEFAULT_EQUITY_CONFIG)

        self.assertEqual(original, snapshot)
        self.assertEqual(original.accounted_trades[0].entry_price, 100.0)


class FullSuiteScenarioTests(unittest.TestCase):
    def test_mixed_tp_sl_unresolved_untradeable_produces_consistent_result(self):
        tp = self._t("tp", 0, TradeExitReason.TP_HIT, 10.0)
        sl = self._t("sl", 10, TradeExitReason.SL_HIT, -5.0)
        unresolved = self._t("unresolved", 20, TradeExitReason.OPEN_UNRESOLVED, None)
        untradeable = executed_trade(trade_id="untradeable", entry_price=100.0, sl_price=70.0, entry_timestamp=T0 + timedelta(minutes=30))

        result = build_equity_curve(account(tp, sl, unresolved, untradeable), DEFAULT_EQUITY_CONFIG)

        # untradeable never opens -> no events for it at all.
        self.assertTrue(all(p.trade_id != "untradeable" for p in result.equity_points))
        # 2 events each for tp/sl, 1 for unresolved.
        self.assertEqual(len(result.equity_points), 5)
        self.assertEqual(result.final_balance_usd, 50_000.0 - 20.0 + 1000.0 - 20.0 - 500.0 - 20.0)

    def _t(self, trade_id, minute_offset, exit_reason, gross_pnl_points):
        entry_ts = T0 + timedelta(minutes=minute_offset)
        exit_price = None if gross_pnl_points is None else 100.0 + gross_pnl_points
        return executed_trade(
            trade_id=trade_id, entry_price=100.0, sl_price=95.0, exit_reason=exit_reason,
            exit_price=exit_price, gross_pnl_points=gross_pnl_points, entry_timestamp=entry_ts,
        )


if __name__ == "__main__":
    unittest.main()
