#!/usr/bin/env python3
"""Accounting/risk layer tests: NQ/MNQ point value and tick size,
closest-to-$500 contract sizing capped at $550 (including the case where
a larger, closer-to-target count would exceed $550 and must be rejected
in favor of a farther-but-valid one, and the exact-tie-favors-smaller
case), an instrument-dependent untradeable trade (NQ rejected, MNQ still
tradeable for the identical stop distance), long/short USD P&L for TP
and SL, the $20 entry cost charged immediately and identically regardless
of outcome (including unresolved trades, which have no exit P&L at all),
RiskConfig validation, and that apply_accounting never mutates its input
BacktestResult."""
from __future__ import annotations

import copy
import unittest
from datetime import datetime, timezone

from tradelab.backtest import BacktestResult, ExecutedTrade, TradeExitReason
from tradelab.strategy_spec import LONG, SHORT
from tradelab.accounting import (
    INSTRUMENT_TICK_SIZE,
    POINT_VALUE_USD,
    AccountBacktestResult,
    AccountedTrade,
    Instrument,
    RiskConfig,
    UntradeableReason,
    UntradeableTrade,
    apply_accounting,
    instrument_tick_size,
    point_value_usd,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)

DEFAULT_RISK_CONFIG = RiskConfig()


def executed_trade(
    trade_id: str = "t1",
    strategy_id: str = "s1",
    direction: str = LONG,
    entry_price: float = 100.0,
    sl_price: float = 95.0,
    tp_price: float = 110.0,
    exit_reason: TradeExitReason = TradeExitReason.TP_HIT,
    exit_price: float | None = 110.0,
    gross_pnl_points: float | None = 10.0,
) -> ExecutedTrade:
    return ExecutedTrade(
        trade_id=trade_id,
        strategy_id=strategy_id,
        direction=direction,
        signal_known_at=T0,
        entry_timestamp=T0,
        entry_price=entry_price,
        tp_price=tp_price,
        tp_source_type="liquidity",
        tp_source_id="tp-1",
        sl_price=sl_price,
        sl_source_type="liquidity",
        sl_source_id="sl-1",
        exit_timestamp=T0 if exit_price is not None else None,
        exit_price=exit_price,
        exit_reason=exit_reason,
        gross_pnl_points=gross_pnl_points,
    )


def bt_result(*trades: ExecutedTrade, strategy_id: str = "s1") -> BacktestResult:
    tp = sum(1 for t in trades if t.exit_reason == TradeExitReason.TP_HIT)
    sl = sum(1 for t in trades if t.exit_reason == TradeExitReason.SL_HIT)
    ou = sum(1 for t in trades if t.exit_reason == TradeExitReason.OPEN_UNRESOLVED)
    gross_total = sum(t.gross_pnl_points for t in trades if t.gross_pnl_points is not None)
    return BacktestResult(
        strategy_id=strategy_id,
        trades=list(trades),
        skipped_signals=[],
        trade_count=len(trades),
        tp_hit_count=tp,
        sl_hit_count=sl,
        open_unresolved_count=ou,
        gross_pnl_points_total=gross_total,
    )


class InstrumentTests(unittest.TestCase):
    def test_nq_point_value_and_tick_size(self):
        self.assertEqual(point_value_usd(Instrument.NQ), 20.0)
        self.assertEqual(instrument_tick_size(Instrument.NQ), 0.25)

    def test_mnq_point_value_and_tick_size(self):
        self.assertEqual(point_value_usd(Instrument.MNQ), 2.0)
        self.assertEqual(instrument_tick_size(Instrument.MNQ), 0.25)

    def test_only_nq_and_mnq_are_defined(self):
        self.assertEqual(set(Instrument), {Instrument.NQ, Instrument.MNQ})
        self.assertEqual(set(POINT_VALUE_USD), {Instrument.NQ, Instrument.MNQ})
        self.assertEqual(set(INSTRUMENT_TICK_SIZE), {Instrument.NQ, Instrument.MNQ})


class RiskConfigTests(unittest.TestCase):
    def test_defaults_are_500_550_20_and_30_contract_cap(self):
        config = RiskConfig()
        self.assertEqual(config.target_risk_usd, 500.0)
        self.assertEqual(config.max_stop_risk_usd, 550.0)
        self.assertEqual(config.entry_cost_usd, 20.0)
        self.assertEqual(config.max_contracts, 30)

    def test_non_positive_target_risk_rejected(self):
        with self.assertRaises(ValueError):
            RiskConfig(target_risk_usd=0)
        with self.assertRaises(ValueError):
            RiskConfig(target_risk_usd=-500)

    def test_non_positive_max_stop_risk_rejected(self):
        with self.assertRaises(ValueError):
            RiskConfig(max_stop_risk_usd=0)

    def test_non_positive_entry_cost_rejected(self):
        with self.assertRaises(ValueError):
            RiskConfig(entry_cost_usd=0)
        with self.assertRaises(ValueError):
            RiskConfig(entry_cost_usd=-20)

    def test_non_positive_or_non_integer_contract_cap_rejected(self):
        for cap in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                RiskConfig(max_contracts=cap)

    def test_target_risk_above_max_stop_risk_rejected(self):
        with self.assertRaises(ValueError):
            RiskConfig(target_risk_usd=600, max_stop_risk_usd=550)

    def test_target_risk_equal_to_max_stop_risk_is_accepted(self):
        RiskConfig(target_risk_usd=550, max_stop_risk_usd=550)  # no raise


class QuantitySelectionTests(unittest.TestCase):
    def test_chooses_quantity_closest_to_target_risk(self):
        # risk_per_contract=50 (NQ, 2.5pt stop): 10 contracts = $500 exactly.
        trade = executed_trade(entry_price=100.0, sl_price=97.5)
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        self.assertEqual(result.accounted_trades[0].quantity, 10)
        self.assertEqual(result.accounted_trades[0].planned_risk_usd, 500.0)

    def test_a_larger_count_closer_to_target_but_over_550_is_rejected(self):
        # risk_per_contract=137.6 (NQ, 6.88pt stop): 4 contracts = $550.4
        # (closer to 500 than 3 contracts) but exceeds the $550 ceiling,
        # so 3 contracts ($412.8, farther from target) must be chosen
        # instead -- never rounded up past $550.
        trade = executed_trade(entry_price=100.0, sl_price=93.12)
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        accounted = result.accounted_trades[0]
        self.assertEqual(accounted.quantity, 3)
        self.assertAlmostEqual(accounted.planned_risk_usd, 412.8, places=6)
        self.assertLessEqual(accounted.planned_risk_usd, 550.0)

    def test_exact_tie_between_two_counts_favors_the_smaller(self):
        # risk_per_contract = 1000/21 (MNQ): 10 contracts (~$476.19) and
        # 11 contracts (~$523.81) are exactly equidistant from $500 (both
        # ~$23.81 away) and both stay under $550 -- 10 (the smaller) must
        # win.
        risk_per_contract = 1000.0 / 21.0
        sl_distance_points = risk_per_contract / point_value_usd(Instrument.MNQ)
        trade = executed_trade(entry_price=100.0, sl_price=100.0 - sl_distance_points)
        result = apply_accounting(bt_result(trade), Instrument.MNQ, DEFAULT_RISK_CONFIG)
        self.assertEqual(result.accounted_trades[0].quantity, 10)

    def test_never_rounds_up_past_max_stop_risk(self):
        for _ in range(1):
            trade = executed_trade(entry_price=100.0, sl_price=93.12)
            result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
            self.assertLessEqual(result.accounted_trades[0].planned_risk_usd, 550.0)

    def test_mnq_position_is_capped_at_30_instead_of_being_skipped(self):
        # A 1pt MNQ stop risks $2/contract: $500 target would need 250
        # contracts. The setup remains tradeable at the explicit 30-MNQ cap.
        trade = executed_trade(entry_price=100.0, sl_price=99.0)
        result = apply_accounting(bt_result(trade), Instrument.MNQ, DEFAULT_RISK_CONFIG)
        accounted = result.accounted_trades[0]
        self.assertEqual(accounted.quantity, 30)
        self.assertEqual(accounted.planned_risk_usd, 60.0)


class UntradeableTests(unittest.TestCase):
    def test_nq_untradeable_with_wide_stop_mnq_still_tradeable(self):
        # 30pt stop: NQ risk_per_contract = 30*20 = $600 > $550 (untradeable
        # even at 1 contract); MNQ risk_per_contract = 30*2 = $60 (8
        # contracts = $480, nearest to $500 without exceeding $550).
        trade = executed_trade(entry_price=100.0, sl_price=70.0)

        nq_result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        self.assertEqual(nq_result.trade_count, 0)
        self.assertEqual(nq_result.untradeable_count, 1)
        self.assertEqual(nq_result.untradeable_trades[0].reason, UntradeableReason.STOP_RISK_EXCEEDS_MAX)
        self.assertEqual(nq_result.untradeable_trades[0].risk_per_contract_usd, 600.0)

        mnq_result = apply_accounting(bt_result(trade), Instrument.MNQ, DEFAULT_RISK_CONFIG)
        self.assertEqual(mnq_result.trade_count, 1)
        self.assertEqual(mnq_result.untradeable_count, 0)
        self.assertEqual(mnq_result.accounted_trades[0].quantity, 8)

    def test_untradeable_trade_is_not_charged_the_entry_cost(self):
        trade = executed_trade(entry_price=100.0, sl_price=70.0)
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        self.assertEqual(result.entry_costs_charged_usd_total, 0.0)


class ProfitAndLossTests(unittest.TestCase):
    def test_long_gross_and_net_pnl_usd(self):
        # long, 10pt gain, quantity from a 5pt/$100-per-contract stop (NQ) -> 5 contracts closest to 500.
        trade = executed_trade(
            direction=LONG, entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.TP_HIT,
            exit_price=110.0, gross_pnl_points=10.0,
        )
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        accounted = result.accounted_trades[0]
        self.assertEqual(accounted.quantity, 5)  # risk_per_contract=100, 5*100=500 exact
        self.assertEqual(accounted.gross_pnl_usd, 10.0 * 20.0 * 5)  # 1000.0
        self.assertEqual(accounted.net_pnl_usd, 1000.0 - 20.0)

    def test_short_gross_and_net_pnl_usd(self):
        trade = executed_trade(
            direction=SHORT, entry_price=100.0, sl_price=105.0, exit_reason=TradeExitReason.TP_HIT,
            exit_price=90.0, gross_pnl_points=10.0,
        )
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        accounted = result.accounted_trades[0]
        self.assertEqual(accounted.quantity, 5)  # risk_per_contract=100
        self.assertEqual(accounted.gross_pnl_usd, 1000.0)
        self.assertEqual(accounted.net_pnl_usd, 980.0)

    def test_sl_hit_produces_negative_pnl(self):
        trade = executed_trade(
            direction=LONG, entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.SL_HIT,
            exit_price=95.0, gross_pnl_points=-5.0,
        )
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        accounted = result.accounted_trades[0]
        self.assertEqual(accounted.gross_pnl_usd, -5.0 * 20.0 * 5)  # -500.0
        self.assertEqual(accounted.net_pnl_usd, -500.0 - 20.0)  # -520.0: SL risk + entry cost

    def test_realized_totals_sum_only_over_accounted_trades(self):
        t1 = executed_trade(trade_id="t1", entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.TP_HIT, exit_price=110.0, gross_pnl_points=10.0)
        t2 = executed_trade(trade_id="t2", entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.SL_HIT, exit_price=95.0, gross_pnl_points=-5.0)
        result = apply_accounting(bt_result(t1, t2), Instrument.NQ, DEFAULT_RISK_CONFIG)
        self.assertEqual(result.realized_gross_pnl_usd_total, 1000.0 + (-500.0))
        self.assertEqual(result.realized_net_pnl_usd_total, (1000.0 - 20.0) + (-500.0 - 20.0))


class EntryCostTests(unittest.TestCase):
    def test_entry_cost_is_20_for_every_accounted_trade(self):
        for exit_reason, exit_price, gross_points in (
            (TradeExitReason.TP_HIT, 110.0, 10.0),
            (TradeExitReason.SL_HIT, 95.0, -5.0),
            (TradeExitReason.OPEN_UNRESOLVED, None, None),
        ):
            trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=exit_reason, exit_price=exit_price, gross_pnl_points=gross_points)
            result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
            self.assertEqual(result.accounted_trades[0].entry_cost_usd, 20.0)

    def test_20_is_deducted_from_net_pnl_on_both_tp_and_sl(self):
        tp_trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.TP_HIT, exit_price=110.0, gross_pnl_points=10.0)
        sl_trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.SL_HIT, exit_price=95.0, gross_pnl_points=-5.0)
        tp_result = apply_accounting(bt_result(tp_trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        sl_result = apply_accounting(bt_result(sl_trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        tp_accounted = tp_result.accounted_trades[0]
        sl_accounted = sl_result.accounted_trades[0]
        self.assertEqual(tp_accounted.net_pnl_usd, tp_accounted.gross_pnl_usd - 20.0)
        self.assertEqual(sl_accounted.net_pnl_usd, sl_accounted.gross_pnl_usd - 20.0)

    def test_unresolved_trade_has_no_exit_pnl_but_entry_cost_is_still_charged(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.OPEN_UNRESOLVED, exit_price=None, gross_pnl_points=None)
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        accounted = result.accounted_trades[0]
        self.assertIsNone(accounted.gross_pnl_usd)
        self.assertIsNone(accounted.net_pnl_usd)
        self.assertEqual(accounted.entry_cost_usd, 20.0)
        self.assertEqual(result.open_unresolved_count, 1)

    def test_entry_costs_total_includes_unresolved_trades(self):
        closed = executed_trade(trade_id="t1", entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.TP_HIT, exit_price=110.0, gross_pnl_points=10.0)
        unresolved = executed_trade(trade_id="t2", entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.OPEN_UNRESOLVED, exit_price=None, gross_pnl_points=None)
        result = apply_accounting(bt_result(closed, unresolved), Instrument.NQ, DEFAULT_RISK_CONFIG)
        self.assertEqual(result.entry_costs_charged_usd_total, 40.0)  # $20 each, both charged
        self.assertEqual(result.realized_gross_pnl_usd_total, 1000.0)  # unresolved contributes nothing here


class MaxLossCapTests(unittest.TestCase):
    def test_worst_case_sl_loss_is_up_to_550_plus_20_entry_cost(self):
        # 5pt stop, NQ: risk_per_contract=100, 5 contracts is closest to
        # $500 (exactly $500) -- worst SL loss here is $500 gross + $20
        # entry cost = $520 net loss, i.e. within the documented $570 cap.
        trade = executed_trade(direction=LONG, entry_price=100.0, sl_price=95.0, exit_reason=TradeExitReason.SL_HIT, exit_price=95.0, gross_pnl_points=-5.0)
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        accounted = result.accounted_trades[0]
        max_possible_net_loss = -(DEFAULT_RISK_CONFIG.max_stop_risk_usd + DEFAULT_RISK_CONFIG.entry_cost_usd)
        self.assertGreaterEqual(accounted.net_pnl_usd, max_possible_net_loss)  # -520 >= -570


class ImmutabilityTests(unittest.TestCase):
    def test_apply_accounting_does_not_modify_the_input_backtest_result(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0)
        original = bt_result(trade)
        snapshot = copy.deepcopy(original)

        apply_accounting(original, Instrument.NQ, DEFAULT_RISK_CONFIG)

        self.assertEqual(original, snapshot)
        self.assertEqual(original.trades[0].entry_price, 100.0)
        self.assertEqual(original.trades[0].sl_price, 95.0)


class ResultShapeTests(unittest.TestCase):
    def test_result_carries_strategy_id_instrument_and_risk_config(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0)
        result = apply_accounting(bt_result(trade, strategy_id="my-strategy"), Instrument.MNQ, DEFAULT_RISK_CONFIG)
        self.assertIsInstance(result, AccountBacktestResult)
        self.assertEqual(result.strategy_id, "my-strategy")
        self.assertEqual(result.instrument, Instrument.MNQ)
        self.assertEqual(result.risk_config, DEFAULT_RISK_CONFIG)

    def test_quantity_is_always_a_positive_integer(self):
        trade = executed_trade(entry_price=100.0, sl_price=95.0)
        result = apply_accounting(bt_result(trade), Instrument.NQ, DEFAULT_RISK_CONFIG)
        quantity = result.accounted_trades[0].quantity
        self.assertIsInstance(quantity, int)
        self.assertGreater(quantity, 0)

    def test_accounted_trade_and_untradeable_trade_are_distinct_types(self):
        tradeable = executed_trade(trade_id="t1", entry_price=100.0, sl_price=95.0)
        untradeable = executed_trade(trade_id="t2", entry_price=100.0, sl_price=70.0)
        result = apply_accounting(bt_result(tradeable, untradeable), Instrument.NQ, DEFAULT_RISK_CONFIG)
        self.assertEqual(len(result.accounted_trades), 1)
        self.assertEqual(len(result.untradeable_trades), 1)
        self.assertIsInstance(result.accounted_trades[0], AccountedTrade)
        self.assertIsInstance(result.untradeable_trades[0], UntradeableTrade)


if __name__ == "__main__":
    unittest.main()
