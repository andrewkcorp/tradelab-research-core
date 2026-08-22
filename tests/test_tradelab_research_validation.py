from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tradelab.accounting import AccountBacktestResult, AccountedTrade, Instrument, RiskConfig
from tradelab.backtest import TradeExitReason
from tradelab.research_validation import (
    benjamini_hochberg,
    deflated_sharpe_probability,
    stationary_block_bootstrap_mean_p_value,
    stop_slippage_sensitivity,
    summarize_account_result,
)

UTC = timezone.utc


def trade(name, day, pnl, reason=TradeExitReason.TP_HIT):
    ts = datetime(2026, 1, day, 15, tzinfo=UTC)
    return AccountedTrade(
        trade_id=name, strategy_id="s", direction="long", instrument=Instrument.MNQ,
        entry_timestamp=ts, exit_timestamp=ts, entry_price=100, sl_price=99,
        tp_price=102, quantity=2, risk_per_contract_usd=2, planned_risk_usd=4,
        entry_cost_usd=20, exit_reason=reason, gross_pnl_usd=pnl + 20,
        net_pnl_usd=pnl,
    )


def account(trades):
    return AccountBacktestResult(
        strategy_id="s", instrument=Instrument.MNQ, risk_config=RiskConfig(),
        accounted_trades=list(trades), untradeable_trades=[], trade_count=len(trades),
        untradeable_count=0, open_unresolved_count=0,
        realized_gross_pnl_usd_total=sum(t.gross_pnl_usd for t in trades),
        realized_net_pnl_usd_total=sum(t.net_pnl_usd for t in trades),
        entry_costs_charged_usd_total=20 * len(trades),
    )


class PerformanceValidationTests(unittest.TestCase):
    def test_summary_uses_net_pnl_r_and_zero_trade_weekdays(self):
        result = summarize_account_result(account([trade("a", 5, 100), trade("b", 7, -50, TradeExitReason.SL_HIT)]))
        self.assertEqual(result.gross_profit_usd, 100)
        self.assertEqual(result.gross_loss_usd, 50)
        self.assertEqual(result.profit_factor, 2)
        self.assertEqual(result.expectancy_usd, 25)
        self.assertEqual(result.total_r, 12.5)
        self.assertIsNotNone(result.annualized_daily_sharpe)
        self.assertEqual(result.top_winner_share_of_positive_pnl, 1)
        self.assertEqual(result.profitable_year_count, 1)
        self.assertEqual(result.evaluated_year_count, 1)
        self.assertEqual(result.yearly_net_pnl_usd, {"2026": 50.0})

    def test_stop_slippage_is_only_charged_on_stop_exits(self):
        value = account([trade("a", 5, 100), trade("b", 7, -50, TradeExitReason.SL_HIT)])
        # MNQ point value $2, 0.25 tick, qty 2 => $1 per extra stop tick.
        self.assertEqual(stop_slippage_sensitivity(value, tick_size=.25, extra_stop_slippage_ticks=(0, 3)), {0: 50, 3: 47})

    def test_bh_is_monotone_and_returns_original_order(self):
        self.assertEqual(benjamini_hochberg([0.04, 0.001, 0.2], false_discovery_rate=.10), (True, True, False))

    def test_stationary_bootstrap_is_reproducible_and_detects_large_edge(self):
        values = [10 + (i % 3) for i in range(100)]
        first = stationary_block_bootstrap_mean_p_value(values, resamples=200, seed=7)
        second = stationary_block_bootstrap_mean_p_value(values, resamples=200, seed=7)
        self.assertEqual(first, second)
        self.assertLess(first, .02)

    def test_deflated_sharpe_penalizes_more_strategy_trials(self):
        few = deflated_sharpe_probability(
            observed_sharpe=.20, observations=252, independent_trials=2,
            trial_sharpe_variance=.01,
        )
        many = deflated_sharpe_probability(
            observed_sharpe=.20, observations=252, independent_trials=10_000,
            trial_sharpe_variance=.01,
        )
        self.assertGreater(few, many)
        self.assertTrue(0 <= many <= 1)

    def test_deflated_sharpe_rejects_annualization_agnostic_bad_inputs(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_probability(
                observed_sharpe=.2, observations=1, independent_trials=1,
                trial_sharpe_variance=0,
            )


if __name__ == "__main__":
    unittest.main()
