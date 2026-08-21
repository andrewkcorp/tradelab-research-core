from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.accounting import Instrument, RiskConfig, apply_accounting
from tradelab.backtest import BacktestResult, ExecutedTrade, TradeExitReason
from tradelab.outlier_audit import audit_trade_outliers

UTC = timezone.utc


def executed(trade_id: str, points: float) -> ExecutedTrade:
    timestamp = datetime(2026, 1, 5, 15, tzinfo=UTC) + timedelta(minutes=len(trade_id))
    return ExecutedTrade(
        trade_id=trade_id, strategy_id="s", direction="long", signal_known_at=timestamp,
        entry_timestamp=timestamp, entry_price=100.0, tp_price=110.0, tp_source_type="liquidity",
        tp_source_id=f"tp-{trade_id}", sl_price=95.0, sl_source_type="liquidity",
        sl_source_id=f"sl-{trade_id}", exit_timestamp=timestamp + timedelta(minutes=1),
        exit_price=100.0 + points, exit_reason=TradeExitReason.TP_HIT if points > 0 else TradeExitReason.SL_HIT,
        gross_pnl_points=points,
    )


class OutlierAuditTests(unittest.TestCase):
    def test_reports_tails_and_pnl_without_largest_winner(self):
        trades = [executed("a", 10.0), executed("b", -5.0), executed("c", 100.0)]
        result = BacktestResult("s", trades, [], 3, 2, 1, 0, 105.0)
        account = apply_accounting(result, Instrument.MNQ, RiskConfig())
        audit = audit_trade_outliers(trades, account, limit=1)
        self.assertEqual(audit.largest_winners[0].trade_id, "c")
        self.assertEqual(audit.largest_losers[0].trade_id, "b")
        self.assertEqual(audit.largest_winners[0].tp_source_id, "tp-c")
        self.assertEqual(audit.realised_net_pnl_without_largest_winner_usd, 260.0)
        self.assertGreater(audit.largest_winner_share_of_positive_pnl, 0.9)

    def test_rejects_non_positive_limit(self):
        result = BacktestResult("s", [], [], 0, 0, 0, 0, 0.0)
        account = apply_accounting(result, Instrument.MNQ, RiskConfig())
        with self.assertRaises(ValueError):
            audit_trade_outliers([], account, limit=0)
