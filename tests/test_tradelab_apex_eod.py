"""Apex 50K EOD audit: reporting only, never changes trade execution."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.accounting import AccountBacktestResult, AccountedTrade, Instrument, RiskConfig
from tradelab.apex_eod import ApexEODConfig, ApexLimit, audit_apex_eod, trading_day_id
from tradelab.backtest import TradeExitReason

UTC = timezone.utc
RISK = RiskConfig()


def trade(trade_id: str, entry: datetime, gross_pnl: float, *, exit_minutes: int = 1) -> AccountedTrade:
    return AccountedTrade(
        trade_id=trade_id, strategy_id="s", direction="long", instrument=Instrument.MNQ,
        entry_timestamp=entry, exit_timestamp=entry + timedelta(minutes=exit_minutes),
        entry_price=100.0, sl_price=99.0, tp_price=101.0, quantity=1,
        risk_per_contract_usd=2.0, planned_risk_usd=2.0, entry_cost_usd=20.0,
        exit_reason=TradeExitReason.TP_HIT, gross_pnl_usd=gross_pnl,
        net_pnl_usd=gross_pnl - 20.0,
    )


def account(*trades: AccountedTrade) -> AccountBacktestResult:
    return AccountBacktestResult(
        strategy_id="s", instrument=Instrument.MNQ, risk_config=RISK,
        accounted_trades=list(trades), untradeable_trades=[], trade_count=len(trades),
        untradeable_count=0, open_unresolved_count=0,
        realized_gross_pnl_usd_total=sum(t.gross_pnl_usd for t in trades if t.gross_pnl_usd is not None),
        realized_net_pnl_usd_total=sum(t.net_pnl_usd for t in trades if t.net_pnl_usd is not None),
        entry_costs_charged_usd_total=sum(t.entry_cost_usd for t in trades),
    )


class TradingDayTests(unittest.TestCase):
    def test_cme_day_starts_at_18_ny_with_dst_aware_conversion(self):
        # 2026-01-06 10:00 NY = 15:00 UTC, belongs to session started Jan 5.
        self.assertEqual(trading_day_id(datetime(2026, 1, 6, 15, tzinfo=UTC)), "2026-01-05")
        # 18:00 NY = 23:00 UTC in January, starts the next CME day.
        self.assertEqual(trading_day_id(datetime(2026, 1, 6, 23, tzinfo=UTC)), "2026-01-06")


class ApexAuditTests(unittest.TestCase):
    def test_no_breach_for_small_loss_and_input_remains_untouched(self):
        original = account(trade("t1", datetime(2026, 1, 6, 15, tzinfo=UTC), -500.0))
        audit = audit_apex_eod(original)
        self.assertFalse(audit.has_breach)
        self.assertEqual(original.trade_count, 1)
        self.assertEqual([t.trade_id for t in original.accounted_trades], ["t1"])

    def test_reports_daily_loss_and_eod_breach_without_stopping_later_trade(self):
        start = datetime(2026, 1, 6, 15, tzinfo=UTC)
        original = account(
            trade("loss", start, -1_100.0),  # -$1,120 after entry fee
            trade("later", start + timedelta(minutes=10), 100.0),
        )
        audit = audit_apex_eod(original)
        self.assertTrue(audit.has_breach)
        self.assertEqual({b.limit for b in audit.breaches}, {ApexLimit.DAILY_LOSS_LIMIT})
        # It is only an audit: later trade is still present and evaluated.
        self.assertEqual([t.trade_id for t in original.accounted_trades], ["loss", "later"])
        self.assertGreaterEqual(audit.max_daily_loss_usd, 1_120.0)

    def test_eod_floor_tightens_only_after_a_completed_profitable_day(self):
        # Jan 5 CME day ends with +$3,000, so its next-day floor rises
        # from $48,000 to $50,980 (includes the $20 entry fee).
        day_one = datetime(2026, 1, 5, 23, tzinfo=UTC)  # 18:00 NY
        day_two = datetime(2026, 1, 6, 23, tzinfo=UTC)
        audit = audit_apex_eod(account(
            trade("winner", day_one, 3_000.0),
            trade("next-loss", day_two, -2_100.0),
        ))
        eod = [b for b in audit.breaches if b.limit == ApexLimit.EOD_DRAWDOWN]
        self.assertEqual(len(eod), 1)
        self.assertEqual(eod[0].trade_id, "next-loss")
        self.assertAlmostEqual(eod[0].limit_floor_usd, 50_980.0)

    def test_config_validation(self):
        for kwargs in ({"eod_drawdown_usd": 0}, {"daily_loss_limit_usd": -1}, {"max_mnq_contracts": 0}):
            with self.assertRaises(ValueError):
                ApexEODConfig(**kwargs)
