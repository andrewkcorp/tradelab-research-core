from __future__ import annotations

import unittest
import random
import dataclasses
from datetime import datetime, timedelta, timezone

from tradelab.accounting import Instrument, RiskConfig, apply_accounting
from tradelab.backtest import BacktestResult, ExecutedTrade, TradeExitReason
from tradelab.bar import Bar1m
from tradelab.bar_range_index import BarRangeIndex
from tradelab.trade_path_audit import (
    _audit_account_mark_to_market_reference,
    audit_account_mark_to_market,
    audit_trade_paths,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 5, 15, tzinfo=UTC)


def bar(offset: int, high: float, low: float) -> Bar1m:
    return Bar1m(T0 + timedelta(minutes=offset), 100.0, high, low, 100.0, 1.0, "MNQH6")


def execution(direction: str) -> ExecutedTrade:
    return ExecutedTrade(
        trade_id="t", strategy_id="s", direction=direction, signal_known_at=T0, entry_timestamp=T0,
        entry_price=100.0, tp_price=110.0, tp_source_type="x", tp_source_id="tp", sl_price=95.0,
        sl_source_type="x", sl_source_id="sl", exit_timestamp=T0 + timedelta(minutes=2), exit_price=101.0,
        exit_reason=TradeExitReason.TP_HIT, gross_pnl_points=1.0,
    )


class TradePathAuditTests(unittest.TestCase):
    def test_long_mae_mfe_includes_entry_and_exit_bars(self):
        trade = execution("long")
        account = apply_accounting(BacktestResult("s", [trade], [], 1, 1, 0, 0, 1.0), Instrument.MNQ, RiskConfig())
        audit = audit_trade_paths([trade], account, [bar(0, 101, 99), bar(1, 104, 96), bar(2, 103, 98)])[0]
        self.assertEqual((audit.mae_points, audit.mfe_points), (4.0, 4.0))
        self.assertEqual((audit.mae_price_usd, audit.mfe_price_usd), (240.0, 240.0))  # 30 MNQ × $2/pt

    def test_short_uses_high_for_mae_and_low_for_mfe(self):
        trade = execution("short")
        account = apply_accounting(BacktestResult("s", [trade], [], 1, 1, 0, 0, 1.0), Instrument.MNQ, RiskConfig())
        audit = audit_trade_paths([trade], account, [bar(0, 101, 99), bar(1, 104, 96), bar(2, 103, 98)])[0]
        self.assertEqual((audit.mae_points, audit.mfe_points), (4.0, 4.0))

    def test_missing_exit_bar_is_not_silently_ignored(self):
        trade = execution("long")
        account = apply_accounting(BacktestResult("s", [trade], [], 1, 1, 0, 0, 1.0), Instrument.MNQ, RiskConfig())
        with self.assertRaises(ValueError):
            audit_trade_paths([trade], account, [bar(0, 101, 99)])

    def test_account_mtm_marks_long_to_intrabar_low_without_changing_cash_result(self):
        trade = execution("long")
        account = apply_accounting(BacktestResult("s", [trade], [], 1, 1, 0, 0, 1.0), Instrument.MNQ, RiskConfig())
        audit = audit_account_mark_to_market([trade], account, [bar(0, 101, 99), bar(1, 104, 96), bar(2, 103, 98)])
        # 30 MNQ × $2/point × 4 adverse points, plus the $20 entry cost.
        self.assertEqual(audit.worst_equity_usd, 49_740.0)
        self.assertEqual(audit.max_mark_to_market_drawdown_usd, 260.0)

    def test_sparse_mtm_replay_matches_original_full_bar_reference(self):
        rng = random.Random(20260821)
        bars = []
        price = 100.0
        for offset in range(240):
            move = rng.uniform(-2.0, 2.0)
            close = price + move
            bars.append(Bar1m(
                T0 + timedelta(minutes=offset), price,
                max(price, close) + rng.uniform(0.0, 1.0),
                min(price, close) - rng.uniform(0.0, 1.0),
                close, 1.0, "MNQH6",
            ))
            price = close

        trades = []
        for sequence, start in enumerate(range(5, 210, 20), start=1):
            end = start + 7
            direction = "long" if sequence % 2 else "short"
            entry_price = bars[start].open
            exit_price = bars[end].close
            points = exit_price - entry_price if direction == "long" else entry_price - exit_price
            trades.append(ExecutedTrade(
                trade_id=f"t-{sequence}", strategy_id="s", direction=direction,
                signal_known_at=bars[start].timestamp, entry_timestamp=bars[start].timestamp,
                entry_price=entry_price, tp_price=entry_price + (10 if direction == "long" else -10),
                tp_source_type="x", tp_source_id="tp",
                sl_price=entry_price + (-5 if direction == "long" else 5), sl_source_type="x", sl_source_id="sl",
                exit_timestamp=bars[end].timestamp, exit_price=exit_price,
                exit_reason=TradeExitReason.TP_HIT if points >= 0 else TradeExitReason.SL_HIT,
                gross_pnl_points=points,
            ))
        backtest = BacktestResult(
            "s", trades, [], len(trades),
            sum(t.exit_reason == TradeExitReason.TP_HIT for t in trades),
            sum(t.exit_reason == TradeExitReason.SL_HIT for t in trades), 0,
            sum(t.gross_pnl_points or 0.0 for t in trades),
        )
        account = apply_accounting(backtest, Instrument.MNQ, RiskConfig())
        expected = _audit_account_mark_to_market_reference(trades, account, bars)
        reusable_index = BarRangeIndex.build(bars)
        actual = audit_account_mark_to_market(trades, account, bars, bar_range_index=reusable_index)
        self.assertEqual(actual, expected)

    def test_sparse_mtm_replay_matches_reference_for_unresolved_final_position(self):
        trade = dataclasses.replace(
            execution("short"), exit_timestamp=None, exit_price=None,
            exit_reason=TradeExitReason.OPEN_UNRESOLVED, gross_pnl_points=None,
        )
        bars = [bar(0, 101, 99), bar(1, 108, 96), bar(2, 103, 98), bar(3, 105, 97)]
        account = apply_accounting(BacktestResult("s", [trade], [], 1, 0, 0, 1, 0.0), Instrument.MNQ, RiskConfig())
        self.assertEqual(
            audit_account_mark_to_market([trade], account, bars, bar_range_index=BarRangeIndex.build(bars)),
            _audit_account_mark_to_market_reference([trade], account, bars),
        )
