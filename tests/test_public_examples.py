"""Small, deterministic onboarding tests; no vendor or private dataset."""
import contextlib
from datetime import timedelta
import io
import json
import unittest

from tradelab.backtest import TradeExitReason
from tradelab.synthetic_demo import START,main,run_example,synthetic_bars


class PublicExamples(unittest.TestCase):
    def test_actual_engine_next_open_and_target(self):
        result = run_example()
        self.assertEqual(result.trade_count,1)
        trade = result.trades[0]
        self.assertEqual(trade.signal_known_at,START+timedelta(minutes=12))
        self.assertEqual(trade.entry_timestamp,START+timedelta(minutes=12))
        self.assertEqual(trade.entry_price,99)
        self.assertEqual((trade.sl_price,trade.tp_price),(90,120))
        self.assertEqual(trade.exit_reason,TradeExitReason.TP_HIT)
        self.assertEqual(trade.gross_pnl_points,21)

    def test_both_barriers_resolve_to_stop_not_optimistic_target(self):
        result = run_example(ambiguous=True)
        self.assertEqual(result.trade_count,1)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason,TradeExitReason.SL_HIT)
        self.assertEqual(trade.exit_price,90)
        self.assertEqual(trade.gross_pnl_points,-9)

    def test_fixture_is_explicitly_synthetic_and_deterministic(self):
        self.assertEqual(synthetic_bars(),synthetic_bars())
        self.assertTrue(all(bar.contract == 'SYNTHETIC' for bar in synthetic_bars()))
        self.assertEqual(run_example(),run_example())

    def test_cli_labels_real_engine_versus_real_market(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output): self.assertEqual(main(),0)
        report = json.loads(output.getvalue())
        self.assertTrue(report['synthetic'])
        self.assertTrue(report['actual_engine_invoked'])
        self.assertIn('not real market performance',report['warning'])
        self.assertEqual(report['target_case']['trades'],1)


if __name__ == '__main__': unittest.main()
