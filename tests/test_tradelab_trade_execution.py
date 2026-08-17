#!/usr/bin/env python3
"""Trade execution tests: long/short TP and SL (including gap/slippage
fills on SL), market entry on the first tick strictly after
signal_known_at (never the signal tick itself), strict (timestamp,
sequence) ordering enforcement, OPEN_UNRESOLVED when the tick stream
runs out before TP/SL, NO_ENTRY when it runs out before any entry,
invalid TP/SL-vs-entry relationships, no lookahead/no cherry-picking of
entry price, and gross P&L in points for both directions."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.trade_execution import (
    LONG,
    SHORT,
    MarketTradeRequest,
    TickPriceEvent,
    TradeExecutionResult,
    TradeStatus,
    simulate_trade,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def tick(minutes: float, price: float, sequence: int = 0, base: datetime = T0) -> TickPriceEvent:
    return TickPriceEvent(timestamp=base + timedelta(minutes=minutes), sequence=sequence, price=price)


def req(
    trade_id: str = "t1",
    direction: str = LONG,
    signal_known_at: datetime = T0,
    tp_price: float = 110.0,
    sl_price: float = 95.0,
) -> MarketTradeRequest:
    return MarketTradeRequest(
        trade_id=trade_id, direction=direction, signal_known_at=signal_known_at, tp_price=tp_price, sl_price=sl_price
    )


class TickPriceEventValidationTests(unittest.TestCase):
    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            TickPriceEvent(timestamp=datetime(2026, 1, 5, 14, 30), sequence=0, price=100.0)

    def test_non_positive_price_rejected(self):
        with self.assertRaises(ValueError):
            TickPriceEvent(timestamp=T0, sequence=0, price=0.0)
        with self.assertRaises(ValueError):
            TickPriceEvent(timestamp=T0, sequence=0, price=-1.0)

    def test_timestamp_normalized_to_utc(self):
        from zoneinfo import ZoneInfo

        ny = ZoneInfo("America/New_York")
        t = TickPriceEvent(timestamp=datetime(2026, 1, 5, 9, 30, tzinfo=ny), sequence=0, price=100.0)
        self.assertEqual(t.timestamp.tzinfo, UTC)


class MarketTradeRequestValidationTests(unittest.TestCase):
    def test_empty_trade_id_rejected(self):
        with self.assertRaises(ValueError):
            MarketTradeRequest(trade_id="", direction=LONG, signal_known_at=T0, tp_price=110.0, sl_price=95.0)

    def test_invalid_direction_rejected(self):
        with self.assertRaises(ValueError):
            MarketTradeRequest(trade_id="t1", direction="up", signal_known_at=T0, tp_price=110.0, sl_price=95.0)

    def test_naive_signal_known_at_rejected(self):
        with self.assertRaises(ValueError):
            MarketTradeRequest(trade_id="t1", direction=LONG, signal_known_at=datetime(2026, 1, 5, 14, 30), tp_price=110.0, sl_price=95.0)

    def test_non_positive_tp_or_sl_rejected(self):
        with self.assertRaises(ValueError):
            MarketTradeRequest(trade_id="t1", direction=LONG, signal_known_at=T0, tp_price=0.0, sl_price=95.0)
        with self.assertRaises(ValueError):
            MarketTradeRequest(trade_id="t1", direction=LONG, signal_known_at=T0, tp_price=110.0, sl_price=-5.0)


class LongTPTests(unittest.TestCase):
    def test_long_tp_hit_fills_exactly_at_tp_price(self):
        ticks = [tick(1, 100.0), tick(2, 105.0), tick(3, 110.0)]
        result = simulate_trade(req(direction=LONG, tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.status, TradeStatus.TP_HIT)
        self.assertEqual(result.entry_price, 100.0)
        self.assertEqual(result.entry_timestamp, T0 + timedelta(minutes=1))
        self.assertEqual(result.exit_price, 110.0)
        self.assertEqual(result.exit_timestamp, T0 + timedelta(minutes=3))
        self.assertEqual(result.gross_pnl_points, 10.0)

    def test_long_tp_fill_price_is_tp_even_when_tick_overshoots(self):
        # tick prints 115 (overshoots TP=110); limit fill must still be exactly 110.
        ticks = [tick(1, 100.0), tick(2, 115.0)]
        result = simulate_trade(req(direction=LONG, tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.status, TradeStatus.TP_HIT)
        self.assertEqual(result.exit_price, 110.0)
        self.assertEqual(result.gross_pnl_points, 10.0)


class LongSLTests(unittest.TestCase):
    def test_long_sl_hit_with_gap_fills_at_tick_price_not_sl_price(self):
        # tick jumps straight through SL=95 down to 90 -- stop-market fills at 90, the tick's own price.
        ticks = [tick(1, 100.0), tick(2, 90.0)]
        result = simulate_trade(req(direction=LONG, tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.status, TradeStatus.SL_HIT)
        self.assertEqual(result.entry_price, 100.0)
        self.assertEqual(result.exit_price, 90.0)
        self.assertEqual(result.gross_pnl_points, -10.0)


class ShortTPTests(unittest.TestCase):
    def test_short_tp_hit_fills_exactly_at_tp_price(self):
        ticks = [tick(1, 100.0), tick(2, 95.0), tick(3, 90.0)]
        result = simulate_trade(req(direction=SHORT, tp_price=90.0, sl_price=105.0), ticks)
        self.assertEqual(result.status, TradeStatus.TP_HIT)
        self.assertEqual(result.entry_price, 100.0)
        self.assertEqual(result.exit_price, 90.0)
        self.assertEqual(result.gross_pnl_points, 10.0)

    def test_short_tp_fill_price_is_tp_even_when_tick_undershoots_further(self):
        ticks = [tick(1, 100.0), tick(2, 85.0)]  # overshoots TP=90 downward
        result = simulate_trade(req(direction=SHORT, tp_price=90.0, sl_price=105.0), ticks)
        self.assertEqual(result.exit_price, 90.0)
        self.assertEqual(result.gross_pnl_points, 10.0)


class ShortSLTests(unittest.TestCase):
    def test_short_sl_hit_with_gap_fills_at_tick_price_not_sl_price(self):
        # tick jumps straight through SL=105 up to 110 -- stop-market fills at 110.
        ticks = [tick(1, 100.0), tick(2, 110.0)]
        result = simulate_trade(req(direction=SHORT, tp_price=90.0, sl_price=105.0), ticks)
        self.assertEqual(result.status, TradeStatus.SL_HIT)
        self.assertEqual(result.entry_price, 100.0)
        self.assertEqual(result.exit_price, 110.0)
        self.assertEqual(result.gross_pnl_points, -10.0)


class MarketEntryTests(unittest.TestCase):
    def test_first_tick_strictly_after_signal_known_at_is_the_entry(self):
        ticks = [tick(1, 100.0), tick(2, 102.0), tick(3, 110.0)]
        result = simulate_trade(req(signal_known_at=T0, tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.entry_timestamp, T0 + timedelta(minutes=1))
        self.assertEqual(result.entry_price, 100.0)

    def test_tick_exactly_at_signal_known_at_cannot_be_the_entry(self):
        # tick(0, ...) sits exactly at T0 == signal_known_at -- must be skipped.
        ticks = [tick(0, 999.0), tick(1, 100.0), tick(2, 110.0)]
        result = simulate_trade(req(signal_known_at=T0, tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.entry_timestamp, T0 + timedelta(minutes=1))
        self.assertEqual(result.entry_price, 100.0)
        self.assertNotEqual(result.entry_price, 999.0)

    def test_no_tick_after_signal_known_at_yields_no_entry(self):
        ticks = [tick(-2, 100.0), tick(-1, 101.0)]  # both strictly before signal_known_at
        result = simulate_trade(req(signal_known_at=T0), ticks)
        self.assertEqual(result.status, TradeStatus.NO_ENTRY)
        self.assertIsNone(result.entry_timestamp)
        self.assertIsNone(result.entry_price)
        self.assertIsNone(result.exit_timestamp)
        self.assertIsNone(result.exit_price)
        self.assertIsNone(result.gross_pnl_points)

    def test_empty_tick_stream_yields_no_entry(self):
        result = simulate_trade(req(), [])
        self.assertEqual(result.status, TradeStatus.NO_ENTRY)


class NoLookaheadTests(unittest.TestCase):
    def test_entry_is_always_the_first_qualifying_tick_never_a_later_better_price(self):
        # A later tick (95.0) would be a "better" long entry price than the
        # first one (105.0) -- entry must still use the first tick, never
        # cherry-pick a later, more favorable one.
        ticks = [tick(1, 105.0), tick(2, 95.0), tick(3, 110.0)]
        result = simulate_trade(req(direction=LONG, tp_price=110.0, sl_price=90.0), ticks)
        self.assertEqual(result.entry_price, 105.0)
        self.assertNotEqual(result.entry_price, 95.0)

    def test_ticks_before_entry_are_never_evaluated_for_exit(self):
        # A pre-signal tick at 110 (>= a would-be TP) must never be treated
        # as a TP hit -- only ticks strictly after signal_known_at even
        # enter the exit scan, and only after the entry tick itself.
        ticks = [tick(-1, 110.0), tick(1, 100.0), tick(2, 102.0)]
        result = simulate_trade(req(signal_known_at=T0, tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.status, TradeStatus.OPEN_UNRESOLVED)
        self.assertEqual(result.entry_price, 100.0)


class OrderingTests(unittest.TestCase):
    def test_same_timestamp_ticks_ordered_by_sequence_are_accepted(self):
        same_ts = T0 + timedelta(minutes=1)
        ticks = [
            TickPriceEvent(timestamp=same_ts, sequence=0, price=100.0),
            TickPriceEvent(timestamp=same_ts, sequence=1, price=101.0),
            tick(2, 110.0),
        ]
        result = simulate_trade(req(tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.entry_price, 100.0)  # sequence 0 wins as the entry tick
        self.assertEqual(result.status, TradeStatus.TP_HIT)

    def test_out_of_order_sequence_at_same_timestamp_is_rejected(self):
        same_ts = T0 + timedelta(minutes=1)
        ticks = [
            TickPriceEvent(timestamp=same_ts, sequence=1, price=100.0),
            TickPriceEvent(timestamp=same_ts, sequence=0, price=101.0),
        ]
        with self.assertRaises(ValueError):
            simulate_trade(req(), ticks)

    def test_out_of_order_timestamps_are_rejected(self):
        ticks = [tick(2, 100.0), tick(1, 101.0)]
        with self.assertRaises(ValueError):
            simulate_trade(req(), ticks)

    def test_duplicate_timestamp_and_sequence_is_rejected(self):
        same_ts = T0 + timedelta(minutes=1)
        ticks = [
            TickPriceEvent(timestamp=same_ts, sequence=0, price=100.0),
            TickPriceEvent(timestamp=same_ts, sequence=0, price=101.0),
        ]
        with self.assertRaises(ValueError):
            simulate_trade(req(), ticks)


class OpenUnresolvedTests(unittest.TestCase):
    def test_stream_ends_after_entry_without_tp_or_sl_is_open_unresolved(self):
        ticks = [tick(1, 100.0), tick(2, 102.0), tick(3, 103.0)]
        result = simulate_trade(req(tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.status, TradeStatus.OPEN_UNRESOLVED)
        self.assertEqual(result.entry_price, 100.0)
        self.assertIsNone(result.exit_timestamp)
        self.assertIsNone(result.exit_price)
        self.assertIsNone(result.gross_pnl_points)
        # position is never artificially closed
        self.assertNotEqual(result.status, TradeStatus.TP_HIT)
        self.assertNotEqual(result.status, TradeStatus.SL_HIT)


class InvalidTPSLRelationshipTests(unittest.TestCase):
    def test_long_entry_price_not_below_tp_is_rejected(self):
        # entry fills at 100, but tp_price=99 is below entry -- violates sl < entry < tp.
        ticks = [tick(1, 100.0)]
        with self.assertRaises(ValueError):
            simulate_trade(req(direction=LONG, tp_price=99.0, sl_price=95.0), ticks)

    def test_long_entry_price_not_above_sl_is_rejected(self):
        # entry fills at 100, but sl_price=101 is above entry -- violates sl < entry.
        ticks = [tick(1, 100.0)]
        with self.assertRaises(ValueError):
            simulate_trade(req(direction=LONG, tp_price=110.0, sl_price=101.0), ticks)

    def test_short_entry_price_not_above_tp_is_rejected(self):
        ticks = [tick(1, 100.0)]
        with self.assertRaises(ValueError):
            simulate_trade(req(direction=SHORT, tp_price=101.0, sl_price=105.0), ticks)

    def test_short_entry_price_not_below_sl_is_rejected(self):
        ticks = [tick(1, 100.0)]
        with self.assertRaises(ValueError):
            simulate_trade(req(direction=SHORT, tp_price=90.0, sl_price=99.0), ticks)


class GrossPnlTests(unittest.TestCase):
    def test_long_gross_pnl_is_exit_minus_entry(self):
        ticks = [tick(1, 100.0), tick(2, 110.0)]
        result = simulate_trade(req(direction=LONG, tp_price=110.0, sl_price=95.0), ticks)
        self.assertEqual(result.gross_pnl_points, result.exit_price - result.entry_price)

    def test_short_gross_pnl_is_entry_minus_exit(self):
        ticks = [tick(1, 100.0), tick(2, 90.0)]
        result = simulate_trade(req(direction=SHORT, tp_price=90.0, sl_price=105.0), ticks)
        self.assertEqual(result.gross_pnl_points, result.entry_price - result.exit_price)

    def test_no_entry_and_open_unresolved_have_no_pnl(self):
        no_entry = simulate_trade(req(signal_known_at=T0), [tick(-1, 100.0)])
        self.assertIsNone(no_entry.gross_pnl_points)
        open_unresolved = simulate_trade(req(tp_price=110.0, sl_price=95.0), [tick(1, 100.0), tick(2, 102.0)])
        self.assertIsNone(open_unresolved.gross_pnl_points)


class ResultShapeTests(unittest.TestCase):
    def test_result_carries_trade_id_and_direction(self):
        ticks = [tick(1, 100.0), tick(2, 110.0)]
        result = simulate_trade(req(trade_id="my-trade", direction=LONG, tp_price=110.0, sl_price=95.0), ticks)
        self.assertIsInstance(result, TradeExecutionResult)
        self.assertEqual(result.trade_id, "my-trade")
        self.assertEqual(result.direction, LONG)


if __name__ == "__main__":
    unittest.main()
