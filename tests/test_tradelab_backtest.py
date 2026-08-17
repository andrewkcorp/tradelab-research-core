#!/usr/bin/env python3
"""Single-strategy backtester integration tests: long/short, entry on
the next 1m candle's open, TP hit, SL hit (with a gap straight through
the level), same-candle TP+SL conflict resolved SL-first, red-folder
news block (boundaries inclusive), single active position (a second
signal is skipped), missing required confluence, bias mismatch, missing
TP/SL target, no lookahead, and BacktestResult's own aggregate counts.

All fixtures are small, synthetic 1-minute bar series built from flat
"resting" candles (repeated identical OHLC, which can never register as
a swing pivot or an FVG gap) punctuated by isolated spike candles that
deliberately form exactly one swing high/low at a time -- this keeps
every liquidity level in each fixture fully deliberate, with no
incidental extra levels competing for TP/SL rank #1. Every fixture is
far too short to complete a single 5m/15m/1H/4H/1D bucket (the smallest,
5m, still needs a fully aligned run of 1-minute bars), so higher
timeframes never contribute a level or a bias change here -- tests that
care about that (bias mismatch, missing confluence) rely on exactly
this fact rather than constructing HTF data.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.bar import Bar1m
from tradelab.news import NewsEvent
from tradelab.sl_targets import SLSourceType
from tradelab.strategy_spec import LONG, SHORT, BiasPolicy, SLChoice, StrategySpec, TPChoice
from tradelab.tp_targets import TPSourceType
from tradelab.backtest import SkipReason, TradeExitReason, backtest_strategy

UTC = timezone.utc
T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
PIVOT_STRENGTH = 1
RECLAIM_WINDOW_BARS = 3


def bar(i: int, o: float, h: float, l: float, c: float, base: datetime = T0) -> Bar1m:
    return Bar1m(timestamp=base + timedelta(minutes=i), open=o, high=h, low=l, close=c, volume=1.0, contract="NQ")


def flat(i: int, price: float = 100.0) -> Bar1m:
    return bar(i, price, price, price, price)


def long_base_bars() -> list[Bar1m]:
    """0-1: filler. 2: SL anchor swing low = 90. 3-4: filler. 5: TP anchor
    swing high = 120. 6-7: filler. 8: trigger swing low = 95. 9-10:
    filler. 11: take candle (low=94 < 95, takes the trigger swing). 12:
    entry candle (open=99 == entry_price), timestamp == the take's
    known_at -- the earliest possible entry bar."""
    return [
        flat(0), flat(1),
        bar(2, 90.5, 91, 90, 90.5),
        flat(3), flat(4),
        bar(5, 119.5, 120, 119, 119.5),
        flat(6), flat(7),
        bar(8, 95.5, 96, 95, 95.5),
        flat(9), flat(10),
        bar(11, 99, 100.5, 94, 94.5),
        bar(12, 99, 100, 98, 99.5),
    ]


def short_base_bars() -> list[Bar1m]:
    """Mirror image of long_base_bars: SL anchor swing high = 120, TP
    anchor swing low = 90, trigger swing high = 105 taken by a later
    candle -- entry candle open = 101 == entry_price."""
    return [
        flat(0), flat(1),
        bar(2, 119.5, 120, 119, 119.5),
        flat(3), flat(4),
        bar(5, 90.5, 91, 90, 90.5),
        flat(6), flat(7),
        bar(8, 104, 105, 104, 104.5),
        flat(9), flat(10),
        bar(11, 101, 106, 99.5, 100.5),
        bar(12, 101, 102, 100, 101.5),
    ]


def long_strategy(
    strategy_id: str = "s-long",
    required_confluence_selectors: tuple = (),
    bias_policy: BiasPolicy = BiasPolicy.IGNORE,
    tp_rank: int = 1,
    sl_rank: int = 1,
) -> StrategySpec:
    return StrategySpec(
        strategy_id=strategy_id,
        direction=LONG,
        primary_entry_selector="liquidity-1m-swing_equal-swing_low-take-reversal",
        required_confluence_selectors=required_confluence_selectors,
        bias_policy=bias_policy,
        tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY, rank=tp_rank),
        sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=sl_rank),
    )


def short_strategy(strategy_id: str = "s-short") -> StrategySpec:
    return StrategySpec(
        strategy_id=strategy_id,
        direction=SHORT,
        primary_entry_selector="liquidity-1m-swing_equal-swing_high-take-reversal",
        required_confluence_selectors=(),
        bias_policy=BiasPolicy.IGNORE,
        tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY, rank=1),
        sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1),
    )


def run(strategy, bars, news_events=()):
    return backtest_strategy(
        strategy, bars, list(news_events), pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS
    )


ENTRY_TIMESTAMP = T0 + timedelta(minutes=12)  # long_base_bars' / short_base_bars' entry candle


class LongTradeTests(unittest.TestCase):
    def test_long_entry_fills_at_next_candle_open(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        result = run(long_strategy(), bars)
        self.assertEqual(result.trade_count, 1)
        trade = result.trades[0]
        self.assertEqual(trade.direction, LONG)
        self.assertEqual(trade.entry_timestamp, ENTRY_TIMESTAMP)
        self.assertEqual(trade.entry_price, 99.0)  # the entry candle's own open
        self.assertEqual(trade.signal_known_at, ENTRY_TIMESTAMP)

    def test_long_tp_hit(self):
        bars = long_base_bars() + [bar(13, 115, 121, 114, 120)]
        result = run(long_strategy(), bars)
        trade = result.trades[0]
        self.assertEqual(trade.tp_price, 120.0)
        self.assertEqual(trade.sl_price, 90.0)
        self.assertEqual(trade.exit_reason, TradeExitReason.TP_HIT)
        self.assertEqual(trade.exit_price, 120.0)
        self.assertEqual(trade.exit_timestamp, T0 + timedelta(minutes=13))
        self.assertEqual(trade.gross_pnl_points, 21.0)  # 120 - 99
        self.assertEqual(result.tp_hit_count, 1)
        self.assertEqual(result.sl_hit_count, 0)
        self.assertEqual(result.gross_pnl_points_total, 21.0)

    def test_long_sl_hit_with_gap_through_the_level(self):
        # low=85 gaps straight through SL=90 -- fills at the level (85 is
        # not used: no finer-grained price path is available at 1m OHLC).
        bars = long_base_bars() + [bar(13, 95, 96, 85, 88)]
        result = run(long_strategy(), bars)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, TradeExitReason.SL_HIT)
        self.assertEqual(trade.exit_price, 90.0)
        self.assertEqual(trade.gross_pnl_points, -9.0)  # 90 - 99
        self.assertEqual(result.sl_hit_count, 1)
        self.assertEqual(result.tp_hit_count, 0)

    def test_same_candle_tp_and_sl_conflict_resolves_sl_first(self):
        bars = long_base_bars() + [bar(13, 100, 125, 80, 100)]  # both TP(120) and SL(90) touched
        result = run(long_strategy(), bars)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, TradeExitReason.SL_HIT)
        self.assertEqual(trade.exit_price, 90.0)

    def test_same_candle_conflict_on_the_entry_candle_itself_resolves_sl_first(self):
        # widen the entry candle's own range to touch both TP and SL.
        wide_bars = long_base_bars()
        wide_bars[-1] = bar(12, 99, 125, 80, 99.5)
        result = run(long_strategy(), wide_bars)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, TradeExitReason.SL_HIT)
        self.assertEqual(trade.exit_timestamp, ENTRY_TIMESTAMP)

    def test_no_tp_or_sl_by_end_of_data_is_open_unresolved(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        result = run(long_strategy(), bars)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, TradeExitReason.OPEN_UNRESOLVED)
        self.assertIsNone(trade.exit_timestamp)
        self.assertIsNone(trade.exit_price)
        self.assertIsNone(trade.gross_pnl_points)
        self.assertEqual(result.open_unresolved_count, 1)
        self.assertEqual(result.gross_pnl_points_total, 0.0)  # unresolved trades never contribute


class ShortTradeTests(unittest.TestCase):
    def test_short_entry_and_tp_hit(self):
        bars = short_base_bars() + [bar(13, 95, 96, 89, 90)]
        result = run(short_strategy(), bars)
        self.assertEqual(result.trade_count, 1)
        trade = result.trades[0]
        self.assertEqual(trade.direction, SHORT)
        self.assertEqual(trade.entry_price, 101.0)
        self.assertEqual(trade.tp_price, 90.0)
        self.assertEqual(trade.sl_price, 120.0)
        self.assertEqual(trade.exit_reason, TradeExitReason.TP_HIT)
        self.assertEqual(trade.exit_price, 90.0)
        self.assertEqual(trade.gross_pnl_points, 11.0)  # 101 - 90

    def test_short_sl_hit_with_gap(self):
        bars = short_base_bars() + [bar(13, 121, 130, 119, 122)]  # gaps through SL=120
        result = run(short_strategy(), bars)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, TradeExitReason.SL_HIT)
        self.assertEqual(trade.exit_price, 120.0)
        self.assertEqual(trade.gross_pnl_points, -19.0)  # 101 - 120


class NewsBlockTests(unittest.TestCase):
    def _event_at(self, minutes_from_t0: int) -> NewsEvent:
        return NewsEvent(
            event_id="EV1",
            scheduled_at=T0 + timedelta(minutes=minutes_from_t0),
            country="United States",
            currency="USD",
            event_type="CPI",
            source_agency="BLS",
            source_url="https://bls.gov/x",
            source_reference="ref-1",
            time_is_exact=True,
            schema_version="1.0",
        )

    def test_entry_exactly_at_t_minus_15_is_blocked(self):
        # entry bar is at T0+12; scheduling the event 15 minutes later
        # puts the entry exactly on the window's inclusive left boundary.
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        result = run(long_strategy(), bars, news_events=[self._event_at(27)])
        self.assertEqual(result.trade_count, 0)
        self.assertEqual(len(result.skipped_signals), 1)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.NEWS_BLOCK)
        self.assertEqual(result.skipped_signals[0].timestamp, ENTRY_TIMESTAMP)

    def test_entry_one_minute_before_the_window_is_not_blocked(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        result = run(long_strategy(), bars, news_events=[self._event_at(28)])
        self.assertEqual(result.trade_count, 1)
        self.assertEqual(result.skipped_signals, [])

    def test_entry_exactly_at_t_plus_5_is_blocked(self):
        # event 5 minutes before the entry bar -> entry sits exactly on
        # the window's inclusive right boundary.
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        result = run(long_strategy(), bars, news_events=[self._event_at(7)])
        self.assertEqual(result.trade_count, 0)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.NEWS_BLOCK)

    def test_an_already_open_position_is_never_closed_by_a_later_news_window(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        result = run(long_strategy(), bars, news_events=[self._event_at(100)])  # far away, irrelevant
        self.assertEqual(result.trade_count, 1)
        self.assertEqual(result.trades[0].exit_reason, TradeExitReason.OPEN_UNRESOLVED)


class ActivePositionTests(unittest.TestCase):
    def test_second_signal_while_position_open_is_skipped_not_traded(self):
        bars = long_base_bars() + [
            bar(13, 99, 100, 99, 99.5),  # stays flat, first trade never resolves
            flat(14), flat(15),
            bar(16, 95.5, 96, 95, 95.5),  # a second, independent trigger swing low
            flat(17), flat(18),
            bar(19, 99, 100.5, 94, 94.5),  # takes it -> a second primary signal known here
            bar(20, 99, 101, 97, 100),
        ]
        result = run(long_strategy(), bars)
        self.assertEqual(result.trade_count, 1)  # only the first trade ever opens
        self.assertEqual(result.trades[0].entry_timestamp, ENTRY_TIMESTAMP)
        self.assertEqual(result.trades[0].exit_reason, TradeExitReason.OPEN_UNRESOLVED)
        self.assertEqual(len(result.skipped_signals), 1)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.ACTIVE_POSITION)
        self.assertEqual(result.skipped_signals[0].timestamp, T0 + timedelta(minutes=20))


class MissingRequiredConfluenceTests(unittest.TestCase):
    def test_confluence_selector_never_active_in_this_data_is_skipped(self):
        # "fvg-5m-fvg_touch" is a syntactically valid selector, but this
        # fixture is far too short to ever complete a 5m bucket, so it
        # never has an active signal.
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        strategy = long_strategy(required_confluence_selectors=("fvg-5m-fvg_touch",))
        result = run(strategy, bars)
        self.assertEqual(result.trade_count, 0)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.MISSING_REQUIRED_CONFLUENCE)


class BiasMismatchTests(unittest.TestCase):
    def test_require_aligned_with_no_htf_data_is_always_neutral_and_mismatches(self):
        # no 1H/4H/1D bucket ever completes in this fixture, so HTF bias
        # stays neutral forever -- REQUIRE_ALIGNED (wants bullish for a
        # long) can never be satisfied.
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        strategy = long_strategy(bias_policy=BiasPolicy.REQUIRE_ALIGNED)
        result = run(strategy, bars)
        self.assertEqual(result.trade_count, 0)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.BIAS_MISMATCH)

    def test_require_neutral_with_no_htf_data_passes(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        strategy = long_strategy(bias_policy=BiasPolicy.REQUIRE_NEUTRAL)
        result = run(strategy, bars)
        self.assertEqual(result.trade_count, 1)

    def test_ignore_policy_never_checks_bias(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        strategy = long_strategy(bias_policy=BiasPolicy.IGNORE)
        result = run(strategy, bars)
        self.assertEqual(result.trade_count, 1)


class MissingTargetTests(unittest.TestCase):
    def test_tp_rank_with_no_such_candidate_is_missing_target(self):
        # only one qualifying liquidity high (the rank-1 TP anchor)
        # exists in this fixture -- rank 2 can never be found.
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        strategy = long_strategy(tp_rank=2)
        result = run(strategy, bars)
        self.assertEqual(result.trade_count, 0)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.MISSING_TARGET)

    def test_sl_rank_with_no_such_candidate_is_missing_target(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        strategy = long_strategy(sl_rank=2)
        result = run(strategy, bars)
        self.assertEqual(result.trade_count, 0)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.MISSING_TARGET)


class NoLookaheadTests(unittest.TestCase):
    def test_entry_and_target_selection_are_identical_whether_or_not_future_bars_exist(self):
        full_bars = long_base_bars() + [bar(13, 115, 121, 114, 120)]
        truncated_bars = long_base_bars()  # ends exactly at the entry candle

        full_result = run(long_strategy(), full_bars)
        truncated_result = run(long_strategy(), truncated_bars)

        self.assertEqual(truncated_result.trade_count, 1)
        self.assertEqual(truncated_result.trades[0].exit_reason, TradeExitReason.OPEN_UNRESOLVED)

        full_trade = full_result.trades[0]
        truncated_trade = truncated_result.trades[0]
        self.assertEqual(full_trade.entry_price, truncated_trade.entry_price)
        self.assertEqual(full_trade.entry_timestamp, truncated_trade.entry_timestamp)
        self.assertEqual(full_trade.tp_price, truncated_trade.tp_price)
        self.assertEqual(full_trade.tp_source_id, truncated_trade.tp_source_id)
        self.assertEqual(full_trade.sl_price, truncated_trade.sl_price)
        self.assertEqual(full_trade.sl_source_id, truncated_trade.sl_source_id)

    def test_a_level_formed_after_entry_known_at_is_never_selected(self):
        # bar 13 forms a brand-new swing low at 97 (lower than idx12's
        # low=98, higher than idx14's), strictly after the entry bar --
        # it must never be picked as the SL even though it is nearer to
        # entry_price than the real (pre-entry) SL anchor at 90.
        bars = long_base_bars() + [
            bar(13, 99, 100, 97, 99),
            flat(14),
            bar(15, 99, 101, 97, 100),
        ]
        result = run(long_strategy(), bars)
        trade = result.trades[0]
        self.assertEqual(trade.sl_price, 90.0)
        self.assertNotEqual(trade.sl_price, 97.0)


class BacktestResultAggregateTests(unittest.TestCase):
    def test_result_carries_strategy_id_and_consistent_counts(self):
        bars = long_base_bars() + [bar(13, 115, 121, 114, 120)]
        strategy = long_strategy(strategy_id="my-strategy")
        result = run(strategy, bars)
        self.assertEqual(result.strategy_id, "my-strategy")
        self.assertEqual(result.trade_count, len(result.trades))
        self.assertEqual(result.tp_hit_count + result.sl_hit_count + result.open_unresolved_count, result.trade_count)


if __name__ == "__main__":
    unittest.main()
