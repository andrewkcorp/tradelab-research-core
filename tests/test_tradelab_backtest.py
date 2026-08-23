#!/usr/bin/env python3
"""Single-strategy backtester integration tests: long/short, entry on
the next 1m candle's open, TP hit, SL hit (with a gap straight through
the level), same-candle TP+SL conflict resolved SL-first, red-folder
news block (boundaries inclusive), single active position (a second
signal is skipped), missing required confluence, bias mismatch, missing
TP/SL target, no lookahead, FVG-boundary SL variant selection, data-wick
TP nearest selection, and BacktestResult's own aggregate counts.

Most fixtures are small, synthetic 1-minute bar series built from flat
"resting" candles (repeated identical OHLC, which can never register as
a swing pivot or an FVG gap) punctuated by isolated spike candles that
deliberately form exactly one swing high/low at a time -- this keeps
every liquidity level in each fixture fully deliberate, with no
incidental extra levels competing for TP/SL rank #1. Most of them are
also far too short to complete a single 5m/15m/1H/4H/1D bucket (the
smallest, 5m, still needs a fully aligned run of 1-minute bars), so
higher timeframes never contribute a level or a bias change there --
tests that care about that (bias mismatch, missing confluence) rely on
exactly this fact rather than constructing HTF data. The FVG-boundary
SL tests are the exception: they deliberately build complete 5m buckets
(this file's ``T0`` is 09:30 NY, already 5m/15m-aligned) using flat,
multi-bar plateaus to form exactly two 5m FVG zones with no incidental
extra ones -- see ``fvg_boundary_variant_bars()`` for the construction
notes.
"""
from __future__ import annotations

import dataclasses
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone

from tradelab.bar import Bar1m
from tradelab.news import NewsEvent
from tradelab.session_calendar import LEGACY_SPECIAL_SCHEDULE_SCHEMA_VERSION, SpecialSessionSchedule
from tradelab.sl_targets import SLSourceType
from tradelab.strategy_spec import LONG, SHORT, BiasPolicy, SLChoice, StrategySpec, TPChoice
from tradelab.tp_targets import TPSourceType
from tradelab import backtest as backtest_module
from tradelab.backtest import (
    SIMULATION_UNUSED_CONTEXT_FIELDS,
    PreparedBacktestContext,
    SkipReason,
    TradeExitReason,
    backtest_prepared_strategy,
    backtest_strategy,
    prepare_backtest_context,
    release_discovery_inputs,
)

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

    def test_long_sl_hit_with_opening_gap_through_the_level(self):
        # The bar opens below SL=90. A stop-market cannot fill at the now
        # unavailable stop price, so the conservative executable fill is 85.
        bars = long_base_bars() + [bar(13, 85, 96, 84, 88)]
        result = run(long_strategy(), bars)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, TradeExitReason.SL_HIT)
        self.assertEqual(trade.exit_price, 85.0)
        self.assertEqual(trade.gross_pnl_points, -14.0)  # 85 - 99
        self.assertEqual(result.sl_hit_count, 1)
        self.assertEqual(result.tp_hit_count, 0)

    def test_long_intrabar_stop_touch_without_opening_gap_fills_at_stop(self):
        bars = long_base_bars() + [bar(13, 95, 96, 85, 88)]
        trade = run(long_strategy(), bars).trades[0]
        self.assertEqual((trade.exit_reason, trade.exit_price), (TradeExitReason.SL_HIT, 90.0))

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
        self.assertEqual(trade.exit_price, 121.0)
        self.assertEqual(trade.gross_pnl_points, -20.0)  # 101 - 121

    def test_short_intrabar_stop_touch_without_opening_gap_fills_at_stop(self):
        bars = short_base_bars() + [bar(13, 119, 130, 118, 122)]
        trade = run(short_strategy(), bars).trades[0]
        self.assertEqual((trade.exit_reason, trade.exit_price), (TradeExitReason.SL_HIT, 120.0))


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


def flat_group(start_idx: int, price: float, n: int = 5) -> list[Bar1m]:
    return [flat(start_idx + k, price) for k in range(n)]


def spike_bucket(start_idx: int, baseline: float, spike: tuple, spike_pos: int = 2) -> list[Bar1m]:
    """One 5m-aligned bucket (5 bars) of ``baseline`` flat bars with a
    single ``spike`` bar (o, h, l, c) at position ``spike_pos`` -- the
    spike's own high/low only ever extends *one* side of ``baseline``
    (this file's spikes never cross both), so this bucket's own 5m
    aggregate high or low moves but the other stays pinned at
    ``baseline``, exactly like every other flat-bucket boundary in this
    file: never enough to register an incidental FVG gap two buckets
    away."""
    bars = []
    for k in range(5):
        if k == spike_pos:
            bars.append(bar(start_idx + k, *spike))
        else:
            bars.append(flat(start_idx + k, baseline))
    return bars


def fvg_boundary_variant_bars() -> list[Bar1m]:
    """A complete, 5m-aligned fixture with exactly two 5m bullish FVG
    zones, both touched (``is_entry_trigger=True``, both LONG) and still
    simultaneously active at entry -- built entirely from multi-bar flat
    plateaus (>=2 identical bars each), which structurally cannot
    register a 1m-level swing pivot (a strict pivot needs a candle
    strictly better than *both* neighbors; an identical same-plateau
    neighbor always fails that strictly).

    Zone A [100, 110] forms from buckets 0/1/2 (14:30-14:45), touched by
    bucket 3 (a shallow 109.9 dip -- shallow enough that looking two
    buckets back never crosses the 5-tick FVG threshold and forms an
    unwanted extra gap). Zone B [110, 118] forms similarly from buckets
    5/6/7, touched by bucket 8. Bucket 10 then *rises* (118 -> 200) to a
    much higher baseline for the liquidity trigger structure -- rising
    (rather than dropping back toward zone A/B's own price range) can
    only ever form extra *bullish* (same-direction, harmless) gaps,
    never a bearish one that could invalidate the two real zones' LONG
    touch signals before entry (an earlier version of this fixture
    dropped back to the original 100 baseline instead, and an incidental
    bearish gap from that drop ended up getting touched by the liquidity
    swing spikes' own 5m aggregates, invalidating both real zones before
    entry -- rising avoids the whole failure mode structurally).

    The liquidity trigger (SL anchor / TP anchor / trigger swing / take
    / entry) then follows around the new 200 baseline, each spike
    isolated in its own 5m bucket via ``spike_bucket`` so no bucket ever
    moves both its high and low away from baseline at once.
    """
    bars: list[Bar1m] = []
    bars += flat_group(0, 100)
    bars += flat_group(5, 100)
    bars += flat_group(10, 110)  # zone A candle_3 -> [100, 110]
    bars += flat_group(15, 109.9)  # touch zone A (shallow)
    bars += flat_group(20, 110)  # recover
    bars += flat_group(25, 110)  # zone B candle_1
    bars += flat_group(30, 110)  # zone B candle_2
    bars += flat_group(35, 118)  # zone B candle_3 -> [110, 118]
    bars += flat_group(40, 117.9)  # touch zone B (shallow)
    bars += flat_group(45, 118)  # recover
    bars += flat_group(50, 200)  # rise to the liquidity baseline (harmless bullish gap only)
    bars += flat_group(55, 200)  # buffer
    bars += spike_bucket(60, 200, (190.5, 191, 190, 190.5))  # SL anchor swing low = 190
    bars += flat_group(65, 200)
    bars += spike_bucket(70, 200, (219.5, 220, 219, 219.5))  # TP anchor swing high = 220
    bars += flat_group(75, 200)
    bars += spike_bucket(80, 200, (195.5, 196, 195, 195.5))  # trigger swing low = 195
    bars.append(bar(85, 199, 200.5, 194, 194.5))  # take candle (low=194 < 195)
    bars.append(bar(86, 199, 200, 198, 199.5))  # entry candle (open=199 == entry_price)
    bars += flat_group(87, 200)
    return bars


FVG_VARIANT_ENTRY_TIMESTAMP = T0 + timedelta(minutes=86)


def fvg_boundary_strategy(fvg_variant_number: int, strategy_id: str = "s-fvg-sl") -> StrategySpec:
    return StrategySpec(
        strategy_id=strategy_id,
        direction=LONG,
        primary_entry_selector="liquidity-1m-swing_equal-swing_low-take-reversal",
        required_confluence_selectors=("fvg-5m-fvg_touch",),
        bias_policy=BiasPolicy.IGNORE,
        tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY, rank=1),
        sl_choice=SLChoice(
            source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector="fvg-5m-fvg_touch", fvg_variant_number=fvg_variant_number
        ),
    )


class FVGBoundaryVariantSelectionTests(unittest.TestCase):
    """Proves the review fix: FVG-boundary SL no longer silently picks
    the nearest qualifying boundary -- it picks exactly the position
    ``fvg_variant_number`` names in sl_targets' own deterministic order
    (ascending distance, then known_at, then sl_target_id), and a
    variant number with no matching position is MISSING_TARGET, not a
    silent fallback to "nearest" or a crash."""

    def test_variant_1_and_variant_2_select_different_boundaries(self):
        bars = fvg_boundary_variant_bars()
        result_v1 = run(fvg_boundary_strategy(1), bars)
        result_v2 = run(fvg_boundary_strategy(2), bars)

        self.assertEqual(result_v1.trade_count, 1)
        self.assertEqual(result_v2.trade_count, 1)
        trade_v1 = result_v1.trades[0]
        trade_v2 = result_v2.trades[0]

        self.assertEqual(trade_v1.sl_source_type, "fvg_boundary")
        self.assertEqual(trade_v2.sl_source_type, "fvg_boundary")
        self.assertNotEqual(trade_v1.sl_price, trade_v2.sl_price)
        self.assertNotEqual(trade_v1.sl_source_id, trade_v2.sl_source_id)
        # variant 1 is nearer to entry_price than variant 2 (sl_targets'
        # own ascending-distance order) -- entry_price is 199 for both.
        self.assertLess(abs(trade_v1.entry_price - trade_v1.sl_price), abs(trade_v2.entry_price - trade_v2.sl_price))

    def test_nonexistent_variant_number_is_missing_target(self):
        bars = fvg_boundary_variant_bars()
        result = run(fvg_boundary_strategy(99), bars)
        self.assertEqual(result.trade_count, 0)
        self.assertEqual(len(result.skipped_signals), 1)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.MISSING_TARGET)
        self.assertEqual(result.skipped_signals[0].timestamp, FVG_VARIANT_ENTRY_TIMESTAMP)

    def test_variant_1_result_is_deterministic_across_runs(self):
        bars = fvg_boundary_variant_bars()
        result_a = run(fvg_boundary_strategy(1), bars)
        result_b = run(fvg_boundary_strategy(1), bars)
        self.assertEqual(result_a.trades[0].sl_price, result_b.trades[0].sl_price)
        self.assertEqual(result_a.trades[0].sl_source_id, result_b.trades[0].sl_source_id)


def data_wick_tp_bars() -> list[Bar1m]:
    """60 flat baseline candles (data_wicks' own mandatory, gapless
    60-candle baseline run -- with every range 0, the baseline median is
    0, so any candle with a real range trivially clears the 2x-baseline
    qualification threshold), then two upper-wick candles -- each inside
    its own news event's [T, T+5min] window -- at two different
    distances from the eventual entry_price, then the same isolated-
    spike liquidity trigger structure used throughout this file."""
    bars = [flat(i) for i in range(60)]
    bars.append(bar(60, 100, 115, 100, 101))  # far upper wick, level=115
    bars.append(bar(61, 100, 108, 100, 100.5))  # near upper wick, level=108
    bars += [flat(i) for i in range(62, 65)]
    bars.append(bar(65, 100, 100, 90, 100))  # SL anchor swing low = 90
    bars += [flat(i) for i in range(66, 68)]
    bars.append(bar(68, 95.5, 96, 95, 95.5))  # trigger swing low = 95
    bars += [flat(i) for i in range(69, 71)]
    bars.append(bar(71, 99, 100.5, 94, 94.5))  # take candle (low=94 < 95)
    bars.append(bar(72, 99, 100, 98, 99.5))  # entry candle
    bars += [flat(i) for i in range(73, 76)]
    return bars


def _news_event(event_id: str, minutes_from_t0: int, event_type: str = "CPI") -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        scheduled_at=T0 + timedelta(minutes=minutes_from_t0),
        country="United States",
        currency="USD",
        event_type=event_type,
        source_agency="BLS",
        source_url="https://bls.gov/x",
        source_reference=f"ref-{event_id}",
        time_is_exact=True,
        schema_version="1.0",
    )


class DataWickTPRegressionTests(unittest.TestCase):
    """Confirms the FVG-boundary SL fix left data-wick TP selection
    untouched: it still always takes the single nearest qualifying data
    wick (tp_targets already returns data_wick_targets sorted nearest-
    first, uncapped -- this module never ranks it, matching TPChoice's
    own contract, unlike the FVG-boundary SL fix above)."""

    def test_nearest_qualifying_data_wick_is_selected_over_a_farther_one(self):
        bars = data_wick_tp_bars()
        events = [_news_event("EV1", 60), _news_event("EV2", 61, event_type="PPI")]
        strategy = StrategySpec(
            strategy_id="s-wick",
            direction=LONG,
            primary_entry_selector="liquidity-1m-swing_equal-swing_low-take-reversal",
            required_confluence_selectors=(),
            bias_policy=BiasPolicy.IGNORE,
            tp_choice=TPChoice(source_type=TPSourceType.DATA_WICK),
            sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1),
        )
        result = run(strategy, bars, news_events=events)
        self.assertEqual(result.trade_count, 1)
        trade = result.trades[0]
        self.assertEqual(trade.tp_source_type, "data_wick")
        self.assertEqual(trade.tp_price, 108.0)  # the nearer wick (115 is farther from entry_price=99)
        self.assertNotEqual(trade.tp_price, 115.0)


class PreparedContextEquivalenceTests(unittest.TestCase):
    """backtest_strategy is now a thin wrapper -- proves it still gives
    exactly the same result as calling prepare_backtest_context() +
    backtest_prepared_strategy() by hand, across several of this file's
    existing scenarios."""

    def test_long_tp_scenario_matches(self):
        bars = long_base_bars() + [bar(13, 115, 121, 114, 120)]
        strategy = long_strategy()
        via_wrapper = backtest_strategy(strategy, bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        via_split = backtest_prepared_strategy(strategy, context)

        self.assertEqual(via_wrapper, via_split)
        self.assertEqual(via_wrapper.trades[0].exit_reason, TradeExitReason.TP_HIT)

    def test_short_scenario_matches(self):
        bars = short_base_bars() + [bar(13, 95, 96, 89, 90)]
        strategy = short_strategy()
        via_wrapper = backtest_strategy(strategy, bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        via_split = backtest_prepared_strategy(strategy, context)

        self.assertEqual(via_wrapper, via_split)

    def test_fvg_boundary_variant_scenario_matches(self):
        bars = fvg_boundary_variant_bars()
        strategy = fvg_boundary_strategy(1)
        via_wrapper = backtest_strategy(strategy, bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        via_split = backtest_prepared_strategy(strategy, context)

        self.assertEqual(via_wrapper, via_split)
        self.assertEqual(via_wrapper.trades[0].sl_source_type, "fvg_boundary")

    def test_news_block_scenario_matches(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        event = _news_event("EV1", 27)
        strategy = long_strategy()
        via_wrapper = backtest_strategy(strategy, bars, [event], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        context = prepare_backtest_context(bars, [event], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        via_split = backtest_prepared_strategy(strategy, context)

        self.assertEqual(via_wrapper, via_split)
        self.assertEqual(via_wrapper.skipped_signals[0].reason, SkipReason.NEWS_BLOCK)

    def test_data_wick_tp_scenario_matches(self):
        bars = data_wick_tp_bars()
        events = [_news_event("EV1", 60), _news_event("EV2", 61, event_type="PPI")]
        strategy = StrategySpec(
            strategy_id="s-wick",
            direction=LONG,
            primary_entry_selector="liquidity-1m-swing_equal-swing_low-take-reversal",
            required_confluence_selectors=(),
            bias_policy=BiasPolicy.IGNORE,
            tp_choice=TPChoice(source_type=TPSourceType.DATA_WICK),
            sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1),
        )
        via_wrapper = backtest_strategy(strategy, bars, events, pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        context = prepare_backtest_context(bars, events, pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        via_split = backtest_prepared_strategy(strategy, context)

        self.assertEqual(via_wrapper, via_split)
        self.assertEqual(via_wrapper.trades[0].tp_source_type, "data_wick")


def _snapshot_context(context: PreparedBacktestContext) -> dict:
    """A plain, value-comparable snapshot of every PreparedBacktestContext
    field -- dict(...) over its read-only mapping fields (a mappingproxy
    itself isn't deepcopy/pickle-compatible), plain values everywhere
    else. Two snapshots taken before/after some operation compare equal
    iff no field's *value* changed."""
    return {
        "bars_1m": context.bars_1m,
        "news_events": context.news_events,
        "tick_size": context.tick_size,
        "pivot_strength": context.pivot_strength,
        "reclaim_window_bars": context.reclaim_window_bars,
        "fvg_results": dict(context.fvg_results),
        "fvg_touch_results": dict(context.fvg_touch_results),
        "fvg_entry_result": context.fvg_entry_result,
        "swing_equal_liquidity_results": dict(context.swing_equal_liquidity_results),
        "session_liquidity_results": dict(context.session_liquidity_results),
        "liquidity_entry_result": context.liquidity_entry_result,
        "data_wick_result": context.data_wick_result,
        "htf_bias_result": context.htf_bias_result,
        "timeline": context.timeline,
    }


class SharedContextMultiStrategyTests(unittest.TestCase):
    """One prepared context, several different strategies -- proves no
    cross-contamination and no dependence on call order."""

    def test_context_is_a_prepared_backtest_context(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        self.assertIsInstance(context, PreparedBacktestContext)

    def test_same_context_produces_correct_independent_results_for_two_strategies(self):
        bars = fvg_boundary_variant_bars()
        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        # long_strategy uses a plain LIQUIDITY SL/TP on the same primary
        # trigger this fixture's liquidity structure already fires --
        # a different SL source entirely from the two FVG-boundary
        # strategies below, proving the shared context serves genuinely
        # different strategy configurations correctly and independently.
        result_liquidity_sl = backtest_prepared_strategy(long_strategy(), context)
        result_variant_1 = backtest_prepared_strategy(fvg_boundary_strategy(1), context)
        result_variant_2 = backtest_prepared_strategy(fvg_boundary_strategy(2), context)

        self.assertEqual(result_liquidity_sl.trade_count, 1)
        self.assertEqual(result_liquidity_sl.trades[0].sl_source_type, "liquidity")
        self.assertEqual(result_variant_1.trades[0].sl_source_type, "fvg_boundary")
        self.assertEqual(result_variant_2.trades[0].sl_source_type, "fvg_boundary")
        self.assertNotEqual(result_variant_1.trades[0].sl_price, result_variant_2.trades[0].sl_price)
        self.assertNotEqual(result_liquidity_sl.trades[0].sl_price, result_variant_1.trades[0].sl_price)

    def test_result_does_not_depend_on_the_order_strategies_are_run_in(self):
        bars = fvg_boundary_variant_bars()
        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        strategy_1 = fvg_boundary_strategy(1)
        strategy_2 = fvg_boundary_strategy(2)

        forward_1 = backtest_prepared_strategy(strategy_1, context)
        forward_2 = backtest_prepared_strategy(strategy_2, context)

        reverse_2 = backtest_prepared_strategy(strategy_2, context)
        reverse_1 = backtest_prepared_strategy(strategy_1, context)

        self.assertEqual(forward_1, reverse_1)
        self.assertEqual(forward_2, reverse_2)

    def test_context_fields_are_unchanged_after_backtesting_several_strategies(self):
        # context's own dict fields are MappingProxyType (not deepcopy-able
        # -- Python's copy module can't pickle a mappingproxy), so this
        # snapshots by value instead: dict(...) over each read-only mapping,
        # plain equality over everything else. That is exactly what this
        # test needs to prove -- no field's *value* changed -- without
        # requiring context itself to be deepcopy-compatible.
        bars = fvg_boundary_variant_bars()
        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        snapshot = _snapshot_context(context)

        backtest_prepared_strategy(long_strategy(), context)
        backtest_prepared_strategy(fvg_boundary_strategy(1), context)
        backtest_prepared_strategy(fvg_boundary_strategy(2), context)

        self.assertEqual(_snapshot_context(context), snapshot)


class ContextImmutabilityTests(unittest.TestCase):
    """Proves the review fix: PreparedBacktestContext takes an immutable
    snapshot of its inputs (bars_1m/news_events as tuples, built before
    any detection runs) and exposes its dict-shaped results as genuinely
    read-only mappings -- a caller mutating the list they originally
    passed to prepare_backtest_context can never desynchronize an
    already-built context from the facts it describes, and nothing can
    write into its mapping fields after the fact."""

    def test_mutating_the_original_bars_list_after_preparation_does_not_affect_the_context(self):
        bars = long_base_bars() + [bar(13, 115, 121, 114, 120)]
        original_bars_snapshot = tuple(bars)
        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        bars.append(bar(14, 200, 200, 200, 200))
        bars.clear()  # even a full clear of the caller's own list

        self.assertEqual(context.bars_1m, original_bars_snapshot)
        result = backtest_prepared_strategy(long_strategy(), context)
        self.assertEqual(result.trade_count, 1)
        self.assertEqual(result.trades[0].exit_reason, TradeExitReason.TP_HIT)

    def test_mutating_the_original_news_list_after_preparation_does_not_affect_the_context(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        events = [_news_event("EV1", 27)]  # blocks the entry bar (T-15 boundary)
        context = prepare_backtest_context(bars, events, pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        events.clear()
        events.append(_news_event("EV2", 1000))

        self.assertEqual(len(context.news_events), 1)
        self.assertEqual(context.news_events[0].event_id, "EV1")
        result = backtest_prepared_strategy(long_strategy(), context)
        # if news_events had been aliased instead of snapshotted, clearing
        # the original list would have silently unblocked this entry.
        self.assertEqual(result.trade_count, 0)
        self.assertEqual(result.skipped_signals[0].reason, SkipReason.NEWS_BLOCK)

    def test_assigning_into_a_context_result_mapping_is_rejected(self):
        bars = long_base_bars() + [bar(13, 99, 101, 97, 100)]
        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        with self.assertRaises(TypeError):
            context.fvg_results["1m"] = None
        with self.assertRaises(TypeError):
            context.fvg_touch_results["1m"] = None
        with self.assertRaises(TypeError):
            context.swing_equal_liquidity_results["1m"] = None
        with self.assertRaises(TypeError):
            context.session_liquidity_results["1m"] = None


class DetectorsRunOnceTests(unittest.TestCase):
    """Every strategy-independent detector must run exactly once per
    prepare_backtest_context() call, never again for any number of
    backtest_prepared_strategy() calls against that same context."""

    def test_detectors_are_not_recalled_across_multiple_strategies(self):
        bars = fvg_boundary_variant_bars()
        originals = {
            name: getattr(backtest_module, name)
            for name in (
                "aggregate_bars",
                "detect_fvgs",
                "detect_fvg_touch_events",
                "detect_fvg_entry_candidates",
                "detect_liquidity",
                "detect_session_liquidity",
                "detect_liquidity_entry_candidates",
                "detect_data_wicks",
                "compute_htf_bias",
                "build_entry_signal_timeline",
                "validate_bars",
            )
        }
        patchers = [
            unittest.mock.patch.object(backtest_module, name, wraps=func) for name, func in originals.items()
        ]
        mocks = [p.start() for p in patchers]
        self.addCleanup(lambda: [p.stop() for p in patchers])

        context = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        call_counts_after_prepare = {name: getattr(backtest_module, name).call_count for name in originals}

        backtest_prepared_strategy(long_strategy(), context)
        backtest_prepared_strategy(fvg_boundary_strategy(1), context)
        backtest_prepared_strategy(fvg_boundary_strategy(2), context)

        for name in originals:
            current = getattr(backtest_module, name).call_count
            self.assertEqual(
                current,
                call_counts_after_prepare[name],
                f"{name} was called {current} times after 3 strategies, expected {call_counts_after_prepare[name]} (unchanged since preparation)",
            )
        # sanity: preparation itself did call each detector.
        self.assertGreater(call_counts_after_prepare["detect_fvgs"], 0)
        self.assertGreater(call_counts_after_prepare["aggregate_bars"], 0)


class PrepareBacktestContextValidationTests(unittest.TestCase):
    def test_non_positive_tick_size_rejected_at_preparation(self):
        bars = long_base_bars()
        with self.assertRaises(ValueError):
            prepare_backtest_context(bars, [], tick_size=0, pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

    def test_invalid_pivot_strength_rejected_at_preparation(self):
        bars = long_base_bars()
        with self.assertRaises(ValueError):
            prepare_backtest_context(bars, [], pivot_strength=0, reclaim_window_bars=RECLAIM_WINDOW_BARS)
        with self.assertRaises(ValueError):
            prepare_backtest_context(bars, [], pivot_strength="1", reclaim_window_bars=RECLAIM_WINDOW_BARS)

    def test_invalid_reclaim_window_bars_rejected_at_preparation(self):
        bars = long_base_bars()
        with self.assertRaises(ValueError):
            prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=0)

    def test_invalid_bars_rejected_at_preparation(self):
        # out-of-order timestamps -- validate_bars' own hard invariant.
        bad_bars = [bar(1, 100, 100, 100, 100), bar(0, 100, 100, 100, 100)]
        with self.assertRaises(ValueError):
            prepare_backtest_context(bad_bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)


_EXPECTED_PHASE_ORDER = [
    "1m_validation_loading",
    "timeframe_aggregation",
    "fvg_detection:15m",
    "fvg_detection:1D",
    "fvg_detection:1H",
    "fvg_detection:1m",
    "fvg_detection:4H",
    "fvg_detection:5m",
    "fvg_entry_candidate_construction",
    "liquidity_detection:swing_equal:15m",
    "liquidity_detection:swing_equal:1D",
    "liquidity_detection:swing_equal:1H",
    "liquidity_detection:swing_equal:1m",
    "liquidity_detection:swing_equal:4H",
    "liquidity_detection:swing_equal:5m",
    "liquidity_detection:session:15m",
    "liquidity_detection:session:1m",
    "liquidity_detection:session:5m",
    "liquidity_entry_candidate_construction",
    "data_wick_detection",
    "htf_bias_construction",
    "entry_signal_timeline_construction",
    "total",
]


class PhaseCallbackTests(unittest.TestCase):
    """``prepare_backtest_context``'s optional ``phase_callback`` --
    diagnostic-only instrumentation for ``tradelab.diagnose_context_
    preparation``. Must never change what ``PreparedBacktestContext`` a
    given input produces, must fire in a fixed, deterministic order
    (``_FVG_TIMEFRAMES``/``_LIQUIDITY_TIMEFRAMES``/``_SESSION_
    TIMEFRAMES`` are ``frozenset``s -- sorted explicitly so hash-order
    randomization can never reorder these calls across processes), and
    must report sane (non-negative) durations."""

    def test_callback_receives_every_phase_in_the_exact_deterministic_order(self):
        observed = []
        prepare_backtest_context(
            long_base_bars(), [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS,
            phase_callback=lambda phase, duration, peak_rss: observed.append(phase),
        )
        self.assertEqual(observed, _EXPECTED_PHASE_ORDER)

    def test_special_schedule_is_retained_and_forwarded_to_every_aggregate(self):
        schedule = SpecialSessionSchedule(
            schema_version=LEGACY_SPECIAL_SCHEDULE_SCHEMA_VERSION,
            source_url="https://www.cmegroup.com/trading-hours.html",
            retrieved_at=datetime(2026, 8, 22, tzinfo=UTC),
            closed_intervals=(),
        )
        with unittest.mock.patch.object(
            backtest_module,
            "aggregate_bars",
            wraps=backtest_module.aggregate_bars,
        ) as aggregate_mock:
            context = prepare_backtest_context(
                long_base_bars(),
                [],
                pivot_strength=PIVOT_STRENGTH,
                reclaim_window_bars=RECLAIM_WINDOW_BARS,
                special_schedule=schedule,
            )

        self.assertIs(context.special_session_schedule, schedule)
        self.assertEqual(aggregate_mock.call_count, 5)
        self.assertTrue(
            all(call.kwargs["special_schedule"] is schedule for call in aggregate_mock.call_args_list)
        )

    def test_phase_order_is_identical_across_repeated_calls(self):
        first_run: list[str] = []
        second_run: list[str] = []
        prepare_backtest_context(
            long_base_bars(), [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS,
            phase_callback=lambda phase, duration, peak_rss: first_run.append(phase),
        )
        prepare_backtest_context(
            long_base_bars(), [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS,
            phase_callback=lambda phase, duration, peak_rss: second_run.append(phase),
        )
        self.assertEqual(first_run, second_run)

    def test_each_phase_name_is_reported_exactly_once(self):
        observed = []
        prepare_backtest_context(
            long_base_bars(), [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS,
            phase_callback=lambda phase, duration, peak_rss: observed.append(phase),
        )
        self.assertEqual(len(observed), len(set(observed)))

    def test_durations_are_non_negative_floats_and_total_covers_the_sub_phases(self):
        durations: dict[str, float] = {}
        prepare_backtest_context(
            long_base_bars(), [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS,
            phase_callback=lambda phase, duration, peak_rss: durations.__setitem__(phase, duration),
        )
        for phase, duration in durations.items():
            self.assertIsInstance(duration, float, f"{phase} duration should be a float")
            self.assertGreaterEqual(duration, 0.0, f"{phase} duration should never be negative")
        sub_phase_total = sum(d for p, d in durations.items() if p != "total")
        self.assertGreaterEqual(durations["total"], sub_phase_total)

    def test_peak_rss_is_none_or_a_non_negative_int(self):
        peak_rss_values = []
        prepare_backtest_context(
            long_base_bars(), [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS,
            phase_callback=lambda phase, duration, peak_rss: peak_rss_values.append(peak_rss),
        )
        for value in peak_rss_values:
            if value is not None:
                self.assertIsInstance(value, int)
                self.assertGreaterEqual(value, 0)

    def test_omitting_the_callback_produces_an_identical_context(self):
        bars = long_base_bars()
        with_callback = prepare_backtest_context(
            bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS,
            phase_callback=lambda phase, duration, peak_rss: None,
        )
        without_callback = prepare_backtest_context(bars, [], pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS)

        self.assertEqual(with_callback.bars_1m, without_callback.bars_1m)
        self.assertEqual(dict(with_callback.fvg_results), dict(without_callback.fvg_results))
        self.assertEqual(dict(with_callback.swing_equal_liquidity_results), dict(without_callback.swing_equal_liquidity_results))
        self.assertEqual(dict(with_callback.session_liquidity_results), dict(without_callback.session_liquidity_results))
        self.assertEqual(with_callback.fvg_entry_result, without_callback.fvg_entry_result)
        self.assertEqual(with_callback.liquidity_entry_result, without_callback.liquidity_entry_result)
        self.assertEqual(with_callback.data_wick_result, without_callback.data_wick_result)
        self.assertEqual(with_callback.htf_bias_result, without_callback.htf_bias_result)
        self.assertEqual(with_callback.timeline, without_callback.timeline)


class ReleaseDiscoveryInputsTests(unittest.TestCase):
    """``release_discovery_inputs`` drops exactly the context fields that
    only entry discovery reads, and nothing else.

    The dangerous failure mode is the list going stale: someone starts
    reading one of these fields from the simulation path, the list is not
    updated, and every run silently simulates against ``None``. So the first
    test does not restate the list -- it parses ``tradelab/backtest_v2.py``
    and proves that every single ``context.<field>`` access there is an
    argument to ``discover_entry_strategy_universe``."""

    @staticmethod
    def _context() -> PreparedBacktestContext:
        bars = [
            Bar1m(timestamp=T0 + timedelta(minutes=i), open=100.0, high=101.0, low=99.0, close=100.0,
                  volume=10.0, contract="TEST")
            for i in range(30)
        ]
        return prepare_backtest_context(bars, [], tick_size=0.25, pivot_strength=2, reclaim_window_bars=3)

    def _simulation_module_source(self) -> str:
        from pathlib import Path

        import tradelab.backtest_v2 as module

        return Path(module.__file__).read_text(encoding="utf-8")

    def test_released_fields_are_only_ever_read_by_entry_discovery(self):
        import ast

        tree = ast.parse(self._simulation_module_source())
        discovery_arg_nodes: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name == "discover_entry_strategy_universe":
                    for argument in list(node.args) + [kw.value for kw in node.keywords]:
                        for inner in ast.walk(argument):
                            discovery_arg_nodes.add(id(inner))

        offenders = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in SIMULATION_UNUSED_CONTEXT_FIELDS
                and isinstance(node.value, ast.Name)
                and node.value.id == "context"
                and id(node) not in discovery_arg_nodes
            ):
                offenders.append(f"context.{node.attr} at line {node.lineno}")
        self.assertEqual(
            offenders, [],
            "these context fields are released after discovery but are read elsewhere in the "
            "simulation module; either stop reading them or remove them from "
            "SIMULATION_UNUSED_CONTEXT_FIELDS:\n  " + "\n  ".join(offenders),
        )

    def test_releases_exactly_the_named_fields_and_preserves_the_rest(self):
        context = self._context()
        released = release_discovery_inputs(context)

        for name in SIMULATION_UNUSED_CONTEXT_FIELDS:
            self.assertIsNotNone(getattr(context, name), f"fixture must actually populate {name}")
            self.assertIsNone(getattr(released, name), f"{name} should have been released")

        preserved = [f.name for f in dataclasses.fields(context) if f.name not in SIMULATION_UNUSED_CONTEXT_FIELDS]
        self.assertIn("bars_1m", preserved)
        for name in preserved:
            self.assertIs(
                getattr(released, name), getattr(context, name),
                f"{name} must be carried through untouched, not rebuilt",
            )

    def test_is_idempotent(self):
        context = self._context()
        once = release_discovery_inputs(context)
        twice = release_discovery_inputs(once)
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
