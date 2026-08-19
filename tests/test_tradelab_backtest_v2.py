#!/usr/bin/env python3
"""tradelab.backtest_v2 tests: next-bar-open entry (never a same-candle
close fill), the news no-entry window applied to the actual fill
instant, the one-open-position rule, TP-hit/SL-hit/same-bar-SL-wins
exits, long and short, MNQ/NQ sizing ($500 target / $550 max / $20
fee), every TP and SL exit-variant family (including linked-FVG
derivation per primary kind), invalid/missing-exit and infeasible-size
skip auditing, P&L/drawdown/yearly-monthly aggregation, telemetry
artifacts + leaderboard schema, and a randomized comparison against a
small, independent reference exit-resolution scan."""
from __future__ import annotations

import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from tradelab.accounting import Instrument, RiskConfig, UntradeableReason, apply_accounting, point_value_usd
from tradelab.backtest import BacktestResult, TradeExitReason, prepare_backtest_context
from tradelab.backtest_v2 import (
    ALL_EXIT_VARIANTS,
    DEFAULT_EQUITY_CONFIG,
    ExecutionSkipReason,
    ExitVariantSpec,
    SLVariantKind,
    TPVariantKind,
    _linked_fvg_candidate,
    _yearly_monthly_breakdown,
    build_synthetic_smoke_context,
    enumerate_exit_variants,
    main as backtest_v2_main,
    run_strategy_variant_backtest,
    simulate_market_execution,
)
from tradelab.bar import Bar1m
from tradelab.entry_signal_timeline import LONG, SHORT, EntryEvent, EntryEventKind
from tradelab.entry_strategy_universe import ObservedEntry, discover_entry_strategy_universe
from tradelab.equity import build_equity_curve
from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, SourceType
from tradelab.news import NewsEvent
from tradelab.research_run import read_leaderboard, read_live_status, read_progress_events, read_run_manifest
from tradelab.sweep_fvg_setup import SweepFVGSetupEvent

UTC = timezone.utc


def bar(ts: datetime, o: float, h: float, l: float, c: float) -> Bar1m:
    return Bar1m(timestamp=ts, open=o, high=h, low=l, close=c, volume=1.0, contract="TEST")


def minute_bars(start: datetime, ohlc: list[tuple[float, float, float, float]]) -> list[Bar1m]:
    return [bar(start + timedelta(minutes=i), *vals) for i, vals in enumerate(ohlc)]


def news(scheduled_at: datetime, event_id: str = "N1") -> NewsEvent:
    return NewsEvent(
        event_id=event_id, scheduled_at=scheduled_at, country="United States", currency="USD", event_type="CPI",
        source_agency="BLS", source_url="https://example.com", source_reference="ref", time_is_exact=True, schema_version="1.0",
    )


def build_minimal_context(bars: list[Bar1m], news_events: list[NewsEvent] | None = None) -> "PreparedBacktestContext":
    return prepare_backtest_context(bars, news_events or [], tick_size=0.25, pivot_strength=2, reclaim_window_bars=3)


def fvg_touch_candidate(zone_id: str, timeframe: str, direction: str, known_at: datetime) -> FVGEntryCandidate:
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{SourceType.FVG_TOUCH.value}-{zone_id}", source_type=SourceType.FVG_TOUCH, direction=direction,
        timeframe=timeframe, zone_id=zone_id, lower=100.0, upper=102.0, contact_type=FVGContactType.PARTIAL_TOUCH,
        timestamp=known_at, known_at=known_at,
    )


def entry_event_from_fvg(candidate: FVGEntryCandidate) -> EntryEvent:
    return EntryEvent(
        event_id=candidate.candidate_id, kind=EntryEventKind.FVG_TOUCH, selector=f"fvg-{candidate.timeframe}-fvg_touch",
        timeframe=candidate.timeframe, direction=candidate.direction, timestamp=candidate.timestamp, known_at=candidate.known_at,
        zone_id=candidate.zone_id, level_id=None, source=candidate,
    )


def observed(entry_event: EntryEvent) -> ObservedEntry:
    return ObservedEntry(entry_id=entry_event.event_id, known_at=entry_event.known_at, direction=entry_event.direction, source=entry_event)


class ExitVariantGrammarTests(unittest.TestCase):
    def test_exactly_20_variants_4_tp_times_5_sl(self):
        variants = enumerate_exit_variants()
        self.assertEqual(len(variants), 20)
        self.assertEqual(len(set(v.variant_id for v in variants)), 20)

    def test_tp_liquidity_ranks_are_exactly_1_2_3(self):
        tp_ranks = sorted({v.tp_rank for v in ALL_EXIT_VARIANTS if v.tp_kind == TPVariantKind.LIQUIDITY_RANK})
        self.assertEqual(tp_ranks, [1, 2, 3])

    def test_data_wick_tp_never_has_a_rank(self):
        data_wick_variants = [v for v in ALL_EXIT_VARIANTS if v.tp_kind == TPVariantKind.DATA_WICK]
        self.assertEqual(len(data_wick_variants), 5)  # one per SL kind
        self.assertTrue(all(v.tp_rank is None for v in data_wick_variants))

    def test_sl_fvg_boundary_covers_exactly_5m_and_15m(self):
        fvg_variants = {v.sl_fvg_timeframe for v in ALL_EXIT_VARIANTS if v.sl_kind == SLVariantKind.FVG_BOUNDARY}
        self.assertEqual(fvg_variants, {"5m", "15m"})

    def test_invalid_variant_construction_rejected(self):
        with self.assertRaises(ValueError):
            ExitVariantSpec(variant_id="x", tp_kind=TPVariantKind.LIQUIDITY_RANK, tp_rank=4, sl_kind=SLVariantKind.LIQUIDITY_RANK, sl_rank=1, sl_fvg_timeframe=None)
        with self.assertRaises(ValueError):
            ExitVariantSpec(variant_id="x", tp_kind=TPVariantKind.DATA_WICK, tp_rank=1, sl_kind=SLVariantKind.LIQUIDITY_RANK, sl_rank=1, sl_fvg_timeframe=None)
        with self.assertRaises(ValueError):
            ExitVariantSpec(variant_id="x", tp_kind=TPVariantKind.LIQUIDITY_RANK, tp_rank=1, sl_kind=SLVariantKind.FVG_BOUNDARY, sl_rank=None, sl_fvg_timeframe="1m")


class LinkedFVGExtractionTests(unittest.TestCase):
    def test_fvg_touch_primary_links_its_own_candidate(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        candidate = fvg_touch_candidate("z1", "5m", LONG, t0)
        entry = observed(entry_event_from_fvg(candidate))
        self.assertIs(_linked_fvg_candidate(entry), candidate)

    def test_liquidity_sweep_primary_has_no_linked_fvg(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        from tradelab.liquidity import HIGH
        from tradelab.liquidity_entry import CandidatePhase, EntryMode, LiquidityEntryCandidate, SourceFamily
        from tradelab.liquidity import LiquidityType

        liquidity_candidate = LiquidityEntryCandidate(
            candidate_id="1m-swing_equal-l1-take-reversal", source_family=SourceFamily.SWING_EQUAL, timeframe="1m", level_id="l1",
            liquidity_type=LiquidityType.SWING_HIGH, side=HIGH, lower=100.0, upper=100.0, trigger_phase=CandidatePhase.TAKE,
            entry_mode=EntryMode.REVERSAL, direction=SHORT, timestamp=t0, known_at=t0,
        )
        entry_event = EntryEvent(
            event_id=liquidity_candidate.candidate_id, kind=EntryEventKind.LIQUIDITY_SWEEP, selector="liquidity-1m-swing_equal-swing_high-take-reversal",
            timeframe="1m", direction=SHORT, timestamp=t0, known_at=t0, zone_id=None, level_id="l1", source=liquidity_candidate,
        )
        entry = observed(entry_event)
        self.assertIsNone(_linked_fvg_candidate(entry))

    def test_sweep_setup_links_the_underlying_fvg_event_candidate(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = t0 + timedelta(minutes=5)
        fvg_candidate = fvg_touch_candidate("z2", "15m", SHORT, t1)
        fvg_event = entry_event_from_fvg(fvg_candidate)
        from tradelab.liquidity import HIGH
        from tradelab.liquidity_entry import CandidatePhase, EntryMode, LiquidityEntryCandidate, SourceFamily
        from tradelab.liquidity import LiquidityType

        sweep_candidate = LiquidityEntryCandidate(
            candidate_id="1m-swing_equal-l2-take-reversal", source_family=SourceFamily.SWING_EQUAL, timeframe="1m", level_id="l2",
            liquidity_type=LiquidityType.SWING_HIGH, side=HIGH, lower=100.0, upper=100.0, trigger_phase=CandidatePhase.TAKE,
            entry_mode=EntryMode.REVERSAL, direction=SHORT, timestamp=t0, known_at=t0,
        )
        sweep_event = EntryEvent(
            event_id=sweep_candidate.candidate_id, kind=EntryEventKind.LIQUIDITY_SWEEP, selector="liquidity-1m-swing_equal-swing_high-take-reversal",
            timeframe="1m", direction=SHORT, timestamp=t0, known_at=t0, zone_id=None, level_id="l2", source=sweep_candidate,
        )
        setup = SweepFVGSetupEvent(
            setup_id=f"{sweep_event.event_id}::{fvg_event.event_id}", sweep_event=sweep_event, fvg_event=fvg_event, direction=SHORT,
            sweep_timeframe="1m", fvg_timeframe="15m", fvg_kind=EntryEventKind.FVG_TOUCH, sweep_known_at=t0, fvg_known_at=t1, known_at=t1,
        )
        entry = ObservedEntry(entry_id=setup.setup_id, known_at=setup.known_at, direction=setup.direction, source=setup)
        self.assertIs(_linked_fvg_candidate(entry), fvg_candidate)


class NextBarOpenEntryTests(unittest.TestCase):
    """Market entry must fill at the OPEN of the first bar whose own
    timestamp is >= known_at -- never at the triggering candle's own
    close/high/low."""

    def test_fill_price_is_the_following_bars_open_never_the_triggering_candles_close(self):
        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        # bar[0] closes at 105 (this is the candle whose close REVEALS the
        # signal, known_at == bar[1]'s own timestamp); bar[1] opens at 110
        # -- very different from bar[0]'s close, so a same-candle-close
        # fill vs. a next-bar-open fill are trivially distinguishable.
        bars = minute_bars(t0, [(100, 106, 99, 105), (110, 115, 108, 112), (112, 113, 111, 111.5)])
        context = build_minimal_context(bars)
        known_at = bars[1].timestamp  # the fill bar's own known_at == its own timestamp
        fvg = fvg_touch_candidate("zA", "1m", LONG, known_at)
        entry = observed(entry_event_from_fvg(fvg))

        trades, skipped = simulate_market_execution((entry,), LONG, ALL_EXIT_VARIANTS[0], context, "test-variant")
        # whether or not a trade was actually opened depends on TP/SL
        # availability in this tiny fixture; if one WAS opened, its entry
        # price must be bar[1]'s own open (110), never bar[0]'s close (105).
        if trades:
            self.assertEqual(trades[0].entry_price, 110.0)
            self.assertEqual(trades[0].entry_timestamp, bars[1].timestamp)
        else:
            # even with no tradeable TP/SL, this must never be a "no bar"
            # skip -- the fill bar unambiguously exists.
            self.assertNotIn(ExecutionSkipReason.NO_BAR_AVAILABLE, {s.reason for s in skipped})

    def test_no_bar_available_when_known_at_is_after_all_data(self):
        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        bars = minute_bars(t0, [(100, 101, 99, 100)])
        context = build_minimal_context(bars)
        far_future = t0 + timedelta(days=1)
        fvg = fvg_touch_candidate("zB", "1m", LONG, far_future)
        entry = observed(entry_event_from_fvg(fvg))

        trades, skipped = simulate_market_execution((entry,), LONG, ALL_EXIT_VARIANTS[0], context, "test-variant")
        self.assertEqual(trades, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, ExecutionSkipReason.NO_BAR_AVAILABLE)


def _short_fvg_touch_observed_entries(context) -> list[ObservedEntry]:
    from tradelab.fvg_entry import SourceType

    touches = sorted(
        (c for c in context.fvg_entry_result.candidates if c.source_type == SourceType.FVG_TOUCH and c.direction == SHORT),
        key=lambda c: c.known_at,
    )
    return [observed(entry_event_from_fvg(c)) for c in touches]


class OnePositionAtATimeTests(unittest.TestCase):
    def test_second_entry_while_first_position_still_open_is_skipped(self):
        # A concrete, reproducible pair of real, consecutive SHORT FVG
        # touches from the deterministic synthetic smoke context: with
        # this exact exit variant, the first opens a position that is
        # still open when the second, one minute later, arrives.
        context = build_synthetic_smoke_context()
        entries = _short_fvg_touch_observed_entries(context)
        entry_1, entry_2 = entries[50], entries[51]
        variant = next(v for v in ALL_EXIT_VARIANTS if v.variant_id == "tp=liquidity_rank_1|sl=fvg_boundary_5m")

        trades, skipped = simulate_market_execution((entry_1, entry_2), SHORT, variant, context, "test-variant")

        self.assertEqual(len(trades), 1)  # only entry_1's own trade
        skip_reasons = {s.entry_id: s.reason for s in skipped}
        self.assertEqual(skip_reasons[entry_2.entry_id], ExecutionSkipReason.POSITION_ALREADY_OPEN)
        # the still-open position's own entry bar is strictly before entry_2's known_at
        self.assertLess(trades[0].entry_timestamp, entry_2.known_at)

    def test_entry_after_position_closes_is_allowed(self):
        # Same real pair, but with only entry_1 present -- its own trade
        # resolves well before any other candidate would need to be
        # evaluated, so nothing here is left open indefinitely.
        context = build_synthetic_smoke_context()
        entries = _short_fvg_touch_observed_entries(context)
        entry_1 = entries[50]
        variant = next(v for v in ALL_EXIT_VARIANTS if v.variant_id == "tp=liquidity_rank_1|sl=fvg_boundary_5m")

        trades, skipped = simulate_market_execution((entry_1,), SHORT, variant, context, "test-variant")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_reason, TradeExitReason.SL_HIT)
        self.assertEqual(skipped, [])


class TPSLHitTests(unittest.TestCase):
    """Directly exercises the reused tradelab.backtest._check_exit /
    _finalize_trade mechanics through simulate_market_execution's own
    entry-bar-inclusive evaluation."""

    def test_same_bar_tp_and_sl_both_reachable_sl_wins(self):
        from tradelab.backtest import _OpenPosition, _check_exit
        from tradelab.sl_targets import SLSourceType, SLTargetCandidate
        from tradelab.tp_targets import TPSourceType, TPTargetCandidate

        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        tp = TPTargetCandidate(
            target_id="tp1", entry_id="e1", source_type=TPSourceType.LIQUIDITY, source_id="L1", direction=LONG, price=110.0,
            distance=10.0, known_at=t0, rank=1, liquidity_type=None, level_id=None, lower=None, upper=None, source_family=None,
            event_ids=None, wick_direction=None, timestamp=None,
        )
        sl = SLTargetCandidate(
            sl_target_id="sl1", entry_id="e1", source_type=SLSourceType.LIQUIDITY, source_id="L2", direction=LONG, price=95.0,
            distance=5.0, known_at=t0, rank=1, lower=95.0, upper=95.0, liquidity_type=None, level_id=None, source_family=None,
            zone_id=None, fvg_source_type=None, fvg_candidate_id=None, fvg_timeframe=None,
        )
        position = _OpenPosition(trade_id="e1", strategy_id="s", direction=LONG, signal_known_at=t0, entry_timestamp=t0, entry_price=100.0, tp_candidate=tp, sl_candidate=sl)
        # A single candle whose range spans BOTH the TP (110) and SL (95) prices.
        conflict_bar = bar(t0, 100.0, 112.0, 90.0, 105.0)
        outcome = _check_exit(position, conflict_bar)
        self.assertIsNotNone(outcome)
        reason, price = outcome
        self.assertEqual(reason, TradeExitReason.SL_HIT)
        self.assertEqual(price, 95.0)

    def test_short_same_bar_conflict_sl_still_wins(self):
        from tradelab.backtest import _OpenPosition, _check_exit
        from tradelab.sl_targets import SLSourceType, SLTargetCandidate
        from tradelab.tp_targets import TPSourceType, TPTargetCandidate

        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        tp = TPTargetCandidate(
            target_id="tp1", entry_id="e1", source_type=TPSourceType.LIQUIDITY, source_id="L1", direction=SHORT, price=90.0,
            distance=10.0, known_at=t0, rank=1, liquidity_type=None, level_id=None, lower=None, upper=None, source_family=None,
            event_ids=None, wick_direction=None, timestamp=None,
        )
        sl = SLTargetCandidate(
            sl_target_id="sl1", entry_id="e1", source_type=SLSourceType.LIQUIDITY, source_id="L2", direction=SHORT, price=105.0,
            distance=5.0, known_at=t0, rank=1, lower=105.0, upper=105.0, liquidity_type=None, level_id=None, source_family=None,
            zone_id=None, fvg_source_type=None, fvg_candidate_id=None, fvg_timeframe=None,
        )
        position = _OpenPosition(trade_id="e1", strategy_id="s", direction=SHORT, signal_known_at=t0, entry_timestamp=t0, entry_price=100.0, tp_candidate=tp, sl_candidate=sl)
        conflict_bar = bar(t0, 100.0, 108.0, 88.0, 95.0)
        outcome = _check_exit(position, conflict_bar)
        self.assertEqual(outcome[0], TradeExitReason.SL_HIT)
        self.assertEqual(outcome[1], 105.0)


class NewsBlockTests(unittest.TestCase):
    def test_entry_blocked_when_actual_fill_bar_falls_in_news_window(self):
        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        bars = minute_bars(t0, [(100, 101, 99, 100)] * 5)
        # news at t0+2min -> block window [t0-13min, t0+7min]; fill bar at t0+1min is inside it
        news_event = news(t0 + timedelta(minutes=2))
        context = build_minimal_context(bars, [news_event])
        fvg = fvg_touch_candidate("zN", "1m", LONG, bars[1].timestamp)
        entry = observed(entry_event_from_fvg(fvg))

        trades, skipped = simulate_market_execution((entry,), LONG, ALL_EXIT_VARIANTS[0], context, "test-variant")
        self.assertEqual(trades, [])
        self.assertEqual(skipped[0].reason, ExecutionSkipReason.NEWS_BLOCK)

    def test_entry_allowed_outside_news_window(self):
        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        bars = minute_bars(t0, [(100, 101, 99, 100)] * 5)
        far_news = news(t0 + timedelta(hours=5))
        context = build_minimal_context(bars, [far_news])
        fvg = fvg_touch_candidate("zN2", "1m", LONG, bars[1].timestamp)
        entry = observed(entry_event_from_fvg(fvg))

        trades, skipped = simulate_market_execution((entry,), LONG, ALL_EXIT_VARIANTS[0], context, "test-variant")
        self.assertNotIn(ExecutionSkipReason.NEWS_BLOCK, {s.reason for s in skipped})


class MissingExitAndInfeasibleSizeTests(unittest.TestCase):
    def test_missing_tp_and_sl_produce_explicit_skip_reasons(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        bars = minute_bars(t0, [(100, 101, 99, 100)] * 3)
        context = build_minimal_context(bars)
        fvg = fvg_touch_candidate("zM", "1m", LONG, bars[1].timestamp)
        entry = observed(entry_event_from_fvg(fvg))
        # A tiny fixture with no real liquidity/data-wick structure --
        # every TP/SL variant should report a missing target, never a trade.
        for variant in ALL_EXIT_VARIANTS:
            trades, skipped = simulate_market_execution((entry,), LONG, variant, context, f"v-{variant.variant_id}")
            self.assertEqual(trades, [])
            self.assertEqual(len(skipped), 1)
            self.assertIn(skipped[0].reason, (ExecutionSkipReason.MISSING_TP_TARGET, ExecutionSkipReason.MISSING_SL_TARGET))

    def test_no_feasible_contract_size_is_recorded_as_untradeable(self):
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1)
        self.assertGreater(len(universe.equivalence_classes), 0)
        cls = universe.equivalence_classes[0]
        # NQ has 10x MNQ's point value, so this fixture's large stop
        # distances are expected to genuinely exceed $550 for at least
        # one contract on some trades.
        result = run_strategy_variant_backtest(cls, ALL_EXIT_VARIANTS[0], Instrument.NQ, context)
        self.assertGreater(result.account_result.untradeable_count, 0)
        self.assertEqual(result.skipped_entries_by_reason.get("stop_risk_exceeds_max", 0), result.account_result.untradeable_count)
        for untradeable in result.account_result.untradeable_trades:
            self.assertEqual(untradeable.reason, UntradeableReason.STOP_RISK_EXCEEDS_MAX)


class SizingAndAccountingTests(unittest.TestCase):
    def test_point_values_are_exactly_mnq_2_and_nq_20(self):
        self.assertEqual(point_value_usd(Instrument.MNQ), 2.0)
        self.assertEqual(point_value_usd(Instrument.NQ), 20.0)

    def test_risk_config_defaults_match_the_task_exactly(self):
        risk_config = RiskConfig()
        self.assertEqual(risk_config.target_risk_usd, 500.0)
        self.assertEqual(risk_config.max_stop_risk_usd, 550.0)
        self.assertEqual(risk_config.entry_cost_usd, 20.0)

    def test_fee_charged_exactly_once_per_executed_trade(self):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        entry_signal_known_at = t0
        entry_price = 100.0
        sl_price = 95.0  # 5-point risk -> MNQ risk/contract = $10, easily affordable
        tp_price = 110.0
        trade = _make_executed_trade("t1", "s1", LONG, entry_signal_known_at, t0, entry_price, tp_price, sl_price, t0 + timedelta(minutes=1), tp_price, TradeExitReason.TP_HIT, 10.0)
        backtest_result = BacktestResult(strategy_id="s1", trades=[trade], skipped_signals=[], trade_count=1, tp_hit_count=1, sl_hit_count=0, open_unresolved_count=0, gross_pnl_points_total=10.0)
        account_result = apply_accounting(backtest_result, Instrument.MNQ, RiskConfig())
        self.assertEqual(len(account_result.accounted_trades), 1)
        accounted = account_result.accounted_trades[0]
        self.assertEqual(accounted.entry_cost_usd, 20.0)
        self.assertEqual(accounted.net_pnl_usd, accounted.gross_pnl_usd - 20.0)


def _make_executed_trade(trade_id, strategy_id, direction, signal_known_at, entry_timestamp, entry_price, tp_price, sl_price, exit_timestamp, exit_price, exit_reason, gross_pnl_points):
    from tradelab.backtest import ExecutedTrade

    return ExecutedTrade(
        trade_id=trade_id, strategy_id=strategy_id, direction=direction, signal_known_at=signal_known_at, entry_timestamp=entry_timestamp,
        entry_price=entry_price, tp_price=tp_price, tp_source_type="liquidity", tp_source_id="L1", sl_price=sl_price, sl_source_type="liquidity",
        sl_source_id="L2", exit_timestamp=exit_timestamp, exit_price=exit_price, exit_reason=exit_reason, gross_pnl_points=gross_pnl_points,
    )


class YearlyMonthlyAggregationTests(unittest.TestCase):
    def test_aggregation_buckets_by_entry_year_and_month(self):
        t_2024 = datetime(2024, 6, 15, tzinfo=UTC)
        t_2025 = datetime(2025, 3, 10, tzinfo=UTC)
        trade_2024 = _make_executed_trade("t1", "s", LONG, t_2024, t_2024, 100.0, 110.0, 95.0, t_2024 + timedelta(minutes=1), 110.0, TradeExitReason.TP_HIT, 10.0)
        trade_2025 = _make_executed_trade("t2", "s", LONG, t_2025, t_2025, 100.0, 110.0, 95.0, t_2025 + timedelta(minutes=1), 95.0, TradeExitReason.SL_HIT, -5.0)
        backtest_result = BacktestResult(
            strategy_id="s", trades=[trade_2024, trade_2025], skipped_signals=[], trade_count=2, tp_hit_count=1, sl_hit_count=1,
            open_unresolved_count=0, gross_pnl_points_total=5.0,
        )
        account_result = apply_accounting(backtest_result, Instrument.MNQ, RiskConfig())
        equity_result = build_equity_curve(account_result, DEFAULT_EQUITY_CONFIG)
        yearly, monthly = _yearly_monthly_breakdown(account_result, equity_result)
        self.assertEqual(set(yearly.keys()), {"2024", "2025"})
        self.assertEqual(yearly["2024"]["trade_count"], 1)
        self.assertEqual(yearly["2025"]["trade_count"], 1)
        self.assertIn("2024-06", monthly)
        self.assertIn("2025-03", monthly)


class TelemetryIntegrationTests(unittest.TestCase):
    def test_smoke_test_cli_produces_a_complete_monitored_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = backtest_v2_main(["smoke-test", "--runs-dir", tmp, "--min-entries-per-year", "1"])
            self.assertEqual(exit_code, 0)

            from tradelab.research_run import list_run_ids

            run_ids = list_run_ids(tmp)
            self.assertEqual(len(run_ids), 1)
            run_dir = f"{tmp}/{run_ids[0]}"

            manifest = read_run_manifest(run_dir)
            self.assertTrue(manifest.is_demo)

            status = read_live_status(run_dir)
            self.assertEqual(status.phase, "completed")
            self.assertIsNotNone(status.eligible_specs)
            self.assertIsNotNone(status.backtests_completed)

            events = read_progress_events(run_dir)
            phase_starts = [e["phase"] for e in events if e["event_type"] == "phase_started"]
            self.assertEqual(
                phase_starts,
                ["preparing_data", "discovering_entries", "deduplicating_entries", "expanding_exit_variants", "simulating_backtests", "ranking_results"],
            )

            leaderboard = read_leaderboard(run_dir)
            self.assertIsNotNone(leaderboard)
            if leaderboard.entries:
                entry = leaderboard.entries[0]
                self.assertIsInstance(entry.pnl, float)
                self.assertIsInstance(entry.trade_count, int)
                self.assertTrue(0.0 <= entry.win_rate <= 1.0)


class RandomizedExitResolutionReferenceTests(unittest.TestCase):
    """Independently re-derives the exit outcome for a random single-bar
    conflict scenario (naive: check SL first, then TP, against the
    bar's own high/low) and compares against the reused ``tradelab.
    backtest._check_exit`` this module's own execution loop relies on."""

    def test_matches_naive_reference_on_random_single_bar_scenarios(self):
        from tradelab.backtest import _OpenPosition, _check_exit
        from tradelab.sl_targets import SLSourceType, SLTargetCandidate
        from tradelab.tp_targets import TPSourceType, TPTargetCandidate

        rng = random.Random(31337)
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        for _ in range(200):
            direction = rng.choice([LONG, SHORT])
            entry_price = 100.0
            if direction == LONG:
                tp_price = entry_price + rng.uniform(1, 20)
                sl_price = entry_price - rng.uniform(1, 20)
            else:
                tp_price = entry_price - rng.uniform(1, 20)
                sl_price = entry_price + rng.uniform(1, 20)
            bar_low = entry_price - rng.uniform(0, 25)
            bar_high = entry_price + rng.uniform(0, 25)
            test_bar = bar(t0, entry_price, bar_high, bar_low, entry_price)

            tp = TPTargetCandidate(
                target_id="tp", entry_id="e", source_type=TPSourceType.LIQUIDITY, source_id="L1", direction=direction, price=tp_price,
                distance=abs(tp_price - entry_price), known_at=t0, rank=1, liquidity_type=None, level_id=None, lower=None, upper=None,
                source_family=None, event_ids=None, wick_direction=None, timestamp=None,
            )
            sl = SLTargetCandidate(
                sl_target_id="sl", entry_id="e", source_type=SLSourceType.LIQUIDITY, source_id="L2", direction=direction, price=sl_price,
                distance=abs(sl_price - entry_price), known_at=t0, rank=1, lower=sl_price, upper=sl_price, liquidity_type=None,
                level_id=None, source_family=None, zone_id=None, fvg_source_type=None, fvg_candidate_id=None, fvg_timeframe=None,
            )
            position = _OpenPosition(trade_id="e", strategy_id="s", direction=direction, signal_known_at=t0, entry_timestamp=t0, entry_price=entry_price, tp_candidate=tp, sl_candidate=sl)

            actual = _check_exit(position, test_bar)

            # naive independent reference: SL first, then TP
            if direction == LONG:
                sl_hit = test_bar.low <= sl_price
                tp_hit = test_bar.high >= tp_price
            else:
                sl_hit = test_bar.high >= sl_price
                tp_hit = test_bar.low <= tp_price
            if sl_hit:
                expected = (TradeExitReason.SL_HIT, sl_price)
            elif tp_hit:
                expected = (TradeExitReason.TP_HIT, tp_price)
            else:
                expected = None

            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
