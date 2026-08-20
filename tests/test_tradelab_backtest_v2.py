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

import dataclasses
import hashlib
import json
import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tradelab.accounting import Instrument, RiskConfig, UntradeableReason, apply_accounting, point_value_usd
from tradelab.entry_target_cache import EntryTargetCache
from tradelab.backtest import BacktestResult, TradeExitReason, prepare_backtest_context
from tradelab.backtest_v2 import (
    ALL_EXIT_VARIANTS,
    DEFAULT_EQUITY_CONFIG,
    DEFAULT_INSTRUMENTS,
    DEFAULT_RISK_CONFIG,
    ExecutionSkipReason,
    ExitVariantSpec,
    SLVariantKind,
    TPVariantKind,
    VariantSummaryRow,
    _expand_and_simulate,
    _find_first_exit,
    _leaderboard_sort_key,
    _linked_fvg_candidate,
    _open_journal,
    _resolve_sl_candidate,
    _simulate_market_execution_reference,
    _TopNTracker,
    _yearly_monthly_breakdown,
    build_synthetic_smoke_context,
    enumerate_exit_variants,
    main as backtest_v2_main,
    read_preflight,
    run_execution_acceleration_benchmark,
    run_preflight_v2,
    run_production_backtest_v2,
    run_strategy_variant_backtest,
    run_strategy_variant_backtest_all_instruments,
    simulate_market_execution,
    simulate_variant_execution,
)
from tradelab.bar import Bar1m
from tradelab.bar_range_index import BarRangeIndex
from tradelab.entry_signal_timeline import LONG, SHORT, EntryEvent, EntryEventKind
from tradelab.entry_strategy_universe import ObservedEntry, discover_entry_strategy_universe
from tradelab.equity import build_equity_curve
from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, SourceType
from tradelab.liquidity import HIGH, LOW, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType
from tradelab.market_data_artifact import MarketDataArtifact, SeriesKind
from tradelab.news import NewsEvent
from tradelab.research_run import ProgressThrottle, RunReporter, create_run, read_leaderboard, read_live_status, read_progress_events, read_run_manifest
from tradelab.sweep_fvg_setup import SweepFVGSetupEvent

UTC = timezone.utc


def _write_smoke_market_artifact(tmp: Path) -> tuple[Path, Path]:
    """Writes the exact same zigzag 1m bar series ``build_synthetic_smoke_
    context`` builds in-memory (reliable, real FVGs/swing highs+lows on
    1m/5m/15m in few bars -- see that function's own docstring) to a real
    market-data-artifact CSV + manifest + an empty news CSV, so file-based
    entry points (``run_production_backtest_v2``/``run_preflight_v2``) can
    be exercised end-to-end against a non-trivial (non-zero equivalence
    class count) universe without needing real market data."""
    start = datetime(2024, 12, 31, 23, 5, tzinfo=UTC)
    lines = ["timestamp,open,high,low,close,volume,contract\n"]
    price = 20000.0
    leg_length = 30
    for i in range(900):
        ts = start + timedelta(minutes=i)
        cycle_bar = i % (2 * leg_length)
        cycle_index = i // (2 * leg_length)
        amplitude = 20.0 + cycle_index * 5.0
        move = amplitude if cycle_bar < leg_length else -amplitude
        open_p = price
        close_p = price + move
        high_p = max(open_p, close_p) + 3.0
        low_p = min(open_p, close_p) - 3.0
        lines.append(f"{ts.strftime('%Y-%m-%dT%H:%M:%SZ')},{open_p},{high_p},{low_p},{close_p},100,SMOKE\n")
        price = close_p

    bars_path = tmp / "bars.csv"
    bars_path.write_text("".join(lines), encoding="utf-8")
    sha256 = hashlib.sha256(bars_path.read_bytes()).hexdigest()

    artifact = MarketDataArtifact(
        schema_version="1.0", source_name="test-fixture", instrument_root=Instrument.NQ, bar_interval="1m",
        timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
        coverage_start=start, coverage_end=start + timedelta(minutes=900), sha256=sha256,
        normalization_method="additive_back_adjustment",
    )
    manifest_path = tmp / "manifest.json"
    manifest_path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")

    news_csv_path = tmp / "news.csv"
    news_csv_path.write_text(
        "event_id,scheduled_at,country,currency,event_type,source_agency,source_url,source_reference,time_is_exact,schema_version\n",
        encoding="utf-8",
    )
    return manifest_path, news_csv_path


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


def _liquidity_level(level_id: str, side: str, price: float, formed_at: datetime) -> LiquidityLevel:
    return LiquidityLevel(
        level_id=level_id, timeframe="1m", liquidity_type=(LiquidityType.SWING_HIGH if side == HIGH else LiquidityType.SWING_LOW),
        side=side, lower=price, upper=price, member_count=1, member_timestamps=(formed_at,), formed_at=formed_at,
        status=LiquidityStatus.ACTIVE, taken_at=None, absorbed_by_level_id=None,
    )


class SourceEligibilityAtActualFillInstantTests(unittest.TestCase):
    """A data-gap scenario where the signal's own known_at precedes the
    first available fill bar by several minutes, and a target source
    becomes known strictly *between* the two: it must be eligible at
    the actual fill (this is the fix -- MarketEntryContext.entry_known_
    at must be fill_bar.timestamp, not entry.known_at), and a second
    source that only becomes known *after* the fill must never be
    eligible, regardless of how close its price is to entry."""

    def test_target_known_between_signal_and_gap_delayed_fill_is_eligible_at_fill_not_at_signal(self):
        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        signal_known_at = t0 + timedelta(minutes=2)  # no bar exists at this exact instant (see gap below)
        during_gap_known_at = t0 + timedelta(minutes=5)  # strictly between signal_known_at and the fill bar
        after_fill_known_at = t0 + timedelta(minutes=11)  # strictly after the fill bar

        # A genuine data gap: bars at t0, t0+1min, then a jump straight to
        # t0+10min (tradelab.validation.validate_bars permits gaps).
        bars = [bar(t0, 100, 101, 99, 100), bar(t0 + timedelta(minutes=1), 100, 101, 99, 100), bar(t0 + timedelta(minutes=10), 100, 101, 99, 100)]
        context = build_minimal_context(bars)
        fill_bar_timestamp = bars[2].timestamp  # the first bar at/after signal_known_at
        self.assertGreater(fill_bar_timestamp, signal_known_at)  # confirms this scenario actually exercises the gap

        # TP: trivially satisfiable, favorable (HIGH) side, formed well before the signal.
        tp_level = _liquidity_level("tp1", HIGH, 110.0, t0 - timedelta(minutes=30))
        # SL candidate A: becomes known DURING the gap (after the signal,
        # at-or-before the actual fill) -- must be eligible, and, being
        # farther from entry than B, must win rank 1 once B is correctly excluded.
        sl_during_gap = _liquidity_level("sl-during-gap", LOW, 90.0, during_gap_known_at)
        # SL candidate B: becomes known only AFTER the fill, and is
        # deliberately NEARER to entry (95 vs 90) than A -- if it were
        # wrongly included, it would win rank 1 instead of A.
        sl_after_fill = _liquidity_level("sl-after-fill", LOW, 95.0, after_fill_known_at)

        rigged_liquidity_results = {"1m": LiquidityDetectionResult(levels=[tp_level, sl_during_gap, sl_after_fill], takes=[], reclaims=[])}
        rigged_context = dataclasses.replace(context, swing_equal_liquidity_results=rigged_liquidity_results)

        entry_event = EntryEvent(
            event_id="sig-1", kind=EntryEventKind.FVG_TOUCH, selector="fvg-1m-fvg_touch", timeframe="1m", direction=LONG,
            timestamp=signal_known_at, known_at=signal_known_at, zone_id="zGap", level_id=None,
            source=fvg_touch_candidate("zGap", "1m", LONG, signal_known_at),
        )
        entry = observed(entry_event)
        variant = next(v for v in ALL_EXIT_VARIANTS if v.variant_id == "tp=liquidity_rank_1|sl=liquidity_rank_1")

        trades, skipped = simulate_market_execution((entry,), LONG, variant, rigged_context, "gap-test-variant")

        self.assertEqual(skipped, [])
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.entry_timestamp, fill_bar_timestamp)  # next-bar-open fill, still correct
        self.assertEqual(trade.signal_known_at, signal_known_at)  # original signal instant preserved for audit
        # The decisive assertion: SL resolved to the during-the-gap level
        # (90), never the nearer but after-fill level (95) -- proving
        # both halves of the fix at once.
        self.assertEqual(trade.sl_price, 90.0)

    def test_counterfactual_using_the_raw_signal_known_at_would_have_missed_the_target(self):
        """Directly demonstrates the bug this fix removes: resolving
        against the raw signal known_at (the pre-fix behavior) excludes
        a target that has since become known by the actual fill."""
        from tradelab.tp_targets import MarketEntryContext

        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        signal_known_at = t0 + timedelta(minutes=2)
        during_gap_known_at = t0 + timedelta(minutes=5)
        fill_bar_timestamp = t0 + timedelta(minutes=10)

        sl_during_gap = _liquidity_level("sl-during-gap", LOW, 90.0, during_gap_known_at)
        liquidity_results = {"1m": LiquidityDetectionResult(levels=[sl_during_gap], takes=[], reclaims=[])}
        empty_results: dict = {}

        entry_event = EntryEvent(
            event_id="sig-2", kind=EntryEventKind.FVG_TOUCH, selector="fvg-1m-fvg_touch", timeframe="1m", direction=LONG,
            timestamp=signal_known_at, known_at=signal_known_at, zone_id="zGap2", level_id=None,
            source=fvg_touch_candidate("zGap2", "1m", LONG, signal_known_at),
        )
        entry = observed(entry_event)
        variant = next(v for v in ALL_EXIT_VARIANTS if v.variant_id == "tp=liquidity_rank_1|sl=liquidity_rank_1")

        pre_fix_market_entry = MarketEntryContext(entry_id="x", direction=LONG, entry_price=100.0, entry_known_at=signal_known_at)
        self.assertIsNone(_resolve_sl_candidate(variant, entry, pre_fix_market_entry, liquidity_results, empty_results))

        post_fix_market_entry = MarketEntryContext(entry_id="x", direction=LONG, entry_price=100.0, entry_known_at=fill_bar_timestamp)
        resolved = _resolve_sl_candidate(variant, entry, post_fix_market_entry, liquidity_results, empty_results)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.price, 90.0)


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


class TopNTrackerTests(unittest.TestCase):
    """Direct proof that the bounded top-N structure never grows past
    its own cap, and that it produces the exact same ordering as an
    independent brute-force sort over everything ever offered to it."""

    def _make_result(self, strategy_variant_id: str, net_pnl_usd: float, max_drawdown_usd: float = 100.0):
        from tradelab.backtest_v2 import ExitVariantSpec as _EV
        from tradelab.entry_strategy_universe import EntryStrategyEquivalenceClass as _Cls

        variant = ALL_EXIT_VARIANTS[0]
        equity_result = build_equity_curve(
            apply_accounting(BacktestResult(strategy_id="s", trades=[], skipped_signals=[], trade_count=0, tp_hit_count=0, sl_hit_count=0, open_unresolved_count=0, gross_pnl_points_total=0.0), Instrument.MNQ, RiskConfig()),
            DEFAULT_EQUITY_CONFIG,
        )
        from tradelab.backtest_v2 import StrategyVariantResult

        return StrategyVariantResult(
            strategy_variant_id=strategy_variant_id, equivalence_class_id="cls", exit_variant=variant, instrument=Instrument.MNQ,
            direction=LONG, trades=(), account_result=apply_accounting(BacktestResult(strategy_id="s", trades=[], skipped_signals=[], trade_count=0, tp_hit_count=0, sl_hit_count=0, open_unresolved_count=0, gross_pnl_points_total=0.0), Instrument.MNQ, RiskConfig()),
            equity_result=equity_result, net_pnl_usd=net_pnl_usd, roi=net_pnl_usd / 50000.0, win_rate=0.5, trade_count=0,
            max_drawdown_usd=max_drawdown_usd, drawdowns_over_2000_count=0, skipped_entries_by_reason={}, yearly={}, monthly={},
        )

    def test_never_grows_past_its_own_cap(self):
        tracker = _TopNTracker(3)
        for i in range(50):
            tracker.offer(self._make_result(f"s{i}", net_pnl_usd=float(i)))
            self.assertLessEqual(len(tracker), 3)
        self.assertEqual(len(tracker), 3)

    def test_matches_brute_force_sort_over_everything_ever_offered(self):
        rng = random.Random(4242)
        results = [self._make_result(f"s{i}", net_pnl_usd=rng.uniform(-1000, 1000), max_drawdown_usd=rng.uniform(0, 500)) for i in range(80)]
        tracker = _TopNTracker(5)
        for r in results:
            tracker.offer(r)
        actual = [r.strategy_variant_id for r in tracker.top_results()]
        expected = [r.strategy_variant_id for r in sorted(results, key=_leaderboard_sort_key, reverse=True)[:5]]
        self.assertEqual(actual, expected)

    def test_deterministic_tie_break_by_drawdown_then_id(self):
        tracker = _TopNTracker(2)
        # identical net_pnl_usd -- lower max_drawdown wins; a further tie
        # (same pnl AND same drawdown) breaks by strategy_variant_id.
        tracker.offer(self._make_result("s-high-dd", net_pnl_usd=100.0, max_drawdown_usd=500.0))
        tracker.offer(self._make_result("s-low-dd", net_pnl_usd=100.0, max_drawdown_usd=100.0))
        tracker.offer(self._make_result("s-tied-b", net_pnl_usd=100.0, max_drawdown_usd=100.0))
        tracker.offer(self._make_result("s-tied-a", net_pnl_usd=100.0, max_drawdown_usd=100.0))
        top = tracker.top_results()
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0].strategy_variant_id, "s-tied-b")  # "s-tied-b" > "s-tied-a" lexicographically
        self.assertEqual(top[1].strategy_variant_id, "s-tied-a")

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            _TopNTracker(0)


class VariantSummaryRowTests(unittest.TestCase):
    def test_round_trips_to_a_json_serializable_dict(self):
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1)
        result = run_strategy_variant_backtest(universe.equivalence_classes[0], ALL_EXIT_VARIANTS[0], Instrument.MNQ, context)
        row = VariantSummaryRow.from_result(result)
        d = row.to_dict()
        import json

        json.dumps(d)  # must not raise -- every value is JSON-serializable
        self.assertEqual(d["strategy_variant_id"], result.strategy_variant_id)
        self.assertEqual(d["net_pnl_usd"], result.net_pnl_usd)
        self.assertEqual(d["exit_variant_id"], result.exit_variant.variant_id)
        self.assertNotIn("trades", d)  # deliberately lean -- no trade-level detail
        self.assertNotIn("equity_curve", d)


class MemoryBoundedProductionOrchestrationTests(unittest.TestCase):
    """Proves _expand_and_simulate (a) durably records every single
    evaluated variant, regardless of how many; (b) keeps the in-memory/
    reported leaderboard strictly bounded to top_n_leaderboard, even
    though far more variants than that are evaluated; and (c) that
    bounded leaderboard is exactly correct against an independent
    brute-force recomputation from the full durable summary stream --
    i.e. boundedness never comes at the cost of correctness."""

    def test_every_variant_is_durably_summarized_while_leaderboard_stays_bounded_and_correct(self):
        from tradelab.research_run import create_run

        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1)
        self.assertGreater(universe.equivalence_class_count, 0)

        top_n = 3  # deliberately much smaller than the total variant count below
        total_expected = universe.equivalence_class_count * len(ALL_EXIT_VARIANTS) * len(DEFAULT_INSTRUMENTS)
        self.assertGreater(total_expected, top_n * 5)  # confirms this test actually exercises bounding, not a no-op

        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={}, is_demo=True)
            journal = _open_journal(
                reporter, market_artifact_path=None, news_csv_path=None, parameters={},
                risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG,
                exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
            )
            completed, final_leaderboard = _expand_and_simulate(
                reporter, journal, universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, context,
                risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, top_n_leaderboard=top_n, checkpoint_every=7,
            )
            journal.close()

            self.assertEqual(completed, total_expected)
            self.assertLessEqual(len(final_leaderboard), top_n)

            # (a) durable: every single variant landed in the summary stream.
            import json as _json

            with open(reporter.run_dir / "variant_summaries.jsonl") as f:
                all_rows = [_json.loads(line) for line in f]
            self.assertEqual(len(all_rows), total_expected)

            # (b) bounded: the reported leaderboard never exceeds top_n.
            leaderboard = read_leaderboard(reporter.run_dir)
            self.assertLessEqual(len(leaderboard.entries), top_n)

            # (c) correct: the bounded leaderboard exactly matches an
            # independent brute-force recomputation over the FULL durable
            # summary stream (all_rows), using the same deterministic key.
            brute_force_ranked = sorted(all_rows, key=lambda r: (r["net_pnl_usd"], -r["max_drawdown_usd"], r["strategy_variant_id"]), reverse=True)
            expected_top_ids = [r["strategy_variant_id"] for r in brute_force_ranked[:top_n]]
            actual_top_ids = [e.strategy_id for e in leaderboard.entries]
            self.assertEqual(actual_top_ids, expected_top_ids)


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


class ResumabilityTests(unittest.TestCase):
    """Proves the SQLite journal actually delivers resumability at the
    orchestration layer (task 043): a "crash" mid-run (a handful of
    variants committed directly to the journal, exactly as ``_expand_
    and_simulate`` itself would have, but without ever calling that
    function) followed by a fresh process re-attaching to the same run
    directory (a) skips every already-completed variant rather than
    re-running or duplicating it, (b) still reaches exactly the same
    total completed count as an uninterrupted run would, and (c)
    reconstructs a leaderboard -- both the durable historical top-N and
    the final merged one -- that exactly matches an independent
    brute-force recomputation over the *entire* durable summary stream
    (both the pre-crash and post-resume rows), proving durable Top-N
    reconstruction is exact, not approximate."""

    def _universe_and_context(self):
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1
        )
        self.assertGreater(universe.equivalence_class_count, 0)
        return context, universe

    def test_interrupted_run_resumes_without_duplicating_completed_work(self):
        context, universe = self._universe_and_context()
        total_expected = universe.equivalence_class_count * len(ALL_EXIT_VARIANTS) * len(DEFAULT_INSTRUMENTS)
        top_n = 3
        self.assertGreater(total_expected, 10)  # confirms there is real work left after the "crash"

        with tempfile.TemporaryDirectory() as tmp:
            run_id = "resume-test-run"

            # --- "first process": completes a handful of variants directly
            # against the journal (exactly what _expand_and_simulate itself
            # would persist), then crashes -- it never calls _expand_and_
            # simulate, and never reports the run as completed.
            reporter1 = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={}, run_id=run_id, is_demo=True)
            journal1 = _open_journal(
                reporter1, market_artifact_path=None, news_csv_path=None, parameters={}, risk_config=DEFAULT_RISK_CONFIG,
                equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
            )
            precomputed_ids = []
            combos = [
                (equivalence_class, variant, instrument)
                for equivalence_class in universe.equivalence_classes
                for variant in ALL_EXIT_VARIANTS
                for instrument in DEFAULT_INSTRUMENTS
            ][:5]
            for equivalence_class, variant, instrument in combos:
                result = run_strategy_variant_backtest(equivalence_class, variant, instrument, context, risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG)
                journal1.commit_summary(VariantSummaryRow.from_result(result).to_dict())
                precomputed_ids.append(result.strategy_variant_id)
            self.assertEqual(journal1.completed_count(), 5)
            created_at_before_resume = reporter1.manifest.created_at
            journal1.close()

            # --- "second process": resumes against the exact same run_dir/run_id.
            reporter2 = RunReporter(reporter1.run_dir, read_run_manifest(reporter1.run_dir))
            self.assertEqual(reporter2.manifest.created_at, created_at_before_resume)  # never re-created
            journal2 = _open_journal(
                reporter2, market_artifact_path=None, news_csv_path=None, parameters={}, risk_config=DEFAULT_RISK_CONFIG,
                equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
            )
            self.assertEqual(journal2.completed_count(), 5)  # durable across the reopen

            completed, final_leaderboard = _expand_and_simulate(
                reporter2, journal2, universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, context,
                risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, top_n_leaderboard=top_n, checkpoint_every=7,
            )

            # (a)/(b): exactly total_expected, never total_expected + the 5
            # precomputed duplicates -- if the skip logic were broken, the
            # journal's own PRIMARY KEY would have raised sqlite3.
            # IntegrityError on the first duplicate INSERT rather than
            # silently double-counting, so this also proves no duplicate
            # commit was even attempted.
            self.assertEqual(completed, total_expected)
            self.assertEqual(journal2.completed_count(), total_expected)
            for pid in precomputed_ids:
                self.assertTrue(journal2.is_completed(pid))

            # (c): durable Top-N reconstruction is exact against a brute
            # force recomputation over every row in the durable journal
            # (both the pre-crash and post-resume ones).
            all_rows = journal2.top_n_summaries(total_expected)
            self.assertEqual(len(all_rows), total_expected)
            brute_force_ranked = sorted(all_rows, key=lambda s: (s.net_pnl_usd, -s.max_drawdown_usd, s.strategy_variant_id), reverse=True)
            expected_top_ids = [s.strategy_variant_id for s in brute_force_ranked[:top_n]]
            self.assertEqual([e.strategy_id for e in final_leaderboard], expected_top_ids)

            # the dashboard-visible leaderboard.json matches too.
            dashboard_leaderboard = read_leaderboard(reporter2.run_dir)
            self.assertEqual([e.strategy_id for e in dashboard_leaderboard.entries], expected_top_ids)

            journal2.close()

    def test_resume_rebuilds_dashboard_visible_progress_immediately(self):
        context, universe = self._universe_and_context()
        with tempfile.TemporaryDirectory() as tmp:
            run_id = "resume-progress-run"
            reporter1 = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={}, run_id=run_id, is_demo=True)
            journal1 = _open_journal(
                reporter1, market_artifact_path=None, news_csv_path=None, parameters={}, risk_config=DEFAULT_RISK_CONFIG,
                equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
            )
            equivalence_class = universe.equivalence_classes[0]
            variant = ALL_EXIT_VARIANTS[0]
            instrument = DEFAULT_INSTRUMENTS[0]
            result = run_strategy_variant_backtest(equivalence_class, variant, instrument, context, risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG)
            journal1.commit_summary(VariantSummaryRow.from_result(result).to_dict())
            journal1.close()

            reporter2 = RunReporter(reporter1.run_dir, read_run_manifest(reporter1.run_dir))
            journal2 = _open_journal(
                reporter2, market_artifact_path=None, news_csv_path=None, parameters={}, risk_config=DEFAULT_RISK_CONFIG,
                equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
            )
            # _expand_and_simulate reports progress/leaderboard from the
            # durable journal BEFORE simulating anything new -- so the
            # dashboard is never left showing stale pre-crash progress
            # while the resumed process works through the remainder.
            _expand_and_simulate(
                reporter2, journal2, universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, context,
                risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, top_n_leaderboard=5, checkpoint_every=1000000,
            )
            journal2.close()

            status = read_live_status(reporter2.run_dir)
            self.assertGreaterEqual(status.items_completed, 1)


class IncompatibleResumeRejectionTests(unittest.TestCase):
    """Resuming with any changed compatibility field must be refused
    clearly, naming the mismatched field, and must never touch the
    already-durable data."""

    def _fresh_run(self, tmp, **journal_kwargs):
        reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="incompat-run", is_demo=True)
        kwargs = dict(
            market_artifact_path=None, news_csv_path=None, parameters={"min_entries_per_year": 1}, risk_config=DEFAULT_RISK_CONFIG,
            equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
        )
        kwargs.update(journal_kwargs)
        journal = _open_journal(reporter, **kwargs)
        journal.close()
        return reporter

    def test_resuming_with_different_parameters_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = self._fresh_run(tmp)
            reporter2 = RunReporter(reporter.run_dir, read_run_manifest(reporter.run_dir))
            with self.assertRaises(ValueError) as ctx:
                _open_journal(
                    reporter2, market_artifact_path=None, news_csv_path=None, parameters={"min_entries_per_year": 999},
                    risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
                )
            self.assertIn("parameters_json", str(ctx.exception))

    def test_resuming_with_different_risk_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = self._fresh_run(tmp)
            reporter2 = RunReporter(reporter.run_dir, read_run_manifest(reporter.run_dir))
            other_risk_config = dataclasses.replace(DEFAULT_RISK_CONFIG, target_risk_usd=DEFAULT_RISK_CONFIG.target_risk_usd + 1.0)
            with self.assertRaises(ValueError) as ctx:
                _open_journal(
                    reporter2, market_artifact_path=None, news_csv_path=None, parameters={"min_entries_per_year": 1},
                    risk_config=other_risk_config, equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
                )
            self.assertIn("risk_config_json", str(ctx.exception))

    def test_resuming_with_different_instruments_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = self._fresh_run(tmp)
            reporter2 = RunReporter(reporter.run_dir, read_run_manifest(reporter.run_dir))
            with self.assertRaises(ValueError) as ctx:
                _open_journal(
                    reporter2, market_artifact_path=None, news_csv_path=None, parameters={"min_entries_per_year": 1},
                    risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=(Instrument.MNQ,),
                )
            self.assertIn("instruments_json", str(ctx.exception))

    def test_resuming_with_different_exit_variants_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = self._fresh_run(tmp)
            reporter2 = RunReporter(reporter.run_dir, read_run_manifest(reporter.run_dir))
            with self.assertRaises(ValueError) as ctx:
                _open_journal(
                    reporter2, market_artifact_path=None, news_csv_path=None, parameters={"min_entries_per_year": 1},
                    risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS[:5], instruments=DEFAULT_INSTRUMENTS,
                )
            self.assertIn("exit_variant_ids_json", str(ctx.exception))

    def test_resuming_under_a_different_current_git_commit_is_refused_before_any_simulation(self):
        # _open_journal must independently rediscover the CURRENT commit at
        # execution time (never trust reporter.manifest.git_commit, which
        # is only the commit that produced the *original*, possibly now-
        # stale, run) -- patch tradelab.backtest_v2._current_git_provenance
        # itself (the exact call site) to simulate the code under test
        # actually changing between the interrupted run and the resume.
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("commit-aaa", False)):
                reporter = self._fresh_run(tmp)

            reporter2 = RunReporter(reporter.run_dir, read_run_manifest(reporter.run_dir))
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("commit-bbb", False)):
                with self.assertRaises(ValueError) as ctx:
                    _open_journal(
                        reporter2, market_artifact_path=None, news_csv_path=None, parameters={"min_entries_per_year": 1},
                        risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
                    )
            self.assertIn("git_commit", str(ctx.exception))

    def test_resuming_under_the_same_current_git_commit_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("commit-aaa", False)):
                reporter = self._fresh_run(tmp)

            reporter2 = RunReporter(reporter.run_dir, read_run_manifest(reporter.run_dir))
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("commit-aaa", False)):
                journal2 = _open_journal(  # must not raise -- identical current commit both times
                    reporter2, market_artifact_path=None, news_csv_path=None, parameters={"min_entries_per_year": 1},
                    risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
                )
            journal2.close()


class ProductionRunnerResumeIntegrationTests(unittest.TestCase):
    """The same resumability contract, but through the real, file-based
    ``run_production_backtest_v2`` entry point (never the internal
    ``_expand_and_simulate`` helper directly) -- proving the CLI-adjacent
    ``run_id`` reuse mechanism itself, not just the journal underneath
    it: calling it twice with the same ``run_id`` re-attaches (preserving
    the original ``created_at``) rather than creating a second run, and
    calling it a second time with a changed parameter is refused clearly
    before any new work is attempted."""

    def test_same_run_id_reattaches_and_preserves_created_at(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            # Pinned clean/stable so this test is immune to the *actual*
            # ambient repo's own working-tree state (e.g. mid-development,
            # with genuinely uncommitted changes) -- see
            # DirtyWorkingTreeRejectionTests for the dedicated dirty-tree
            # coverage.
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", False)):
                reporter1 = run_production_backtest_v2(
                    market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                    pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                    run_id="prod-resume-run", command="test",
                )
                created_at_1 = reporter1.manifest.created_at
                self.assertEqual(read_live_status(reporter1.run_dir).phase, "completed")

                reporter2 = run_production_backtest_v2(
                    market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                    pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                    run_id="prod-resume-run", command="test",
                )
            self.assertEqual(reporter2.run_dir, reporter1.run_dir)
            self.assertEqual(reporter2.manifest.created_at, created_at_1)  # re-attached, never re-created
            self.assertEqual(read_live_status(reporter2.run_dir).phase, "completed")

    def test_resuming_with_a_different_pivot_strength_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", False)):
                run_production_backtest_v2(
                    market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                    pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                    run_id="prod-mismatch-run", command="test",
                )
                with self.assertRaises(ValueError) as ctx:
                    run_production_backtest_v2(
                        market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                        pivot_strength=3, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                        run_id="prod-mismatch-run", command="test",
                    )
            self.assertIn("parameters_json", str(ctx.exception))


class DirtyWorkingTreeRejectionTests(unittest.TestCase):
    """Task 043's final research-integrity guard: ``run_production_
    backtest_v2`` must refuse to start OR resume when the working tree
    is dirty, even though the commit hash itself may be unchanged --
    a matching commit alone doesn't guarantee the code that actually ran
    was reproducible if uncommitted local edits sat on top of it. Every
    test here pins ``tradelab.backtest_v2._current_git_provenance``
    directly (never relying on this *actual* repo's own ambient
    working-tree state, which is dirty as a matter of course during
    active development)."""

    def _artifact(self, tmp: Path) -> tuple[Path, Path, Path]:
        manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
        return manifest_path, news_csv_path, tmp / "runs"

    def test_clean_tree_succeeds_on_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            manifest_path, news_csv_path, runs_dir = self._artifact(Path(tmp_str))
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", False)):
                reporter = run_production_backtest_v2(
                    market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                    pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                    run_id="clean-fresh-run", command="test",
                )
            self.assertEqual(read_live_status(reporter.run_dir).phase, "completed")

    def test_dirty_tree_is_rejected_on_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path, runs_dir = self._artifact(tmp)
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", True)):
                with self.assertRaises(ValueError) as ctx:
                    run_production_backtest_v2(
                        market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                        pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                        run_id="dirty-fresh-run", command="test",
                    )
            message = str(ctx.exception)
            self.assertIn("uncommitted", message)
            self.assertIn("commit", message.lower())

    def test_dirty_tree_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path, runs_dir = self._artifact(tmp)

            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", False)):
                reporter1 = run_production_backtest_v2(
                    market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                    pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                    run_id="dirty-resume-run", command="test",
                )
            manifest_before = (reporter1.run_dir / "run_manifest.json").read_text(encoding="utf-8")
            status_before = (reporter1.run_dir / "live_status.json").read_text(encoding="utf-8")

            # the tree becomes dirty (e.g. uncommitted local edits) before
            # the second, resuming invocation -- even though the commit
            # hash reported here is deliberately left identical.
            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", True)):
                with self.assertRaises(ValueError) as ctx:
                    run_production_backtest_v2(
                        market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                        pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                        run_id="dirty-resume-run", command="test",
                    )
            self.assertIn("uncommitted", str(ctx.exception))

            # nothing about the already-durable run was touched by the
            # refused resume attempt.
            self.assertEqual((reporter1.run_dir / "run_manifest.json").read_text(encoding="utf-8"), manifest_before)
            self.assertEqual((reporter1.run_dir / "live_status.json").read_text(encoding="utf-8"), status_before)

    def test_rejection_happens_before_any_data_preparation_or_simulation_work(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path, runs_dir = self._artifact(tmp)

            with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", True)):
                with self.assertRaises(ValueError):
                    run_production_backtest_v2(
                        market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                        pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                        run_id="dirty-before-anything-run", command="test",
                    )

            # not even a run directory was created -- the guard fires
            # before create_run()/RunReporter re-attach, before the
            # journal is opened, before prepare_backtest_context, before
            # discover_entry_strategy_universe, and before any simulation.
            self.assertFalse((runs_dir / "dirty-before-anything-run").exists())


class LivenessHeartbeatWiringTests(unittest.TestCase):
    """task 046: run_production_backtest_v2, run_preflight_v2, and the
    CLI smoke test all start the background liveness thread before doing
    any real work and stop it exactly once the run reaches a terminal
    state -- proven via a call-count spy (never relying on timing), plus
    a genuine failure case proving the thread is stopped even when the
    run itself raises."""

    def test_production_run_starts_and_stops_liveness_heartbeat_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            starts, stops = [], []
            orig_start, orig_stop = RunReporter.start_liveness_heartbeat, RunReporter.stop_liveness_heartbeat

            def spy_start(self, **kwargs):
                starts.append(1)
                return orig_start(self, **kwargs)

            def spy_stop(self, **kwargs):
                stops.append(1)
                return orig_stop(self, **kwargs)

            with patch.object(RunReporter, "start_liveness_heartbeat", spy_start), patch.object(RunReporter, "stop_liveness_heartbeat", spy_stop):
                with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", False)):
                    run_production_backtest_v2(
                        market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                        pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1, command="test",
                    )
            self.assertEqual(len(starts), 1)
            self.assertEqual(len(stops), 1)

    def test_preflight_starts_and_stops_liveness_heartbeat_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            starts, stops = [], []
            orig_start, orig_stop = RunReporter.start_liveness_heartbeat, RunReporter.stop_liveness_heartbeat

            def spy_start(self, **kwargs):
                starts.append(1)
                return orig_start(self, **kwargs)

            def spy_stop(self, **kwargs):
                stops.append(1)
                return orig_stop(self, **kwargs)

            with patch.object(RunReporter, "start_liveness_heartbeat", spy_start), patch.object(RunReporter, "stop_liveness_heartbeat", spy_stop):
                run_preflight_v2(
                    market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                    pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1, command="test",
                )
            self.assertEqual(len(starts), 1)
            self.assertEqual(len(stops), 1)

    def test_smoke_test_starts_and_stops_liveness_heartbeat_exactly_once(self):
        starts, stops = [], []
        orig_start, orig_stop = RunReporter.start_liveness_heartbeat, RunReporter.stop_liveness_heartbeat

        def spy_start(self, **kwargs):
            starts.append(1)
            return orig_start(self, **kwargs)

        def spy_stop(self, **kwargs):
            stops.append(1)
            return orig_stop(self, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(RunReporter, "start_liveness_heartbeat", spy_start), patch.object(RunReporter, "stop_liveness_heartbeat", spy_stop):
                exit_code = backtest_v2_main(["smoke-test", "--runs-dir", tmp, "--min-entries-per-year", "1"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(stops), 1)

    def test_liveness_heartbeat_is_stopped_even_when_the_run_fails(self):
        # a real failure (an incompatible resume, refused before any
        # simulation) must still stop the thread it started -- a thread
        # leaked on the failure path would be a real bug, not cosmetic.
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            stops = []
            orig_stop = RunReporter.stop_liveness_heartbeat

            def spy_stop(self, **kwargs):
                stops.append(1)
                return orig_stop(self, **kwargs)

            with patch.object(RunReporter, "stop_liveness_heartbeat", spy_stop):
                with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", False)):
                    run_production_backtest_v2(
                        market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                        pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                        run_id="fail-liveness-run", command="test",
                    )
                    with self.assertRaises(ValueError):
                        run_production_backtest_v2(
                            market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                            pivot_strength=3, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                            run_id="fail-liveness-run", command="test",
                        )
            # the second call's own journal-mismatch ValueError is raised
            # before _open_journal (and thus before start_liveness_
            # heartbeat) is ever reached -- so only the first, successful
            # call's own start/stop pair happened. This proves the
            # *successful* run's thread was cleanly stopped, not leaked;
            # see DirtyWorkingTreeRejectionTests for a case that fails
            # after start_liveness_heartbeat has already run.
            self.assertEqual(len(stops), 1)

    def test_liveness_heartbeat_stopped_after_a_failure_that_occurs_after_starting_it(self):
        # unlike the mismatch case above, a failure *inside* the try block
        # (after start_liveness_heartbeat has already run) must still
        # reach stop_liveness_heartbeat via the finally clause.
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            stops = []
            orig_stop = RunReporter.stop_liveness_heartbeat

            def spy_stop(self, **kwargs):
                stops.append(1)
                return orig_stop(self, **kwargs)

            with patch.object(RunReporter, "stop_liveness_heartbeat", spy_stop):
                with patch("tradelab.backtest_v2._current_git_provenance", return_value=("test-commit", False)):
                    with patch("tradelab.backtest_v2.discover_entry_strategy_universe", side_effect=RuntimeError("boom")):
                        with self.assertRaises(RuntimeError):
                            run_production_backtest_v2(
                                market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                                pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1, command="test",
                            )
            self.assertEqual(len(stops), 1)


class PreflightTests(unittest.TestCase):
    """Preflight correctness (task 043 requirement 2): the artifact's
    counts must exactly match an independent, direct call to ``discover_
    entry_strategy_universe`` over the same prepared context (proving
    preflight can never omit or alter the final universe), the total
    simulation count must be the exact product of the three cardinalities,
    it must never write a journal or simulate every variant by default,
    and it must integrate with the same dashboard-visible telemetry model
    a production run uses."""

    def test_preflight_counts_match_a_direct_universe_discovery_and_never_simulates_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            reporter = run_preflight_v2(
                market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1, command="test",
            )

            preflight = read_preflight(reporter.run_dir)
            self.assertEqual(preflight["bar_count"], 900)
            self.assertIsNotNone(preflight["covered_time_range"]["start"])
            self.assertIsNotNone(preflight["covered_time_range"]["end"])

            # independent, direct recomputation -- must match exactly.
            expected_context = build_synthetic_smoke_context()
            expected_universe = discover_entry_strategy_universe(
                expected_context.fvg_results, expected_context.fvg_entry_result, expected_context.liquidity_entry_result,
                expected_context.htf_bias_result, min_entries_per_year=1,
            )
            self.assertGreater(expected_universe.equivalence_class_count, 0)
            self.assertEqual(preflight["syntactically_enumerated_spec_count"], expected_universe.syntactically_enumerated_spec_count)
            self.assertEqual(preflight["eligible_spec_count"], expected_universe.eligible_spec_count)
            self.assertEqual(preflight["equivalence_class_count"], expected_universe.equivalence_class_count)
            self.assertEqual(preflight["exit_variant_count"], len(ALL_EXIT_VARIANTS))
            self.assertEqual(sorted(preflight["instruments"]), sorted(i.value for i in DEFAULT_INSTRUMENTS))

            expected_total = expected_universe.equivalence_class_count * len(ALL_EXIT_VARIANTS) * len(DEFAULT_INSTRUMENTS)
            self.assertEqual(preflight["total_strategy_variant_simulations"], expected_total)
            self.assertGreater(expected_total, 0)

            self.assertIn("disk_space_estimate", preflight)
            self.assertGreater(preflight["disk_space_estimate"]["estimated_total_bytes"], 0)
            self.assertIsNone(preflight["throughput_estimate"])  # default sample size 0 -- never measured

            # never simulates anything by default: no journal, no summaries export.
            self.assertFalse((reporter.run_dir / "backtest_v2_journal.sqlite3").exists())
            self.assertFalse((reporter.run_dir / "variant_summaries.jsonl").exists())

            # dashboard-compatible: same telemetry artifacts a production run writes.
            self.assertEqual(read_live_status(reporter.run_dir).phase, "completed")
            self.assertIsNotNone(read_run_manifest(reporter.run_dir))

    def test_throughput_sample_size_measures_a_bounded_real_sample_without_completing_the_universe(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            reporter = run_preflight_v2(
                market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                throughput_sample_size=4, command="test",
            )
            preflight = read_preflight(reporter.run_dir)
            sample = preflight["throughput_estimate"]
            self.assertIsNotNone(sample)
            self.assertEqual(sample["sample_size_requested"], 4)
            self.assertEqual(sample["sample_size_measured"], 4)
            self.assertLess(sample["sample_size_measured"], preflight["total_strategy_variant_simulations"])
            self.assertIsNotNone(sample["variants_per_second"])
            self.assertIn("MEASURED", sample["note"])
            self.assertEqual(preflight["disk_space_estimate"]["basis"].split(":")[0], "measured")

            # still never wrote a journal or completed the universe.
            self.assertFalse((reporter.run_dir / "backtest_v2_journal.sqlite3").exists())


class BarProgressCallbackTests(unittest.TestCase):
    """task 044: on_bar_progress is purely observational -- it reports
    monotonically non-decreasing (bars_scanned, bars_total) pairs as the
    simulation loop advances, and never changes trades, skipped entries,
    P&L, or any other field of the returned StrategyVariantResult."""

    def _one_real_variant(self):
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1
        )
        self.assertGreater(universe.equivalence_class_count, 0)
        return context, universe.equivalence_classes[0], ALL_EXIT_VARIANTS[0], DEFAULT_INSTRUMENTS[0]

    def test_callback_reports_monotonically_non_decreasing_progress(self):
        context, equivalence_class, variant, instrument = self._one_real_variant()
        calls = []
        run_strategy_variant_backtest(
            equivalence_class, variant, instrument, context, on_bar_progress=lambda scanned, total: calls.append((scanned, total))
        )
        self.assertGreater(len(calls), 0)
        self.assertTrue(all(total == len(context.bars_1m) for _scanned, total in calls))
        scanned_values = [scanned for scanned, _total in calls]
        self.assertEqual(scanned_values, sorted(scanned_values))
        self.assertTrue(all(0 < s <= len(context.bars_1m) for s in scanned_values))

    def test_callback_never_changes_the_simulated_result(self):
        context, equivalence_class, variant, instrument = self._one_real_variant()
        without_callback = run_strategy_variant_backtest(equivalence_class, variant, instrument, context)
        calls = []
        with_callback = run_strategy_variant_backtest(
            equivalence_class, variant, instrument, context, on_bar_progress=lambda s, t: calls.append((s, t))
        )
        self.assertGreater(len(calls), 0)  # confirms the callback path was genuinely exercised
        self.assertEqual(without_callback.trades, with_callback.trades)
        self.assertEqual(without_callback.trade_count, with_callback.trade_count)
        self.assertEqual(without_callback.net_pnl_usd, with_callback.net_pnl_usd)
        self.assertEqual(without_callback.max_drawdown_usd, with_callback.max_drawdown_usd)
        self.assertEqual(without_callback.skipped_entries_by_reason, with_callback.skipped_entries_by_reason)
        self.assertEqual(without_callback.strategy_variant_id, with_callback.strategy_variant_id)

    def test_a_throttled_callback_never_receives_a_call_per_bar(self):
        # A ProgressThrottle wrapping a spy, exactly as production wires
        # it -- proves the actual wrapped callback fires far less often
        # than the number of bars scanned, even though maybe() itself is
        # invoked once per bar internally.
        context, equivalence_class, variant, instrument = self._one_real_variant()
        fired = []
        throttle = ProgressThrottle(lambda s, t: fired.append((s, t)), check_every=50)
        run_strategy_variant_backtest(equivalence_class, variant, instrument, context, on_bar_progress=throttle.maybe)
        self.assertGreaterEqual(len(fired), 1)
        self.assertLess(len(fired), len(context.bars_1m))


class EntryProgressCallbackTests(unittest.TestCase):
    """task 046: on_entry_progress reports real (entry_index, total_
    entries) -- 1-based -- as simulate_market_execution processes each
    observed entry, purely observationally (never changes trades,
    skipped entries, or any other result field), and independently of
    on_bar_progress (many entries can be resolved while bars barely
    advance)."""

    def _one_real_variant(self):
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1
        )
        self.assertGreater(universe.equivalence_class_count, 0)
        equivalence_class = max(universe.equivalence_classes, key=lambda c: len(c.observed_entries))
        self.assertGreater(len(equivalence_class.observed_entries), 10)  # a genuinely multi-entry scenario
        return context, equivalence_class, ALL_EXIT_VARIANTS[0], DEFAULT_INSTRUMENTS[0]

    def test_callback_reports_every_entry_in_order_starting_at_one(self):
        context, equivalence_class, variant, instrument = self._one_real_variant()
        calls = []
        run_strategy_variant_backtest(
            equivalence_class, variant, instrument, context, on_entry_progress=lambda idx, total: calls.append((idx, total))
        )
        total_entries = len(equivalence_class.observed_entries)
        self.assertEqual(calls, [(i, total_entries) for i in range(1, total_entries + 1)])  # exactly once per entry, in order, no gaps/dupes

    def test_callback_never_changes_the_simulated_result(self):
        context, equivalence_class, variant, instrument = self._one_real_variant()
        without_callback = run_strategy_variant_backtest(equivalence_class, variant, instrument, context)
        calls = []
        with_callback = run_strategy_variant_backtest(
            equivalence_class, variant, instrument, context, on_entry_progress=lambda i, t: calls.append((i, t))
        )
        self.assertGreater(len(calls), 0)
        self.assertEqual(without_callback.trades, with_callback.trades)
        self.assertEqual(without_callback.trade_count, with_callback.trade_count)
        self.assertEqual(without_callback.net_pnl_usd, with_callback.net_pnl_usd)
        self.assertEqual(without_callback.skipped_entries_by_reason, with_callback.skipped_entries_by_reason)
        self.assertEqual(without_callback.strategy_variant_id, with_callback.strategy_variant_id)

    def test_entry_progress_and_bar_progress_are_independent_signals(self):
        # both callbacks fire during the same simulation -- entry progress
        # doesn't suppress bar progress or vice versa.
        context, equivalence_class, variant, instrument = self._one_real_variant()
        entry_calls = []
        bar_calls = []
        run_strategy_variant_backtest(
            equivalence_class, variant, instrument, context,
            on_entry_progress=lambda i, t: entry_calls.append((i, t)), on_bar_progress=lambda s, t: bar_calls.append((s, t)),
        )
        self.assertGreater(len(entry_calls), 0)
        self.assertGreater(len(bar_calls), 0)


class ExpandAndSimulateHeartbeatTests(unittest.TestCase):
    """task 044: _expand_and_simulate reports a live heartbeat (current
    strategy_variant_id, 1-based position among total_variant_runs) for
    every variant -- proven here to advance in order and to never change
    the completed count or final leaderboard versus an unobserved run."""

    def _universe_and_context(self):
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1
        )
        self.assertGreater(universe.equivalence_class_count, 0)
        return context, universe

    def _run(self, tmp, run_id, universe, context, *, spy=None):
        reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={}, run_id=run_id, is_demo=True)
        journal = _open_journal(
            reporter, market_artifact_path=None, news_csv_path=None, parameters={}, risk_config=DEFAULT_RISK_CONFIG,
            equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
        )
        if spy is not None:
            with patch.object(RunReporter, "heartbeat", spy):
                completed, final_leaderboard = _expand_and_simulate(
                    reporter, journal, universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, context, risk_config=DEFAULT_RISK_CONFIG,
                    equity_config=DEFAULT_EQUITY_CONFIG, top_n_leaderboard=3, checkpoint_every=1000,
                )
        else:
            completed, final_leaderboard = _expand_and_simulate(
                reporter, journal, universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, context, risk_config=DEFAULT_RISK_CONFIG,
                equity_config=DEFAULT_EQUITY_CONFIG, top_n_leaderboard=3, checkpoint_every=1000,
            )
        journal.close()
        return completed, final_leaderboard

    def test_heartbeat_advances_across_variants_without_changing_the_outcome(self):
        context, universe = self._universe_and_context()
        total_expected = universe.equivalence_class_count * len(ALL_EXIT_VARIANTS) * len(DEFAULT_INSTRUMENTS)

        calls = []
        orig = RunReporter.heartbeat

        def spy(self, **kwargs):
            calls.append(dict(kwargs))
            return orig(self, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            completed_observed, leaderboard_observed = self._run(tmp, "observed", universe, context, spy=spy)
            completed_baseline, leaderboard_baseline = self._run(tmp, "baseline", universe, context)

        self.assertEqual(completed_observed, total_expected)
        sim_calls = [c for c in calls if c.get("operation") == "simulating_backtests"]
        self.assertGreater(len(sim_calls), 0)
        self.assertTrue(all(c.get("variant_id") for c in sim_calls))
        unit_indices = [c["unit_index"] for c in sim_calls]
        self.assertEqual(unit_indices, sorted(unit_indices))  # 1-based position never goes backwards
        self.assertEqual(unit_indices[0], 1)

        # zero semantic change: an unobserved run reaches exactly the
        # same completed count and the exact same final leaderboard.
        self.assertEqual(completed_observed, completed_baseline)
        self.assertEqual([e.to_dict() for e in leaderboard_observed], [e.to_dict() for e in leaderboard_baseline])

    def test_building_range_index_and_entry_resolution_progress_are_reported_without_faking_bars(self):
        # task 046: the two operations that previously left live_status
        # frozen (BarRangeIndex construction, and target resolution for
        # entries clustered in a small bar range) must both actually
        # report real progress, and neither may ever claim a bars_scanned
        # value -- no bar is genuinely being scanned during either.
        context, universe = self._universe_and_context()

        calls = []
        orig = RunReporter.heartbeat

        def spy(self, **kwargs):
            calls.append(dict(kwargs))
            return orig(self, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, "observed", universe, context, spy=spy)

        index_calls = [c for c in calls if c["operation"].startswith("building_range_index")]
        entry_calls = [c for c in calls if c["operation"] == "resolving_entry_targets"]

        self.assertGreater(len(index_calls), 0)  # the build genuinely got observed at least once
        self.assertGreater(len(entry_calls), 0)

        # real, exact sparse-table cell progress -- never bars.
        self.assertTrue(all(c.get("unit_index") is not None and c.get("unit_total") is not None for c in index_calls))
        self.assertTrue(all(c.get("bars_scanned") is None for c in index_calls))

        # real entry index/total -- never bars, and every call carries a
        # real strategy_variant_id so the dashboard knows which variant.
        self.assertTrue(all(c.get("unit_index") is not None and c.get("unit_total") is not None for c in entry_calls))
        self.assertTrue(all(c.get("bars_scanned") is None for c in entry_calls))
        self.assertTrue(all(c.get("variant_id") for c in entry_calls))

    def test_rate_limit_is_never_violated_even_when_every_variant_completes_within_the_same_instant(self):
        # The production analogue of the preflight adversarial test below:
        # freeze the heartbeat-relevant clock entirely so every one of
        # this small synthetic universe's variants (and every bar-scan
        # callback inside each) genuinely happens at the exact same
        # monotonic instant, and prove the number of actual live_status.
        # json writes heartbeat() causes stays small and bounded --
        # nowhere close to one per variant, no matter how many variants
        # exist.
        context, universe = self._universe_and_context()
        total_expected = universe.equivalence_class_count * len(ALL_EXIT_VARIANTS) * len(DEFAULT_INSTRUMENTS)
        self.assertGreater(total_expected, 20)  # confirms this is a genuine many-variant scenario

        write_count = 0
        orig = RunReporter.heartbeat

        def spy(self, **kwargs):
            nonlocal write_count
            wrote = orig(self, **kwargs)
            if wrote:
                write_count += 1
            return wrote

        with tempfile.TemporaryDirectory() as tmp:
            with patch("tradelab.research_run.time.monotonic", return_value=100.0):
                completed, _leaderboard = self._run(tmp, "frozen-clock", universe, context, spy=spy)

        self.assertEqual(completed, total_expected)
        # exactly one successful heartbeat write for the entire
        # simulating_backtests phase (its own first call) -- the frozen
        # clock means nothing after it can ever write again, regardless
        # of how many dozens of variants (and many more bar-scan callback
        # attempts within each) were actually simulated.
        self.assertLessEqual(write_count, 1)


class PreflightThroughputSampleProgressTests(unittest.TestCase):
    """task 044 (amended): --throughput-sample-size N must visibly
    advance 0/N -> 1/N -> ... -> N/N over time (never jump straight from
    0 to N at the very end), with intra-sample bar progress too -- while
    RunReporter.heartbeat()'s own hard, non-bypassable once-per-second
    write cap is never violated, no matter how many samples complete
    within the same second. There is no longer a "force" escape hatch:
    fast-completing samples are coalesced (only the latest state is
    written once a write is actually due), never all individually
    written."""

    def test_sample_completion_is_attempted_in_order_with_advancing_unit_index(self):
        # Proves the call-site orchestration itself is correct -- each
        # completed sample is reported with the right, advancing state --
        # independent of whether RunReporter.heartbeat() actually writes
        # every single one of those attempts (that write-rate guarantee
        # is proven separately, deterministically, below and in
        # tests/test_tradelab_research_run.py's own HeartbeatTests).
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            calls = []
            orig = RunReporter.heartbeat

            def spy(self, **kwargs):
                calls.append(dict(kwargs))
                return orig(self, **kwargs)

            with patch.object(RunReporter, "heartbeat", spy):
                reporter = run_preflight_v2(
                    market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                    pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                    throughput_sample_size=5, command="test",
                )

            preflight = read_preflight(reporter.run_dir)
            self.assertEqual(preflight["throughput_estimate"]["sample_size_measured"], 5)

            sample_calls = [c for c in calls if c.get("operation") == "sampling_throughput"]
            self.assertGreater(len(sample_calls), 0)

            # the per-sample-completion attempts (bars_scanned ==
            # bars_total) must show unit_index advancing 1, 2, 3, 4, 5 in
            # order -- whether or not each one actually earns a write.
            completions = [c["unit_index"] for c in sample_calls if c.get("bars_scanned") == c.get("bars_total")]
            self.assertEqual(completions, [1, 2, 3, 4, 5])

            # every sample carried a real strategy_variant_id and the
            # correct, fixed unit_total throughout.
            self.assertTrue(all(c.get("variant_id") for c in sample_calls))
            self.assertTrue(all(c.get("unit_total") == 5 for c in sample_calls))

    def test_rate_limit_is_never_violated_even_when_every_sample_completes_within_the_same_instant(self):
        # The most adversarial case this bug report described: freeze the
        # heartbeat-relevant clock entirely (every sample, and every
        # bar-scan callback within it, genuinely happens at the exact
        # same monotonic instant) and prove the number of actual
        # live_status.json writes heartbeat() causes stays small and
        # bounded -- never anywhere close to one per sample.
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"

            write_count = 0
            last_active_work_by_operation = {}
            orig = RunReporter.heartbeat

            def spy(self, **kwargs):
                nonlocal write_count
                wrote = orig(self, **kwargs)
                if wrote:
                    write_count += 1
                # captured regardless of whether this particular call
                # earned a write -- proves the in-memory snapshot itself
                # (what a later, unrelated flush would pick up) really
                # does hold the latest state, not just whatever happened
                # to land on-disk.
                last_active_work_by_operation[kwargs["operation"]] = self._active_work.to_dict()
                return wrote

            with patch("tradelab.research_run.time.monotonic", return_value=100.0):
                with patch.object(RunReporter, "heartbeat", spy):
                    reporter = run_preflight_v2(
                        market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                        pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                        throughput_sample_size=25, command="test",
                    )

            preflight = read_preflight(reporter.run_dir)
            self.assertEqual(preflight["throughput_estimate"]["sample_size_measured"], 25)

            # exactly one successful heartbeat write per distinct phase
            # that ever calls heartbeat at all (preparing_data,
            # discovering_entries, sampling_throughput) -- the frozen
            # clock means nothing beyond each phase's own first call can
            # ever write again. Nowhere close to 25 (one per sample) or
            # 50 (one per sample start + completion).
            self.assertLessEqual(write_count, 5)
            self.assertLess(write_count, 25)

            # the milestone was never lost, only rate-limited: the very
            # last heartbeat attempt (sample 25's own completion) is
            # exactly what the in-memory snapshot held, ready for the
            # next reporter call (progress(), right after the sampling
            # loop) to flush -- even though heartbeat() itself never got
            # to write most of these 25 samples' own completions.
            self.assertEqual(last_active_work_by_operation["sampling_throughput"]["unit_index"], 25)
            self.assertEqual(last_active_work_by_operation["sampling_throughput"]["bars_scanned"], last_active_work_by_operation["sampling_throughput"]["bars_total"])

    def test_throughput_sample_reporter_none_path_is_unaffected(self):
        # reporter=None (no telemetry wired) must still work identically
        # -- _measure_sample_throughput is called this way nowhere in
        # production, but the parameter is optional and must degrade
        # cleanly, never crash, when omitted.
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest_path, news_csv_path = _write_smoke_market_artifact(tmp)
            runs_dir = tmp / "runs"
            reporter = run_preflight_v2(
                market_artifact_path=manifest_path, news_csv_path=news_csv_path, runs_dir=runs_dir,
                pivot_strength=2, reclaim_window_bars=3, tick_size=0.25, min_entries_per_year=1,
                throughput_sample_size=3, command="test",
            )
            preflight = read_preflight(reporter.run_dir)
            self.assertEqual(preflight["throughput_estimate"]["sample_size_measured"], 3)


# ---------------------------------------------------------------------------
# Task 045: execution acceleration -- exactness, instrument dedup, benchmark
# ---------------------------------------------------------------------------


class FindFirstExitTests(unittest.TestCase):
    """Direct, guaranteed (not merely probable) coverage of
    ``_find_first_exit``'s own exit-finding scenarios: TP-only, SL-only,
    same-bar-both (SL wins, exactly as ``_check_exit`` itself decides),
    no-hit-in-range (position survives, ``None``), and finding the exit
    correctly on the far side of a genuine multi-bar data gap."""

    def _position(self, direction, tp_price: float, sl_price: float):
        from tradelab.backtest import _OpenPosition
        from tradelab.sl_targets import SLSourceType, SLTargetCandidate
        from tradelab.tp_targets import TPSourceType, TPTargetCandidate

        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        tp = TPTargetCandidate(
            target_id="tp1", entry_id="e1", source_type=TPSourceType.LIQUIDITY, source_id="L1", direction=direction, price=tp_price,
            distance=abs(tp_price - 100.0), known_at=t0, rank=1, liquidity_type=None, level_id=None, lower=None, upper=None,
            source_family=None, event_ids=None, wick_direction=None, timestamp=None,
        )
        sl = SLTargetCandidate(
            sl_target_id="sl1", entry_id="e1", source_type=SLSourceType.LIQUIDITY, source_id="L2", direction=direction, price=sl_price,
            distance=abs(sl_price - 100.0), known_at=t0, rank=1, lower=sl_price, upper=sl_price, liquidity_type=None, level_id=None,
            source_family=None, zone_id=None, fvg_source_type=None, fvg_candidate_id=None, fvg_timeframe=None,
        )
        return _OpenPosition(trade_id="e1", strategy_id="s", direction=direction, signal_known_at=t0, entry_timestamp=t0, entry_price=100.0, tp_candidate=tp, sl_candidate=sl)

    def _bars_and_index(self, ohlc: list[tuple[float, float, float, float]], *, gap_before_index: int | None = None):
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        bars = []
        ts = t0
        for i, vals in enumerate(ohlc):
            if gap_before_index is not None and i == gap_before_index:
                ts += timedelta(hours=6)  # a genuine multi-bar data gap
            bars.append(bar(ts, *vals))
            ts += timedelta(minutes=1)
        bars = tuple(bars)
        return bars, BarRangeIndex.build(bars)

    def test_tp_only(self):
        position = self._position(LONG, tp_price=110.0, sl_price=90.0)
        bars, index = self._bars_and_index([(100, 101, 99, 100), (100, 105, 99, 104), (100, 111, 99, 108), (100, 103, 99, 102)])
        hit = _find_first_exit(bars, index, position, 0, len(bars))
        self.assertIsNotNone(hit)
        exit_index, reason, price = hit
        self.assertEqual(exit_index, 2)
        self.assertEqual(reason, TradeExitReason.TP_HIT)
        self.assertEqual(price, 110.0)

    def test_sl_only(self):
        position = self._position(SHORT, tp_price=90.0, sl_price=110.0)
        bars, index = self._bars_and_index([(100, 101, 99, 100), (100, 103, 98, 101), (100, 112, 99, 105), (100, 103, 99, 102)])
        hit = _find_first_exit(bars, index, position, 0, len(bars))
        self.assertIsNotNone(hit)
        exit_index, reason, price = hit
        self.assertEqual(exit_index, 2)
        self.assertEqual(reason, TradeExitReason.SL_HIT)
        self.assertEqual(price, 110.0)

    def test_same_bar_tp_and_sl_sl_wins(self):
        position = self._position(LONG, tp_price=110.0, sl_price=95.0)
        # bar 1's own range spans BOTH 110 (TP) and 95 (SL).
        bars, index = self._bars_and_index([(100, 101, 99, 100), (100, 112, 90, 105)])
        hit = _find_first_exit(bars, index, position, 0, len(bars))
        self.assertIsNotNone(hit)
        exit_index, reason, price = hit
        self.assertEqual(exit_index, 1)
        self.assertEqual(reason, TradeExitReason.SL_HIT)
        self.assertEqual(price, 95.0)

    def test_no_hit_survives_the_range(self):
        position = self._position(LONG, tp_price=200.0, sl_price=10.0)
        bars, index = self._bars_and_index([(100, 101, 99, 100)] * 10)
        self.assertIsNone(_find_first_exit(bars, index, position, 0, len(bars)))

    def test_exit_found_correctly_beyond_a_genuine_data_gap(self):
        position = self._position(LONG, tp_price=110.0, sl_price=90.0)
        # no hit before the gap; the qualifying bar is the very first one after it.
        bars, index = self._bars_and_index(
            [(100, 101, 99, 100), (100, 102, 99, 101), (100, 111, 99, 108), (100, 103, 99, 102)], gap_before_index=2,
        )
        hit = _find_first_exit(bars, index, position, 0, len(bars))
        self.assertIsNotNone(hit)
        exit_index, reason, price = hit
        self.assertEqual(exit_index, 2)  # the bar immediately after the gap
        self.assertEqual(reason, TradeExitReason.TP_HIT)

    def test_bounded_range_never_reports_a_hit_beyond_hi(self):
        position = self._position(LONG, tp_price=110.0, sl_price=90.0)
        bars, index = self._bars_and_index([(100, 101, 99, 100), (100, 102, 99, 101), (100, 111, 99, 108)])
        # querying only [0, 2) -- the qualifying bar (index 2) is out of range.
        self.assertIsNone(_find_first_exit(bars, index, position, 0, 2))


def _zigzag_bars_with_gaps(rng: random.Random, *, n_bars: int, leg_length: int, base_amplitude: float, amplitude_growth: float, noise: float, gap_probability: float) -> list[Bar1m]:
    """A randomized variant of ``build_synthetic_smoke_context``'s own
    proven "widening zigzag" recipe (reliably forms real FVGs/swing
    highs+lows on 1m/5m/15m in few bars) -- randomized leg length,
    amplitude growth, per-bar noise, and occasional genuine multi-minute
    data gaps, while staying inside one clean ~15h trading window
    (matching that function's own maintenance-break-avoidance) so
    randomized parameters can never accidentally produce an invalid
    ``known_at``."""
    start = datetime(2024, 12, 31, 23, 5, tzinfo=UTC)
    bars: list[Bar1m] = []
    price = 20000.0
    minute_offset = 0
    i = 0
    while i < n_bars:
        if rng.random() < gap_probability:
            minute_offset += rng.randint(2, 10)
        ts = start + timedelta(minutes=minute_offset)
        cycle_bar = i % (2 * leg_length)
        cycle_index = i // (2 * leg_length)
        amplitude = base_amplitude + cycle_index * amplitude_growth
        move = (amplitude if cycle_bar < leg_length else -amplitude) + rng.uniform(-noise, noise)
        open_p = price
        close_p = price + move
        high_p = max(open_p, close_p) + 3.0 + rng.uniform(0, noise)
        low_p = min(open_p, close_p) - 3.0 - rng.uniform(0, noise)
        bars.append(Bar1m(timestamp=ts, open=open_p, high=high_p, low=low_p, close=close_p, volume=100.0, contract="RAND"))
        price = close_p
        minute_offset += 1
        i += 1
    return bars


def _random_zigzag_context(rng: random.Random, *, n_bars: int) -> "PreparedBacktestContext":
    bars = _zigzag_bars_with_gaps(
        rng, n_bars=n_bars, leg_length=rng.choice([15, 20, 25, 30]), base_amplitude=rng.uniform(10.0, 30.0),
        amplitude_growth=rng.uniform(2.0, 8.0), noise=rng.uniform(0.0, 2.0), gap_probability=rng.uniform(0.0, 0.03),
    )
    news_events = []
    if rng.random() < 0.7:
        for j in range(rng.randint(1, 4)):
            idx = rng.randint(0, len(bars) - 1)
            news_events.append(news(bars[idx].timestamp + timedelta(minutes=rng.randint(-15, 5)), event_id=f"N{j}-{idx}"))
    return prepare_backtest_context(bars, news_events, tick_size=0.25, pivot_strength=2, reclaim_window_bars=3)


class SlowReferenceVsAcceleratedEquivalenceTests(unittest.TestCase):
    """task 045, requirement 5: the accelerated ``simulate_market_
    execution`` must produce byte-identical ``trades``/``skipped`` to
    the slow, unaccelerated ``_simulate_market_execution_reference``,
    for every real (equivalence class, exit variant) a randomized zigzag
    context discovers -- across many random trials. Randomized bar
    counts, leg lengths, amplitudes, noise, gap probability, and news
    placement, applied to real (not hand-picked) equivalence classes and
    all 20 real exit variants, organically covers long/short, data gaps,
    missing targets, news blocks, unresolved (still-open) positions,
    one-position skipping, TP-only, SL-only, and same-bar TP+SL -- proven
    below by asserting every one of those outcomes was actually observed
    across the trials, not merely assumed."""

    def test_accelerated_matches_reference_across_many_randomized_trials(self):
        rng = random.Random(20450)
        total_compared = 0
        exit_reasons_seen = set()
        skip_reasons_seen = set()
        directions_seen = set()

        for trial in range(20):
            context = _random_zigzag_context(rng, n_bars=rng.choice([300, 500, 900]))
            universe = discover_entry_strategy_universe(
                context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1,
            )
            if universe.equivalence_class_count == 0:
                continue
            bar_range_index = BarRangeIndex.build(context.bars_1m)

            for equivalence_class in universe.equivalence_classes:
                direction = equivalence_class.representative_spec.primary.direction
                directions_seen.add(direction)
                for variant in ALL_EXIT_VARIANTS:
                    neutral_id = f"{equivalence_class.class_id}::{variant.variant_id}"
                    ref_trades, ref_skipped = _simulate_market_execution_reference(equivalence_class.observed_entries, direction, variant, context, neutral_id)
                    acc_trades, acc_skipped = simulate_market_execution(
                        equivalence_class.observed_entries, direction, variant, context, neutral_id, bar_range_index=bar_range_index,
                    )
                    self.assertEqual(ref_trades, acc_trades, msg=f"trial={trial} class={equivalence_class.class_id} variant={variant.variant_id}")
                    self.assertEqual(ref_skipped, acc_skipped, msg=f"trial={trial} class={equivalence_class.class_id} variant={variant.variant_id}")
                    total_compared += 1
                    exit_reasons_seen.update(t.exit_reason for t in ref_trades)
                    skip_reasons_seen.update(s.reason for s in ref_skipped)

        self.assertGreater(total_compared, 500)  # confirms this genuinely exercised many real combinations
        self.assertEqual(directions_seen, {LONG, SHORT})
        self.assertIn(TradeExitReason.TP_HIT, exit_reasons_seen)
        self.assertIn(TradeExitReason.SL_HIT, exit_reasons_seen)
        self.assertIn(TradeExitReason.OPEN_UNRESOLVED, exit_reasons_seen)
        self.assertIn(ExecutionSkipReason.POSITION_ALREADY_OPEN, skip_reasons_seen)
        self.assertIn(ExecutionSkipReason.NEWS_BLOCK, skip_reasons_seen)


class EntryTargetCacheEquivalenceTests(unittest.TestCase):
    """task 047, requirement 5: an ``EntryTargetCache`` -- scoped per
    equivalence class, exactly like ``_expand_and_simulate`` now builds
    it -- must produce byte-for-byte identical trades, skips, and
    per-instrument ``StrategyVariantResult`` fields (both MNQ and NQ) to
    running with no cache at all, for every real (equivalence class,
    exit variant) a randomized zigzag context discovers. The cache only
    ever memoizes an already-deterministic computation (``detect_tp_
    target_candidates``/``detect_sl_target_candidates`` keyed on the
    observed entry's own stable id, plus -- for SL -- the FVG confluence
    it was resolved with); it must never change which target is chosen,
    for either exit family.

    Reuses ``_random_zigzag_context`` (same fixture as
    ``SlowReferenceVsAcceleratedEquivalenceTests`` above), so this
    organically covers long/short, same-bar TP+SL ambiguity, news
    windows, both SL families (liquidity-rank and FVG-boundary --
    proven below by asserting SL_kind FVG_BOUNDARY was actually
    resolved into at least one trade across the trials, not merely
    attempted), and liquidity TP targets."""

    def test_cached_matches_uncached_across_many_randomized_trials(self):
        rng = random.Random(74110)
        total_compared = 0
        sl_kinds_with_trades_seen = set()
        directions_seen = set()

        for trial in range(20):
            context = _random_zigzag_context(rng, n_bars=rng.choice([300, 500, 900]))
            universe = discover_entry_strategy_universe(
                context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1,
            )
            if universe.equivalence_class_count == 0:
                continue
            bar_range_index = BarRangeIndex.build(context.bars_1m)

            for equivalence_class in universe.equivalence_classes:
                direction = equivalence_class.representative_spec.primary.direction
                directions_seen.add(direction)
                # One fresh cache per class, shared across all of that
                # class's own exit variants -- exactly how
                # _expand_and_simulate uses it in production.
                entry_target_cache = EntryTargetCache()
                for variant in ALL_EXIT_VARIANTS:
                    uncached_results = run_strategy_variant_backtest_all_instruments(
                        equivalence_class, variant, DEFAULT_INSTRUMENTS, context, bar_range_index=bar_range_index,
                    )
                    cached_results = run_strategy_variant_backtest_all_instruments(
                        equivalence_class, variant, DEFAULT_INSTRUMENTS, context, bar_range_index=bar_range_index,
                        entry_target_cache=entry_target_cache,
                    )
                    for instrument in DEFAULT_INSTRUMENTS:
                        uncached, cached = uncached_results[instrument], cached_results[instrument]
                        msg = f"trial={trial} class={equivalence_class.class_id} variant={variant.variant_id} instrument={instrument.value}"
                        self.assertEqual(uncached.trades, cached.trades, msg=msg)
                        self.assertEqual(uncached.account_result.accounted_trades, cached.account_result.accounted_trades, msg=msg)
                        self.assertEqual(uncached.skipped_entries_by_reason, cached.skipped_entries_by_reason, msg=msg)
                        self.assertEqual(uncached.net_pnl_usd, cached.net_pnl_usd, msg=msg)
                        self.assertEqual(uncached.trade_count, cached.trade_count, msg=msg)
                        self.assertEqual(uncached.max_drawdown_usd, cached.max_drawdown_usd, msg=msg)
                        self.assertEqual(uncached.win_rate, cached.win_rate, msg=msg)
                        self.assertEqual(uncached.yearly, cached.yearly, msg=msg)
                        self.assertEqual(uncached.monthly, cached.monthly, msg=msg)
                        total_compared += 1
                        for trade in uncached.trades:
                            sl_kinds_with_trades_seen.add(variant.sl_kind)

        self.assertGreater(total_compared, 500)  # confirms this genuinely exercised many real combinations
        self.assertEqual(directions_seen, {LONG, SHORT})
        self.assertIn(SLVariantKind.LIQUIDITY_RANK, sl_kinds_with_trades_seen)
        self.assertIn(SLVariantKind.FVG_BOUNDARY, sl_kinds_with_trades_seen)

    def test_cache_is_never_shared_across_equivalence_classes(self):
        """A cache scoped to one class must never leak a memoized result
        into a different class -- passing the SAME cache instance across
        two different classes' own variants (never done in production,
        but must not silently corrupt results if it somehow happened)
        would only be safe because the cache key includes the observed
        entry's own id, which real entries across different classes
        never share -- confirmed here by asserting a shared cache still
        matches a fresh-per-class cache, rather than by construction
        alone."""
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1,
        )
        self.assertGreaterEqual(universe.equivalence_class_count, 2)
        bar_range_index = BarRangeIndex.build(context.bars_1m)
        variant = ALL_EXIT_VARIANTS[0]

        shared_cache = EntryTargetCache()
        for equivalence_class in universe.equivalence_classes[:2]:
            fresh_cache = EntryTargetCache()
            with_shared = run_strategy_variant_backtest_all_instruments(
                equivalence_class, variant, DEFAULT_INSTRUMENTS, context, bar_range_index=bar_range_index, entry_target_cache=shared_cache,
            )
            with_fresh = run_strategy_variant_backtest_all_instruments(
                equivalence_class, variant, DEFAULT_INSTRUMENTS, context, bar_range_index=bar_range_index, entry_target_cache=fresh_cache,
            )
            for instrument in DEFAULT_INSTRUMENTS:
                self.assertEqual(with_shared[instrument].trades, with_fresh[instrument].trades)
                self.assertEqual(with_shared[instrument].net_pnl_usd, with_fresh[instrument].net_pnl_usd)


class InstrumentDedupTests(unittest.TestCase):
    """task 045, requirement 4: market execution runs exactly once per
    (equivalence class, exit variant), shared across every instrument --
    ``run_strategy_variant_backtest_all_instruments`` must return exactly
    what calling ``run_strategy_variant_backtest`` once per instrument
    would have (same trades, P&L, ranking-relevant fields), while
    provably running the shared execution step only once."""

    def _one_real_variant(self):
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1
        )
        self.assertGreater(universe.equivalence_class_count, 0)
        return context, universe.equivalence_classes[0], ALL_EXIT_VARIANTS[0]

    def test_matches_calling_run_strategy_variant_backtest_once_per_instrument(self):
        context, equivalence_class, variant = self._one_real_variant()

        individually = {
            instrument: run_strategy_variant_backtest(equivalence_class, variant, instrument, context)
            for instrument in DEFAULT_INSTRUMENTS
        }
        shared = run_strategy_variant_backtest_all_instruments(equivalence_class, variant, DEFAULT_INSTRUMENTS, context)

        self.assertEqual(set(shared.keys()), set(DEFAULT_INSTRUMENTS))
        for instrument in DEFAULT_INSTRUMENTS:
            a, b = individually[instrument], shared[instrument]
            self.assertEqual(a.strategy_variant_id, b.strategy_variant_id)
            self.assertEqual(a.net_pnl_usd, b.net_pnl_usd)
            self.assertEqual(a.trade_count, b.trade_count)
            self.assertEqual(a.max_drawdown_usd, b.max_drawdown_usd)
            self.assertEqual(a.win_rate, b.win_rate)
            self.assertEqual(a.yearly, b.yearly)
            self.assertEqual(a.monthly, b.monthly)
            self.assertEqual(a.skipped_entries_by_reason, b.skipped_entries_by_reason)

        # the two instrument results are genuinely distinct (different
        # point value/sizing), not accidentally identical objects.
        self.assertNotEqual(shared[Instrument.MNQ].strategy_variant_id, shared[Instrument.NQ].strategy_variant_id)

    def test_market_execution_runs_exactly_once_regardless_of_instrument_count(self):
        context, equivalence_class, variant = self._one_real_variant()
        calls = []
        orig = simulate_variant_execution

        def spy(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        with patch("tradelab.backtest_v2.simulate_variant_execution", spy):
            run_strategy_variant_backtest_all_instruments(equivalence_class, variant, DEFAULT_INSTRUMENTS, context)
        self.assertEqual(len(calls), 1)  # never once per instrument, regardless of len(DEFAULT_INSTRUMENTS)

    def test_expand_and_simulate_shares_execution_across_instruments_on_resume_too(self):
        # If one instrument is already durably completed and the other
        # isn't, execution must still run (exactly once) for the
        # remaining instrument -- and must NOT re-commit the already-
        # completed one.
        context, equivalence_class, variant = self._one_real_variant()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1
        )
        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(runs_dir=tmp, market_artifact_path=None, news_csv_path=None, parameters={}, is_demo=True)
            journal = _open_journal(
                reporter, market_artifact_path=None, news_csv_path=None, parameters={}, risk_config=DEFAULT_RISK_CONFIG,
                equity_config=DEFAULT_EQUITY_CONFIG, exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
            )
            # pre-complete MNQ only, for the very first (class, variant) combo.
            first_class = universe.equivalence_classes[0]
            first_variant = ALL_EXIT_VARIANTS[0]
            precomputed = run_strategy_variant_backtest(first_class, first_variant, Instrument.MNQ, context)
            journal.commit_summary(VariantSummaryRow.from_result(precomputed).to_dict())
            self.assertEqual(journal.completed_count(), 1)

            completed, _leaderboard = _expand_and_simulate(
                reporter, journal, universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, context,
                risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG, top_n_leaderboard=3, checkpoint_every=1000,
            )
            journal.close()

        total_expected = universe.equivalence_class_count * len(ALL_EXIT_VARIANTS) * len(DEFAULT_INSTRUMENTS)
        self.assertEqual(completed, total_expected)  # exactly total, not total+1 (no duplicate re-commit of MNQ)


class ExecutionAccelerationBenchmarkTests(unittest.TestCase):
    """task 045, requirement 5: the benchmark reports *measured* baseline
    vs accelerated timings (never a claimed/projected number) and exact
    result equality, on the deterministic synthetic representative
    fixture."""

    def test_benchmark_reports_measured_timings_and_exact_equality(self):
        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1
        )
        self.assertGreater(universe.equivalence_class_count, 0)

        report = run_execution_acceleration_benchmark(context, universe, ALL_EXIT_VARIANTS, sample_size=25)

        self.assertTrue(report["exact_result_equality"])
        self.assertEqual(report["sample_size"], 25)
        self.assertGreater(report["reference_elapsed_seconds"], 0.0)
        self.assertGreater(report["accelerated_elapsed_seconds"], 0.0)
        self.assertIsNotNone(report["reference_variants_per_second"])
        self.assertIsNotNone(report["accelerated_variants_per_second"])
        self.assertIsNotNone(report["speedup_factor"])

    def test_benchmark_cli_runs_and_prints_a_report(self):
        exit_code = backtest_v2_main(["benchmark", "--sample-size", "10"])
        self.assertEqual(exit_code, 0)


class StageDiagnosticTests(unittest.TestCase):
    """task 047, requirements 1-2: run_stage_diagnostic actually measures
    (never fabricates) both an uncached and a cached pass over a real,
    bounded sample -- reporting wall/CPU time, peak RSS, and all five
    required stage buckets for each, on the deterministic synthetic
    representative fixture (the real-data run is done manually/out of
    band -- see the PR description -- never as part of the automated
    suite, per the task's own "never run the full 3-year backtest" and
    general test-speed constraints)."""

    def test_reports_measured_timings_for_both_passes(self):
        from tradelab.backtest_v2 import run_stage_diagnostic

        context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=1
        )
        self.assertGreater(universe.equivalence_class_count, 0)

        report = run_stage_diagnostic(context, universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, sample_size=5)

        self.assertEqual(report["sample_size"], 5)
        required_stages = {"bar_range_index_build", "entry_target_resolution", "exit_resolution", "accounting_equity", "persistence"}
        for pass_name in ("uncached", "cached"):
            pass_report = report[pass_name]
            self.assertGreater(pass_report["wall_elapsed_seconds"], 0.0)
            self.assertGreaterEqual(pass_report["cpu_elapsed_seconds"], 0.0)
            self.assertGreater(pass_report["peak_rss_bytes"], 0)
            self.assertIsNotNone(pass_report["variants_per_second"])
            self.assertTrue(required_stages.issubset(pass_report["stage_timings"]["totals_seconds"].keys()))
        self.assertEqual(report["uncached"]["use_cache"], False)
        self.assertEqual(report["uncached"]["entry_target_cache_entries"], 0)
        self.assertEqual(report["cached"]["use_cache"], True)
        self.assertGreater(report["cached"]["entry_target_cache_entries"], 0)
        self.assertIsNotNone(report["speedup_factor"])

    def test_stage_diagnostic_cli_runs_and_prints_a_report(self):
        exit_code = backtest_v2_main(["stage-diagnostic", "--sample-size", "3"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
