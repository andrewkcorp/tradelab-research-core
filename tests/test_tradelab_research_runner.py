#!/usr/bin/env python3
"""Research batch runner tests: iter_research_evaluations is a true
generator that never materializes its strategy_specs input and never
calls prepare_backtest_context; NQ/MNQ are evaluated independently from
one shared BacktestResult per strategy; the min-trades-per-window
frequency rule (only accounted trades count, untradeable trades never
do, OPEN_UNRESOLVED trades do, [start, end) boundaries are exact); the
safe prefilter rejects only strategies that cannot possibly qualify and
never produces a false rejection; results do not depend on strategy
input order; and ResearchWindow/ResearchConfig reject malformed or
overlapping windows."""
from __future__ import annotations

import inspect
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone

from tradelab.accounting import Instrument, RiskConfig
from tradelab.backtest import BacktestResult, ExecutedTrade, PreparedBacktestContext, TradeExitReason
from tradelab.entry_signal_timeline import EntrySignal, EntrySignalTimeline, SourceKind
from tradelab.equity import EquityConfig
from tradelab.research_runner import (
    RejectionReason,
    ResearchConfig,
    ResearchEvaluation,
    ResearchWindow,
    WindowTradeCount,
    iter_research_evaluations,
)
from tradelab.sl_targets import SLSourceType
from tradelab.strategy_spec import LONG, BiasPolicy, SLChoice, StrategySpec, TPChoice
from tradelab.tp_targets import TPSourceType

UTC = timezone.utc
T0 = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)

RC = RiskConfig()
EC = EquityConfig()


def window(window_id: str = "w1", start: datetime = T0, end: datetime = T0 + timedelta(days=1)) -> ResearchWindow:
    return ResearchWindow(window_id=window_id, start_utc=start, end_utc=end)


def config(*windows: ResearchWindow, min_trades: int = 3, risk_config: RiskConfig = RC, equity_config: EquityConfig = EC) -> ResearchConfig:
    return ResearchConfig(windows=windows or (window(),), risk_config=risk_config, equity_config=equity_config, min_trades_per_window=min_trades)


def make_signal(selector: str, direction: str, known_at: datetime, timeframe: str = "1m") -> EntrySignal:
    return EntrySignal(
        candidate_id=f"cand-{selector}-{direction}-{known_at.isoformat()}",
        selector=selector,
        source_kind=SourceKind.LIQUIDITY,
        timeframe=timeframe,
        direction=direction,
        timestamp=known_at,
        known_at=known_at,
        trading_day_id="2024-03-04",
        valid_until=known_at + timedelta(hours=8),
        source=None,
    )


def make_context(signals: list) -> PreparedBacktestContext:
    by_selector: dict[str, list] = {}
    for s in signals:
        by_selector.setdefault(s.selector, []).append(s)
    timeline = EntrySignalTimeline(signals=list(signals), signals_by_selector=by_selector)
    # Every other field is untouched by research_runner (which only ever
    # reads context.timeline and passes context opaquely to a mocked
    # backtest_prepared_strategy in these tests), so a real detector-built
    # value is unnecessary here.
    return PreparedBacktestContext(
        bars_1m=(),
        news_events=(),
        tick_size=0.25,
        pivot_strength=3,
        reclaim_window_bars=5,
        fvg_results=None,
        fvg_touch_results=None,
        fvg_entry_result=None,
        swing_equal_liquidity_results=None,
        session_liquidity_results=None,
        liquidity_entry_result=None,
        data_wick_result=None,
        htf_bias_result=None,
        timeline=timeline,
    )


def make_strategy(strategy_id: str, primary: str = "fvg-1m-fvg_touch", direction: str = LONG) -> StrategySpec:
    return StrategySpec(
        strategy_id=strategy_id,
        direction=direction,
        primary_entry_selector=primary,
        required_confluence_selectors=(),
        bias_policy=BiasPolicy.IGNORE,
        tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY, rank=1),
        sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY, rank=1),
    )


def make_trades(
    n: int,
    *,
    strategy_id: str,
    start: datetime,
    direction: str = LONG,
    entry_price: float = 100.0,
    sl_price: float = 99.0,
    tp_price: float = 101.0,
    exit_reason: TradeExitReason = TradeExitReason.TP_HIT,
    step: timedelta = timedelta(minutes=1),
) -> list[ExecutedTrade]:
    trades = []
    for i in range(n):
        ts = start + step * i
        if exit_reason == TradeExitReason.OPEN_UNRESOLVED:
            exit_ts, exit_price = None, None
        elif exit_reason == TradeExitReason.TP_HIT:
            exit_ts, exit_price = ts + timedelta(minutes=1), tp_price
        else:
            exit_ts, exit_price = ts + timedelta(minutes=1), sl_price
        gross = None
        if exit_price is not None:
            gross = (exit_price - entry_price) if direction == LONG else (entry_price - exit_price)
        trades.append(
            ExecutedTrade(
                trade_id=f"{strategy_id}-t{i}",
                strategy_id=strategy_id,
                direction=direction,
                signal_known_at=ts,
                entry_timestamp=ts,
                entry_price=entry_price,
                tp_price=tp_price,
                tp_source_type="liquidity",
                tp_source_id=f"tp-{i}",
                sl_price=sl_price,
                sl_source_type="liquidity",
                sl_source_id=f"sl-{i}",
                exit_timestamp=exit_ts,
                exit_price=exit_price,
                exit_reason=exit_reason,
                gross_pnl_points=gross,
            )
        )
    return trades


def make_backtest_result(trades: list[ExecutedTrade], strategy_id: str) -> BacktestResult:
    tp = sum(1 for t in trades if t.exit_reason == TradeExitReason.TP_HIT)
    sl = sum(1 for t in trades if t.exit_reason == TradeExitReason.SL_HIT)
    ou = sum(1 for t in trades if t.exit_reason == TradeExitReason.OPEN_UNRESOLVED)
    total = sum(t.gross_pnl_points for t in trades if t.gross_pnl_points is not None)
    return BacktestResult(
        strategy_id=strategy_id, trades=trades, skipped_signals=[], trade_count=len(trades),
        tp_hit_count=tp, sl_hit_count=sl, open_unresolved_count=ou, gross_pnl_points_total=total,
    )


class GeneratorLazinessTests(unittest.TestCase):
    def test_returns_a_generator(self):
        gen = iter_research_evaluations((), make_context([]), config())
        self.assertTrue(inspect.isgenerator(gen))

    def test_does_not_materialize_strategy_specs_input(self):
        strategies = [make_strategy(f"s{i}") for i in range(5)]
        pulled: list[str] = []

        def tracking_iter():
            for s in strategies:
                pulled.append(s.strategy_id)
                yield s

        context = make_context([])  # no signals at all -> every strategy prefilter-rejects instantly
        gen = iter_research_evaluations(tracking_iter(), context, config())
        first = next(gen)
        self.assertEqual(pulled, ["s0"])
        self.assertFalse(first.qualifies_frequency)
        self.assertLess(len(pulled), len(strategies))


class ContextPreparationNotRepeatedTests(unittest.TestCase):
    def test_prepare_backtest_context_is_never_called(self):
        with unittest.mock.patch("tradelab.backtest.prepare_backtest_context") as mocked:
            strategies = [make_strategy("s1"), make_strategy("s2")]
            list(iter_research_evaluations(strategies, make_context([]), config()))
            mocked.assert_not_called()


class InstrumentSeparationTests(unittest.TestCase):
    def test_nq_and_mnq_evaluated_independently_from_one_shared_backtest_result(self):
        w = window(end=T0 + timedelta(hours=1))
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(3)]
        context = make_context(signals)
        strategy = make_strategy("s-instr")
        # entry-sl distance of 30 points: NQ (point value 20) needs 600 USD
        # risk for even 1 contract, exceeding the 550 USD max -> untradeable
        # for every trade; MNQ (point value 2) needs only 60 USD -> tradeable.
        trades = make_trades(3, strategy_id="s-instr", start=T0, entry_price=100.0, sl_price=70.0, tp_price=160.0)
        backtest_result = make_backtest_result(trades, "s-instr")
        cfg = config(w, min_trades=3)

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", return_value=backtest_result) as mocked_bt:
            evaluations = list(iter_research_evaluations([strategy], context, cfg))

        self.assertEqual(mocked_bt.call_count, 1)  # shared once, not once per instrument
        self.assertEqual(len(evaluations), 2)
        by_instrument = {e.instrument: e for e in evaluations}
        self.assertIn(Instrument.NQ, by_instrument)
        self.assertIn(Instrument.MNQ, by_instrument)
        self.assertFalse(by_instrument[Instrument.NQ].qualifies_frequency)
        self.assertEqual(by_instrument[Instrument.NQ].window_trade_counts[0].trade_count, 0)
        self.assertTrue(by_instrument[Instrument.MNQ].qualifies_frequency)
        self.assertEqual(by_instrument[Instrument.MNQ].window_trade_counts[0].trade_count, 3)


class UntradeableTradesExcludedTests(unittest.TestCase):
    def test_untradeable_trades_never_count_toward_frequency(self):
        w = window(end=T0 + timedelta(hours=1))
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(2)]
        context = make_context(signals)
        strategy = make_strategy("s-untradeable")
        trades = make_trades(2, strategy_id="s-untradeable", start=T0, entry_price=100.0, sl_price=50.0, tp_price=200.0)
        backtest_result = make_backtest_result(trades, "s-untradeable")
        cfg = config(w, min_trades=2)

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", return_value=backtest_result):
            evaluations = list(iter_research_evaluations([strategy], context, cfg, instruments=(Instrument.NQ,)))

        self.assertEqual(len(evaluations), 1)
        evaluation = evaluations[0]
        self.assertFalse(evaluation.prefiltered)
        self.assertFalse(evaluation.qualifies_frequency)
        self.assertEqual(evaluation.rejection_reason, RejectionReason.INSUFFICIENT_TRADE_FREQUENCY)
        self.assertEqual(evaluation.window_trade_counts[0].trade_count, 0)
        self.assertIsNone(evaluation.backtest_result)
        self.assertIsNone(evaluation.account_result)
        self.assertIsNone(evaluation.equity_result)


class OpenUnresolvedCountsTests(unittest.TestCase):
    def test_open_unresolved_trades_count_as_opened(self):
        w = window(end=T0 + timedelta(hours=1))
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(3)]
        context = make_context(signals)
        strategy = make_strategy("s-open")
        trades = make_trades(3, strategy_id="s-open", start=T0, exit_reason=TradeExitReason.OPEN_UNRESOLVED)
        backtest_result = make_backtest_result(trades, "s-open")
        cfg = config(w, min_trades=3)

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", return_value=backtest_result):
            evaluations = list(iter_research_evaluations([strategy], context, cfg, instruments=(Instrument.NQ,)))

        evaluation = evaluations[0]
        self.assertTrue(evaluation.qualifies_frequency)
        self.assertEqual(evaluation.window_trade_counts[0].trade_count, 3)
        self.assertEqual(evaluation.account_result.open_unresolved_count, 3)
        self.assertIsNotNone(evaluation.equity_result)


class WindowBoundaryTests(unittest.TestCase):
    def test_half_open_start_inclusive_end_exclusive(self):
        w = window(end=T0 + timedelta(minutes=3))
        # 3rd signal/trade lands exactly on end_utc -> excluded both times.
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in (0, 1, 3)]
        context = make_context(signals)
        strategy = make_strategy("s-boundary")
        trades = make_trades(1, strategy_id="s-boundary", start=T0)
        trades += make_trades(1, strategy_id="s-boundary-2", start=T0 + timedelta(minutes=1))
        trades += make_trades(1, strategy_id="s-boundary-3", start=T0 + timedelta(minutes=3))
        backtest_result = make_backtest_result(trades, "s-boundary")
        cfg = config(w, min_trades=2)

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", return_value=backtest_result):
            evaluations = list(iter_research_evaluations([strategy], context, cfg, instruments=(Instrument.NQ,)))

        evaluation = evaluations[0]
        self.assertEqual(evaluation.window_trade_counts[0].trade_count, 2)
        self.assertTrue(evaluation.window_trade_counts[0].meets_minimum)
        self.assertTrue(evaluation.qualifies_frequency)


class PrefilterCorrectnessTests(unittest.TestCase):
    def test_prefilter_rejects_when_raw_signals_fall_short_in_any_window(self):
        w = window(end=T0 + timedelta(hours=1))
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(1)]  # only 1 raw signal
        context = make_context(signals)
        strategy = make_strategy("s-prefiltered")
        cfg = config(w, min_trades=3)

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy") as mocked_bt:
            evaluations = list(iter_research_evaluations([strategy], context, cfg))

        mocked_bt.assert_not_called()
        self.assertEqual(len(evaluations), 2)  # still one per instrument
        for evaluation in evaluations:
            self.assertTrue(evaluation.prefiltered)
            self.assertFalse(evaluation.qualifies_frequency)
            self.assertEqual(evaluation.rejection_reason, RejectionReason.INSUFFICIENT_TRADE_FREQUENCY)
            self.assertEqual(evaluation.window_trade_counts[0].trade_count, 1)
            self.assertFalse(evaluation.window_trade_counts[0].meets_minimum)
            self.assertIsNone(evaluation.backtest_result)

    def test_prefilter_never_produces_a_false_rejection(self):
        # 5 raw signals clear the min=3 bar, but only 2 of them convert into
        # accounted trades in the (mocked) real backtest -- the strategy
        # must still run the real backtest (never prefiltered) and only
        # then correctly fail on the real, lower count.
        w = window(end=T0 + timedelta(hours=1))
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
        context = make_context(signals)
        strategy = make_strategy("s-close-call")
        trades = make_trades(2, strategy_id="s-close-call", start=T0)
        backtest_result = make_backtest_result(trades, "s-close-call")
        cfg = config(w, min_trades=3)

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", return_value=backtest_result) as mocked_bt:
            evaluations = list(iter_research_evaluations([strategy], context, cfg, instruments=(Instrument.NQ,)))

        mocked_bt.assert_called_once()
        evaluation = evaluations[0]
        self.assertFalse(evaluation.prefiltered)
        self.assertFalse(evaluation.qualifies_frequency)
        self.assertEqual(evaluation.window_trade_counts[0].trade_count, 2)

    def test_prefilter_lets_through_a_genuinely_qualifying_strategy(self):
        w = window(end=T0 + timedelta(hours=1))
        signals = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(3)]
        context = make_context(signals)
        strategy = make_strategy("s-qualifies")
        trades = make_trades(3, strategy_id="s-qualifies", start=T0)
        backtest_result = make_backtest_result(trades, "s-qualifies")
        cfg = config(w, min_trades=3)

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", return_value=backtest_result):
            evaluations = list(iter_research_evaluations([strategy], context, cfg, instruments=(Instrument.NQ,)))

        evaluation = evaluations[0]
        self.assertFalse(evaluation.prefiltered)
        self.assertTrue(evaluation.qualifies_frequency)
        self.assertIsNotNone(evaluation.backtest_result)
        self.assertIsNotNone(evaluation.account_result)
        self.assertIsNotNone(evaluation.equity_result)


class OrderIndependenceTests(unittest.TestCase):
    def test_results_do_not_depend_on_strategy_order(self):
        w = window(end=T0 + timedelta(hours=1))
        signals_a = [make_signal("fvg-1m-fvg_touch", LONG, T0 + timedelta(minutes=i)) for i in range(5)]
        signals_b = [make_signal("fvg-5m-ifvg", LONG, T0 + timedelta(minutes=i)) for i in range(1)]  # too few, prefiltered
        context = make_context(signals_a + signals_b)
        strategy_a = make_strategy("s-a", primary="fvg-1m-fvg_touch")
        strategy_b = make_strategy("s-b", primary="fvg-5m-ifvg")
        trades_a = make_trades(5, strategy_id="s-a", start=T0)
        results_by_id = {"s-a": make_backtest_result(trades_a, "s-a")}
        cfg = config(w, min_trades=3)

        def fake_backtest(strategy, ctx):
            return results_by_id[strategy.strategy_id]

        def summarize(evaluations):
            return {
                (e.strategy_id, e.instrument.value): (
                    e.qualifies_frequency,
                    e.rejection_reason,
                    e.prefiltered,
                    tuple((wc.window_id, wc.trade_count, wc.meets_minimum) for wc in e.window_trade_counts),
                )
                for e in evaluations
            }

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", side_effect=fake_backtest):
            forward = summarize(iter_research_evaluations([strategy_a, strategy_b], context, cfg))
            backward = summarize(iter_research_evaluations([strategy_b, strategy_a], context, cfg))

        self.assertEqual(forward, backward)
        self.assertTrue(forward[("s-a", "NQ")][0])
        self.assertFalse(forward[("s-b", "NQ")][0])
        self.assertTrue(forward[("s-b", "NQ")][2])  # prefiltered


class QualificationBoundaryTests(unittest.TestCase):
    """The task's own headline rule: at least 100 trades in EVERY passed
    window qualifies; 99 in even one window rejects."""

    def setUp(self):
        self.window_2024 = ResearchWindow("y2024", datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC))
        self.window_2025 = ResearchWindow("y2025", datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
        self.cfg = config(self.window_2024, self.window_2025, min_trades=100)

    def _context_with_raw_signals(self, count_2024: int, count_2025: int) -> PreparedBacktestContext:
        signals = [make_signal("fvg-1m-fvg_touch", LONG, self.window_2024.start_utc + timedelta(minutes=i)) for i in range(count_2024)]
        signals += [make_signal("fvg-1m-fvg_touch", LONG, self.window_2025.start_utc + timedelta(minutes=i)) for i in range(count_2025)]
        return make_context(signals)

    def test_100_trades_in_every_window_qualifies(self):
        context = self._context_with_raw_signals(100, 100)
        strategy = make_strategy("s-100")
        trades = make_trades(100, strategy_id="s-100-a", start=self.window_2024.start_utc)
        trades += make_trades(100, strategy_id="s-100-b", start=self.window_2025.start_utc)
        backtest_result = make_backtest_result(trades, "s-100")

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", return_value=backtest_result):
            evaluations = list(iter_research_evaluations([strategy], context, self.cfg, instruments=(Instrument.NQ,)))

        evaluation = evaluations[0]
        self.assertTrue(evaluation.qualifies_frequency)
        counts = {wc.window_id: wc for wc in evaluation.window_trade_counts}
        self.assertEqual(counts["y2024"].trade_count, 100)
        self.assertTrue(counts["y2024"].meets_minimum)
        self.assertEqual(counts["y2025"].trade_count, 100)
        self.assertTrue(counts["y2025"].meets_minimum)

    def test_99_trades_in_one_window_rejects(self):
        context = self._context_with_raw_signals(100, 100)  # raw signals still clear the bar in both windows
        strategy = make_strategy("s-99")
        trades = make_trades(100, strategy_id="s-99-a", start=self.window_2024.start_utc)
        trades += make_trades(99, strategy_id="s-99-b", start=self.window_2025.start_utc)  # one short
        backtest_result = make_backtest_result(trades, "s-99")

        with unittest.mock.patch("tradelab.research_runner.backtest_prepared_strategy", return_value=backtest_result):
            evaluations = list(iter_research_evaluations([strategy], context, self.cfg, instruments=(Instrument.NQ,)))

        evaluation = evaluations[0]
        self.assertFalse(evaluation.qualifies_frequency)
        self.assertFalse(evaluation.prefiltered)
        self.assertEqual(evaluation.rejection_reason, RejectionReason.INSUFFICIENT_TRADE_FREQUENCY)
        counts = {wc.window_id: wc for wc in evaluation.window_trade_counts}
        self.assertEqual(counts["y2024"].trade_count, 100)
        self.assertTrue(counts["y2024"].meets_minimum)
        self.assertEqual(counts["y2025"].trade_count, 99)
        self.assertFalse(counts["y2025"].meets_minimum)


class ResearchWindowValidationTests(unittest.TestCase):
    def test_naive_start_rejected(self):
        with self.assertRaises(ValueError):
            ResearchWindow("w", datetime(2024, 1, 1), datetime(2024, 1, 2, tzinfo=UTC))

    def test_naive_end_rejected(self):
        with self.assertRaises(ValueError):
            ResearchWindow("w", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2))

    def test_start_not_before_end_rejected(self):
        with self.assertRaises(ValueError):
            ResearchWindow("w", datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC))
        with self.assertRaises(ValueError):
            ResearchWindow("w", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC))

    def test_empty_window_id_rejected(self):
        with self.assertRaises(ValueError):
            ResearchWindow("", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))


class ResearchConfigValidationTests(unittest.TestCase):
    def test_overlapping_windows_rejected(self):
        w1 = ResearchWindow("w1", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC))
        w2 = ResearchWindow("w2", datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 9, 1, tzinfo=UTC))
        with self.assertRaises(ValueError):
            ResearchConfig(windows=(w1, w2), risk_config=RC, equity_config=EC)

    def test_duplicate_window_id_rejected(self):
        w1 = ResearchWindow("dup", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC))
        w2 = ResearchWindow("dup", datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 9, 1, tzinfo=UTC))
        with self.assertRaises(ValueError):
            ResearchConfig(windows=(w1, w2), risk_config=RC, equity_config=EC)

    def test_empty_windows_rejected(self):
        with self.assertRaises(ValueError):
            ResearchConfig(windows=(), risk_config=RC, equity_config=EC)

    def test_windows_must_be_a_tuple(self):
        w1 = window()
        with self.assertRaises(ValueError):
            ResearchConfig(windows=[w1], risk_config=RC, equity_config=EC)

    def test_adjacent_touching_windows_are_allowed(self):
        w1 = ResearchWindow("w1", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC))
        w2 = ResearchWindow("w2", datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 9, 1, tzinfo=UTC))
        ResearchConfig(windows=(w1, w2), risk_config=RC, equity_config=EC)  # no raise

    def test_non_positive_min_trades_per_window_rejected(self):
        with self.assertRaises(ValueError):
            ResearchConfig(windows=(window(),), risk_config=RC, equity_config=EC, min_trades_per_window=0)

    def test_wrong_risk_config_type_rejected(self):
        with self.assertRaises(ValueError):
            ResearchConfig(windows=(window(),), risk_config=EC, equity_config=EC)

    def test_wrong_equity_config_type_rejected(self):
        with self.assertRaises(ValueError):
            ResearchConfig(windows=(window(),), risk_config=RC, equity_config=RC)


if __name__ == "__main__":
    unittest.main()
