"""Fixed R-multiple take-profit targets (task 048, backlog item A)."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tradelab.fixed_rr_targets import (
    FIXED_RR_MULTIPLES,
    build_fixed_rr_tp_candidate,
    fixed_rr_variant_label,
)
from tradelab.tp_targets import MarketEntryContext, TPSourceType

UTC = timezone.utc
TICK = 0.25


def entry(direction: str, price: float) -> MarketEntryContext:
    return MarketEntryContext(
        entry_id="e1", direction=direction, entry_price=price,
        entry_known_at=datetime(2024, 5, 6, 14, 30, tzinfo=UTC),
    )


class GeometryTests(unittest.TestCase):
    def test_long_1r_and_2r_measured_from_entry_and_stop(self):
        e = entry("long", 18_000.00)
        sl = 17_990.00  # R = 10.00
        one = build_fixed_rr_tp_candidate(e, sl_price=sl, multiple=1.0, tick_size=TICK, source_id="sl")
        two = build_fixed_rr_tp_candidate(e, sl_price=sl, multiple=2.0, tick_size=TICK, source_id="sl")
        self.assertEqual(one.price, 18_010.00)
        self.assertEqual(two.price, 18_020.00)
        self.assertEqual(one.distance, 10.00)
        self.assertEqual(two.distance, 20.00)

    def test_short_1r_and_2r_are_the_exact_mirror(self):
        e = entry("short", 18_000.00)
        sl = 18_012.75  # R = 12.75
        one = build_fixed_rr_tp_candidate(e, sl_price=sl, multiple=1.0, tick_size=TICK, source_id="sl")
        two = build_fixed_rr_tp_candidate(e, sl_price=sl, multiple=2.0, tick_size=TICK, source_id="sl")
        self.assertEqual(one.price, 18_000.00 - 12.75)
        self.assertEqual(two.price, 18_000.00 - 25.50)

    def test_r_uses_the_actual_tick_aligned_prices_not_a_planned_distance(self):
        """R is recomputed from whatever entry and stop the trade actually got.
        Moving the stop by one tick moves the target by exactly one tick at 1R
        and two ticks at 2R -- nothing is carried over from a nominal plan."""
        e = entry("long", 18_000.00)
        base = build_fixed_rr_tp_candidate(e, sl_price=17_990.00, multiple=2.0, tick_size=TICK, source_id="sl")
        moved = build_fixed_rr_tp_candidate(e, sl_price=17_989.75, multiple=2.0, tick_size=TICK, source_id="sl")
        self.assertEqual(moved.price - base.price, 0.50)

    def test_twenty_dollar_cost_is_not_part_of_the_geometry(self):
        """There is no argument by which a cost could enter, and the target for
        a given (entry, stop, multiple) is a pure function of those three."""
        e = entry("long", 18_000.00)
        first = build_fixed_rr_tp_candidate(e, sl_price=17_995.00, multiple=1.0, tick_size=TICK, source_id="sl")
        self.assertEqual(first.price, 18_005.00)
        self.assertEqual(first.distance, e.entry_price - 17_995.00)


class TickGridTests(unittest.TestCase):
    def test_integral_multiples_of_tick_aligned_inputs_stay_on_grid(self):
        e = entry("long", 18_000.25)
        for sl in (18_000.00, 17_999.75, 17_987.50, 17_912.25):
            for multiple in FIXED_RR_MULTIPLES:
                candidate = build_fixed_rr_tp_candidate(e, sl_price=sl, multiple=multiple, tick_size=TICK, source_id="sl")
                ticks = candidate.price / TICK
                self.assertAlmostEqual(ticks, round(ticks), places=9, msg=f"{sl} x{multiple}")

    def test_off_grid_result_snaps_away_from_entry_never_toward_it(self):
        """A fractional multiple is the only way to land off-grid. It must never
        round to a target that is reachable at less than the nominal R."""
        e = entry("long", 18_000.00)
        candidate = build_fixed_rr_tp_candidate(e, sl_price=17_999.00, multiple=1.5, tick_size=TICK, source_id="sl")
        self.assertEqual(candidate.price, 18_001.50)

        e_short = entry("short", 18_000.00)
        odd = build_fixed_rr_tp_candidate(e_short, sl_price=18_001.00, multiple=1.1, tick_size=TICK, source_id="sl")
        # exact = 17998.90 -> snapped DOWN (away from entry, further for a short)
        self.assertEqual(odd.price, 17_998.75)
        self.assertGreaterEqual(odd.distance, 1.10)


class DegenerateInputTests(unittest.TestCase):
    def test_zero_risk_yields_no_target(self):
        e = entry("long", 18_000.00)
        self.assertIsNone(build_fixed_rr_tp_candidate(e, sl_price=18_000.00, multiple=1.0, tick_size=TICK, source_id="sl"))

    def test_stop_on_the_wrong_side_yields_no_target(self):
        self.assertIsNone(build_fixed_rr_tp_candidate(entry("long", 18_000.0), sl_price=18_010.0, multiple=1.0, tick_size=TICK, source_id="sl"))
        self.assertIsNone(build_fixed_rr_tp_candidate(entry("short", 18_000.0), sl_price=17_990.0, multiple=1.0, tick_size=TICK, source_id="sl"))

    def test_invalid_multiple_and_tick_size_rejected(self):
        e = entry("long", 18_000.0)
        with self.assertRaises(ValueError):
            build_fixed_rr_tp_candidate(e, sl_price=17_990.0, multiple=0.0, tick_size=TICK, source_id="sl")
        with self.assertRaises(ValueError):
            build_fixed_rr_tp_candidate(e, sl_price=17_990.0, multiple=1.0, tick_size=0.0, source_id="sl")


class AuditTests(unittest.TestCase):
    def test_candidate_records_its_multiple_and_the_stop_it_measured(self):
        e = entry("long", 18_000.0)
        candidate = build_fixed_rr_tp_candidate(e, sl_price=17_990.0, multiple=2.0, tick_size=TICK, source_id="sl-level-7")
        self.assertEqual(candidate.source_type, TPSourceType.FIXED_RR)
        self.assertEqual(candidate.rr_multiple, 2.0)
        self.assertEqual(candidate.rr_sl_price, 17_990.0)
        self.assertEqual(candidate.source_id, "sl-level-7")
        self.assertEqual(candidate.known_at, e.entry_known_at)

    def test_no_liquidity_or_data_wick_audit_fields_are_invented(self):
        candidate = build_fixed_rr_tp_candidate(entry("long", 18_000.0), sl_price=17_990.0, multiple=1.0, tick_size=TICK, source_id="sl")
        for field in ("liquidity_type", "level_id", "lower", "upper", "source_family", "event_ids", "wick_direction", "timestamp", "rank"):
            self.assertIsNone(getattr(candidate, field), field)

    def test_variant_labels(self):
        self.assertEqual(fixed_rr_variant_label(1.0), "fixed_rr_1")
        self.assertEqual(fixed_rr_variant_label(2.0), "fixed_rr_2")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Through the real execution loops
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402
from datetime import timedelta  # noqa: E402

from tradelab.backtest import TradeExitReason, prepare_backtest_context  # noqa: E402
from tradelab.backtest_v2 import (  # noqa: E402
    ALL_EXIT_VARIANTS,
    ExecutionSkipReason,
    TPVariantKind,
    _ambiguous_same_bar_exit_count,
    _simulate_market_execution_reference,
    simulate_market_execution,
    simulate_market_execution_batched,
)
from tradelab.bar import Bar1m  # noqa: E402
from tradelab.bar_range_index import BarRangeIndex  # noqa: E402
from tradelab.entry_signal_timeline import EntryEvent, EntryEventKind  # noqa: E402
from tradelab.entry_strategy_universe import ObservedEntry  # noqa: E402
from tradelab.fvg_entry import FVGEntryCandidate, SourceType  # noqa: E402
from tradelab.fvg_touch import FVGContactType  # noqa: E402
from tradelab.liquidity import (  # noqa: E402
    HIGH, LOW, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType,
)

LONG = "long"


def _bar(ts, o, h, l, c) -> Bar1m:
    return Bar1m(timestamp=ts, open=o, high=h, low=l, close=c, volume=1.0, contract="TEST")


def _level(level_id, side, price, formed_at) -> LiquidityLevel:
    return LiquidityLevel(
        level_id=level_id, timeframe="1m",
        liquidity_type=(LiquidityType.SWING_HIGH if side == HIGH else LiquidityType.SWING_LOW),
        side=side, lower=price, upper=price, member_count=1, member_timestamps=(formed_at,),
        formed_at=formed_at, status=LiquidityStatus.ACTIVE, taken_at=None, absorbed_by_level_id=None,
    )


def _observed(known_at, entry_id="sig-1") -> ObservedEntry:
    candidate = FVGEntryCandidate(
        candidate_id="1m-fvg_touch-z1", source_type=SourceType.FVG_TOUCH, direction=LONG, timeframe="1m",
        zone_id="z1", lower=99.0, upper=101.0, contact_type=FVGContactType.PARTIAL_TOUCH,
        timestamp=known_at, known_at=known_at,
    )
    event = EntryEvent(
        event_id=entry_id, kind=EntryEventKind.FVG_TOUCH, selector="fvg-1m-fvg_touch", timeframe="1m",
        direction=LONG, timestamp=known_at, known_at=known_at, zone_id="z1", level_id=None, source=candidate,
    )
    return ObservedEntry(entry_id=event.event_id, known_at=event.known_at, direction=event.direction, source=event)


def _context(bars, *, tp_price, sl_price):
    context = prepare_backtest_context(bars, [], tick_size=TICK, pivot_strength=2, reclaim_window_bars=3)
    formed_at = bars[0].timestamp - timedelta(minutes=30)
    return dataclasses.replace(
        context,
        swing_equal_liquidity_results={
            "1m": LiquidityDetectionResult(
                levels=[_level("tp", HIGH, tp_price, formed_at), _level("sl", LOW, sl_price, formed_at)],
                takes=[], reclaims=[],
            )
        },
    )


FIXED_1R = next(v for v in ALL_EXIT_VARIANTS if v.variant_id == "tp=fixed_rr_1|sl=liquidity_rank_1")
FIXED_2R = next(v for v in ALL_EXIT_VARIANTS if v.variant_id == "tp=fixed_rr_2|sl=liquidity_rank_1")
LIQ_RANK_1 = next(v for v in ALL_EXIT_VARIANTS if v.variant_id == "tp=liquidity_rank_1|sl=liquidity_rank_1")


class ExecutionTests(unittest.TestCase):
    """Bars start at 09:00 New York, far from the session-close blackout, so
    these tests are about the TP family alone."""

    START = datetime(2024, 5, 6, 13, 0, tzinfo=UTC)  # 09:00 NY

    def _flat(self, count, price=100.0):
        return [_bar(self.START + timedelta(minutes=i), price, price + 0.5, price - 0.5, price) for i in range(count)]

    def test_tp_price_is_the_r_multiple_of_the_variants_own_stop(self):
        bars = self._flat(40)
        # Entry fills at 100.0; the only SL level is at 90.0, so R = 10.
        context = _context(bars, tp_price=500.0, sl_price=90.0)
        entry = _observed(bars[5].timestamp)
        for variant, expected_tp in ((FIXED_1R, 110.0), (FIXED_2R, 120.0)):
            trades, skipped = simulate_market_execution((entry,), LONG, variant, context, "v")
            self.assertEqual(skipped, [], variant.variant_id)
            self.assertEqual(len(trades), 1, variant.variant_id)
            self.assertEqual(trades[0].tp_price, expected_tp, variant.variant_id)
            self.assertEqual(trades[0].sl_price, 90.0, variant.variant_id)

    def test_1r_is_hit_where_the_liquidity_target_would_not_be(self):
        bars = self._flat(40)
        hit_index = 20
        bars[hit_index] = _bar(bars[hit_index].timestamp, 100.0, 112.0, 99.5, 100.0)
        # The liquidity TP sits at 200 -- unreachable. The 1R target is 110.
        context = _context(bars, tp_price=200.0, sl_price=90.0)
        entry = _observed(bars[5].timestamp)

        fixed_trades, _ = simulate_market_execution((entry,), LONG, FIXED_1R, context, "v")
        self.assertEqual(fixed_trades[0].exit_reason, TradeExitReason.TP_HIT)
        self.assertEqual(fixed_trades[0].exit_price, 110.0)
        self.assertEqual(fixed_trades[0].exit_timestamp, bars[hit_index].timestamp)

        liquidity_trades, _ = simulate_market_execution((entry,), LONG, LIQ_RANK_1, context, "v")
        self.assertEqual(liquidity_trades[0].exit_reason, TradeExitReason.OPEN_UNRESOLVED)

    def test_same_bar_tp_and_sl_still_resolves_sl_first(self):
        bars = self._flat(40)
        both_index = 20
        # One bar spanning both the 1R target (110) and the stop (90), with an
        # open that does not gap through the stop.
        bars[both_index] = _bar(bars[both_index].timestamp, 100.0, 115.0, 85.0, 100.0)
        context = _context(bars, tp_price=500.0, sl_price=90.0)
        entry = _observed(bars[5].timestamp)
        trades, _ = simulate_market_execution((entry,), LONG, FIXED_1R, context, "v")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_reason, TradeExitReason.SL_HIT)
        self.assertEqual(trades[0].exit_price, 90.0)
        self.assertEqual(trades[0].exit_timestamp, bars[both_index].timestamp)
        # The bar genuinely spans both levels, so this is a real ambiguity the
        # SL-first rule resolved, not a bar that only touched the stop.
        self.assertGreaterEqual(bars[both_index].high, 110.0)
        self.assertLessEqual(bars[both_index].low, 90.0)
        self.assertEqual(
            _ambiguous_same_bar_exit_count(trades, tuple(bars), BarRangeIndex.build(tuple(bars))), 1
        )

    def test_missing_sl_is_reported_as_missing_sl_never_as_missing_tp(self):
        """For this family the TP cannot exist without the stop, so the skip
        must name the thing that was actually absent."""
        bars = self._flat(20)
        context = _context(bars, tp_price=500.0, sl_price=90.0)
        context = dataclasses.replace(
            context,
            swing_equal_liquidity_results={
                "1m": LiquidityDetectionResult(
                    levels=[_level("tp", HIGH, 500.0, bars[0].timestamp - timedelta(minutes=30))],
                    takes=[], reclaims=[],
                )
            },
        )
        trades, skipped = simulate_market_execution((_observed(bars[5].timestamp),), LONG, FIXED_1R, context, "v")
        self.assertEqual(trades, [])
        self.assertEqual([s.reason for s in skipped], [ExecutionSkipReason.MISSING_SL_TARGET])

    def test_all_three_execution_paths_agree_for_a_fixed_rr_variant(self):
        bars = self._flat(60)
        bars[30] = _bar(bars[30].timestamp, 100.0, 125.0, 99.0, 100.0)
        context = _context(bars, tp_price=500.0, sl_price=90.0)
        entries = (_observed(bars[5].timestamp, "s1"), _observed(bars[40].timestamp, "s2"))
        for variant in (FIXED_1R, FIXED_2R):
            accelerated, _ = simulate_market_execution(entries, LONG, variant, context, "v")
            reference, _ = _simulate_market_execution_reference(entries, LONG, variant, context, "v")
            batched, _ = simulate_market_execution_batched(entries, LONG, (variant,), context, "cls")[variant.variant_id]
            key = lambda ts: [(t.entry_timestamp, t.exit_timestamp, t.exit_reason, t.tp_price, t.sl_price) for t in ts]
            self.assertEqual(key(accelerated), key(reference), variant.variant_id)
            self.assertEqual(key(accelerated), key(batched), variant.variant_id)


class UnchangedVariantEquivalenceTests(unittest.TestCase):
    """Adding the fixed-RR family must not perturb a single pre-existing
    variant. The resolution ORDER changed for fixed RR only; everything else
    must produce byte-identical trades."""

    START = datetime(2024, 5, 6, 13, 0, tzinfo=UTC)  # 09:00 NY

    def test_every_pre_existing_variant_is_unchanged_by_the_new_family(self):
        bars = [
            _bar(self.START + timedelta(minutes=i), 100.0 + (i % 7) * 0.25, 100.0 + (i % 7) * 0.25 + 3.0,
                 100.0 + (i % 7) * 0.25 - 3.0, 100.0 + (i % 7) * 0.25)
            for i in range(120)
        ]
        context = _context(bars, tp_price=104.0, sl_price=96.0)
        entries = tuple(_observed(bars[i].timestamp, f"s{i}") for i in (5, 30, 60, 90))
        original = [v for v in ALL_EXIT_VARIANTS if v.tp_kind != TPVariantKind.FIXED_RR]
        self.assertEqual(len(original), 20)

        # Run each original variant alone, and then batched together WITH the
        # ten new ones. Both must give the original its exact same trades.
        batched_all = simulate_market_execution_batched(entries, LONG, ALL_EXIT_VARIANTS, context, "cls")
        for variant in original:
            alone, alone_skipped = simulate_market_execution(entries, LONG, variant, context, "v")
            together, together_skipped = batched_all[variant.variant_id]
            key = lambda ts: [
                (t.entry_timestamp, t.entry_price, t.exit_timestamp, t.exit_price, t.exit_reason, t.tp_price, t.sl_price)
                for t in ts
            ]
            self.assertEqual(key(alone), key(together), variant.variant_id)
            self.assertEqual([s.reason for s in alone_skipped], [s.reason for s in together_skipped], variant.variant_id)
