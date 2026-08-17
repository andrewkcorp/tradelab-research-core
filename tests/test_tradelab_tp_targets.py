#!/usr/bin/env python3
"""TP target candidate tests: long/short symmetry, wrong-side/at-entry
rejection, top-3 nearest liquidity targets with deterministic sorting,
swing/equal/session/data-wick sources, taken-before-vs-after-entry
filtering, source known_at after entry rejection, data wick as TP-only,
no lookahead, DST/timezone, malformed-input validation, and duplicate
targets."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.data_wicks import DataWickDetectionResult, DataWickDirection, DataWickEvent
from tradelab.liquidity import HIGH, LOW, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType
from tradelab.tp_targets import (
    LONG,
    SHORT,
    LiquiditySourceFamily,
    MarketEntryContext,
    TPSourceType,
    detect_tp_target_candidates,
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def entry(
    entry_id: str = "e1",
    direction: str = LONG,
    entry_price: float = 100.0,
    entry_known_at: datetime = T0,
) -> MarketEntryContext:
    return MarketEntryContext(entry_id=entry_id, direction=direction, entry_price=entry_price, entry_known_at=entry_known_at)


def level(
    level_id: str = "1m-swing_high-1",
    timeframe: str = "1m",
    liquidity_type: LiquidityType = LiquidityType.SWING_HIGH,
    side: str = HIGH,
    price: float = 105.0,
    status: LiquidityStatus = LiquidityStatus.ACTIVE,
    formed_at: datetime = T0 - timedelta(hours=1),
    taken_at: datetime | None = None,
) -> LiquidityLevel:
    return LiquidityLevel(
        level_id=level_id,
        timeframe=timeframe,
        liquidity_type=liquidity_type,
        side=side,
        lower=price,
        upper=price,
        member_count=1,
        member_timestamps=(formed_at - timedelta(minutes=1),),
        formed_at=formed_at,
        status=status,
        taken_at=taken_at,
        absorbed_by_level_id=None,
    )


def liq_result(*levels: LiquidityLevel) -> LiquidityDetectionResult:
    return LiquidityDetectionResult(levels=list(levels), takes=[], reclaims=[])


def wick_event(
    direction: DataWickDirection = DataWickDirection.UPPER,
    level_price: float = 106.0,
    known_at: datetime = T0 - timedelta(minutes=30),
    event_ids: tuple[str, ...] = ("EVT-1",),
) -> DataWickEvent:
    return DataWickEvent(
        event_ids=event_ids,
        direction=direction,
        timestamp=known_at - timedelta(minutes=1),
        known_at=known_at,
        level=level_price,
        candle_range=10.0,
        baseline_median=2.0,
        wick_length=6.0,
    )


def wick_result(*events: DataWickEvent) -> DataWickDetectionResult:
    return DataWickDetectionResult(events=list(events))


EMPTY_LIQ: dict = {}
EMPTY_WICKS = wick_result()


class LongShortSymmetryTests(unittest.TestCase):
    def test_long_liquidity_tp_is_a_high_above_entry(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=HIGH, price=105.0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(len(result.liquidity_targets), 1)
        self.assertEqual(result.liquidity_targets[0].price, 105.0)

    def test_short_liquidity_tp_is_a_low_below_entry(self):
        e = entry(direction=SHORT, entry_price=100.0)
        lvl = level(level_id="1m-swing_low-1", liquidity_type=LiquidityType.SWING_LOW, side=LOW, price=95.0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(len(result.liquidity_targets), 1)
        self.assertEqual(result.liquidity_targets[0].price, 95.0)

    def test_long_data_wick_tp_is_upper_above_entry(self):
        e = entry(direction=LONG, entry_price=100.0)
        w = wick_event(direction=DataWickDirection.UPPER, level_price=106.0)
        result = detect_tp_target_candidates(e, EMPTY_LIQ, EMPTY_LIQ, wick_result(w))
        self.assertEqual(len(result.data_wick_targets), 1)
        self.assertEqual(result.data_wick_targets[0].price, 106.0)

    def test_short_data_wick_tp_is_lower_below_entry(self):
        e = entry(direction=SHORT, entry_price=100.0)
        w = wick_event(direction=DataWickDirection.LOWER, level_price=94.0)
        result = detect_tp_target_candidates(e, EMPTY_LIQ, EMPTY_LIQ, wick_result(w))
        self.assertEqual(len(result.data_wick_targets), 1)
        self.assertEqual(result.data_wick_targets[0].price, 94.0)


class WrongSideAndAtEntryTests(unittest.TestCase):
    def test_long_rejects_liquidity_low_entirely(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(level_id="1m-swing_low-1", liquidity_type=LiquidityType.SWING_LOW, side=LOW, price=95.0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(result.liquidity_targets, [])

    def test_long_rejects_high_below_entry_price(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=HIGH, price=95.0)  # below entry despite being a "high" type
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(result.liquidity_targets, [])

    def test_long_rejects_high_exactly_at_entry_price(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=HIGH, price=100.0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(result.liquidity_targets, [])

    def test_short_rejects_low_exactly_at_entry_price(self):
        e = entry(direction=SHORT, entry_price=100.0)
        lvl = level(level_id="1m-swing_low-1", liquidity_type=LiquidityType.SWING_LOW, side=LOW, price=100.0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(result.liquidity_targets, [])

    def test_long_rejects_lower_data_wick_entirely(self):
        e = entry(direction=LONG, entry_price=100.0)
        w = wick_event(direction=DataWickDirection.LOWER, level_price=94.0)
        result = detect_tp_target_candidates(e, EMPTY_LIQ, EMPTY_LIQ, wick_result(w))
        self.assertEqual(result.data_wick_targets, [])

    def test_data_wick_exactly_at_entry_price_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        w = wick_event(direction=DataWickDirection.UPPER, level_price=100.0)
        result = detect_tp_target_candidates(e, EMPTY_LIQ, EMPTY_LIQ, wick_result(w))
        self.assertEqual(result.data_wick_targets, [])


class TopThreeNearestTests(unittest.TestCase):
    def test_only_three_nearest_liquidity_targets_are_returned_ranked(self):
        e = entry(direction=LONG, entry_price=100.0)
        levels = [
            level(level_id="L1", price=110.0),
            level(level_id="L2", price=101.0),
            level(level_id="L3", price=105.0),
            level(level_id="L4", price=103.0),
        ]
        result = detect_tp_target_candidates(e, {"1m": liq_result(*levels)}, {}, EMPTY_WICKS)
        self.assertEqual(len(result.liquidity_targets), 3)
        self.assertEqual([t.level_id for t in result.liquidity_targets], ["L2", "L4", "L3"])
        self.assertEqual([t.rank for t in result.liquidity_targets], [1, 2, 3])

    def test_all_qualifying_data_wick_targets_are_returned_uncapped(self):
        e = entry(direction=LONG, entry_price=100.0)
        events = [
            wick_event(level_price=101.0, event_ids=("E1",), known_at=T0 - timedelta(minutes=25)),
            wick_event(level_price=110.0, event_ids=("E2",), known_at=T0 - timedelta(minutes=24)),
            wick_event(level_price=105.0, event_ids=("E3",), known_at=T0 - timedelta(minutes=23)),
            wick_event(level_price=103.0, event_ids=("E4",), known_at=T0 - timedelta(minutes=22)),
            wick_event(level_price=120.0, event_ids=("E5",), known_at=T0 - timedelta(minutes=21)),
        ]
        result = detect_tp_target_candidates(e, EMPTY_LIQ, EMPTY_LIQ, wick_result(*events))
        self.assertEqual(len(result.data_wick_targets), 5)
        prices = [t.price for t in result.data_wick_targets]
        self.assertEqual(prices, sorted(prices))
        for t in result.data_wick_targets:
            self.assertIsNone(t.rank)

    def test_deterministic_sort_order_on_repeated_calls(self):
        e = entry(direction=LONG, entry_price=100.0)
        levels = [level(level_id="L1", price=110.0), level(level_id="L2", price=101.0), level(level_id="L3", price=105.0)]
        result_1 = detect_tp_target_candidates(e, {"1m": liq_result(*levels)}, {}, EMPTY_WICKS)
        result_2 = detect_tp_target_candidates(e, {"1m": liq_result(*levels)}, {}, EMPTY_WICKS)
        self.assertEqual([t.level_id for t in result_1.liquidity_targets], [t.level_id for t in result_2.liquidity_targets])


class SourceFamilyTests(unittest.TestCase):
    def test_swing_and_equal_and_session_all_produce_targets(self):
        e = entry(direction=LONG, entry_price=100.0)
        swing = level(level_id="SW", liquidity_type=LiquidityType.SWING_HIGH, side=HIGH, price=101.0)
        equal = level(level_id="EQ", liquidity_type=LiquidityType.EQUAL_HIGHS, side=HIGH, price=102.0)
        session = level(level_id="SESS", liquidity_type=LiquidityType.SESSION_LONDON_HIGH, side=HIGH, price=103.0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(swing, equal)}, {"1m": liq_result(session)}, EMPTY_WICKS)
        families = {t.level_id: t.source_family for t in result.liquidity_targets}
        self.assertEqual(families["SW"], LiquiditySourceFamily.SWING)
        self.assertEqual(families["EQ"], LiquiditySourceFamily.EQUAL)
        self.assertEqual(families["SESS"], LiquiditySourceFamily.SESSION)

    def test_absorbed_level_never_produces_its_own_target(self):
        e = entry(direction=LONG, entry_price=100.0)
        absorbed = level(side=HIGH, price=110.0, status=LiquidityStatus.ABSORBED_IN_EQUAL_CLUSTER)
        result = detect_tp_target_candidates(e, {"1m": liq_result(absorbed)}, {}, EMPTY_WICKS)
        self.assertEqual(result.liquidity_targets, [])

    def test_session_level_filed_under_swing_equal_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        session_type_level = level(liquidity_type=LiquidityType.SESSION_LONDON_HIGH, side=HIGH, price=101.0)
        with self.assertRaises(ValueError):
            detect_tp_target_candidates(e, {"1m": liq_result(session_type_level)}, {}, EMPTY_WICKS)

    def test_swing_level_filed_under_session_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        swing_level = level(liquidity_type=LiquidityType.SWING_HIGH, side=HIGH, price=101.0)
        with self.assertRaises(ValueError):
            detect_tp_target_candidates(e, {}, {"1m": liq_result(swing_level)}, EMPTY_WICKS)


class TakenBeforeVsAfterEntryTests(unittest.TestCase):
    def test_level_taken_before_entry_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=HIGH, price=105.0, status=LiquidityStatus.TAKEN, formed_at=T0 - timedelta(hours=2), taken_at=T0 - timedelta(minutes=1))
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(result.liquidity_targets, [])

    def test_level_taken_exactly_at_entry_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=HIGH, price=105.0, status=LiquidityStatus.TAKEN, formed_at=T0 - timedelta(hours=2), taken_at=T0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(result.liquidity_targets, [])

    def test_level_taken_after_entry_is_accepted(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=HIGH, price=105.0, status=LiquidityStatus.TAKEN, formed_at=T0 - timedelta(hours=2), taken_at=T0 + timedelta(minutes=1))
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(len(result.liquidity_targets), 1)

    def test_active_never_taken_level_is_accepted(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=HIGH, price=105.0, status=LiquidityStatus.ACTIVE, formed_at=T0 - timedelta(hours=2))
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(len(result.liquidity_targets), 1)


class SourceKnownAtAfterEntryTests(unittest.TestCase):
    def test_level_formed_after_entry_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=HIGH, price=105.0, formed_at=T0 + timedelta(minutes=1))
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(result.liquidity_targets, [])

    def test_level_formed_exactly_at_entry_is_accepted(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=HIGH, price=105.0, formed_at=T0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(len(result.liquidity_targets), 1)

    def test_data_wick_known_after_entry_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        w = wick_event(level_price=106.0, known_at=T0 + timedelta(minutes=1))
        result = detect_tp_target_candidates(e, EMPTY_LIQ, EMPTY_LIQ, wick_result(w))
        self.assertEqual(result.data_wick_targets, [])

    def test_data_wick_known_exactly_at_entry_is_accepted(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        w = wick_event(level_price=106.0, known_at=T0)
        result = detect_tp_target_candidates(e, EMPTY_LIQ, EMPTY_LIQ, wick_result(w))
        self.assertEqual(len(result.data_wick_targets), 1)


class DataWickIsTPOnlyTests(unittest.TestCase):
    def test_data_wick_source_type_is_labeled_correctly(self):
        e = entry(direction=LONG, entry_price=100.0)
        w = wick_event(level_price=106.0)
        result = detect_tp_target_candidates(e, EMPTY_LIQ, EMPTY_LIQ, wick_result(w))
        t = result.data_wick_targets[0]
        self.assertEqual(t.source_type, TPSourceType.DATA_WICK)
        self.assertIsNone(t.liquidity_type)
        self.assertIsNone(t.level_id)
        self.assertEqual(t.event_ids, ("EVT-1",))
        self.assertEqual(t.wick_direction, DataWickDirection.UPPER)

    def test_liquidity_target_has_no_data_wick_fields(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=HIGH, price=105.0)
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        t = result.liquidity_targets[0]
        self.assertEqual(t.source_type, TPSourceType.LIQUIDITY)
        self.assertIsNone(t.event_ids)
        self.assertIsNone(t.wick_direction)


class NoLookaheadTests(unittest.TestCase):
    def test_a_later_liquidity_level_never_leaks_into_an_earlier_entrys_targets(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        early = level(level_id="EARLY", side=HIGH, price=105.0, formed_at=T0 - timedelta(minutes=30))
        later = level(level_id="LATER", side=HIGH, price=101.0, formed_at=T0 + timedelta(minutes=30))
        result = detect_tp_target_candidates(e, {"1m": liq_result(early, later)}, {}, EMPTY_WICKS)
        self.assertEqual([t.level_id for t in result.liquidity_targets], ["EARLY"])


class DSTAndTimezoneTests(unittest.TestCase):
    def test_entry_known_at_normalized_to_utc_from_ny_local(self):
        # 2026-03-08: spring forward. entry at 10:00 NY (EDT after transition).
        ny_time = datetime(2026, 3, 8, 10, 0, tzinfo=NY)
        e = entry(entry_known_at=ny_time)
        self.assertEqual(e.entry_known_at, ny_time.astimezone(UTC))
        self.assertEqual(e.entry_known_at.tzinfo, UTC)

    def test_level_formed_just_before_entry_across_dst_transition_is_accepted(self):
        entry_ny = datetime(2026, 3, 8, 10, 0, tzinfo=NY)
        formed_ny = datetime(2026, 3, 8, 1, 0, tzinfo=NY)  # before the 02:00->03:00 jump, same calendar window
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=entry_ny.astimezone(UTC))
        lvl = level(side=HIGH, price=105.0, formed_at=formed_ny.astimezone(UTC))
        result = detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(len(result.liquidity_targets), 1)

    def test_naive_entry_known_at_is_rejected(self):
        with self.assertRaises(ValueError):
            MarketEntryContext(entry_id="e1", direction=LONG, entry_price=100.0, entry_known_at=datetime(2026, 1, 5, 10, 0))


class ValidationTests(unittest.TestCase):
    def test_empty_entry_id_is_rejected(self):
        with self.assertRaises(ValueError):
            MarketEntryContext(entry_id="", direction=LONG, entry_price=100.0, entry_known_at=T0)

    def test_invalid_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            MarketEntryContext(entry_id="e1", direction="up", entry_price=100.0, entry_known_at=T0)

    def test_non_positive_entry_price_is_rejected(self):
        for price in (0.0, -5.0):
            with self.subTest(price=price):
                with self.assertRaises(ValueError):
                    MarketEntryContext(entry_id="e1", direction=LONG, entry_price=price, entry_known_at=T0)

    def test_level_timeframe_mismatch_with_dict_key_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=HIGH, price=105.0, timeframe="5m")
        with self.assertRaises(ValueError):
            detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)

    def test_unsupported_swing_equal_timeframe_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=HIGH, price=105.0, timeframe="3m")
        with self.assertRaises(ValueError):
            detect_tp_target_candidates(e, {"3m": liq_result(lvl)}, {}, EMPTY_WICKS)

    def test_unsupported_session_timeframe_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(liquidity_type=LiquidityType.SESSION_LONDON_HIGH, side=HIGH, price=105.0, timeframe="1H")
        with self.assertRaises(ValueError):
            detect_tp_target_candidates(e, {}, {"1H": liq_result(lvl)}, EMPTY_WICKS)

    def test_taken_level_with_taken_at_not_after_formed_at_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=HIGH, price=105.0, status=LiquidityStatus.TAKEN, formed_at=T0, taken_at=T0)  # equal, not strictly after
        with self.assertRaises(ValueError):
            detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)

    def test_taken_level_with_no_taken_at_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=HIGH, price=105.0, status=LiquidityStatus.TAKEN, taken_at=None)
        with self.assertRaises(ValueError):
            detect_tp_target_candidates(e, {"1m": liq_result(lvl)}, {}, EMPTY_WICKS)

    def test_duplicate_level_id_within_a_source_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl1 = level(level_id="L1", side=HIGH, price=105.0)
        lvl2 = level(level_id="L1", side=HIGH, price=106.0)
        with self.assertRaises(ValueError):
            detect_tp_target_candidates(e, {"1m": liq_result(lvl1, lvl2)}, {}, EMPTY_WICKS)


class DuplicateTargetTests(unittest.TestCase):
    def test_far_htf_swing_liquidity_target_is_allowed(self):
        # TP is not restricted to setup timeframes -- an HTF level is a
        # perfectly valid target, unlike an entry trigger.
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(level_id="4H-swing_high-1", side=HIGH, price=105.0, timeframe="4H")
        result = detect_tp_target_candidates(e, {"4H": liq_result(lvl)}, {}, EMPTY_WICKS)
        self.assertEqual(len(result.liquidity_targets), 1)

    def test_result_target_ids_are_unique(self):
        e = entry(direction=LONG, entry_price=100.0)
        levels = [level(level_id=f"L{i}", side=HIGH, price=101.0 + i) for i in range(3)]
        events = [
            wick_event(level_price=110.0 + i, event_ids=(f"E{i}",), known_at=T0 - timedelta(minutes=30 - i))
            for i in range(3)
        ]
        result = detect_tp_target_candidates(e, {"1m": liq_result(*levels)}, {}, wick_result(*events))
        all_ids = [t.target_id for t in result.liquidity_targets] + [t.target_id for t in result.data_wick_targets]
        self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
    unittest.main()
