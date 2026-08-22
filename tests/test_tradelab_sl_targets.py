#!/usr/bin/env python3
"""SL target candidate tests: long/short symmetry, top-3 nearest
liquidity SL, swing/equal/session sources, FVG boundary restricted to
5m/15m and only the linked confluence, wrong-side/at-entry rejection,
taken-before-vs-after-entry, known_at/no-lookahead, malformed FVG
links/timeframes/duplicates, and data wicks never appearing in SL."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.fvg_entry import FVGEntryCandidate
from tradelab.fvg_entry import SourceType as FVGSourceType
from tradelab.liquidity import HIGH, LOW, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType
from tradelab.sl_targets import (
    LONG,
    SHORT,
    SLSourceType,
    SelectedFVGConfluence,
    detect_sl_target_candidates,
)
from tradelab.tp_targets import LiquiditySourceFamily, MarketEntryContext

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
    level_id: str = "1m-swing_low-1",
    timeframe: str = "1m",
    liquidity_type: LiquidityType = LiquidityType.SWING_LOW,
    side: str = LOW,
    price: float = 95.0,
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


def fvg_candidate(
    zone_id: str = "5m-FVG-1",
    timeframe: str = "5m",
    source_type: FVGSourceType = FVGSourceType.FVG_TOUCH,
    direction: str = LONG,
    lower: float = 95.0,
    upper: float = 97.0,
    known_at: datetime = T0 - timedelta(minutes=30),
) -> FVGEntryCandidate:
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{source_type.value}-{zone_id}",
        source_type=source_type,
        direction=direction,
        timeframe=timeframe,
        zone_id=zone_id,
        lower=lower,
        upper=upper,
        contact_type=None,
        timestamp=known_at - timedelta(minutes=1),
        known_at=known_at,
    )


def selected(entry_id: str = "e1", *candidates: FVGEntryCandidate) -> SelectedFVGConfluence:
    return SelectedFVGConfluence(entry_id=entry_id, fvg_candidates=tuple(candidates))


EMPTY_LIQ: dict = {}
NO_FVG = selected()


class LongShortSymmetryTests(unittest.TestCase):
    def test_long_liquidity_sl_is_a_low_below_entry(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=LOW, price=95.0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(len(result.liquidity_targets), 1)
        self.assertEqual(result.liquidity_targets[0].price, 95.0)

    def test_short_liquidity_sl_is_a_high_above_entry(self):
        e = entry(direction=SHORT, entry_price=100.0)
        lvl = level(level_id="1m-swing_high-1", liquidity_type=LiquidityType.SWING_HIGH, side=HIGH, price=105.0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(len(result.liquidity_targets), 1)
        self.assertEqual(result.liquidity_targets[0].price, 105.0)

    def test_long_fvg_sl_uses_the_zones_lower_bound(self):
        e = entry(direction=LONG, entry_price=100.0)
        c = fvg_candidate(direction=LONG, lower=95.0, upper=97.0)
        result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
        self.assertEqual(len(result.fvg_targets), 1)
        self.assertEqual(result.fvg_targets[0].price, 95.0)

    def test_short_fvg_sl_uses_the_zones_upper_bound(self):
        e = entry(direction=SHORT, entry_price=100.0)
        c = fvg_candidate(direction=SHORT, lower=103.0, upper=105.0)
        result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
        self.assertEqual(len(result.fvg_targets), 1)
        self.assertEqual(result.fvg_targets[0].price, 105.0)


class TopThreeNearestLiquidityTests(unittest.TestCase):
    def test_only_three_nearest_liquidity_sl_are_returned_ranked(self):
        e = entry(direction=LONG, entry_price=100.0)
        levels = [
            level(level_id="L1", price=80.0),
            level(level_id="L2", price=99.0),
            level(level_id="L3", price=90.0),
            level(level_id="L4", price=95.0),
        ]
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(*levels)}, {})
        self.assertEqual(len(result.liquidity_targets), 3)
        self.assertEqual([t.level_id for t in result.liquidity_targets], ["L2", "L4", "L3"])
        self.assertEqual([t.rank for t in result.liquidity_targets], [1, 2, 3])

    def test_deterministic_sort_order_on_repeated_calls(self):
        e = entry(direction=LONG, entry_price=100.0)
        levels = [level(level_id="L1", price=80.0), level(level_id="L2", price=99.0), level(level_id="L3", price=90.0)]
        r1 = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(*levels)}, {})
        r2 = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(*levels)}, {})
        self.assertEqual([t.level_id for t in r1.liquidity_targets], [t.level_id for t in r2.liquidity_targets])


class SourceFamilyTests(unittest.TestCase):
    def test_swing_and_equal_and_session_all_produce_targets(self):
        e = entry(direction=LONG, entry_price=100.0)
        swing = level(level_id="SW", liquidity_type=LiquidityType.SWING_LOW, side=LOW, price=95.0)
        equal = level(level_id="EQ", liquidity_type=LiquidityType.EQUAL_LOWS, side=LOW, price=90.0)
        session = level(level_id="SESS", liquidity_type=LiquidityType.SESSION_LONDON_LOW, side=LOW, price=85.0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(swing, equal)}, {"1m": liq_result(session)})
        families = {t.level_id: t.source_family for t in result.liquidity_targets}
        self.assertEqual(families["SW"], LiquiditySourceFamily.SWING)
        self.assertEqual(families["EQ"], LiquiditySourceFamily.EQUAL)
        self.assertEqual(families["SESS"], LiquiditySourceFamily.SESSION)

    def test_absorbed_level_never_produces_its_own_target(self):
        e = entry(direction=LONG, entry_price=100.0)
        absorbed = level(side=LOW, price=90.0, status=LiquidityStatus.ABSORBED_IN_EQUAL_CLUSTER)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(absorbed)}, {})
        self.assertEqual(result.liquidity_targets, [])

    def test_session_level_filed_under_swing_equal_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(liquidity_type=LiquidityType.SESSION_LONDON_LOW, side=LOW, price=95.0)
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})

    def test_swing_level_filed_under_session_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(liquidity_type=LiquidityType.SWING_LOW, side=LOW, price=95.0)
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, NO_FVG, {}, {"1m": liq_result(lvl)})


class FVGBoundaryTimeframeAndLinkTests(unittest.TestCase):
    def test_5m_and_15m_fvg_qualify(self):
        for timeframe in ("5m", "15m"):
            with self.subTest(timeframe=timeframe):
                e = entry(direction=LONG, entry_price=100.0)
                c = fvg_candidate(zone_id=f"{timeframe}-FVG-1", timeframe=timeframe, lower=95.0, upper=97.0)
                result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
                self.assertEqual(len(result.fvg_targets), 1)

    def test_1m_fvg_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        c = fvg_candidate(zone_id="1m-FVG-1", timeframe="1m", lower=95.0, upper=97.0)
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})

    def test_1h_4h_1d_fvg_are_rejected(self):
        for timeframe in ("1H", "4H", "1D"):
            with self.subTest(timeframe=timeframe):
                e = entry(direction=LONG, entry_price=100.0)
                c = fvg_candidate(zone_id=f"{timeframe}-FVG-1", timeframe=timeframe, lower=95.0, upper=97.0)
                with self.assertRaises(ValueError):
                    detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})

    def test_only_the_linked_zone_produces_an_fvg_sl_not_others(self):
        e = entry(direction=LONG, entry_price=100.0)
        linked = fvg_candidate(zone_id="5m-FVG-LINKED", lower=95.0, upper=97.0)
        result = detect_sl_target_candidates(e, selected("e1", linked), EMPTY_LIQ, {})
        self.assertEqual(len(result.fvg_targets), 1)
        self.assertEqual(result.fvg_targets[0].zone_id, "5m-FVG-LINKED")
        # an unlinked candidate, never passed to SelectedFVGConfluence, cannot appear
        unrelated_zone_ids = {t.zone_id for t in result.fvg_targets}
        self.assertNotIn("5m-FVG-RANDOM-OLD", unrelated_zone_ids)

    def test_multiple_linked_zones_each_produce_their_own_target(self):
        e = entry(direction=LONG, entry_price=100.0)
        c1 = fvg_candidate(zone_id="5m-FVG-1", timeframe="5m", lower=95.0, upper=97.0)
        c2 = fvg_candidate(zone_id="15m-FVG-1", timeframe="15m", lower=90.0, upper=92.0)
        result = detect_sl_target_candidates(e, selected("e1", c1, c2), EMPTY_LIQ, {})
        self.assertEqual(len(result.fvg_targets), 2)
        self.assertEqual({t.zone_id for t in result.fvg_targets}, {"5m-FVG-1", "15m-FVG-1"})

    def test_limited_fvg_targets_keeps_the_same_nearest_stop(self):
        e = entry(direction=LONG, entry_price=100.0)
        candidates = tuple(
            fvg_candidate(zone_id=f"5m-FVG-{index}", lower=99.0 - index, upper=100.0 - index)
            for index in range(20)
        )
        full = detect_sl_target_candidates(e, selected("e1", *candidates), EMPTY_LIQ, {})
        limited = detect_sl_target_candidates(
            e, selected("e1", *candidates), EMPTY_LIQ, {}, max_fvg_targets=1,
        )
        self.assertEqual(len(limited.fvg_targets), 1)
        self.assertEqual(limited.fvg_targets[0], full.fvg_targets[0])

    def test_ifvg_confluence_boundary_also_qualifies(self):
        e = entry(direction=LONG, entry_price=100.0)
        c = fvg_candidate(source_type=FVGSourceType.IFVG, direction=LONG, lower=95.0, upper=97.0)
        result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
        self.assertEqual(len(result.fvg_targets), 1)
        self.assertEqual(result.fvg_targets[0].fvg_source_type, FVGSourceType.IFVG)

    def test_entry_id_mismatch_between_entry_and_selected_confluence_is_rejected(self):
        e = entry(entry_id="e1", direction=LONG, entry_price=100.0)
        c = fvg_candidate(lower=95.0, upper=97.0)
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, selected("e2", c), EMPTY_LIQ, {})

    def test_duplicate_zone_id_in_selected_confluence_is_rejected(self):
        c1 = fvg_candidate(zone_id="5m-FVG-1", source_type=FVGSourceType.FVG_TOUCH)
        c2 = fvg_candidate(zone_id="5m-FVG-1", source_type=FVGSourceType.IFVG)
        with self.assertRaises(ValueError):
            selected("e1", c1, c2)

    def test_no_fvg_confluence_yields_empty_fvg_targets(self):
        e = entry(direction=LONG, entry_price=100.0)
        result = detect_sl_target_candidates(e, NO_FVG, EMPTY_LIQ, {})
        self.assertEqual(result.fvg_targets, [])


class WrongSideAndAtEntryTests(unittest.TestCase):
    def test_long_rejects_liquidity_high_entirely(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(level_id="1m-swing_high-1", liquidity_type=LiquidityType.SWING_HIGH, side=HIGH, price=105.0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(result.liquidity_targets, [])

    def test_long_rejects_low_above_entry_price(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=LOW, price=105.0)  # above entry despite being a "low" type
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(result.liquidity_targets, [])

    def test_long_rejects_low_exactly_at_entry_price(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=LOW, price=100.0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(result.liquidity_targets, [])

    def test_short_rejects_high_exactly_at_entry_price(self):
        e = entry(direction=SHORT, entry_price=100.0)
        lvl = level(level_id="1m-swing_high-1", liquidity_type=LiquidityType.SWING_HIGH, side=HIGH, price=100.0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(result.liquidity_targets, [])

    def test_long_fvg_boundary_above_entry_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        c = fvg_candidate(lower=101.0, upper=103.0)  # lower still above entry_price
        result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
        self.assertEqual(result.fvg_targets, [])

    def test_fvg_boundary_exactly_at_entry_price_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        c = fvg_candidate(lower=100.0, upper=103.0)
        result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
        self.assertEqual(result.fvg_targets, [])


class TakenBeforeVsAfterEntryTests(unittest.TestCase):
    def test_level_taken_before_entry_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=LOW, price=95.0, status=LiquidityStatus.TAKEN, formed_at=T0 - timedelta(hours=2), taken_at=T0 - timedelta(minutes=1))
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(result.liquidity_targets, [])

    def test_level_taken_exactly_at_entry_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=LOW, price=95.0, status=LiquidityStatus.TAKEN, formed_at=T0 - timedelta(hours=2), taken_at=T0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(result.liquidity_targets, [])

    def test_level_taken_after_entry_is_accepted(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=LOW, price=95.0, status=LiquidityStatus.TAKEN, formed_at=T0 - timedelta(hours=2), taken_at=T0 + timedelta(minutes=1))
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(len(result.liquidity_targets), 1)

    def test_active_never_taken_level_is_accepted(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=LOW, price=95.0, status=LiquidityStatus.ACTIVE, formed_at=T0 - timedelta(hours=2))
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(len(result.liquidity_targets), 1)


class KnownAtNoLookaheadTests(unittest.TestCase):
    def test_level_formed_after_entry_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=LOW, price=95.0, formed_at=T0 + timedelta(minutes=1))
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(result.liquidity_targets, [])

    def test_level_formed_exactly_at_entry_is_accepted(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        lvl = level(side=LOW, price=95.0, formed_at=T0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        self.assertEqual(len(result.liquidity_targets), 1)

    def test_fvg_candidate_known_after_entry_is_excluded(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        c = fvg_candidate(lower=95.0, upper=97.0, known_at=T0 + timedelta(minutes=1))
        result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
        self.assertEqual(result.fvg_targets, [])

    def test_fvg_candidate_known_exactly_at_entry_is_accepted(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        c = fvg_candidate(lower=95.0, upper=97.0, known_at=T0)
        result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
        self.assertEqual(len(result.fvg_targets), 1)

    def test_a_later_liquidity_level_never_leaks_into_an_earlier_entrys_targets(self):
        e = entry(direction=LONG, entry_price=100.0, entry_known_at=T0)
        early = level(level_id="EARLY", side=LOW, price=90.0, formed_at=T0 - timedelta(minutes=30))
        later = level(level_id="LATER", side=LOW, price=99.0, formed_at=T0 + timedelta(minutes=30))
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(early, later)}, {})
        self.assertEqual([t.level_id for t in result.liquidity_targets], ["EARLY"])


class MalformedInputTests(unittest.TestCase):
    def test_level_timeframe_mismatch_with_dict_key_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=LOW, price=95.0, timeframe="5m")
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})

    def test_unsupported_swing_equal_timeframe_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=LOW, price=95.0, timeframe="3m")
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, NO_FVG, {"3m": liq_result(lvl)}, {})

    def test_unsupported_session_timeframe_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(liquidity_type=LiquidityType.SESSION_LONDON_LOW, side=LOW, price=95.0, timeframe="1H")
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, NO_FVG, {}, {"1H": liq_result(lvl)})

    def test_taken_level_with_taken_at_not_after_formed_at_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=LOW, price=95.0, status=LiquidityStatus.TAKEN, formed_at=T0, taken_at=T0)
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})

    def test_duplicate_level_id_within_a_source_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl1 = level(level_id="L1", side=LOW, price=95.0)
        lvl2 = level(level_id="L1", side=LOW, price=96.0)
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl1, lvl2)}, {})

    def test_fvg_candidate_with_non_increasing_bounds_is_rejected(self):
        e = entry(direction=LONG, entry_price=100.0)
        c = fvg_candidate(lower=97.0, upper=95.0)
        with self.assertRaises(ValueError):
            detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})

    def test_result_sl_target_ids_are_unique(self):
        e = entry(direction=LONG, entry_price=100.0)
        levels = [level(level_id=f"L{i}", side=LOW, price=99.0 - i) for i in range(3)]
        c = fvg_candidate(lower=95.0, upper=97.0)
        result = detect_sl_target_candidates(e, selected("e1", c), {"1m": liq_result(*levels)}, {})
        all_ids = [t.sl_target_id for t in result.liquidity_targets] + [t.sl_target_id for t in result.fvg_targets]
        self.assertEqual(len(all_ids), len(set(all_ids)))


class DataWicksNeverAppearTests(unittest.TestCase):
    def test_result_never_references_data_wicks(self):
        import tradelab.sl_targets as sl_targets_module

        with open(sl_targets_module.__file__) as f:
            source = f.read()
        self.assertNotIn("data_wick", source.lower())

    def test_sl_target_candidate_has_no_data_wick_fields(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=LOW, price=95.0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        field_names = set(result.liquidity_targets[0].__dataclass_fields__)
        self.assertNotIn("wick_direction", field_names)
        self.assertNotIn("event_ids", field_names)


class AuditTests(unittest.TestCase):
    def test_liquidity_sl_has_no_fvg_fields(self):
        e = entry(direction=LONG, entry_price=100.0)
        lvl = level(side=LOW, price=95.0)
        result = detect_sl_target_candidates(e, NO_FVG, {"1m": liq_result(lvl)}, {})
        t = result.liquidity_targets[0]
        self.assertEqual(t.source_type, SLSourceType.LIQUIDITY)
        self.assertIsNone(t.zone_id)
        self.assertIsNone(t.fvg_source_type)
        self.assertIsNone(t.fvg_candidate_id)

    def test_fvg_sl_has_no_liquidity_fields_and_links_back_to_candidate(self):
        e = entry(direction=LONG, entry_price=100.0)
        c = fvg_candidate(lower=95.0, upper=97.0)
        result = detect_sl_target_candidates(e, selected("e1", c), EMPTY_LIQ, {})
        t = result.fvg_targets[0]
        self.assertEqual(t.source_type, SLSourceType.FVG_BOUNDARY)
        self.assertIsNone(t.liquidity_type)
        self.assertIsNone(t.level_id)
        self.assertEqual(t.fvg_candidate_id, c.candidate_id)
        self.assertEqual(t.fvg_timeframe, c.timeframe)
        self.assertEqual(t.lower, c.lower)
        self.assertEqual(t.upper, c.upper)


if __name__ == "__main__":
    unittest.main()
