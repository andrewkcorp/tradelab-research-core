#!/usr/bin/env python3
"""HTF bias tests: FVG/iFVG-driven per-timeframe state, persistence,
same-instant conflicts, global bias combination, asof queries, timeframe/
chronology validation, and no lookahead."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.bias import (
    NEUTRAL,
    SUPPORTED_TIMEFRAMES,
    BiasSnapshot,
    ConfluenceType,
    HTFBiasResult,
    bias_at,
    compute_htf_bias,
)
from tradelab.fvg import BEARISH, BULLISH, Candle, FVGDetectionResult, FVGEvent, FVGStatus, FVGZone, detect_fvgs

T0 = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
_OPPOSITE = {BULLISH: BEARISH, BEARISH: BULLISH}


def zone(
    timeframe: str,
    direction: str,
    zone_id: str,
    formed_at: datetime,
    status: FVGStatus = FVGStatus.ACTIVE,
    resolved_at: datetime | None = None,
    produced_ifvg: bool = False,
    lower: float = 100.0,
    upper: float = 102.0,
) -> FVGZone:
    return FVGZone(
        zone_id=zone_id,
        timeframe=timeframe,
        direction=direction,
        lower=lower,
        upper=upper,
        candle_1_timestamp=formed_at - timedelta(hours=3),
        candle_2_timestamp=formed_at - timedelta(hours=2),
        candle_3_timestamp=formed_at - timedelta(hours=1),
        formed_at=formed_at,
        status=status,
        resolved_at=resolved_at,
        produced_ifvg=produced_ifvg,
    )


def ifvg_event(timeframe: str, zone_id: str, original_direction: str, known_at: datetime, lower: float = 100.0, upper: float = 102.0) -> FVGEvent:
    return FVGEvent(
        zone_id=zone_id,
        timeframe=timeframe,
        direction=_OPPOSITE[original_direction],
        original_zone_direction=original_direction,
        lower=lower,
        upper=upper,
        timestamp=known_at,
        known_at=known_at,
    )


def result_with(zones: list[FVGZone], events: list[FVGEvent]) -> FVGDetectionResult:
    return FVGDetectionResult(zones=zones, events=events)


class FVGSetsDirectionTimingTests(unittest.TestCase):
    def test_fvg_formation_sets_direction_only_from_its_known_at_onward(self):
        formed_at = T0 + timedelta(hours=3)
        z = zone("1H", BULLISH, "1H-FVG-1", formed_at)
        htf = compute_htf_bias({"1H": result_with([z], [])})

        before = bias_at(htf, formed_at - timedelta(microseconds=1))
        at = bias_at(htf, formed_at)
        after = bias_at(htf, formed_at + timedelta(hours=1))

        self.assertEqual(before.timeframe_states["1H"], NEUTRAL)
        self.assertEqual(at.timeframe_states["1H"], BULLISH)
        self.assertEqual(after.timeframe_states["1H"], BULLISH)

    def test_bearish_fvg_formation_sets_bearish(self):
        formed_at = T0 + timedelta(hours=3)
        z = zone("4H", BEARISH, "4H-FVG-1", formed_at)
        htf = compute_htf_bias({"4H": result_with([z], [])})
        self.assertEqual(bias_at(htf, formed_at).timeframe_states["4H"], BEARISH)


class IFVGSymmetryTests(unittest.TestCase):
    def test_bullish_zone_ifvg_flips_to_bearish(self):
        formed_at = T0 + timedelta(hours=3)
        inverted_at = T0 + timedelta(hours=10)
        z = zone("1H", BULLISH, "1H-FVG-1", formed_at, status=FVGStatus.MITIGATED, resolved_at=inverted_at, produced_ifvg=True)
        e = ifvg_event("1H", "1H-FVG-1", BULLISH, inverted_at)
        htf = compute_htf_bias({"1H": result_with([z], [e])})

        self.assertEqual(bias_at(htf, formed_at).timeframe_states["1H"], BULLISH)
        self.assertEqual(bias_at(htf, inverted_at - timedelta(microseconds=1)).timeframe_states["1H"], BULLISH)
        self.assertEqual(bias_at(htf, inverted_at).timeframe_states["1H"], BEARISH)

    def test_bearish_zone_ifvg_flips_to_bullish(self):
        formed_at = T0 + timedelta(hours=3)
        inverted_at = T0 + timedelta(hours=10)
        z = zone("4H", BEARISH, "4H-FVG-1", formed_at, status=FVGStatus.MITIGATED, resolved_at=inverted_at, produced_ifvg=True)
        e = ifvg_event("4H", "4H-FVG-1", BEARISH, inverted_at)
        htf = compute_htf_bias({"4H": result_with([z], [e])})
        self.assertEqual(bias_at(htf, formed_at).timeframe_states["4H"], BEARISH)
        self.assertEqual(bias_at(htf, inverted_at).timeframe_states["4H"], BULLISH)

    def test_end_to_end_via_real_detect_fvgs(self):
        # A genuine bullish 1H FVG, then a candle that fully inverts it.
        def c(i, o, h, l, cl):
            ts = T0 + timedelta(hours=i)
            return Candle(ts, o, h, l, cl, ts + timedelta(hours=1))

        candles = [
            c(0, 100.0, 100.0, 99.0, 99.5),
            c(1, 101.0, 101.5, 100.2, 101.2),
            c(2, 102.3, 102.5, 102.0, 102.35),  # forms bullish [100, 102]
            c(3, 100.5, 100.6, 99.0, 99.5),  # full pass-through + close below 100 -> bearish iFVG
        ]
        fvg_result = detect_fvgs(candles, "1H")
        htf = compute_htf_bias({"1H": fvg_result})
        self.assertEqual(bias_at(htf, candles[2].known_at).timeframe_states["1H"], BULLISH)
        self.assertEqual(bias_at(htf, candles[3].known_at).timeframe_states["1H"], BEARISH)


class MitigationWithoutIFVGTests(unittest.TestCase):
    def test_plain_mitigation_does_not_change_the_persisted_bias(self):
        formed_at = T0 + timedelta(hours=3)
        mitigated_at = T0 + timedelta(hours=10)
        # zone reaches MITIGATED, produced_ifvg=False, and (correctly) has
        # no corresponding FVGEvent -- exactly what detect_fvgs produces
        # for a mitigation-without-inversion.
        z = zone("1H", BULLISH, "1H-FVG-1", formed_at, status=FVGStatus.MITIGATED, resolved_at=mitigated_at, produced_ifvg=False)
        htf = compute_htf_bias({"1H": result_with([z], [])})
        self.assertEqual(bias_at(htf, mitigated_at).timeframe_states["1H"], BULLISH)
        self.assertEqual(bias_at(htf, mitigated_at + timedelta(days=365)).timeframe_states["1H"], BULLISH)

    def test_end_to_end_mitigation_without_ifvg_via_real_detect_fvgs(self):
        def c(i, o, h, l, cl):
            ts = T0 + timedelta(hours=i)
            return Candle(ts, o, h, l, cl, ts + timedelta(hours=1))

        candles = [
            c(0, 100.0, 100.0, 99.0, 99.5),
            c(1, 101.0, 101.5, 100.2, 101.2),
            c(2, 102.3, 102.5, 102.0, 102.35),  # forms bullish [100, 102]
            c(3, 100.5, 100.6, 99.9, 100.5),  # full pass-through (low<=100) but closes back inside -> MITIGATED, no iFVG
        ]
        fvg_result = detect_fvgs(candles, "1H")
        self.assertEqual(fvg_result.events, [])  # sanity: this really is mitigation-without-inversion
        htf = compute_htf_bias({"1H": fvg_result})
        self.assertEqual(bias_at(htf, candles[3].known_at).timeframe_states["1H"], BULLISH)


class SameInstantConflictTests(unittest.TestCase):
    def test_opposite_confluence_at_the_same_known_at_on_one_timeframe_is_neutral(self):
        # A fresh bullish formation vs. a DIFFERENT zone's iFVG that
        # happens to resolve bearish, both at the exact same known_at.
        shared_known_at = T0 + timedelta(hours=5)
        bull_zone = zone("1H", BULLISH, "1H-FVG-1", shared_known_at)
        bearish_original = zone("1H", BULLISH, "1H-FVG-3", T0 + timedelta(hours=1))
        bearish_ifvg = ifvg_event("1H", "1H-FVG-3", BULLISH, shared_known_at)  # bullish zone -> bearish ifvg
        htf = compute_htf_bias({"1H": result_with([bull_zone, bearish_original], [bearish_ifvg])})

        snapshot = bias_at(htf, shared_known_at)
        self.assertEqual(snapshot.timeframe_states["1H"], NEUTRAL)
        change = next(ch for ch in htf.timeframe_changes["1H"] if ch.known_at == shared_known_at)
        self.assertEqual(len(change.sources), 2)
        self.assertEqual({s.direction for s in change.sources}, {BULLISH, BEARISH})

    def test_same_direction_confluence_at_the_same_known_at_simply_confirms_it(self):
        shared_known_at = T0 + timedelta(hours=5)
        zone_a = zone("1H", BULLISH, "1H-FVG-1", shared_known_at)
        zone_b_original = zone("1H", BEARISH, "1H-FVG-2", T0 + timedelta(hours=1))
        bullish_ifvg = ifvg_event("1H", "1H-FVG-2", BEARISH, shared_known_at)  # bearish zone -> bullish ifvg
        htf = compute_htf_bias({"1H": result_with([zone_a, zone_b_original], [bullish_ifvg])})
        change = next(ch for ch in htf.timeframe_changes["1H"] if ch.known_at == shared_known_at)
        self.assertEqual(change.state, BULLISH)
        self.assertEqual(len(change.sources), 2)

    def test_conflict_resolves_to_neutral_and_then_persists_until_next_confluence(self):
        shared_known_at = T0 + timedelta(hours=5)
        later_known_at = T0 + timedelta(hours=20)
        bull_zone = zone("1H", BULLISH, "1H-FVG-1", shared_known_at)
        bearish_original = zone("1H", BULLISH, "1H-FVG-2", T0 + timedelta(hours=1))
        bearish_ifvg = ifvg_event("1H", "1H-FVG-2", BULLISH, shared_known_at)
        later_bearish = zone("1H", BEARISH, "1H-FVG-3", later_known_at)
        htf = compute_htf_bias({"1H": result_with([bull_zone, bearish_original, later_bearish], [bearish_ifvg])})

        self.assertEqual(bias_at(htf, shared_known_at).timeframe_states["1H"], NEUTRAL)
        self.assertEqual(bias_at(htf, shared_known_at + timedelta(hours=1)).timeframe_states["1H"], NEUTRAL)
        self.assertEqual(bias_at(htf, later_known_at).timeframe_states["1H"], BEARISH)


class Conflict1DAnd4HTests(unittest.TestCase):
    def test_bullish_daily_fvg_and_bearish_4h_fvg_is_global_neutral(self):
        # The spec's mandatory example.
        ts = T0 + timedelta(hours=5)
        d1 = zone("1D", BULLISH, "1D-FVG-1", ts)
        h4 = zone("4H", BEARISH, "4H-FVG-1", ts)
        htf = compute_htf_bias({"1D": result_with([d1], []), "4H": result_with([h4], [])})
        snapshot = bias_at(htf, ts)
        self.assertEqual(snapshot.timeframe_states["1D"], BULLISH)
        self.assertEqual(snapshot.timeframe_states["4H"], BEARISH)
        self.assertEqual(snapshot.global_bias, NEUTRAL)

    def test_no_weighting_of_1d_over_4h_or_1h(self):
        ts = T0 + timedelta(hours=5)
        d1 = zone("1D", BULLISH, "1D-FVG-1", ts)
        h1 = zone("1H", BEARISH, "1H-FVG-1", ts)
        htf = compute_htf_bias({"1D": result_with([d1], []), "1H": result_with([h1], [])})
        self.assertEqual(bias_at(htf, ts).global_bias, NEUTRAL)


class GlobalBiasCombinationTests(unittest.TestCase):
    def _snapshot_for(self, states: dict[str, str]) -> BiasSnapshot:
        ts = T0 + timedelta(hours=5)
        fvg_results = {}
        for timeframe, direction in states.items():
            if direction == NEUTRAL:
                continue
            fvg_results[timeframe] = result_with([zone(timeframe, direction, f"{timeframe}-FVG-1", ts)], [])
        htf = compute_htf_bias(fvg_results)
        return bias_at(htf, ts)

    def test_all_neutral_is_global_neutral(self):
        snap = self._snapshot_for({"1H": NEUTRAL, "4H": NEUTRAL, "1D": NEUTRAL})
        self.assertEqual(snap.global_bias, NEUTRAL)

    def test_single_bullish_with_rest_neutral_is_global_bullish(self):
        snap = self._snapshot_for({"1H": BULLISH, "4H": NEUTRAL, "1D": NEUTRAL})
        self.assertEqual(snap.global_bias, BULLISH)

    def test_single_bearish_with_rest_neutral_is_global_bearish(self):
        snap = self._snapshot_for({"1H": NEUTRAL, "4H": BEARISH, "1D": NEUTRAL})
        self.assertEqual(snap.global_bias, BEARISH)

    def test_multiple_bullish_with_neutral_is_global_bullish(self):
        snap = self._snapshot_for({"1H": BULLISH, "4H": BULLISH, "1D": NEUTRAL})
        self.assertEqual(snap.global_bias, BULLISH)

    def test_multiple_bearish_with_neutral_is_global_bearish(self):
        snap = self._snapshot_for({"1H": BEARISH, "4H": BEARISH, "1D": NEUTRAL})
        self.assertEqual(snap.global_bias, BEARISH)

    def test_all_bullish_is_global_bullish(self):
        snap = self._snapshot_for({"1H": BULLISH, "4H": BULLISH, "1D": BULLISH})
        self.assertEqual(snap.global_bias, BULLISH)

    def test_all_bearish_is_global_bearish(self):
        snap = self._snapshot_for({"1H": BEARISH, "4H": BEARISH, "1D": BEARISH})
        self.assertEqual(snap.global_bias, BEARISH)

    def test_bullish_and_bearish_mix_is_global_neutral(self):
        snap = self._snapshot_for({"1H": BULLISH, "4H": BEARISH, "1D": NEUTRAL})
        self.assertEqual(snap.global_bias, NEUTRAL)

    def test_bullish_bearish_and_neutral_together_is_global_neutral(self):
        snap = self._snapshot_for({"1H": BULLISH, "4H": BEARISH, "1D": BULLISH})
        self.assertEqual(snap.global_bias, NEUTRAL)


class AsofQueryTests(unittest.TestCase):
    def test_query_before_between_and_after_events(self):
        first_at = T0 + timedelta(hours=3)
        second_at = T0 + timedelta(hours=20)
        z1 = zone("1H", BULLISH, "1H-FVG-1", first_at)
        z2 = zone("1H", BEARISH, "1H-FVG-2", second_at)
        htf = compute_htf_bias({"1H": result_with([z1, z2], [])})

        before = bias_at(htf, T0)
        between = bias_at(htf, first_at + timedelta(hours=1))
        after = bias_at(htf, second_at + timedelta(days=1))

        self.assertEqual(before.timeframe_states["1H"], NEUTRAL)
        self.assertEqual(between.timeframe_states["1H"], BULLISH)
        self.assertEqual(after.timeframe_states["1H"], BEARISH)

    def test_no_events_at_all_is_always_neutral(self):
        htf = compute_htf_bias({})
        snap = bias_at(htf, T0 + timedelta(days=100))
        self.assertEqual(snap.timeframe_states, {"1H": NEUTRAL, "4H": NEUTRAL, "1D": NEUTRAL})
        self.assertEqual(snap.global_bias, NEUTRAL)

    def test_naive_ts_utc_is_rejected(self):
        htf = compute_htf_bias({})
        with self.assertRaises(ValueError):
            bias_at(htf, datetime(2026, 1, 5))


class ValidationTests(unittest.TestCase):
    def test_unsupported_timeframe_key_is_rejected(self):
        z = zone("5m", BULLISH, "5m-FVG-1", T0 + timedelta(hours=1))
        with self.assertRaises(ValueError):
            compute_htf_bias({"5m": result_with([z], [])})

    def test_zone_timeframe_mismatch_with_dict_key_is_rejected(self):
        z = zone("4H", BULLISH, "4H-FVG-1", T0 + timedelta(hours=1))  # zone says 4H
        with self.assertRaises(ValueError):
            compute_htf_bias({"1H": result_with([z], [])})  # but filed under 1H

    def test_event_referencing_unknown_zone_id_is_rejected(self):
        e = ifvg_event("1H", "1H-FVG-999", BULLISH, T0 + timedelta(hours=5))
        with self.assertRaises(ValueError):
            compute_htf_bias({"1H": result_with([], [e])})

    def test_event_known_at_not_after_its_zones_formed_at_is_rejected(self):
        formed_at = T0 + timedelta(hours=5)
        z = zone("1H", BULLISH, "1H-FVG-1", formed_at)
        e = ifvg_event("1H", "1H-FVG-1", BULLISH, formed_at)  # same instant as formation -- impossible for a real iFVG
        with self.assertRaises(ValueError):
            compute_htf_bias({"1H": result_with([z], [e])})

    def test_event_direction_inconsistent_with_its_zone_is_rejected(self):
        formed_at = T0 + timedelta(hours=1)
        z = zone("1H", BULLISH, "1H-FVG-1", formed_at)
        bad_event = FVGEvent(
            zone_id="1H-FVG-1",
            timeframe="1H",
            direction=BULLISH,  # should be BEARISH (opposite of the bullish zone)
            original_zone_direction=BULLISH,
            lower=100.0,
            upper=102.0,
            timestamp=formed_at + timedelta(hours=5),
            known_at=formed_at + timedelta(hours=5),
        )
        with self.assertRaises(ValueError):
            compute_htf_bias({"1H": result_with([z], [bad_event])})

    def test_all_supported_timeframes_always_present_in_output(self):
        htf = compute_htf_bias({})
        self.assertEqual(set(htf.timeframe_changes), SUPPORTED_TIMEFRAMES)
        self.assertEqual(SUPPORTED_TIMEFRAMES, frozenset({"1H", "4H", "1D"}))


class NoLookaheadTests(unittest.TestCase):
    def test_a_later_zone_never_affects_an_earlier_query(self):
        early_at = T0 + timedelta(hours=1)
        late_at = T0 + timedelta(hours=1000)
        z_late = zone("1H", BEARISH, "1H-FVG-1", late_at)
        htf = compute_htf_bias({"1H": result_with([z_late], [])})
        self.assertEqual(bias_at(htf, early_at).timeframe_states["1H"], NEUTRAL)

    def test_change_sources_carry_confluence_type_direction_id_and_known_at_for_audit(self):
        formed_at = T0 + timedelta(hours=3)
        z = zone("1D", BULLISH, "1D-FVG-1", formed_at)
        htf = compute_htf_bias({"1D": result_with([z], [])})
        change = htf.timeframe_changes["1D"][0]
        self.assertEqual(change.timeframe, "1D")
        self.assertEqual(change.known_at, formed_at)
        source = change.sources[0]
        self.assertEqual(source.confluence_type, ConfluenceType.FVG_FORMATION)
        self.assertEqual(source.direction, BULLISH)
        self.assertEqual(source.source_id, "1D-FVG-1")
        self.assertEqual(source.known_at, formed_at)


if __name__ == "__main__":
    unittest.main()
