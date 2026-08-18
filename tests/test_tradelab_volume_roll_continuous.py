#!/usr/bin/env python3
"""Volume-front-month roll policy tests: parallel-contract sessions
select the single highest-previous-session-volume symbol, strictly
never looking at the current or a later session; ties are broken
deterministically by ascending symbol and recorded; the very first
session in the data is always warm-up and never enters the series;
spreads/wrong-root symbols are filtered out before anything else;
a contract transition derives its RollAdjustment from the last common
minute (both contracts trading) strictly before the rollover; the
final result is exactly what calling tradelab.continuous_futures.
build_back_adjusted_continuous_series directly on the same raw_bars/
rolls would produce; and input records are never mutated."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tradelab.continuous_futures import build_back_adjusted_continuous_series
from tradelab.databento_parent_csv import DatabentoParentRecord
from tradelab.volume_roll_continuous import build_volume_front_month_continuous_series

UTC = timezone.utc

# Session boundaries in this fixture (EST, UTC-5, no DST edge involved):
# 2024-01-15 session: [2024-01-15T23:00Z, 2024-01-16T22:00Z)
# 2024-01-16 session: [2024-01-16T23:00Z, 2024-01-17T22:00Z)
# 2024-01-17 session: [2024-01-17T23:00Z, 2024-01-18T22:00Z)
# 2024-01-18 session: [2024-01-18T23:00Z, 2024-01-19T22:00Z)
S0 = datetime(2024, 1, 15, 23, 0, tzinfo=UTC)
S1 = datetime(2024, 1, 16, 23, 0, tzinfo=UTC)
S2 = datetime(2024, 1, 17, 23, 0, tzinfo=UTC)
S3 = datetime(2024, 1, 18, 23, 0, tzinfo=UTC)


def rec(ts: datetime, symbol: str, close: float, volume: float, *, o: float | None = None, h: float | None = None, l: float | None = None) -> DatabentoParentRecord:
    o = close if o is None else o
    h = close if h is None else h
    l = close if l is None else l
    return DatabentoParentRecord(
        ts_event=ts, rtype=33, publisher_id=1, instrument_id=1, open=o, high=h, low=l, close=close, volume=volume, symbol=symbol
    )


def three_session_transition_fixture() -> list[DatabentoParentRecord]:
    """S0 warm-up: MNQZ3 wins volume (150 vs 15) -> S1 selects MNQZ3.
    S1: MNQU3 wins volume (160 vs 40) -> S2 selects MNQU3 (transition).
    S2/S3: MNQU3 continues."""
    return [
        rec(S0, "MNQZ3", 100.0, 100), rec(S0, "MNQU3", 105.0, 10),
        rec(S0 + _min(1), "MNQZ3", 100.5, 50), rec(S0 + _min(1), "MNQU3", 105.5, 5),
        rec(S1, "MNQZ3", 101.0, 20), rec(S1, "MNQU3", 106.0, 80),
        rec(S1 + _min(1), "MNQZ3", 101.5, 20), rec(S1 + _min(1), "MNQU3", 106.5, 80),
        rec(S2, "MNQU3", 107.0, 90),
        rec(S2 + _min(1), "MNQU3", 107.5, 90),
        rec(S3, "MNQU3", 108.0, 50),
    ]


def _min(n: int):
    from datetime import timedelta

    return timedelta(minutes=n)


class ParallelContractsSameTimestampTests(unittest.TestCase):
    def test_multiple_symbols_at_identical_timestamp_are_all_candidates(self):
        records = three_session_transition_fixture()
        result = build_volume_front_month_continuous_series(records, "MNQ_CONTINUOUS")
        s1_selection = result.session_selections[1]
        candidate_symbols = {c.symbol for c in s1_selection.candidate_volumes}
        self.assertEqual(candidate_symbols, {"MNQZ3", "MNQU3"})


class SessionSelectionFromPreviousVolumeOnlyTests(unittest.TestCase):
    def test_selection_uses_only_previous_session_totals(self):
        records = three_session_transition_fixture()
        result = build_volume_front_month_continuous_series(records, "MNQ_CONTINUOUS")
        by_session = {s.session_id: s for s in result.session_selections}

        self.assertTrue(by_session["2024-01-15"].is_warmup)
        self.assertEqual(by_session["2024-01-16"].selected_symbol, "MNQZ3")  # S0: MNQZ3=150 > MNQU3=15
        self.assertEqual(by_session["2024-01-17"].selected_symbol, "MNQU3")  # S1: MNQU3=160 > MNQZ3=40
        self.assertEqual(by_session["2024-01-18"].selected_symbol, "MNQU3")  # S2: MNQU3=180 only candidate

        candidates_for_s2 = {c.symbol: c.total_volume for c in by_session["2024-01-17"].candidate_volumes}
        self.assertEqual(candidates_for_s2, {"MNQZ3": 40.0, "MNQU3": 160.0})


class NoLookAheadTests(unittest.TestCase):
    def test_earlier_session_selection_unaffected_by_later_session_data(self):
        base = [
            rec(S0, "MNQZ3", 100.0, 100), rec(S0, "MNQU3", 105.0, 10),
            rec(S1, "MNQZ3", 101.0, 10), rec(S1, "MNQU3", 106.0, 5),
            rec(S2, "MNQZ3", 102.0, 10), rec(S2, "MNQU3", 107.0, 5),
        ]
        with_future_spike = base + [rec(S1 + _min(1), "MNQU3", 999.0, 100_000)]
        with_future_spike.sort(key=lambda r: r.ts_event)

        result_base = build_volume_front_month_continuous_series(base, "X")
        result_spike = build_volume_front_month_continuous_series(with_future_spike, "X")

        sel_base = {s.session_id: s.selected_symbol for s in result_base.session_selections}
        sel_spike = {s.session_id: s.selected_symbol for s in result_spike.session_selections}

        # S1 (2024-01-16)'s own selection depends only on S0 -- unaffected by the huge
        # volume added inside S1 itself, even though that volume precedes some of S1's bars.
        self.assertEqual(sel_base["2024-01-16"], sel_spike["2024-01-16"])
        # S2 (2024-01-17) depends on S1 -- it DOES react to the spike, proving the
        # mechanism is genuinely volume-based and not simply inert.
        self.assertEqual(sel_spike["2024-01-17"], "MNQU3")


class DeterministicTieBreakTests(unittest.TestCase):
    def test_tie_is_broken_by_ascending_symbol_and_recorded(self):
        records = [
            rec(S0, "MNQZ3", 100.0, 50), rec(S0, "MNQU3", 105.0, 50),  # exact tie
            rec(S1, "MNQZ3", 101.0, 10), rec(S1, "MNQU3", 106.0, 10),
        ]
        result = build_volume_front_month_continuous_series(records, "X")
        s1 = result.session_selections[1]
        self.assertEqual(s1.selected_symbol, "MNQU3")  # "MNQU3" < "MNQZ3" ascending
        self.assertEqual(s1.tied_symbols, ("MNQU3", "MNQZ3"))

    def test_no_tie_reports_a_single_tied_symbol(self):
        records = three_session_transition_fixture()
        result = build_volume_front_month_continuous_series(records, "X")
        s1 = result.session_selections[1]
        self.assertEqual(s1.tied_symbols, ("MNQZ3",))


class WarmupExcludedTests(unittest.TestCase):
    def test_first_session_never_enters_raw_bars(self):
        records = three_session_transition_fixture()
        result = build_volume_front_month_continuous_series(records, "X")
        self.assertTrue(all(b.timestamp >= S1 for b in result.raw_bars))
        warmup_selection = result.session_selections[0]
        self.assertTrue(warmup_selection.is_warmup)
        self.assertIsNone(warmup_selection.selected_symbol)
        self.assertEqual(warmup_selection.candidate_volumes, ())

    def test_only_warmup_session_present_is_rejected(self):
        records = [rec(S0, "MNQZ3", 100.0, 10)]
        with self.assertRaises(ValueError):
            build_volume_front_month_continuous_series(records, "X")


class SymbolFilterExcludesSpreadsAndWrongRootTests(unittest.TestCase):
    def test_spread_and_wrong_root_symbols_are_excluded_from_selection(self):
        records = three_session_transition_fixture() + [
            rec(S1, "MNQZ3-MNQU3", 1.0, 1_000_000),  # calendar spread -- must never win
            rec(S1, "ESZ3", 1.0, 1_000_000),  # wrong root entirely
        ]
        records.sort(key=lambda r: r.ts_event)
        result = build_volume_front_month_continuous_series(records, "X")
        self.assertIn("MNQZ3-MNQU3", result.excluded_non_outright_symbols)
        self.assertIn("ESZ3", result.excluded_non_outright_symbols)
        by_session = {s.session_id: s for s in result.session_selections}
        # despite the enormous volume on the excluded symbols, S2's selection
        # (based on S1's candidate set) must still only ever be MNQZ3/MNQU3.
        self.assertEqual(by_session["2024-01-17"].selected_symbol, "MNQU3")
        candidate_symbols = {c.symbol for c in by_session["2024-01-17"].candidate_volumes}
        self.assertEqual(candidate_symbols, {"MNQZ3", "MNQU3"})

    def test_all_records_excluded_is_rejected(self):
        records = [rec(S0, "ESZ3", 100.0, 10), rec(S1, "ESH4", 100.0, 10)]
        with self.assertRaises(ValueError):
            build_volume_front_month_continuous_series(records, "X")


class TransitionAdjustmentFromLastCommonMinuteTests(unittest.TestCase):
    def test_transition_uses_the_latest_common_minute_before_rollover(self):
        records = three_session_transition_fixture()
        result = build_volume_front_month_continuous_series(records, "X")
        self.assertEqual(len(result.transitions), 1)
        transition = result.transitions[0]
        self.assertEqual(transition.outgoing_contract, "MNQZ3")
        self.assertEqual(transition.incoming_contract, "MNQU3")
        self.assertEqual(transition.effective_at, S2)  # nominal first minute of the new session
        self.assertEqual(transition.common_minute, S1 + _min(1))  # latest minute both traded, before S2
        self.assertEqual(transition.outgoing_close_at_common_minute, 101.5)
        self.assertEqual(transition.incoming_close_at_common_minute, 106.5)
        self.assertEqual(transition.adjustment_points, 5.0)  # incoming_close - outgoing_close

    def test_no_common_minute_before_rollover_is_rejected(self):
        # MNQU3 never appears in S0 at all, and in S1 it only trades at a
        # *different* minute than MNQZ3's own S1 bar -- MNQZ3 and MNQU3
        # never share a single minute anywhere before the rollover, even
        # though MNQU3 wins S1's volume and forces a transition.
        records = [
            rec(S0, "MNQZ3", 100.0, 100),
            rec(S1, "MNQZ3", 101.0, 1),
            rec(S1 + _min(5), "MNQU3", 106.0, 100),
            rec(S2, "MNQU3", 107.0, 200),
        ]
        with self.assertRaises(ValueError):
            build_volume_front_month_continuous_series(records, "X")

    def test_selected_symbol_missing_from_current_session_is_rejected(self):
        records = [
            rec(S0, "MNQZ3", 100.0, 1), rec(S0, "MNQU3", 105.0, 100),  # MNQU3 wins S0
            rec(S1, "MNQZ3", 101.0, 10),  # but only MNQZ3 actually trades in S1 -- MNQU3 has no bar here
        ]
        with self.assertRaises(ValueError):
            build_volume_front_month_continuous_series(records, "X")


class MatchesDirectContinuousFuturesCallTests(unittest.TestCase):
    def test_result_matches_calling_build_back_adjusted_continuous_series_directly(self):
        records = three_session_transition_fixture()
        result = build_volume_front_month_continuous_series(records, "MNQ_CONTINUOUS")

        expected = build_back_adjusted_continuous_series(list(result.raw_bars), list(_extract_rolls(result)), "MNQ_CONTINUOUS")

        self.assertEqual(result.continuous_series_result.normalized_bars, expected.normalized_bars)
        self.assertEqual(result.continuous_series_result.cumulative_offsets, expected.cumulative_offsets)
        self.assertEqual(result.continuous_series_result.applied_rolls, expected.applied_rolls)


def _extract_rolls(result):
    from tradelab.continuous_futures import RollAdjustment

    return [
        RollAdjustment(
            outgoing_contract=t.outgoing_contract,
            incoming_contract=t.incoming_contract,
            effective_at=t.effective_at,
            adjustment_points=t.adjustment_points,
        )
        for t in result.transitions
    ]


class InputImmutabilityTests(unittest.TestCase):
    def test_input_records_are_never_modified(self):
        records = three_session_transition_fixture()
        snapshot = [(r.ts_event, r.symbol, r.open, r.high, r.low, r.close, r.volume) for r in records]
        build_volume_front_month_continuous_series(records, "X")
        after = [(r.ts_event, r.symbol, r.open, r.high, r.low, r.close, r.volume) for r in records]
        self.assertEqual(snapshot, after)

    def test_input_list_object_is_not_the_result(self):
        records = three_session_transition_fixture()
        result = build_volume_front_month_continuous_series(records, "X")
        self.assertIsNot(result.raw_bars, records)


class ValidationTests(unittest.TestCase):
    def test_empty_records_rejected(self):
        with self.assertRaises(ValueError):
            build_volume_front_month_continuous_series([], "X")

    def test_empty_canonical_contract_rejected(self):
        records = three_session_transition_fixture()
        with self.assertRaises(ValueError):
            build_volume_front_month_continuous_series(records, "")

    def test_decreasing_ts_event_input_rejected(self):
        records = list(reversed(three_session_transition_fixture()))
        with self.assertRaises(ValueError):
            build_volume_front_month_continuous_series(records, "X")

    def test_duplicate_symbol_timestamp_pair_rejected(self):
        records = three_session_transition_fixture() + [rec(S1, "MNQZ3", 999.0, 1)]
        records.sort(key=lambda r: r.ts_event)
        with self.assertRaises(ValueError):
            build_volume_front_month_continuous_series(records, "X")


if __name__ == "__main__":
    unittest.main()
