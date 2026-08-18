#!/usr/bin/env python3
"""Continuous futures normalization tests: additive back-adjustment
chains cumulative offsets correctly across two and three contracts,
shifts open/high/low/close identically while leaving volume/timestamp
untouched, never mutates input bars, leaves the newest contract at a
zero offset, treats an already-single-contract series as a valid
empty-rolls passthrough, rejects every malformed/inconsistent roll
manifest (missing roll event, unknown contract, duplicate/conflicting
roll, naive timestamp, non-finite adjustment, effective_at that
doesn't line up with the real data transition, a contract reappearing
after the series moved past it), and removes the artificial rollover
gap in a synthetic example rather than merely relabeling it."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.bar import Bar1m
from tradelab.continuous_futures import (
    ContinuousSeriesResult,
    ContractOffset,
    RollAdjustment,
    build_back_adjusted_continuous_series,
)

UTC = timezone.utc
T0 = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)


def bar(i: int, contract: str, *, o: float = 100.0, h: float = 101.0, l: float = 99.0, c: float = 100.0, v: float = 10.0, base: datetime = T0) -> Bar1m:
    return Bar1m(timestamp=base + timedelta(minutes=i), open=o, high=h, low=l, close=c, volume=v, contract=contract)


def roll(outgoing: str, incoming: str, minute: int, points: float, base: datetime = T0) -> RollAdjustment:
    return RollAdjustment(outgoing_contract=outgoing, incoming_contract=incoming, effective_at=base + timedelta(minutes=minute), adjustment_points=points)


class SingleContractPassthroughTests(unittest.TestCase):
    def test_single_contract_no_rolls_is_valid_and_unchanged(self):
        bars = [bar(0, "C1", o=100, h=101, l=99, c=100.5), bar(1, "C1", o=100.5, h=102, l=100, c=101)]
        result = build_back_adjusted_continuous_series(bars, [], "NQ_CONTINUOUS")

        self.assertEqual(len(result.normalized_bars), 2)
        self.assertEqual(result.applied_rolls, ())
        self.assertEqual(result.cumulative_offsets, (ContractOffset(contract="C1", cumulative_offset_points=0.0),))
        for original, normalized in zip(bars, result.normalized_bars):
            self.assertEqual(normalized.open, original.open)
            self.assertEqual(normalized.high, original.high)
            self.assertEqual(normalized.low, original.low)
            self.assertEqual(normalized.close, original.close)
            self.assertEqual(normalized.volume, original.volume)
            self.assertEqual(normalized.contract, "NQ_CONTINUOUS")

    def test_empty_bars_with_empty_rolls_is_valid(self):
        result = build_back_adjusted_continuous_series([], [], "NQ_CONTINUOUS")
        self.assertEqual(result.normalized_bars, ())
        self.assertEqual(result.applied_rolls, ())
        self.assertEqual(result.cumulative_offsets, ())

    def test_empty_bars_with_nonempty_rolls_rejected(self):
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series([], [roll("C1", "C2", 0, 5.0)], "NQ_CONTINUOUS")

    def test_nonempty_rolls_on_single_contract_series_rejected(self):
        bars = [bar(0, "C1"), bar(1, "C1")]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, [roll("C1", "C2", 1, 5.0)], "NQ_CONTINUOUS")


class TwoContractAdjustmentTests(unittest.TestCase):
    def test_single_additive_adjustment_applied_to_older_contract(self):
        bars = [
            bar(0, "C1", o=100, h=101, l=99, c=100.5),
            bar(1, "C1", o=100.5, h=101.5, l=100, c=101),
            bar(2, "C2", o=150, h=151, l=149, c=150.5),
            bar(3, "C2", o=150.5, h=151.5, l=150, c=151),
        ]
        rolls = [roll("C1", "C2", 2, 50.0)]
        result = build_back_adjusted_continuous_series(bars, rolls, "NQ_CONTINUOUS")

        n = result.normalized_bars
        self.assertEqual((n[0].open, n[0].high, n[0].low, n[0].close), (150.0, 151.0, 149.0, 150.5))
        self.assertEqual((n[1].open, n[1].high, n[1].low, n[1].close), (150.5, 151.5, 150.0, 151.0))
        # newest contract (C2) stays exactly as it was
        self.assertEqual((n[2].open, n[2].high, n[2].low, n[2].close), (150.0, 151.0, 149.0, 150.5))
        self.assertEqual((n[3].open, n[3].high, n[3].low, n[3].close), (150.5, 151.5, 150.0, 151.0))
        self.assertTrue(all(b.contract == "NQ_CONTINUOUS" for b in n))
        self.assertEqual(
            result.cumulative_offsets,
            (ContractOffset(contract="C1", cumulative_offset_points=50.0), ContractOffset(contract="C2", cumulative_offset_points=0.0)),
        )
        self.assertEqual(result.applied_rolls, tuple(rolls))


class ThreeContractCumulativeOffsetTests(unittest.TestCase):
    def test_offsets_accumulate_correctly_across_three_contracts(self):
        bars = [
            bar(0, "C1", o=100, h=101, l=99, c=100),
            bar(1, "C1", o=100, h=101, l=99, c=100.5),
            bar(2, "C2", o=150, h=151, l=149, c=150),
            bar(3, "C2", o=150, h=151, l=149, c=150.5),
            bar(4, "C3", o=200, h=201, l=199, c=200),
            bar(5, "C3", o=200, h=201, l=199, c=200.5),
        ]
        rolls = [roll("C1", "C2", 2, 50.0), roll("C2", "C3", 4, 50.0)]
        result = build_back_adjusted_continuous_series(bars, rolls, "NQ_CONTINUOUS")

        n = result.normalized_bars
        # C1 shifted by 50 (C1->C2) + 50 (C2->C3) = 100
        self.assertEqual(n[0].open, 200.0)
        self.assertEqual(n[1].close, 200.5)
        # C2 shifted by its own single hop only: 50
        self.assertEqual(n[2].open, 200.0)
        self.assertEqual(n[3].close, 200.5)
        # C3 (newest) unadjusted
        self.assertEqual(n[4].open, 200.0)
        self.assertEqual(n[5].close, 200.5)

        offsets = {o.contract: o.cumulative_offset_points for o in result.cumulative_offsets}
        self.assertEqual(offsets, {"C1": 100.0, "C2": 50.0, "C3": 0.0})
        self.assertEqual([r.outgoing_contract for r in result.applied_rolls], ["C1", "C2"])  # chronological

    def test_asymmetric_adjustments_still_chain_correctly(self):
        bars = [bar(0, "C1", c=100), bar(1, "C2", c=100), bar(2, "C3", c=100)]
        rolls = [roll("C1", "C2", 1, 30.0), roll("C2", "C3", 2, -10.0)]
        result = build_back_adjusted_continuous_series(bars, rolls, "X")
        offsets = {o.contract: o.cumulative_offset_points for o in result.cumulative_offsets}
        self.assertEqual(offsets, {"C1": 20.0, "C2": -10.0, "C3": 0.0})


class OHLCShiftedIdenticallyTests(unittest.TestCase):
    def test_open_high_low_close_all_shifted_by_the_same_offset(self):
        bars = [bar(0, "C1", o=10, h=20, l=5, c=15), bar(1, "C2", o=110, h=120, l=105, c=115)]
        rolls = [roll("C1", "C2", 1, 100.0)]
        result = build_back_adjusted_continuous_series(bars, rolls, "X")
        n0 = result.normalized_bars[0]
        self.assertEqual((n0.open, n0.high, n0.low, n0.close), (110.0, 120.0, 105.0, 115.0))


class InputImmutabilityTests(unittest.TestCase):
    def test_input_bars_are_never_modified(self):
        bars = [bar(0, "C1", o=100, h=101, l=99, c=100), bar(1, "C2", o=150, h=151, l=149, c=150)]
        snapshot = [(b.open, b.high, b.low, b.close, b.volume, b.contract) for b in bars]
        build_back_adjusted_continuous_series(bars, [roll("C1", "C2", 1, 50.0)], "X")
        after = [(b.open, b.high, b.low, b.close, b.volume, b.contract) for b in bars]
        self.assertEqual(snapshot, after)

    def test_input_bars_list_object_is_not_the_result(self):
        bars = [bar(0, "C1"), bar(1, "C1")]
        result = build_back_adjusted_continuous_series(bars, [], "X")
        self.assertIsNot(result.normalized_bars, bars)
        for original, normalized in zip(bars, result.normalized_bars):
            self.assertIsNot(original, normalized)


class MissingRollEventTests(unittest.TestCase):
    def test_contract_change_without_matching_roll_rejected(self):
        bars = [bar(0, "C1"), bar(1, "C2")]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, [], "X")

    def test_one_of_two_transitions_missing_a_roll_rejected(self):
        bars = [bar(0, "C1"), bar(1, "C2"), bar(2, "C3")]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, [roll("C1", "C2", 1, 10.0)], "X")


class UnknownContractTests(unittest.TestCase):
    def test_roll_referencing_contract_not_in_bars_rejected(self):
        bars = [bar(0, "C1"), bar(1, "C2")]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, [roll("C1", "C9", 1, 10.0)], "X")

    def test_roll_between_known_but_nonadjacent_contracts_rejected(self):
        bars = [bar(0, "C1"), bar(1, "C2"), bar(2, "C3")]
        rolls = [roll("C1", "C2", 1, 10.0), roll("C2", "C3", 2, 10.0), roll("C1", "C3", 1, 20.0)]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, rolls, "X")


class DuplicateConflictingRollTests(unittest.TestCase):
    def test_two_rolls_for_the_same_transition_rejected_even_if_values_agree(self):
        bars = [bar(0, "C1"), bar(1, "C2")]
        rolls = [roll("C1", "C2", 1, 10.0), roll("C1", "C2", 1, 10.0)]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, rolls, "X")

    def test_two_conflicting_rolls_for_the_same_transition_rejected(self):
        bars = [bar(0, "C1"), bar(1, "C2")]
        rolls = [roll("C1", "C2", 1, 10.0), roll("C1", "C2", 1, 99.0)]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, rolls, "X")


class EffectiveAtConsistencyTests(unittest.TestCase):
    def test_effective_at_before_previous_bar_rejected(self):
        bars = [bar(0, "C1"), bar(5, "C2")]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, [roll("C1", "C2", 0, 10.0)], "X")

    def test_effective_at_after_transition_bar_rejected(self):
        bars = [bar(0, "C1"), bar(1, "C2")]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, [roll("C1", "C2", 10, 10.0)], "X")

    def test_effective_at_exactly_on_transition_bar_accepted(self):
        bars = [bar(0, "C1"), bar(1, "C2")]
        build_back_adjusted_continuous_series(bars, [roll("C1", "C2", 1, 10.0)], "X")  # no raise


class ReappearingContractTests(unittest.TestCase):
    def test_contract_reappearing_after_series_moved_on_rejected(self):
        bars = [bar(0, "C1"), bar(1, "C2"), bar(2, "C1")]
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series(bars, [roll("C1", "C2", 1, 10.0), roll("C2", "C1", 2, -10.0)], "X")


class RollAdjustmentValidationTests(unittest.TestCase):
    def test_empty_outgoing_contract_rejected(self):
        with self.assertRaises(ValueError):
            RollAdjustment(outgoing_contract="", incoming_contract="C2", effective_at=T0, adjustment_points=1.0)

    def test_empty_incoming_contract_rejected(self):
        with self.assertRaises(ValueError):
            RollAdjustment(outgoing_contract="C1", incoming_contract="", effective_at=T0, adjustment_points=1.0)

    def test_same_outgoing_and_incoming_contract_rejected(self):
        with self.assertRaises(ValueError):
            RollAdjustment(outgoing_contract="C1", incoming_contract="C1", effective_at=T0, adjustment_points=1.0)

    def test_naive_effective_at_rejected(self):
        with self.assertRaises(ValueError):
            RollAdjustment(outgoing_contract="C1", incoming_contract="C2", effective_at=datetime(2024, 1, 1), adjustment_points=1.0)

    def test_non_finite_adjustment_points_rejected(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(ValueError):
                RollAdjustment(outgoing_contract="C1", incoming_contract="C2", effective_at=T0, adjustment_points=bad)

    def test_non_numeric_adjustment_points_rejected(self):
        with self.assertRaises(ValueError):
            RollAdjustment(outgoing_contract="C1", incoming_contract="C2", effective_at=T0, adjustment_points="10")
        with self.assertRaises(ValueError):
            RollAdjustment(outgoing_contract="C1", incoming_contract="C2", effective_at=T0, adjustment_points=True)

    def test_zero_adjustment_points_is_valid(self):
        RollAdjustment(outgoing_contract="C1", incoming_contract="C2", effective_at=T0, adjustment_points=0.0)  # no raise


class CanonicalContractValidationTests(unittest.TestCase):
    def test_empty_canonical_contract_rejected(self):
        with self.assertRaises(ValueError):
            build_back_adjusted_continuous_series([], [], "")


class NoArtificialRolloverGapTests(unittest.TestCase):
    def test_back_adjustment_removes_the_synthetic_contract_switch_jump(self):
        # Raw data: C1's last bar closes at 4999.75; C2's first bar opens
        # at 5050.00 -- a 50.25-point raw jump that is almost entirely a
        # pure contract-switch artifact (the vendor-assessed roll gap is
        # exactly 50.0 points), with only a genuine 0.25-point intrabar
        # move actually happening at the boundary.
        bars = [
            bar(0, "C1", o=4998.00, h=5000.00, l=4997.50, c=4999.75),
            bar(1, "C2", o=5050.00, h=5051.00, l=5049.50, c=5050.25),
        ]
        raw_gap = abs(bars[1].open - bars[0].close)
        self.assertAlmostEqual(raw_gap, 50.25)

        result = build_back_adjusted_continuous_series(bars, [roll("C1", "C2", 1, 50.0)], "NQ_CONTINUOUS")
        adjusted_gap = abs(result.normalized_bars[1].open - result.normalized_bars[0].close)
        self.assertAlmostEqual(adjusted_gap, 0.25)
        self.assertLess(adjusted_gap, raw_gap)


if __name__ == "__main__":
    unittest.main()
