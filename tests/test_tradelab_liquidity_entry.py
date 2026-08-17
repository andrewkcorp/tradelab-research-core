#!/usr/bin/env python3
"""Liquidity entry candidate tests: all four trigger phases, PENDING
producing nothing, high/low x reversal/continuation direction mapping,
swing/equal and session sources, 1m/5m/15m-only restriction, known_at/
no-lookahead, take/reclaim consistency validation, no duplicates, and
full level audit retained on each candidate."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tradelab.fvg import Candle
from tradelab.liquidity import (
    HIGH,
    LOW,
    LiquidityDetectionResult,
    LiquidityLevel,
    LiquidityStatus,
    LiquidityTakeEvent,
    LiquidityType,
    ReclaimOutcome,
    ReclaimResult,
    detect_liquidity,
)
from tradelab.liquidity_entry import (
    CandidatePhase,
    EntryMode,
    SourceFamily,
    detect_liquidity_entry_candidates,
)
from tradelab.session_liquidity import detect_session_liquidity

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)


def candle(offset_minutes: float, o: float, h: float, l: float, c: float, known_offset_minutes: float | None = None) -> Candle:
    ts = T0 + timedelta(minutes=offset_minutes)
    known_at = T0 + timedelta(minutes=known_offset_minutes if known_offset_minutes is not None else offset_minutes + 1)
    return Candle(ts, o, h, l, c, known_at)


def level(
    level_id: str = "1m-swing_high-1",
    timeframe: str = "1m",
    liquidity_type: LiquidityType = LiquidityType.SWING_HIGH,
    side: str = HIGH,
    lower: float = 100.0,
    upper: float = 100.0,
    status: LiquidityStatus = LiquidityStatus.TAKEN,
    taken_at: datetime | None = None,
) -> LiquidityLevel:
    formed_at = T0
    return LiquidityLevel(
        level_id=level_id,
        timeframe=timeframe,
        liquidity_type=liquidity_type,
        side=side,
        lower=lower,
        upper=upper,
        member_count=1,
        member_timestamps=(T0,),
        formed_at=formed_at,
        status=status,
        taken_at=taken_at,
        absorbed_by_level_id=None,
    )


def take_event(
    level_id: str = "1m-swing_high-1",
    timeframe: str = "1m",
    liquidity_type: LiquidityType | None = None,
    side: str = HIGH,
    lower: float = 100.0,
    upper: float = 100.0,
    timestamp: datetime | None = None,
    known_at: datetime = T0 + timedelta(minutes=5),
) -> LiquidityTakeEvent:
    if liquidity_type is None:
        liquidity_type = LiquidityType.SWING_HIGH if side == HIGH else LiquidityType.SWING_LOW
    return LiquidityTakeEvent(
        level_id=level_id,
        timeframe=timeframe,
        liquidity_type=liquidity_type,
        side=side,
        lower=lower,
        upper=upper,
        timestamp=known_at if timestamp is None else timestamp,
        known_at=known_at,
    )


def reclaim_result(
    level_id: str = "1m-swing_high-1",
    timeframe: str = "1m",
    outcome: ReclaimOutcome = ReclaimOutcome.RECLAIM_CLOSE,
    resolved_at: datetime | None = T0 + timedelta(minutes=6),
    bars_after_take: int | None = 1,
    take_timestamp: datetime = T0 + timedelta(minutes=5),
) -> ReclaimResult:
    return ReclaimResult(
        level_id=level_id,
        timeframe=timeframe,
        take_timestamp=take_timestamp,
        outcome=outcome,
        resolved_at=resolved_at,
        bars_after_take=bars_after_take,
    )


def result_with(levels: list[LiquidityLevel], takes: list[LiquidityTakeEvent], reclaims: list[ReclaimResult]) -> LiquidityDetectionResult:
    return LiquidityDetectionResult(levels=levels, takes=takes, reclaims=reclaims)


class TriggerPhaseTests(unittest.TestCase):
    def test_take_phase_is_known_at_the_takes_own_known_at(self):
        take_known_at = T0 + timedelta(minutes=5)
        result = result_with([level()], [take_event(known_at=take_known_at)], [])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        take_candidates = [c for c in out.candidates if c.trigger_phase == CandidatePhase.TAKE]
        self.assertEqual(len(take_candidates), 2)
        for c in take_candidates:
            self.assertEqual(c.known_at, take_known_at)

    def test_reclaim_wick_phase_is_known_at_resolved_at(self):
        resolved_at = T0 + timedelta(minutes=6)
        result = result_with(
            [level()], [take_event()], [reclaim_result(outcome=ReclaimOutcome.RECLAIM_WICK, resolved_at=resolved_at)]
        )
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        wick_candidates = [c for c in out.candidates if c.trigger_phase == CandidatePhase.RECLAIM_WICK]
        self.assertEqual(len(wick_candidates), 2)
        for c in wick_candidates:
            self.assertEqual(c.known_at, resolved_at)

    def test_reclaim_close_phase_is_known_at_resolved_at(self):
        resolved_at = T0 + timedelta(minutes=6)
        result = result_with(
            [level()], [take_event()], [reclaim_result(outcome=ReclaimOutcome.RECLAIM_CLOSE, resolved_at=resolved_at)]
        )
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        close_candidates = [c for c in out.candidates if c.trigger_phase == CandidatePhase.RECLAIM_CLOSE]
        self.assertEqual(len(close_candidates), 2)
        for c in close_candidates:
            self.assertEqual(c.known_at, resolved_at)

    def test_no_reclaim_in_window_phase_is_known_at_resolved_at(self):
        resolved_at = T0 + timedelta(minutes=8)
        result = result_with(
            [level()],
            [take_event()],
            [reclaim_result(outcome=ReclaimOutcome.NO_RECLAIM_IN_WINDOW, resolved_at=resolved_at, bars_after_take=3)],
        )
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        no_reclaim_candidates = [c for c in out.candidates if c.trigger_phase == CandidatePhase.NO_RECLAIM_IN_WINDOW]
        self.assertEqual(len(no_reclaim_candidates), 2)
        for c in no_reclaim_candidates:
            self.assertEqual(c.known_at, resolved_at)

    def test_take_and_resolved_reclaim_together_produce_four_candidates(self):
        result = result_with([level()], [take_event()], [reclaim_result()])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        self.assertEqual(len(out.candidates), 4)
        phases = {c.trigger_phase for c in out.candidates}
        self.assertEqual(phases, {CandidatePhase.TAKE, CandidatePhase.RECLAIM_CLOSE})


class PendingTests(unittest.TestCase):
    def test_pending_reclaim_produces_no_additional_candidates(self):
        result = result_with([level()], [take_event()], [reclaim_result(outcome=ReclaimOutcome.PENDING, resolved_at=None, bars_after_take=None)])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        self.assertEqual(len(out.candidates), 2)  # only the TAKE pair
        self.assertTrue(all(c.trigger_phase == CandidatePhase.TAKE for c in out.candidates))

    def test_take_with_no_reclaim_recorded_at_all_produces_only_take_pair(self):
        result = result_with([level()], [take_event()], [])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        self.assertEqual(len(out.candidates), 2)
        self.assertTrue(all(c.trigger_phase == CandidatePhase.TAKE for c in out.candidates))


class DirectionMappingTests(unittest.TestCase):
    def test_liquidity_high_reversal_is_short(self):
        result = result_with([level(side=HIGH)], [take_event(side=HIGH)], [])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        reversal = next(c for c in out.candidates if c.entry_mode == EntryMode.REVERSAL)
        self.assertEqual(reversal.direction, "short")

    def test_liquidity_high_continuation_is_long(self):
        result = result_with([level(side=HIGH)], [take_event(side=HIGH)], [])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        continuation = next(c for c in out.candidates if c.entry_mode == EntryMode.CONTINUATION)
        self.assertEqual(continuation.direction, "long")

    def test_liquidity_low_reversal_is_long(self):
        result = result_with(
            [level(level_id="1m-swing_low-1", liquidity_type=LiquidityType.SWING_LOW, side=LOW)],
            [take_event(level_id="1m-swing_low-1", side=LOW)],
            [],
        )
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        reversal = next(c for c in out.candidates if c.entry_mode == EntryMode.REVERSAL)
        self.assertEqual(reversal.direction, "long")

    def test_liquidity_low_continuation_is_short(self):
        result = result_with(
            [level(level_id="1m-swing_low-1", liquidity_type=LiquidityType.SWING_LOW, side=LOW)],
            [take_event(level_id="1m-swing_low-1", side=LOW)],
            [],
        )
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        continuation = next(c for c in out.candidates if c.entry_mode == EntryMode.CONTINUATION)
        self.assertEqual(continuation.direction, "short")

    def test_reversal_and_continuation_are_always_disjoint(self):
        result = result_with([level()], [take_event()], [reclaim_result()])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        for phase in (CandidatePhase.TAKE, CandidatePhase.RECLAIM_CLOSE):
            pair = [c for c in out.candidates if c.trigger_phase == phase]
            self.assertEqual({c.entry_mode for c in pair}, {EntryMode.REVERSAL, EntryMode.CONTINUATION})
            self.assertNotEqual(pair[0].direction, pair[1].direction)


class SourceFamilyEndToEndTests(unittest.TestCase):
    def test_swing_equal_liquidity_end_to_end(self):
        candles = [
            candle(0, 99, 99.2, 98.8, 99),
            candle(1, 99.3, 100.0, 99.0, 99.1),  # pivot high = 100
            candle(2, 99, 99.2, 98.8, 99),
            candle(3, 99.9, 100.5, 99.8, 100.0),  # take
            candle(4, 100.0, 100.1, 99.7, 99.8),  # reclaim close
        ]
        liquidity_result = detect_liquidity(candles, "1m", pivot_strength=1, reclaim_window_bars=3)
        out = detect_liquidity_entry_candidates({"1m": liquidity_result}, {})
        self.assertGreaterEqual(len(out.candidates), 4)
        self.assertTrue(all(c.source_family == SourceFamily.SWING_EQUAL for c in out.candidates))

    def test_session_liquidity_end_to_end(self):
        from datetime import date, time, timedelta as td
        from zoneinfo import ZoneInfo

        from tradelab.session_liquidity import _SESSION_WINDOWS, _expected_utc_timestamps

        NY = ZoneInfo("America/New_York")
        anchor = date(2026, 1, 6)
        expected = _expected_utc_timestamps(anchor, _SESSION_WINDOWS["london"], td(minutes=1))
        session_candles = [Candle(ts, 100, 100.1, 99.9, 100, ts + td(minutes=1)) for ts in expected]
        # Confirms the session's close.
        confirming = Candle(session_candles[-1].known_at, 100, 100.1, 99.9, 100, session_candles[-1].known_at + td(minutes=1))
        # Breaks the London high (100.1) by 1 tick.
        take_candle = Candle(confirming.known_at, 100, 100.35, 99.9, 100, confirming.known_at + td(minutes=1))

        session_result = detect_session_liquidity(session_candles + [confirming, take_candle], "1m", reclaim_window_bars=3)
        out = detect_liquidity_entry_candidates({}, {"1m": session_result})
        london_high_candidates = [c for c in out.candidates if c.liquidity_type == LiquidityType.SESSION_LONDON_HIGH]
        self.assertEqual(len(london_high_candidates), 2)
        self.assertTrue(all(c.source_family == SourceFamily.SESSION for c in london_high_candidates))


class TimeframeRestrictionTests(unittest.TestCase):
    def test_1m_5m_15m_are_accepted(self):
        for timeframe in ("1m", "5m", "15m"):
            with self.subTest(timeframe=timeframe):
                result = result_with(
                    [level(level_id=f"{timeframe}-swing_high-1", timeframe=timeframe)],
                    [take_event(level_id=f"{timeframe}-swing_high-1", timeframe=timeframe)],
                    [],
                )
                out = detect_liquidity_entry_candidates({timeframe: result}, {})
                self.assertEqual(len(out.candidates), 2)

    def test_higher_timeframes_are_rejected(self):
        for timeframe in ("1H", "4H", "1D"):
            with self.subTest(timeframe=timeframe):
                result = result_with(
                    [level(level_id=f"{timeframe}-swing_high-1", timeframe=timeframe)],
                    [take_event(level_id=f"{timeframe}-swing_high-1", timeframe=timeframe)],
                    [],
                )
                with self.assertRaises(ValueError):
                    detect_liquidity_entry_candidates({timeframe: result}, {})

    def test_real_detect_liquidity_on_1h_is_rejected_by_entry_candidates(self):
        candles = [
            candle(0, 99, 99.2, 98.8, 99),
            candle(1, 99.3, 100.0, 99.0, 99.1),
            candle(2, 99, 99.2, 98.8, 99),
        ]
        liquidity_result = detect_liquidity(candles, "1H", pivot_strength=1, reclaim_window_bars=3)
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1H": liquidity_result}, {})


class KnownAtNoLookaheadTests(unittest.TestCase):
    def test_take_known_at_matches_the_source_event_exactly(self):
        take_known_at = T0 + timedelta(minutes=5)
        result = result_with([level()], [take_event(known_at=take_known_at)], [])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        for c in out.candidates:
            self.assertEqual(c.known_at, take_known_at)
            self.assertEqual(c.timestamp, take_known_at)

    def test_reclaim_known_at_is_strictly_after_take_known_at(self):
        take_known_at = T0 + timedelta(minutes=5)
        resolved_at = T0 + timedelta(minutes=6)
        result = result_with([level()], [take_event(known_at=take_known_at)], [reclaim_result(resolved_at=resolved_at)])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        take_known_ats = {c.known_at for c in out.candidates if c.trigger_phase == CandidatePhase.TAKE}
        reclaim_known_ats = {c.known_at for c in out.candidates if c.trigger_phase == CandidatePhase.RECLAIM_CLOSE}
        self.assertEqual(take_known_ats, {take_known_at})
        self.assertEqual(reclaim_known_ats, {resolved_at})
        self.assertLess(max(take_known_ats), min(reclaim_known_ats))


class ValidationTests(unittest.TestCase):
    def test_take_referencing_missing_level_is_rejected(self):
        result = result_with([], [take_event()], [])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_reclaim_referencing_nonexistent_take_is_rejected(self):
        result = result_with([level()], [], [reclaim_result()])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_level_timeframe_mismatch_with_dict_key_is_rejected(self):
        result = result_with([level(timeframe="5m")], [], [])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_take_timeframe_mismatch_with_dict_key_is_rejected(self):
        result = result_with([level()], [take_event(timeframe="5m")], [])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_duplicate_level_id_within_a_source_is_rejected(self):
        result = result_with([level(), level()], [], [])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_duplicate_take_for_the_same_level_is_rejected(self):
        result = result_with([level()], [take_event(), take_event()], [])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_duplicate_reclaim_for_the_same_level_is_rejected(self):
        result = result_with([level()], [take_event()], [reclaim_result(), reclaim_result()])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_session_liquidity_type_filed_under_swing_equal_is_rejected(self):
        session_level = level(level_id="1m-session_asia_high-1", liquidity_type=LiquidityType.SESSION_ASIA_HIGH)
        result = result_with([session_level], [], [])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_swing_liquidity_type_filed_under_session_is_rejected(self):
        result = result_with([level()], [], [])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({}, {"1m": result})

    # -- Mutations of an otherwise-valid take+reclaim, one field at a time --

    def test_reclaim_with_take_timestamp_not_matching_its_take_is_rejected(self):
        take_known_at = T0 + timedelta(minutes=5)
        result = result_with(
            [level()],
            [take_event(known_at=take_known_at)],
            [reclaim_result(take_timestamp=take_known_at + timedelta(minutes=1))],  # same level_id, wrong take_timestamp
        )
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_terminal_reclaim_missing_resolved_at_is_rejected(self):
        for outcome in (ReclaimOutcome.RECLAIM_WICK, ReclaimOutcome.RECLAIM_CLOSE, ReclaimOutcome.NO_RECLAIM_IN_WINDOW):
            with self.subTest(outcome=outcome):
                result = result_with([level()], [take_event()], [reclaim_result(outcome=outcome, resolved_at=None)])
                with self.assertRaises(ValueError):
                    detect_liquidity_entry_candidates({"1m": result}, {})

    def test_pending_reclaim_with_a_resolved_at_is_rejected(self):
        result = result_with(
            [level()], [take_event()], [reclaim_result(outcome=ReclaimOutcome.PENDING, resolved_at=T0 + timedelta(minutes=6), bars_after_take=None)]
        )
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_take_with_wrong_side_is_rejected(self):
        # level says HIGH; take claims LOW (liquidity_type kept consistent
        # with the level so only `side` itself is the mismatch).
        result = result_with(
            [level(side=HIGH, liquidity_type=LiquidityType.SWING_HIGH)],
            [take_event(side=LOW, liquidity_type=LiquidityType.SWING_HIGH)],
            [],
        )
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_take_with_wrong_liquidity_type_is_rejected(self):
        result = result_with(
            [level(liquidity_type=LiquidityType.SWING_HIGH, side=HIGH)],
            [take_event(liquidity_type=LiquidityType.EQUAL_HIGHS, side=HIGH)],
            [],
        )
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_take_with_wrong_bounds_is_rejected(self):
        result = result_with([level(lower=100.0, upper=100.0)], [take_event(lower=99.0, upper=99.0)], [])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": result}, {})

    def test_reclaim_known_at_or_before_its_take_is_rejected(self):
        take_known_at = T0 + timedelta(minutes=5)
        at_take = result_with([level()], [take_event(known_at=take_known_at)], [reclaim_result(take_timestamp=take_known_at, resolved_at=take_known_at)])
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": at_take}, {})

        before_take = result_with(
            [level()],
            [take_event(known_at=take_known_at)],
            [reclaim_result(take_timestamp=take_known_at, resolved_at=take_known_at - timedelta(minutes=1))],
        )
        with self.assertRaises(ValueError):
            detect_liquidity_entry_candidates({"1m": before_take}, {})


class NoDuplicatesTests(unittest.TestCase):
    def test_candidate_ids_are_unique_across_a_full_result(self):
        result_1m = result_with([level()], [take_event()], [reclaim_result()])
        result_5m = result_with(
            [level(level_id="5m-swing_low-1", liquidity_type=LiquidityType.SWING_LOW, side=LOW, timeframe="5m")],
            [take_event(level_id="5m-swing_low-1", timeframe="5m", side=LOW)],
            [],
        )
        out = detect_liquidity_entry_candidates({"1m": result_1m, "5m": result_5m}, {})
        candidate_ids = [c.candidate_id for c in out.candidates]
        self.assertEqual(len(candidate_ids), len(set(candidate_ids)))
        self.assertEqual(len(candidate_ids), 6)

    def test_candidate_id_is_stable_and_deterministic(self):
        result = result_with([level()], [take_event()], [])
        out1 = detect_liquidity_entry_candidates({"1m": result}, {})
        out2 = detect_liquidity_entry_candidates({"1m": result}, {})
        self.assertEqual([c.candidate_id for c in out1.candidates], [c.candidate_id for c in out2.candidates])


class AuditTests(unittest.TestCase):
    def test_candidate_retains_full_level_audit_fields(self):
        lvl = level(lower=99.5, upper=100.5)
        result = result_with([lvl], [take_event(lower=99.5, upper=100.5)], [])
        out = detect_liquidity_entry_candidates({"1m": result}, {})
        for c in out.candidates:
            self.assertEqual(c.level_id, lvl.level_id)
            self.assertEqual(c.liquidity_type, lvl.liquidity_type)
            self.assertEqual(c.side, lvl.side)
            self.assertEqual(c.lower, lvl.lower)
            self.assertEqual(c.upper, lvl.upper)
            self.assertEqual(c.timeframe, lvl.timeframe)


if __name__ == "__main__":
    unittest.main()
