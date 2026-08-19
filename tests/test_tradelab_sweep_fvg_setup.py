#!/usr/bin/env python3
"""Sweep -> FVG/iFVG setup chain tests: pending-context creation, same-
selector invalidate/replace, one-time consumption (including deterministic
same-instant ties emitted before consumption), fan-out across independent
selectors, every 3x3 sweep-timeframe x FVG/iFVG-timeframe pairing, and no
look-ahead."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.entry_signal_timeline import LONG, SHORT, EntryEventKind
from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, FVGEntryCandidateResult, SourceType
from tradelab.liquidity import HIGH, LOW, LiquidityType
from tradelab.liquidity_entry import CandidatePhase, EntryMode, LiquidityEntryCandidate, LiquidityEntryCandidateResult, SourceFamily
from tradelab.sweep_fvg_setup import (
    PendingSweepTransitionKind,
    build_sweep_fvg_setup_events,
    iter_sweep_pending_transitions,
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def ny_dt(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


T0 = ny_dt(2026, 1, 5, 10, 0)


def fvg_candidate(
    zone_id: str,
    timeframe: str = "1m",
    source_type: SourceType = SourceType.FVG_TOUCH,
    direction: str = LONG,
    known_at: datetime = T0,
) -> FVGEntryCandidate:
    contact_type = FVGContactType.PARTIAL_TOUCH if source_type == SourceType.FVG_TOUCH else None
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{source_type.value}-{zone_id}",
        source_type=source_type,
        direction=direction,
        timeframe=timeframe,
        zone_id=zone_id,
        lower=100.0,
        upper=102.0,
        contact_type=contact_type,
        timestamp=known_at - timedelta(minutes=1),
        known_at=known_at,
    )


def liquidity_candidate(
    level_id: str,
    timeframe: str = "1m",
    side: str = LOW,
    entry_mode: EntryMode = EntryMode.REVERSAL,
    known_at: datetime = T0,
) -> LiquidityEntryCandidate:
    direction = {
        (HIGH, EntryMode.REVERSAL): SHORT,
        (HIGH, EntryMode.CONTINUATION): LONG,
        (LOW, EntryMode.REVERSAL): LONG,
        (LOW, EntryMode.CONTINUATION): SHORT,
    }[(side, entry_mode)]
    return LiquidityEntryCandidate(
        candidate_id=f"{timeframe}-{SourceFamily.SWING_EQUAL.value}-{level_id}-{CandidatePhase.TAKE.value}-{entry_mode.value}",
        source_family=SourceFamily.SWING_EQUAL,
        timeframe=timeframe,
        level_id=level_id,
        liquidity_type=LiquidityType.SWING_HIGH,
        side=side,
        lower=100.0,
        upper=100.0,
        trigger_phase=CandidatePhase.TAKE,
        entry_mode=entry_mode,
        direction=direction,
        timestamp=known_at,
        known_at=known_at,
    )


def fvg_result(*candidates: FVGEntryCandidate) -> FVGEntryCandidateResult:
    return FVGEntryCandidateResult(candidates=list(candidates), zone_mitigations=())


def liquidity_result(*candidates: LiquidityEntryCandidate) -> LiquidityEntryCandidateResult:
    return LiquidityEntryCandidateResult(candidates=list(candidates))


EMPTY_FVG = fvg_result()
EMPTY_LIQUIDITY = liquidity_result()


class BasicSetupTests(unittest.TestCase):
    def test_sweep_then_compatible_fvg_touch_produces_one_setup_event(self):
        t1 = T0 + timedelta(minutes=5)
        sweep = liquidity_candidate("L1", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t1)

        setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(sweep))

        self.assertEqual(len(setups), 1)
        setup = setups[0]
        self.assertEqual(setup.sweep_event.event_id, sweep.candidate_id)
        self.assertEqual(setup.fvg_event.event_id, touch.candidate_id)
        self.assertEqual(setup.fvg_kind, EntryEventKind.FVG_TOUCH)
        self.assertEqual(setup.direction, LONG)
        self.assertEqual(setup.sweep_known_at, T0)
        self.assertEqual(setup.fvg_known_at, t1)
        self.assertEqual(setup.known_at, t1)
        self.assertEqual(setup.setup_id, f"{sweep.candidate_id}::{touch.candidate_id}")

    def test_sweep_then_compatible_ifvg_produces_one_setup_event(self):
        t1 = T0 + timedelta(minutes=5)
        sweep = liquidity_candidate("L1", side=HIGH, entry_mode=EntryMode.CONTINUATION, known_at=T0)  # LONG
        ifvg = fvg_candidate("Z1", source_type=SourceType.IFVG, direction=LONG, known_at=t1)

        setups = build_sweep_fvg_setup_events(fvg_result(ifvg), liquidity_result(sweep))

        self.assertEqual(len(setups), 1)
        self.assertEqual(setups[0].fvg_kind, EntryEventKind.IFVG)
        self.assertEqual(setups[0].fvg_event.event_id, ifvg.candidate_id)

    def test_incompatible_direction_never_creates_a_setup(self):
        t1 = T0 + timedelta(minutes=5)
        sweep = liquidity_candidate("L1", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=SHORT, known_at=t1)

        setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(sweep))

        self.assertEqual(setups, [])


class InvalidateAndReplaceTests(unittest.TestCase):
    def test_opposite_direction_same_selector_sweep_invalidates_the_pending_sweep(self):
        t1 = T0 + timedelta(minutes=2)  # opposite sweep, same selector
        t2 = T0 + timedelta(minutes=5)  # later long-compatible FVG touch

        long_sweep = liquidity_candidate("La", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        short_sweep = liquidity_candidate("Lb", side=HIGH, entry_mode=EntryMode.REVERSAL, known_at=t1)  # SHORT, same selector
        self.assertEqual(
            f"liquidity-{long_sweep.timeframe}-{long_sweep.source_family.value}-{long_sweep.liquidity_type}-"
            f"{long_sweep.trigger_phase.value}-{long_sweep.entry_mode.value}",
            f"liquidity-{short_sweep.timeframe}-{short_sweep.source_family.value}-{short_sweep.liquidity_type}-"
            f"{short_sweep.trigger_phase.value}-{short_sweep.entry_mode.value}",
        )
        touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t2)

        setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(long_sweep, short_sweep))

        self.assertEqual(setups, [])  # the original LONG sweep was invalidated by the opposite-direction one

        transitions = list(iter_sweep_pending_transitions(fvg_result(touch), liquidity_result(long_sweep, short_sweep)))
        invalidated = [t for t in transitions if t.kind == PendingSweepTransitionKind.INVALIDATED]
        self.assertEqual(len(invalidated), 1)
        self.assertEqual(invalidated[0].context.sweep_event.event_id, long_sweep.candidate_id)
        self.assertEqual(invalidated[0].caused_by.event_id, short_sweep.candidate_id)

    def test_newer_same_direction_same_selector_sweep_replaces_the_older_one(self):
        t1 = T0 + timedelta(minutes=2)
        t2 = T0 + timedelta(minutes=5)

        old_sweep = liquidity_candidate("La", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        new_sweep = liquidity_candidate("Lb", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=t1)  # LONG, same selector
        touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t2)

        setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(old_sweep, new_sweep))

        self.assertEqual(len(setups), 1)
        self.assertEqual(setups[0].sweep_event.event_id, new_sweep.candidate_id)  # the replacement, not the original

        transitions = list(iter_sweep_pending_transitions(fvg_result(touch), liquidity_result(old_sweep, new_sweep)))
        replaced = [t for t in transitions if t.kind == PendingSweepTransitionKind.REPLACED]
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0].context.sweep_event.event_id, old_sweep.candidate_id)
        self.assertEqual(replaced[0].caused_by.event_id, new_sweep.candidate_id)


class OneTimeConsumptionTests(unittest.TestCase):
    def test_consumed_sweep_cannot_produce_another_later_setup(self):
        t1 = T0 + timedelta(minutes=5)
        t2 = T0 + timedelta(minutes=10)

        sweep = liquidity_candidate("L1", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        first_touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t1)
        second_touch = fvg_candidate("Z2", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t2)

        setups = build_sweep_fvg_setup_events(fvg_result(first_touch, second_touch), liquidity_result(sweep))

        self.assertEqual(len(setups), 1)
        self.assertEqual(setups[0].fvg_event.event_id, first_touch.candidate_id)

    def test_simultaneous_compatible_ties_are_all_emitted_before_consumption(self):
        t1 = T0 + timedelta(minutes=5)
        t2 = T0 + timedelta(minutes=10)

        sweep = liquidity_candidate("L1", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        touch_a = fvg_candidate("Za", timeframe="1m", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t1)
        touch_b = fvg_candidate("Zb", timeframe="5m", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t1)
        later_touch = fvg_candidate("Zc", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t2)

        setups = build_sweep_fvg_setup_events(fvg_result(touch_a, touch_b, later_touch), liquidity_result(sweep))

        self.assertEqual(len(setups), 2)
        self.assertEqual({s.fvg_event.event_id for s in setups}, {touch_a.candidate_id, touch_b.candidate_id})
        self.assertTrue(all(s.sweep_event.event_id == sweep.candidate_id for s in setups))
        # deterministic order: matches touch_a before touch_b (event_id order)
        self.assertEqual([s.fvg_event.event_id for s in setups], sorted(s.fvg_event.event_id for s in setups))

        transitions = list(iter_sweep_pending_transitions(fvg_result(touch_a, touch_b, later_touch), liquidity_result(sweep)))
        consumed = [t for t in transitions if t.kind == PendingSweepTransitionKind.CONSUMED]
        self.assertEqual(len(consumed), 1)  # one context, consumed once, even though two events tied on it


class FanOutTests(unittest.TestCase):
    def test_one_fvg_event_can_consume_multiple_independent_pending_sweeps(self):
        t1 = T0 + timedelta(minutes=5)
        sweep_1m = liquidity_candidate("L1", timeframe="1m", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        sweep_5m = liquidity_candidate("L2", timeframe="5m", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t1)

        setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(sweep_1m, sweep_5m))

        self.assertEqual(len(setups), 2)
        self.assertEqual({s.sweep_event.event_id for s in setups}, {sweep_1m.candidate_id, sweep_5m.candidate_id})
        self.assertTrue(all(s.fvg_event.event_id == touch.candidate_id for s in setups))


class TimeframePairingTests(unittest.TestCase):
    def test_every_sweep_tf_by_fvg_tf_pairing_is_supported(self):
        for sweep_tf in ("1m", "5m", "15m"):
            for fvg_tf in ("1m", "5m", "15m"):
                with self.subTest(sweep_tf=sweep_tf, fvg_tf=fvg_tf):
                    t1 = T0 + timedelta(minutes=5)
                    sweep = liquidity_candidate(f"L-{sweep_tf}-{fvg_tf}", timeframe=sweep_tf, side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)
                    touch = fvg_candidate(f"Z-{sweep_tf}-{fvg_tf}", timeframe=fvg_tf, source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t1)

                    setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(sweep))

                    self.assertEqual(len(setups), 1, f"sweep_tf={sweep_tf} fvg_tf={fvg_tf}")
                    self.assertEqual(setups[0].sweep_timeframe, sweep_tf)
                    self.assertEqual(setups[0].fvg_timeframe, fvg_tf)


class SameInstantPairingTests(unittest.TestCase):
    def test_sweep_and_compatible_fvg_at_the_exact_same_known_at_still_form_a_setup(self):
        sweep = liquidity_candidate("L1", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=T0)

        setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(sweep))

        self.assertEqual(len(setups), 1)
        self.assertEqual(setups[0].sweep_known_at, T0)
        self.assertEqual(setups[0].fvg_known_at, T0)


class NoLookaheadTests(unittest.TestCase):
    def test_a_sweep_known_after_an_fvg_event_never_retroactively_forms_a_setup(self):
        t1 = T0 + timedelta(minutes=5)  # FVG touch, known first
        t2 = T0 + timedelta(minutes=10)  # sweep, known later -- too late for the earlier touch

        touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t1)
        sweep = liquidity_candidate("L1", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=t2)  # LONG

        setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(sweep))

        self.assertEqual(setups, [])

    def test_every_setup_known_at_is_at_or_after_its_own_sweeps_known_at(self):
        t1 = T0 + timedelta(minutes=5)
        sweep = liquidity_candidate("L1", side=LOW, entry_mode=EntryMode.REVERSAL, known_at=T0)  # LONG
        touch = fvg_candidate("Z1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=t1)

        setups = build_sweep_fvg_setup_events(fvg_result(touch), liquidity_result(sweep))

        self.assertEqual(len(setups), 1)
        self.assertGreaterEqual(setups[0].known_at, setups[0].sweep_known_at)


class EmptyInputTests(unittest.TestCase):
    def test_no_sweeps_or_fvg_events_yields_nothing(self):
        self.assertEqual(build_sweep_fvg_setup_events(EMPTY_FVG, EMPTY_LIQUIDITY), [])


if __name__ == "__main__":
    unittest.main()
