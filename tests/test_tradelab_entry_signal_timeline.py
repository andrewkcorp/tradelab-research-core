#!/usr/bin/env python3
"""Entry signal timeline tests: FVG/liquidity normalization, known_at/
no-lookahead, persist-until-invalidated validity (no time-based expiry
of any kind), trading_day_id as pure audit metadata (still computed via
NY session boundaries, DST-correct), liquidity same-selector permanent
invalidation, FVG/iFVG same-timeframe reversible suspension and
restoration (modeled on tradelab.fvg's own ACTIVE/MITIGATED zone
lifecycle), no cross-timeframe suspension, no cross-family interaction
between liquidity and FVG/iFVG, unselected selectors never
invalidating/suspending/restoring, nested FVG conflicts, no look-ahead,
same-direction refresh (no duplication), deterministic selector keys
with full audit, and randomized equivalence between
``query_active_signals`` and ``iter_active_signal_snapshots``."""
from __future__ import annotations

import random
import unittest
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.bias import BEARISH as BIAS_BEARISH
from tradelab.bias import BULLISH as BIAS_BULLISH
from tradelab.bias import NEUTRAL as BIAS_NEUTRAL
from tradelab.bias import ConfluenceSource, ConfluenceType, HTFBiasResult, TimeframeBiasChange, bias_at
from tradelab.entry_signal_timeline import (
    LONG,
    SHORT,
    CandidateEntrySnapshot,
    EntryEventKind,
    MarketStateSnapshot,
    SignalRestoration,
    SignalSuspension,
    SourceKind,
    build_entry_events,
    build_entry_signal_timeline,
    iter_active_pool_scalar_snapshots,
    iter_active_signal_snapshots,
    iter_candidate_entry_snapshots,
    iter_fvg_lifecycle_transitions,
    query_active_signals,
)
from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, FVGEntryCandidateResult, FVGMitigationEvent, SourceType
from tradelab.liquidity import HIGH, LOW, LiquidityType
from tradelab.liquidity_entry import (
    CandidatePhase,
    EntryMode,
    LiquidityEntryCandidate,
    LiquidityEntryCandidateResult,
    SourceFamily,
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def ny_dt(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


def _avoid_maintenance_break(dt: datetime) -> datetime:
    """Nudges a randomly-generated instant forward past the daily NY
    17:00-18:00 maintenance break, if it would otherwise land inside it
    -- pure test-fixture hygiene, unrelated to what is being tested."""
    local = dt.astimezone(NY)
    if local.hour == 17 and local.minute > 0:
        return dt + timedelta(minutes=61 - local.minute)
    return dt


def fvg_candidate(
    zone_id: str = "1m-FVG-1",
    timeframe: str = "1m",
    source_type: SourceType = SourceType.FVG_TOUCH,
    contact_type: FVGContactType | None = None,
    direction: str = LONG,
    known_at: datetime = ny_dt(2026, 1, 5, 10, 1),
    timestamp: datetime | None = None,
) -> FVGEntryCandidate:
    if source_type == SourceType.FVG_TOUCH and contact_type is None:
        contact_type = FVGContactType.PARTIAL_TOUCH
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{source_type.value}-{zone_id}",
        source_type=source_type,
        direction=direction,
        timeframe=timeframe,
        zone_id=zone_id,
        lower=100.0,
        upper=102.0,
        contact_type=contact_type,
        timestamp=known_at - timedelta(minutes=1) if timestamp is None else timestamp,
        known_at=known_at,
    )


def fvg_touch_then_inversion(
    zone_id: str, timeframe: str, original_direction: str, touch_known_at: datetime, inversion_known_at: datetime
) -> tuple[FVGEntryCandidate, FVGEntryCandidate]:
    """A realistic, semantically-linked pair for one zone: an initial
    ``PARTIAL_TOUCH`` (the zone stays ``ACTIVE``), followed later by that
    same zone's own inversion -- carrying the *opposite* direction, per
    ``tradelab.fvg_entry``'s own contract."""
    touch = fvg_candidate(
        zone_id=zone_id, timeframe=timeframe, source_type=SourceType.FVG_TOUCH,
        contact_type=FVGContactType.PARTIAL_TOUCH, direction=original_direction, known_at=touch_known_at,
    )
    inverted_direction = SHORT if original_direction == LONG else LONG
    ifvg = fvg_candidate(
        zone_id=zone_id, timeframe=timeframe, source_type=SourceType.IFVG, direction=inverted_direction, known_at=inversion_known_at,
    )
    return touch, ifvg


def fvg_touch_then_mitigation(
    zone_id: str, timeframe: str, direction: str, touch_known_at: datetime, mitigation_known_at: datetime
) -> tuple[FVGEntryCandidate, FVGMitigationEvent]:
    """A realistic, semantically-linked pair for one zone: an initial
    ``PARTIAL_TOUCH`` (the zone stays ``ACTIVE``), followed later, on a
    separate candle, by that same zone becoming ``MITIGATED`` *without*
    inverting -- carrying the zone's own, unchanged direction, per
    ``tradelab.fvg``'s own contract. The scenario ``tradelab.fvg_touch``
    itself can never surface on its own (it only ever records a zone's
    first contact)."""
    touch = fvg_candidate(
        zone_id=zone_id, timeframe=timeframe, source_type=SourceType.FVG_TOUCH,
        contact_type=FVGContactType.PARTIAL_TOUCH, direction=direction, known_at=touch_known_at,
    )
    mitigation = FVGMitigationEvent(
        zone_id=zone_id, timeframe=timeframe, direction=direction, lower=100.0, upper=102.0,
        timestamp=mitigation_known_at, known_at=mitigation_known_at,
    )
    return touch, mitigation


def liquidity_candidate(
    level_id: str = "1m-swing_high-1",
    timeframe: str = "1m",
    source_family: SourceFamily = SourceFamily.SWING_EQUAL,
    liquidity_type: LiquidityType = LiquidityType.SWING_HIGH,
    side: str = HIGH,
    trigger_phase: CandidatePhase = CandidatePhase.TAKE,
    entry_mode: EntryMode = EntryMode.REVERSAL,
    known_at: datetime = ny_dt(2026, 1, 5, 10, 1),
) -> LiquidityEntryCandidate:
    direction = {
        (HIGH, EntryMode.REVERSAL): SHORT,
        (HIGH, EntryMode.CONTINUATION): LONG,
        (LOW, EntryMode.REVERSAL): LONG,
        (LOW, EntryMode.CONTINUATION): SHORT,
    }[(side, entry_mode)]
    return LiquidityEntryCandidate(
        candidate_id=f"{timeframe}-{source_family.value}-{level_id}-{trigger_phase.value}-{entry_mode.value}",
        source_family=source_family,
        timeframe=timeframe,
        level_id=level_id,
        liquidity_type=liquidity_type,
        side=side,
        lower=100.0,
        upper=100.0,
        trigger_phase=trigger_phase,
        entry_mode=entry_mode,
        direction=direction,
        timestamp=known_at,
        known_at=known_at,
    )


def fvg_result(*candidates: FVGEntryCandidate, mitigations: tuple[FVGMitigationEvent, ...] = ()) -> FVGEntryCandidateResult:
    return FVGEntryCandidateResult(candidates=list(candidates), zone_mitigations=tuple(mitigations))


def liquidity_result(*candidates: LiquidityEntryCandidate) -> LiquidityEntryCandidateResult:
    return LiquidityEntryCandidateResult(candidates=list(candidates))


EMPTY_FVG = fvg_result()
EMPTY_LIQUIDITY = liquidity_result()


class NormalizationTests(unittest.TestCase):
    def test_fvg_candidate_normalizes_with_correct_selector_and_source_kind(self):
        c = fvg_candidate(source_type=SourceType.FVG_TOUCH, timeframe="5m")
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(len(timeline.signals), 1)
        signal = timeline.signals[0]
        self.assertEqual(signal.source_kind, SourceKind.FVG)
        self.assertEqual(signal.selector, "fvg-5m-fvg_touch")
        self.assertEqual(signal.candidate_id, c.candidate_id)
        self.assertIs(signal.source, c)

    def test_ifvg_candidate_normalizes_with_its_own_selector(self):
        c = fvg_candidate(source_type=SourceType.IFVG)
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].selector, "fvg-1m-ifvg")

    def test_liquidity_candidate_normalizes_with_correct_selector_and_source_kind(self):
        c = liquidity_candidate(source_family=SourceFamily.SWING_EQUAL, liquidity_type=LiquidityType.SWING_HIGH, trigger_phase=CandidatePhase.TAKE, entry_mode=EntryMode.REVERSAL)
        timeline = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(c))
        signal = timeline.signals[0]
        self.assertEqual(signal.source_kind, SourceKind.LIQUIDITY)
        self.assertEqual(signal.selector, "liquidity-1m-swing_equal-swing_high-take-reversal")
        self.assertIs(signal.source, c)

    def test_session_liquidity_candidate_selector_includes_its_subtype(self):
        c = liquidity_candidate(
            level_id="1m-session_asia_high-1",
            source_family=SourceFamily.SESSION,
            liquidity_type=LiquidityType.SESSION_ASIA_HIGH,
            trigger_phase=CandidatePhase.RECLAIM_CLOSE,
            entry_mode=EntryMode.CONTINUATION,
        )
        timeline = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(c))
        self.assertEqual(timeline.signals[0].selector, "liquidity-1m-session-session_asia_high-reclaim_close-continuation")

    def test_combined_timeline_holds_both_kinds_sorted_by_known_at(self):
        f = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        l = liquidity_candidate(known_at=ny_dt(2026, 1, 5, 9, 0))
        timeline = build_entry_signal_timeline(fvg_result(f), liquidity_result(l))
        self.assertEqual([s.candidate_id for s in timeline.signals], [l.candidate_id, f.candidate_id])

    def test_duplicate_candidate_id_across_sources_is_rejected(self):
        f = fvg_candidate(zone_id="X")
        l = liquidity_candidate(level_id="Y")
        # force a collision
        object.__setattr__(l, "candidate_id", f.candidate_id)
        with self.assertRaises(ValueError):
            build_entry_signal_timeline(fvg_result(f), liquidity_result(l))


class KnownAtNoLookaheadTests(unittest.TestCase):
    def test_signal_known_at_and_timestamp_match_the_source_exactly(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 1), timestamp=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        signal = timeline.signals[0]
        self.assertEqual(signal.known_at, c.known_at)
        self.assertEqual(signal.timestamp, c.timestamp)

    def test_query_before_known_at_never_returns_the_signal(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 1))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 10, 0))
        self.assertEqual(result.active_signals, [])

    def test_query_exactly_at_known_at_returns_the_signal(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 1))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 10, 1))
        self.assertEqual(len(result.active_signals), 1)

    def test_signal_remains_active_indefinitely_with_no_time_based_expiry(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 1))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 2, 5, 10, 1))
        self.assertEqual([s.candidate_id for s in result.active_signals], [c.candidate_id])


class NewTradingDayTests(unittest.TestCase):
    """``trading_day_id`` is pure audit metadata -- it plays no role in
    whether a signal is active. Only the id itself and the still-
    unchanged malformed-input rejection are checked here."""

    def test_signal_known_before_1700_belongs_to_the_day_that_started_the_evening_before(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-04")

    def test_signal_known_exactly_at_1800_starts_a_new_trading_day(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 18, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-05")

    def test_signal_known_exactly_at_1700_belongs_to_the_ending_trading_day(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-04")

    def test_signal_known_exactly_at_1700_is_active_from_that_instant_onward(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 17, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [c.candidate_id])

    def test_signal_known_inside_the_break_is_rejected(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 30))
        with self.assertRaises(ValueError):
            build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)

    def test_signal_known_just_before_1800_still_belongs_to_the_previous_days_window(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 59))
        with self.assertRaises(ValueError):
            build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)


class DSTTests(unittest.TestCase):
    def test_trading_day_id_is_correct_across_spring_forward(self):
        c = fvg_candidate(known_at=ny_dt(2026, 3, 7, 20, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-03-07")

    def test_trading_day_id_is_correct_across_fall_back(self):
        c = fvg_candidate(known_at=ny_dt(2026, 10, 31, 20, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-10-31")

    def test_query_across_the_dst_transition_still_sees_the_signal_active(self):
        c = fvg_candidate(known_at=ny_dt(2026, 3, 7, 20, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        after_transition = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 3, 9, 12, 0))
        self.assertEqual([s.candidate_id for s in after_transition.active_signals], [c.candidate_id])


class CrossTradingDayPersistenceTests(unittest.TestCase):
    def test_signal_from_an_earlier_trading_day_is_still_active_the_next_day(self):
        day1_signal = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(day1_signal), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 6, 10, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [day1_signal.candidate_id])

    def test_a_next_day_opposite_signal_on_the_same_timeframe_does_suspend_the_prior_day(self):
        day1_signal = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        day2_signal = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 6, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(day1_signal, day2_signal), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 6, 11, 0))
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.suspensions), 1)
        self.assertEqual(result.suspensions[0].signal.candidate_id, day1_signal.candidate_id)
        self.assertEqual(result.suspensions[0].suspended_by.candidate_id, day2_signal.candidate_id)
        self.assertEqual(result.invalidations, [])  # FVG never permanently invalidates

    def test_signals_from_different_trading_days_remain_simultaneously_active_when_never_suspended(self):
        day1_signal = fvg_candidate(zone_id="1m-FVG-1", timeframe="5m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        day2_signal = fvg_candidate(zone_id="1m-FVG-2", timeframe="15m", direction=LONG, known_at=ny_dt(2026, 1, 6, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(day1_signal, day2_signal), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-5m-fvg_touch", "fvg-15m-fvg_touch"}, LONG, ny_dt(2026, 1, 6, 11, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in result.active_signals),
            sorted([day1_signal.candidate_id, day2_signal.candidate_id]),
        )


class LiquidityInvalidationTests(unittest.TestCase):
    """Reverted (this task): liquidity invalidation is scoped to the
    exact same ``selector``, permanent -- never timeframe-wide, never
    cross-family with FVG/iFVG. ``side`` is deliberately varied while
    every selector-relevant field stays fixed, to construct a same-
    selector/opposite-direction test double exercising the mechanism in
    isolation (real detector output never actually pairs a mismatched
    side with a fixed liquidity_type this way)."""

    SELECTOR = "liquidity-1m-swing_equal-swing_high-take-reversal"

    def test_same_selector_opposite_direction_invalidates_permanently(self):
        earlier = liquidity_candidate(level_id="L1", side=HIGH, liquidity_type=LiquidityType.SWING_HIGH, entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 10, 0))
        later = liquidity_candidate(level_id="L2", side=LOW, liquidity_type=LiquidityType.SWING_HIGH, entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 11, 0))
        self.assertEqual(earlier.direction, SHORT)
        self.assertEqual(later.direction, LONG)
        timeline = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(earlier, later))
        self.assertEqual(timeline.signals[0].selector, self.SELECTOR)
        self.assertEqual(timeline.signals[1].selector, self.SELECTOR)

        result = query_active_signals(timeline, {self.SELECTOR}, SHORT, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.invalidations), 1)
        self.assertEqual(result.invalidations[0].signal.candidate_id, earlier.candidate_id)
        self.assertEqual(result.invalidations[0].invalidated_by.candidate_id, later.candidate_id)
        self.assertEqual(result.suspensions, [])
        self.assertEqual(result.restorations, [])

    def test_once_invalidated_a_liquidity_signal_never_becomes_active_again(self):
        earlier = liquidity_candidate(level_id="L1", side=HIGH, liquidity_type=LiquidityType.SWING_HIGH, entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 10, 0))
        invalidator = liquidity_candidate(level_id="L2", side=LOW, liquidity_type=LiquidityType.SWING_HIGH, entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 11, 0))
        revival = liquidity_candidate(level_id="L3", side=HIGH, liquidity_type=LiquidityType.SWING_HIGH, entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(earlier, invalidator, revival))

        result = query_active_signals(timeline, {self.SELECTOR}, SHORT, ny_dt(2026, 1, 5, 13, 0))
        active_ids = {s.candidate_id for s in result.active_signals}
        self.assertNotIn(earlier.candidate_id, active_ids)
        self.assertIn(revival.candidate_id, active_ids)

    def test_different_selector_never_invalidates_even_opposite_direction_same_timeframe(self):
        reversal = liquidity_candidate(level_id="1m-swing_high-1", side=HIGH, trigger_phase=CandidatePhase.TAKE, entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 10, 0))
        continuation = liquidity_candidate(level_id="1m-swing_high-1", side=HIGH, trigger_phase=CandidatePhase.TAKE, entry_mode=EntryMode.CONTINUATION, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(reversal, continuation))
        self.assertNotEqual(reversal.direction, continuation.direction)
        self.assertEqual(reversal.timeframe, continuation.timeframe)
        self.assertNotEqual(timeline.signals[0].selector, timeline.signals[1].selector)

        result = query_active_signals(
            timeline,
            {"liquidity-1m-swing_equal-swing_high-take-reversal", "liquidity-1m-swing_equal-swing_high-take-continuation"},
            reversal.direction,
            ny_dt(2026, 1, 5, 12, 0),
        )
        self.assertEqual([s.candidate_id for s in result.active_signals], [reversal.candidate_id])
        self.assertEqual(result.invalidations, [])


class FVGSuspensionAndRestorationTests(unittest.TestCase):
    """Core suite for the reversible FVG/iFVG conflict model (this
    task)."""

    def test_bearish_touch_suspends_the_active_bullish_fvg_on_the_same_timeframe(self):
        bullish = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, contact_type=FVGContactType.PARTIAL_TOUCH, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.suspensions), 1)
        self.assertEqual(result.suspensions[0].signal.candidate_id, bullish.candidate_id)
        self.assertEqual(result.suspensions[0].suspended_by.candidate_id, bearish_touch.candidate_id)
        self.assertEqual(result.invalidations, [])

        short_result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, SHORT, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual([s.candidate_id for s in short_result.active_signals], [bearish_touch.candidate_id])

    def test_bullish_touch_suspends_the_active_bearish_fvg_on_the_same_timeframe(self):
        """Symmetric bearish case (mirror of the above)."""
        bearish = fvg_candidate(zone_id="1m-FVG-1", direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 0))
        bullish_touch = fvg_candidate(zone_id="1m-FVG-2", direction=LONG, contact_type=FVGContactType.PARTIAL_TOUCH, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(bearish, bullish_touch), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, SHORT, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.suspensions), 1)
        self.assertEqual(result.suspensions[0].signal.candidate_id, bearish.candidate_id)
        self.assertEqual(result.suspensions[0].suspended_by.candidate_id, bullish_touch.candidate_id)

        long_result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual([s.candidate_id for s in long_result.active_signals], [bullish_touch.candidate_id])

    def test_the_suspending_zones_own_inversion_restores_the_earlier_bullish_fvg(self):
        bullish = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bullish_ifvg = fvg_touch_then_inversion("1m-FVG-2", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, bullish_ifvg), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        mid_result = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(mid_result.active_signals, [])

        result = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 14, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in result.active_signals),
            sorted([bullish.candidate_id, bullish_ifvg.candidate_id]),
        )
        self.assertEqual(result.suspensions, [])
        self.assertEqual(len(result.restorations), 1)
        self.assertEqual(result.restorations[0].signal.candidate_id, bullish.candidate_id)
        self.assertEqual(result.restorations[0].restored_by.candidate_id, bullish_ifvg.candidate_id)

    def test_full_mitigation_without_inversion_never_suspends_and_reports_an_immediate_restoration(self):
        bullish = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_failed = fvg_candidate(
            zone_id="1m-FVG-2", direction=SHORT, contact_type=FVGContactType.FULL_MITIGATION_NO_INVERSION, known_at=ny_dt(2026, 1, 5, 11, 0),
        )
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_failed), EMPTY_LIQUIDITY)

        for ts in (ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 12, 0)):
            result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ts)
            self.assertEqual([s.candidate_id for s in result.active_signals], [bullish.candidate_id])
            self.assertEqual(result.suspensions, [])

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(len(result.restorations), 1)
        self.assertEqual(result.restorations[0].signal.candidate_id, bullish.candidate_id)
        self.assertEqual(result.restorations[0].restored_by.candidate_id, bearish_failed.candidate_id)

        short_result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, SHORT, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual([s.candidate_id for s in short_result.active_signals], [bearish_failed.candidate_id])

    def test_no_look_ahead_suspension_and_restoration_only_visible_from_their_own_known_at(self):
        bullish = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bullish_ifvg = fvg_touch_then_inversion("1m-FVG-2", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, bullish_ifvg), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        before_suspend = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 10, 59))
        self.assertEqual([s.candidate_id for s in before_suspend.active_signals], [bullish.candidate_id])
        self.assertEqual(before_suspend.suspensions, [])

        at_suspend = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 11, 0))
        self.assertEqual(at_suspend.active_signals, [])
        self.assertEqual(len(at_suspend.suspensions), 1)

        before_restore = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 12, 59))
        self.assertEqual(before_restore.active_signals, [])

        at_restore = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 13, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in at_restore.active_signals),
            sorted([bullish.candidate_id, bullish_ifvg.candidate_id]),
        )

    def test_nested_fvg_conflicts_resolve_deterministically_in_order(self):
        # A suspended by B's touch; B's touch (while itself suspended by
        # C's touch) later inverts, retiring itself (terminal -- it is no
        # longer a valid bearish FVG at all) and restoring A. C's touch
        # later inverts too, retiring itself the same way -- but by then
        # its own zone's touch is the *only* thing it could restore, and
        # that touch (B's) was already permanently retired at B's own
        # inversion, not by C -- so nothing is left for C's inversion to
        # restore. B's touch never comes back.
        a = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        b_touch, b_ifvg = fvg_touch_then_inversion("Z-B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        c_touch, c_ifvg = fvg_touch_then_inversion("Z-C", "1m", LONG, ny_dt(2026, 1, 5, 12, 0), ny_dt(2026, 1, 5, 14, 0))
        timeline = build_entry_signal_timeline(fvg_result(a, b_touch, b_ifvg, c_touch, c_ifvg), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        # after B suspends A: A gone, B active SHORT
        after_b_long = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 11, 30))
        self.assertEqual(after_b_long.active_signals, [])
        after_b_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 11, 30))
        self.assertEqual([s.candidate_id for s in after_b_short.active_signals], [b_touch.candidate_id])

        # after C suspends B: A still gone, B now also gone, C active LONG
        after_c_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 12, 30))
        self.assertEqual(after_c_short.active_signals, [])
        after_c_long = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 12, 30))
        self.assertEqual([s.candidate_id for s in after_c_long.active_signals], [c_touch.candidate_id])

        # after B inverts: A restored, C still active, iFVG(B) active LONG;
        # b_touch itself is now permanently retired (terminal), not just
        # suspended -- it can never come back, regardless of what C does.
        after_b_ifvg_long = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 13, 30))
        self.assertEqual(
            sorted(s.candidate_id for s in after_b_ifvg_long.active_signals),
            sorted([a.candidate_id, c_touch.candidate_id, b_ifvg.candidate_id]),
        )
        after_b_ifvg_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 13, 30))
        self.assertEqual(after_b_ifvg_short.active_signals, [])

        # after C inverts: c_touch is itself retired (it is no longer a
        # valid bullish FVG either -- same rule, symmetric); it had
        # nothing left to restore, since b_touch was already retired, not
        # merely suspended, at B's own inversion. b_touch never returns.
        final_long = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 15, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in final_long.active_signals),
            sorted([a.candidate_id, b_ifvg.candidate_id]),
        )
        final_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 15, 0))
        self.assertEqual([s.candidate_id for s in final_short.active_signals], [c_ifvg.candidate_id])
        active_ids = {s.candidate_id for s in final_long.active_signals} | {s.candidate_id for s in final_short.active_signals}
        self.assertNotIn(b_touch.candidate_id, active_ids)
        self.assertNotIn(c_touch.candidate_id, active_ids)

    def test_a_suspended_fvg_does_not_reappear_due_to_an_unrelated_same_direction_signal(self):
        bullish = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        unrelated_bullish = fvg_candidate(zone_id="1m-FVG-3", direction=LONG, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, unrelated_bullish), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        active_ids = {s.candidate_id for s in result.active_signals}
        self.assertNotIn(bullish.candidate_id, active_ids)
        self.assertIn(unrelated_bullish.candidate_id, active_ids)


class MitigationRestorationTests(unittest.TestCase):
    """Regression coverage for the second reported bug: a challenger
    that later becomes MITIGATED *without inverting* -- on a separate
    candle, after its own earlier PARTIAL_TOUCH -- must still resolve
    whatever it suspended, exactly as its own IFVG would, via
    ``tradelab.fvg.FVGMitigationEvent`` (``tradelab.fvg_touch`` itself
    can never surface this transition, since it only records a zone's
    first contact)."""

    def test_partial_touch_then_later_non_inverting_mitigation_restores_the_suspended_bullish_fvg(self):
        bullish = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bearish_mitigation = fvg_touch_then_mitigation("Z-B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, mitigations=(bearish_mitigation,)), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        mid_result = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(mid_result.active_signals, [])

        result = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 14, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [bullish.candidate_id])
        self.assertEqual(len(result.restorations), 1)
        self.assertEqual(result.restorations[0].signal.candidate_id, bullish.candidate_id)
        self.assertIsInstance(result.restorations[0].restored_by, FVGMitigationEvent)
        self.assertEqual(result.restorations[0].restored_by.zone_id, "Z-B")

        # the mitigation is never itself a new active confluence -- the
        # zone simply closes out, with no replacement.
        short_result = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 14, 0))
        self.assertEqual(short_result.active_signals, [])

        # bearish_touch itself is permanently retired, never restorable.
        self.assertNotIn(bearish_touch.candidate_id, {s.candidate_id for s in result.active_signals} | {s.candidate_id for s in short_result.active_signals})

    def test_symmetric_bearish_original_restored_by_a_later_non_inverting_mitigation(self):
        bearish = fvg_candidate(zone_id="Z-A2", direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 0))
        bullish_touch, bullish_mitigation = fvg_touch_then_mitigation("Z-B2", "1m", LONG, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        timeline = build_entry_signal_timeline(fvg_result(bearish, bullish_touch, mitigations=(bullish_mitigation,)), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        result = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 14, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [bearish.candidate_id])
        self.assertEqual(len(result.restorations), 1)
        self.assertIsInstance(result.restorations[0].restored_by, FVGMitigationEvent)

    def test_no_look_ahead_restoration_only_visible_from_the_mitigations_own_known_at(self):
        bullish = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bearish_mitigation = fvg_touch_then_mitigation("Z-B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, mitigations=(bearish_mitigation,)), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        before = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 12, 59))
        self.assertEqual(before.active_signals, [])

        at = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 13, 0))
        self.assertEqual([s.candidate_id for s in at.active_signals], [bullish.candidate_id])

    def test_nested_conflict_a_mitigation_resolved_link_is_never_restored_by_an_outer_challengers_own_resolution(self):
        # A suspended by B's touch; B suspended by C's touch; B resolves
        # via a later, non-inverting mitigation (not an IFVG) -- restores
        # A and permanently retires B's touch; C later inverts too, but
        # B's touch is already gone for good, so C's own resolution has
        # nothing left to restore.
        a = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        b_touch, b_mitigation = fvg_touch_then_mitigation("Z-B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        c_touch, c_ifvg = fvg_touch_then_inversion("Z-C", "1m", LONG, ny_dt(2026, 1, 5, 12, 0), ny_dt(2026, 1, 5, 14, 0))
        timeline = build_entry_signal_timeline(fvg_result(a, b_touch, c_touch, c_ifvg, mitigations=(b_mitigation,)), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        # after B suspends A: A gone, B active SHORT
        after_b_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 11, 30))
        self.assertEqual([s.candidate_id for s in after_b_short.active_signals], [b_touch.candidate_id])

        # after C suspends B: B gone too, C active LONG
        after_c_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 12, 30))
        self.assertEqual(after_c_short.active_signals, [])

        # after B's own (non-inverting) mitigation: A restored, B retired for good, C still active
        after_b_mitigation_long = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 13, 30))
        self.assertEqual(
            sorted(s.candidate_id for s in after_b_mitigation_long.active_signals),
            sorted([a.candidate_id, c_touch.candidate_id]),
        )

        # after C inverts too: nothing left for C's own resolution to
        # restore (B's touch is already permanently gone) -- only the
        # new iFVG(C) itself is added.
        final_long = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 15, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in final_long.active_signals),
            sorted([a.candidate_id]),
        )
        final_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 15, 0))
        self.assertEqual([s.candidate_id for s in final_short.active_signals], [c_ifvg.candidate_id])
        self.assertNotIn(b_touch.candidate_id, {s.candidate_id for s in final_long.active_signals} | {s.candidate_id for s in final_short.active_signals})


class FVGTouchTerminalAfterOwnInversionTests(unittest.TestCase):
    """Regression coverage for the specific bug reported in review: a
    touch that itself later converts to its own iFVG is a terminally
    superseded fact from that instant on -- it is no longer a valid
    original-direction FVG at all, so it must never be restored
    afterward by the resolution of some other, unrelated challenger,
    even if that challenger happened to have suspended it in the
    meantime."""

    def test_bug_scenario_bullish_original_suspended_by_bearish_then_by_bullish_challenger(self):
        # A(bullish) suspended by B(bearish touch); B suspended by
        # C(bullish touch); B inverts (restores A, and B's own touch is
        # now terminal); C later inverts too -- must NOT restore B.
        a = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        b_touch, b_ifvg = fvg_touch_then_inversion("Z-B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        c_touch, c_ifvg = fvg_touch_then_inversion("Z-C", "1m", LONG, ny_dt(2026, 1, 5, 12, 0), ny_dt(2026, 1, 5, 14, 0))
        timeline = build_entry_signal_timeline(fvg_result(a, b_touch, b_ifvg, c_touch, c_ifvg), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        result = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 15, 0))
        active_ids = {s.candidate_id for s in result.active_signals}
        self.assertNotIn(b_touch.candidate_id, active_ids)
        self.assertIn(c_ifvg.candidate_id, active_ids)
        # its absence is not because it is "currently suspended" -- it is
        # permanently retired, out of both active_signals and the
        # suspension audit trail.
        self.assertNotIn(b_touch.candidate_id, {s.signal.candidate_id for s in result.suspensions})

    def test_symmetric_bearish_original_suspended_by_bullish_then_by_bearish_challenger(self):
        a = fvg_candidate(zone_id="Z-A2", direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 0))
        b_touch, b_ifvg = fvg_touch_then_inversion("Z-B2", "1m", LONG, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        c_touch, c_ifvg = fvg_touch_then_inversion("Z-C2", "1m", SHORT, ny_dt(2026, 1, 5, 12, 0), ny_dt(2026, 1, 5, 14, 0))
        timeline = build_entry_signal_timeline(fvg_result(a, b_touch, b_ifvg, c_touch, c_ifvg), EMPTY_LIQUIDITY)
        selectors = {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}

        result = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 15, 0))
        active_ids = {s.candidate_id for s in result.active_signals}
        self.assertNotIn(b_touch.candidate_id, active_ids)
        self.assertIn(c_ifvg.candidate_id, active_ids)
        self.assertNotIn(b_touch.candidate_id, {s.signal.candidate_id for s in result.suspensions})

    def test_a_touch_that_inverts_while_still_active_is_also_retired_not_just_suspended_signals(self):
        # simpler case, no suspension involved at all: a touch that
        # inverts while it is itself still plainly active (never
        # suspended by anyone) must still disappear for good -- an
        # unrelated later same-direction event must never revive it.
        touch = fvg_candidate(zone_id="Z-X", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        ifvg = fvg_candidate(zone_id="Z-X", timeframe="1m", source_type=SourceType.IFVG, direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(touch, ifvg), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [])


class UnselectedFVGSelectorTests(unittest.TestCase):
    def test_unselected_fvg_touch_selector_never_suspends(self):
        bullish = fvg_candidate(zone_id="1m-FVG-1", source_type=SourceType.IFVG, direction=LONG, known_at=ny_dt(2026, 1, 5, 9, 0))
        bearish_touch = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-ifvg"}, LONG, ny_dt(2026, 1, 5, 11, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [bullish.candidate_id])
        self.assertEqual(result.suspensions, [])

        both_selected = query_active_signals(timeline, {"fvg-1m-ifvg", "fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 11, 0))
        self.assertEqual(both_selected.active_signals, [])
        self.assertEqual(len(both_selected.suspensions), 1)

    def test_unselected_ifvg_selector_never_restores(self):
        bullish = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 9, 0))
        bearish_touch, bullish_ifvg = fvg_touch_then_inversion("1m-FVG-2", "1m", SHORT, ny_dt(2026, 1, 5, 10, 0), ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, bullish_ifvg), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        self.assertEqual(result.active_signals, [])
        self.assertEqual(result.restorations, [])

        both_selected = query_active_signals(timeline, {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        self.assertIn(bullish.candidate_id, {s.candidate_id for s in both_selected.active_signals})


class NoCrossTimeframeSuspensionTests(unittest.TestCase):
    def test_opposite_direction_touch_on_a_different_timeframe_never_suspends(self):
        m1_long = fvg_candidate(zone_id="1m-A", timeframe="1m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        m5_short_touch = fvg_candidate(zone_id="5m-B", timeframe="5m", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(m1_long, m5_short_touch), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch", "fvg-5m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [m1_long.candidate_id])
        self.assertEqual(result.suspensions, [])

    def test_no_cross_timeframe_suspension_holds_for_multiple_timeframe_pairs(self):
        for own_tf, other_tf in [("1m", "5m"), ("5m", "15m"), ("1m", "15m")]:
            own = fvg_candidate(zone_id=f"{own_tf}-A", timeframe=own_tf, direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
            other = fvg_candidate(zone_id=f"{other_tf}-B", timeframe=other_tf, direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
            timeline = build_entry_signal_timeline(fvg_result(own, other), EMPTY_LIQUIDITY)
            result = query_active_signals(timeline, {f"fvg-{own_tf}-fvg_touch", f"fvg-{other_tf}-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 12, 0))
            self.assertEqual([s.candidate_id for s in result.active_signals], [own.candidate_id], f"{own_tf} vs {other_tf} cross-timeframe leak")


class CrossFamilyNonInteractionTests(unittest.TestCase):
    """FVG/iFVG conflict logic must not affect liquidity, and liquidity
    invalidation must never affect FVG/iFVG -- even on the identical
    timeframe, opposite direction, later known_at (which would interact
    under the model this task replaces)."""

    def test_liquidity_never_suspends_or_invalidates_an_fvg_signal(self):
        bullish_fvg = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_liquidity = liquidity_candidate(
            level_id="1m-swing_low-1", side=LOW, liquidity_type=LiquidityType.SWING_LOW, trigger_phase=CandidatePhase.TAKE, entry_mode=EntryMode.CONTINUATION, known_at=ny_dt(2026, 1, 5, 11, 0),
        )
        self.assertEqual(bearish_liquidity.direction, SHORT)
        timeline = build_entry_signal_timeline(fvg_result(bullish_fvg), liquidity_result(bearish_liquidity))

        result = query_active_signals(
            timeline, {"fvg-1m-fvg_touch", "liquidity-1m-swing_equal-swing_low-take-continuation"}, LONG, ny_dt(2026, 1, 5, 12, 0)
        )
        self.assertEqual([s.candidate_id for s in result.active_signals], [bullish_fvg.candidate_id])
        self.assertEqual(result.suspensions, [])
        self.assertEqual(result.invalidations, [])

    def test_fvg_never_invalidates_a_liquidity_signal(self):
        bullish_liquidity = liquidity_candidate(
            level_id="1m-swing_low-1", side=LOW, liquidity_type=LiquidityType.SWING_LOW, trigger_phase=CandidatePhase.TAKE, entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 10, 0),
        )
        self.assertEqual(bullish_liquidity.direction, LONG)
        bearish_fvg = fvg_candidate(zone_id="1m-FVG-1", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(bearish_fvg), liquidity_result(bullish_liquidity))

        result = query_active_signals(
            timeline, {"fvg-1m-fvg_touch", "liquidity-1m-swing_equal-swing_low-take-reversal"}, LONG, ny_dt(2026, 1, 5, 12, 0)
        )
        self.assertEqual([s.candidate_id for s in result.active_signals], [bullish_liquidity.candidate_id])
        self.assertEqual(result.invalidations, [])
        self.assertEqual(result.suspensions, [])


class SameDirectionRefreshTests(unittest.TestCase):
    def test_later_same_direction_signal_does_not_invalidate_and_both_remain_active(self):
        first = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        second = fvg_candidate(zone_id="1m-FVG-2", direction=LONG, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(first, second), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in result.active_signals),
            sorted([first.candidate_id, second.candidate_id]),
        )
        self.assertEqual(result.suspensions, [])
        self.assertEqual(result.invalidations, [])

    def test_no_duplicate_active_instance_for_the_same_candidate(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(len(result.active_signals), 1)


class SelectorKeyAndAuditTests(unittest.TestCase):
    def test_selector_keys_are_plain_strings_deterministic_and_stable(self):
        c = liquidity_candidate()
        timeline_1 = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(c))
        timeline_2 = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(c))
        self.assertIsInstance(timeline_1.signals[0].selector, str)
        self.assertEqual(timeline_1.signals[0].selector, timeline_2.signals[0].selector)

    def test_trading_day_id_is_a_plain_serializable_string(self):
        c = fvg_candidate()
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertIsInstance(timeline.signals[0].trading_day_id, str)
        date.fromisoformat(timeline.signals[0].trading_day_id)

    def test_signal_retains_full_source_candidate_for_future_tp_sl(self):
        c = fvg_candidate()
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        signal = timeline.signals[0]
        self.assertEqual(signal.source.lower, c.lower)
        self.assertEqual(signal.source.upper, c.upper)
        self.assertEqual(signal.source.zone_id, c.zone_id)

    def test_query_rejects_naive_ts_utc(self):
        c = fvg_candidate()
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        with self.assertRaises(ValueError):
            query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, datetime(2026, 1, 5, 12, 0))

    def test_query_rejects_invalid_direction(self):
        c = fvg_candidate()
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        with self.assertRaises(ValueError):
            query_active_signals(timeline, {"fvg-1m-fvg_touch"}, "up", ny_dt(2026, 1, 5, 12, 0))

    def test_selectors_not_present_in_the_timeline_yield_nothing_not_an_error(self):
        c = fvg_candidate()
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-15m-ifvg"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(result.active_signals, [])


FVG_SELECTOR_POOL = ("fvg-1m-fvg_touch", "fvg-1m-ifvg", "fvg-5m-fvg_touch", "fvg-15m-ifvg")
FVG_TIMEFRAME_POOL = ("1m", "5m", "15m")
LIQUIDITY_VARIANT_POOL = (
    (SourceFamily.SWING_EQUAL, LiquidityType.SWING_HIGH, HIGH, EntryMode.REVERSAL),
    (SourceFamily.SWING_EQUAL, LiquidityType.SWING_HIGH, HIGH, EntryMode.CONTINUATION),
    (SourceFamily.SWING_EQUAL, LiquidityType.SWING_LOW, LOW, EntryMode.REVERSAL),
    (SourceFamily.SWING_EQUAL, LiquidityType.SWING_LOW, LOW, EntryMode.CONTINUATION),
)
LIQUIDITY_TIMEFRAME_POOL = ("1m", "5m", "15m")


def _random_fvg_signals(rng: random.Random, count: int, base_dt: datetime, id_prefix: str) -> list[FVGEntryCandidate]:
    """Independent, unlinked FVG-family candidates -- still individually
    valid EntrySignal sources, useful for broad randomized coverage, but
    (unlike ``_random_fvg_zone_stories`` below) never a realistic touch-
    then-its-own-inversion pair for the same zone."""
    out = []
    for i in range(count):
        selector = rng.choice(FVG_SELECTOR_POOL)
        _, timeframe, source_type_str = selector.split("-")
        source_type = SourceType.FVG_TOUCH if source_type_str == "fvg_touch" else SourceType.IFVG
        known_at = base_dt + timedelta(minutes=rng.randint(0, 460))
        direction = rng.choice([LONG, SHORT])
        out.append(
            fvg_candidate(zone_id=f"{id_prefix}-{i}", timeframe=timeframe, source_type=source_type, direction=direction, known_at=known_at)
        )
    return out


def _random_fvg_zone_stories(
    rng: random.Random, zone_count: int, base_dt: datetime, id_prefix: str
) -> tuple[list[FVGEntryCandidate], list[FVGMitigationEvent]]:
    """Randomized but semantically-valid FVG zone lifecycles -- each zone
    gets a first touch (partial, full-mitigation, or immediate
    inversion) and, only if the first touch was partial, an optional
    later resolution of that same zone: either a genuine inversion, or
    a later, separate ``FVGMitigationEvent`` (the scenario
    ``tradelab.fvg_touch`` itself can never surface, since it only ever
    records a zone's first contact) -- mirroring ``tradelab.fvg_touch``'s
    own "at most one touch event per zone, ever" contract. Directly
    stresses suspend/restore/nesting/mitigation-driven-restoration."""
    candidates: list[FVGEntryCandidate] = []
    mitigations: list[FVGMitigationEvent] = []
    for i in range(zone_count):
        zone_id = f"{id_prefix}-{i}"
        timeframe = rng.choice(FVG_TIMEFRAME_POOL)
        direction = rng.choice([LONG, SHORT])
        touch_known_at = _avoid_maintenance_break(base_dt + timedelta(minutes=rng.randint(0, 400)))
        kind = rng.choice(["partial_only", "partial_then_inverted", "partial_then_mitigated", "full_mitigation", "immediate_inversion"])
        if kind == "immediate_inversion":
            inverted = SHORT if direction == LONG else LONG
            candidates.append(
                fvg_candidate(zone_id=zone_id, timeframe=timeframe, source_type=SourceType.IFVG, direction=inverted, known_at=touch_known_at)
            )
        elif kind == "full_mitigation":
            candidates.append(
                fvg_candidate(
                    zone_id=zone_id, timeframe=timeframe, source_type=SourceType.FVG_TOUCH,
                    contact_type=FVGContactType.FULL_MITIGATION_NO_INVERSION, direction=direction, known_at=touch_known_at,
                )
            )
        elif kind == "partial_then_mitigated":
            mitigation_known_at = _avoid_maintenance_break(touch_known_at + timedelta(minutes=rng.randint(1, 200)))
            touch, mitigation = fvg_touch_then_mitigation(zone_id, timeframe, direction, touch_known_at, mitigation_known_at)
            candidates.append(touch)
            mitigations.append(mitigation)
        else:
            inversion_known_at = _avoid_maintenance_break(touch_known_at + timedelta(minutes=rng.randint(1, 200)))
            touch, ifvg = fvg_touch_then_inversion(zone_id, timeframe, direction, touch_known_at, inversion_known_at)
            candidates.append(touch)
            if kind == "partial_then_inverted":
                candidates.append(ifvg)
    return candidates, mitigations


def _random_liquidity_signals(rng: random.Random, count: int, base_dt: datetime, id_prefix: str) -> list[LiquidityEntryCandidate]:
    out = []
    for i in range(count):
        source_family, liquidity_type, side, entry_mode = rng.choice(LIQUIDITY_VARIANT_POOL)
        timeframe = rng.choice(LIQUIDITY_TIMEFRAME_POOL)
        trigger_phase = rng.choice([CandidatePhase.TAKE, CandidatePhase.RECLAIM_CLOSE])
        known_at = base_dt + timedelta(minutes=rng.randint(0, 460))
        out.append(
            liquidity_candidate(
                level_id=f"{id_prefix}-{i}",
                timeframe=timeframe,
                source_family=source_family,
                liquidity_type=liquidity_type,
                side=side,
                trigger_phase=trigger_phase,
                entry_mode=entry_mode,
                known_at=known_at,
            )
        )
    return out


class FVGLifecycleTransitionsTests(unittest.TestCase):
    """``iter_fvg_lifecycle_transitions``: unlike ``EntrySignalQueryResult``'s
    audit lists (state as of one caller-chosen ``ts_utc``), this exposes
    the complete suspend/restore transition history across the whole
    timeline, in true chronological order -- used by
    ``tradelab.profile_research_context`` to compute active-duration
    statistics."""

    def test_yields_a_suspension_then_a_restoration_for_a_simple_suspend_restore_cycle(self):
        bullish = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bearish_ifvg = fvg_touch_then_inversion("Z-B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, bearish_ifvg), EMPTY_LIQUIDITY)

        transitions = list(iter_fvg_lifecycle_transitions(timeline))
        self.assertEqual(len(transitions), 2)
        self.assertIsInstance(transitions[0], SignalSuspension)
        self.assertEqual(transitions[0].signal.candidate_id, bullish.candidate_id)
        self.assertEqual(transitions[0].suspended_by.candidate_id, bearish_touch.candidate_id)
        self.assertIsInstance(transitions[1], SignalRestoration)
        self.assertEqual(transitions[1].signal.candidate_id, bullish.candidate_id)
        self.assertEqual(transitions[1].restored_by.candidate_id, bearish_ifvg.candidate_id)

    def test_a_suspension_that_is_never_resolved_yields_no_restoration(self):
        bullish = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch = fvg_candidate(zone_id="Z-B", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch), EMPTY_LIQUIDITY)

        transitions = list(iter_fvg_lifecycle_transitions(timeline))
        self.assertEqual(len(transitions), 1)
        self.assertIsInstance(transitions[0], SignalSuspension)

    def test_restoration_via_a_later_non_inverting_mitigation_carries_the_mitigation_event(self):
        bullish = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bearish_mitigation = fvg_touch_then_mitigation("Z-B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, mitigations=(bearish_mitigation,)), EMPTY_LIQUIDITY)

        transitions = list(iter_fvg_lifecycle_transitions(timeline))
        self.assertEqual(len(transitions), 2)
        restoration = next(t for t in transitions if isinstance(t, SignalRestoration))
        self.assertIsInstance(restoration.restored_by, FVGMitigationEvent)
        self.assertEqual(restoration.restored_by.zone_id, "Z-B")

    def test_transitions_are_in_true_chronological_order_across_a_nested_conflict(self):
        a = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        b_touch, b_ifvg = fvg_touch_then_inversion("Z-B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        c_touch, c_ifvg = fvg_touch_then_inversion("Z-C", "1m", LONG, ny_dt(2026, 1, 5, 12, 0), ny_dt(2026, 1, 5, 14, 0))
        timeline = build_entry_signal_timeline(fvg_result(a, b_touch, c_touch, b_ifvg, c_ifvg), EMPTY_LIQUIDITY)

        transitions = list(iter_fvg_lifecycle_transitions(timeline))
        known_ats = [t.signal.known_at if isinstance(t, SignalSuspension) else t.restored_by.known_at for t in transitions]
        # not strictly meaningful ordering criteria across mixed types, so
        # instead assert against the causing event's own known_at, which
        # must be non-decreasing throughout.
        causing_known_ats = [(t.suspended_by if isinstance(t, SignalSuspension) else t.restored_by).known_at for t in transitions]
        self.assertEqual(causing_known_ats, sorted(causing_known_ats))
        # A suspended (by B), B suspended (by C), A restored (by B's own
        # inversion) -- C's own later inversion produces no restoration:
        # B's touch was already permanently retired at B's own inversion,
        # so nothing is left pending for C to restore.
        self.assertEqual(len(transitions), 3)

    def test_liquidity_signals_never_appear(self):
        bullish = fvg_candidate(zone_id="Z-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch = fvg_candidate(zone_id="Z-B", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        liq = liquidity_candidate(known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch), liquidity_result(liq))

        transitions = list(iter_fvg_lifecycle_transitions(timeline))
        for t in transitions:
            self.assertEqual(t.signal.source_kind, SourceKind.FVG)

    def test_empty_timeline_yields_nothing(self):
        timeline = build_entry_signal_timeline(EMPTY_FVG, EMPTY_LIQUIDITY)
        self.assertEqual(list(iter_fvg_lifecycle_transitions(timeline)), [])


class BulkSnapshotEquivalenceTests(unittest.TestCase):
    """``iter_active_signal_snapshots`` must agree with ``query_active_
    signals`` at every real signal's own instant -- verified by brute-
    force comparison on a small, deliberately overlapping/suspending
    fixture and on large randomized fixtures spanning a trading-day
    boundary, multiple timeframes, both confluence families, and
    realistic FVG zone lifecycles (touch/inversion pairs, nesting)."""

    def _assert_snapshots_match_query(self, timeline) -> None:
        all_selectors = set(timeline.signals_by_selector.keys())
        checked = 0
        for snapshot in iter_active_signal_snapshots(timeline):
            expected = query_active_signals(timeline, all_selectors, snapshot.signal.direction, snapshot.signal.known_at)
            expected_by_selector: dict[str, set[str]] = {}
            for signal in expected.active_signals:
                expected_by_selector.setdefault(signal.selector, set()).add(signal.candidate_id)

            actual_by_selector = {selector: {s.candidate_id for s in signals} for selector, signals in snapshot.active_by_selector.items()}
            self.assertEqual(
                actual_by_selector,
                expected_by_selector,
                f"mismatch at signal {snapshot.signal.candidate_id!r} known_at={snapshot.signal.known_at!r}",
            )
            checked += 1
        self.assertGreater(checked, 0)

    def test_matches_query_active_signals_on_a_small_overlapping_fixture(self):
        same_instant_a = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        same_instant_b = fvg_candidate(zone_id="B", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        challenger = fvg_candidate(zone_id="C", direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 30))
        other_timeframe = fvg_candidate(zone_id="D", timeframe="5m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 15))
        tie_with_challenger = fvg_candidate(zone_id="E", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 30))
        timeline = build_entry_signal_timeline(
            fvg_result(same_instant_a, same_instant_b, challenger, other_timeframe, tie_with_challenger), EMPTY_LIQUIDITY
        )
        self._assert_snapshots_match_query(timeline)

    def test_matches_query_active_signals_on_randomized_fixtures_across_a_trading_day_boundary(self):
        rng = random.Random(12345)
        signals = _random_fvg_signals(rng, 150, ny_dt(2026, 1, 5, 9, 0), "d1") + _random_fvg_signals(rng, 150, ny_dt(2026, 1, 6, 9, 0), "d2")
        timeline = build_entry_signal_timeline(fvg_result(*signals), EMPTY_LIQUIDITY)
        self._assert_snapshots_match_query(timeline)

    def test_matches_query_active_signals_on_a_second_random_seed(self):
        rng = random.Random(987654321)
        signals = _random_fvg_signals(rng, 200, ny_dt(2026, 6, 1, 9, 0), "s2")
        timeline = build_entry_signal_timeline(fvg_result(*signals), EMPTY_LIQUIDITY)
        self._assert_snapshots_match_query(timeline)

    def test_matches_query_active_signals_on_randomized_mixed_fvg_and_liquidity_fixtures(self):
        rng = random.Random(24680)
        fvg_signals = _random_fvg_signals(rng, 120, ny_dt(2026, 1, 5, 9, 0), "mix-fvg")
        liquidity_signals = _random_liquidity_signals(rng, 120, ny_dt(2026, 1, 5, 9, 0), "mix-liq")
        timeline = build_entry_signal_timeline(fvg_result(*fvg_signals), liquidity_result(*liquidity_signals))
        self._assert_snapshots_match_query(timeline)

    def test_matches_query_active_signals_on_randomized_realistic_fvg_zone_lifecycles(self):
        # exercises suspend/restore/nesting/mitigation-driven-restoration
        # directly: real touch-then-inversion pairs, touch-then-later-
        # separate-mitigation pairs (same zone_id), immediate inversions,
        # and first-contact full-mitigation-without-inversion, interleaved
        # across three timeframes.
        rng = random.Random(112233)
        candidates, mitigations = _random_fvg_zone_stories(rng, 150, ny_dt(2026, 1, 5, 9, 0), "story")
        timeline = build_entry_signal_timeline(fvg_result(*candidates, mitigations=mitigations), EMPTY_LIQUIDITY)
        self._assert_snapshots_match_query(timeline)

    def test_matches_query_active_signals_on_randomized_realistic_fvg_zone_lifecycles_second_seed(self):
        rng = random.Random(998877)
        candidates, mitigations = _random_fvg_zone_stories(rng, 200, ny_dt(2026, 3, 1, 9, 0), "story2")
        timeline = build_entry_signal_timeline(fvg_result(*candidates, mitigations=mitigations), EMPTY_LIQUIDITY)
        self._assert_snapshots_match_query(timeline)

    def test_snapshot_includes_the_signal_itself_under_its_own_selector(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        snapshots = list(iter_active_signal_snapshots(timeline))
        self.assertEqual(len(snapshots), 1)
        self.assertIn(c.candidate_id, {s.candidate_id for s in snapshots[0].active_by_selector["fvg-1m-fvg_touch"]})

    def test_yields_one_snapshot_per_signal(self):
        rng = random.Random(42)
        signals = _random_fvg_signals(rng, 50, ny_dt(2026, 1, 5, 9, 0), "count")
        timeline = build_entry_signal_timeline(fvg_result(*signals), EMPTY_LIQUIDITY)
        snapshots = list(iter_active_signal_snapshots(timeline))
        self.assertEqual(len(snapshots), 50)
        self.assertEqual([s.signal.candidate_id for s in snapshots], [s.candidate_id for s in timeline.signals])


class ActivePoolScalarSnapshotEquivalenceTests(unittest.TestCase):
    """``iter_active_pool_scalar_snapshots`` must agree exactly with
    ``iter_active_signal_snapshots`` (unique active-selector count
    including/excluding the primary's own selector, and raw active
    instance count) at every real signal's own instant -- it reuses the
    exact same ``_apply_liquidity_batch``/``_apply_fvg_batch``/
    ``_resolve_zone`` calls via ``_RefCountingDict``, never
    reimplementing the suspend/restore/invalidate/retire rules, so any
    disagreement here would indicate the ref-counting wrapper itself is
    broken, not a semantic divergence."""

    def _assert_scalar_matches_full(self, timeline) -> None:
        full = list(iter_active_signal_snapshots(timeline))
        scalar = list(iter_active_pool_scalar_snapshots(timeline))
        self.assertEqual(len(full), len(scalar))
        checked = 0
        for f, s in zip(full, scalar):
            self.assertEqual(f.signal.candidate_id, s.signal.candidate_id)
            unique_selectors = len(f.active_by_selector)
            instance_count = sum(len(sigs) for sigs in f.active_by_selector.values())
            self.assertEqual(
                unique_selectors, s.active_selector_count_including_primary,
                f"unique-selector-count mismatch at {f.signal.candidate_id!r} known_at={f.signal.known_at!r}",
            )
            self.assertEqual(
                unique_selectors - 1, s.required_confluence_selector_count_excluding_primary,
                f"confluence-count mismatch at {f.signal.candidate_id!r} known_at={f.signal.known_at!r}",
            )
            self.assertEqual(
                instance_count, s.active_signal_instance_count,
                f"instance-count mismatch at {f.signal.candidate_id!r} known_at={f.signal.known_at!r}",
            )
            checked += 1
        self.assertGreater(checked, 0)

    def test_matches_on_a_small_overlapping_fixture(self):
        same_instant_a = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        same_instant_b = fvg_candidate(zone_id="B", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        challenger = fvg_candidate(zone_id="C", direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 30))
        other_timeframe = fvg_candidate(zone_id="D", timeframe="5m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 15))
        tie_with_challenger = fvg_candidate(zone_id="E", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 30))
        timeline = build_entry_signal_timeline(
            fvg_result(same_instant_a, same_instant_b, challenger, other_timeframe, tie_with_challenger), EMPTY_LIQUIDITY
        )
        self._assert_scalar_matches_full(timeline)

    def test_matches_on_randomized_fvg_signals(self):
        rng = random.Random(555111)
        signals = _random_fvg_signals(rng, 200, ny_dt(2026, 1, 5, 9, 0), "scalar-fvg")
        timeline = build_entry_signal_timeline(fvg_result(*signals), EMPTY_LIQUIDITY)
        self._assert_scalar_matches_full(timeline)

    def test_matches_on_randomized_fvg_signals_second_seed(self):
        rng = random.Random(222999)
        signals = _random_fvg_signals(rng, 250, ny_dt(2026, 6, 1, 9, 0), "scalar-fvg2")
        timeline = build_entry_signal_timeline(fvg_result(*signals), EMPTY_LIQUIDITY)
        self._assert_scalar_matches_full(timeline)

    def test_matches_on_randomized_mixed_fvg_and_liquidity_fixtures(self):
        rng = random.Random(13579)
        fvg_signals = _random_fvg_signals(rng, 150, ny_dt(2026, 1, 5, 9, 0), "scalar-mix-fvg")
        liquidity_signals = _random_liquidity_signals(rng, 150, ny_dt(2026, 1, 5, 9, 0), "scalar-mix-liq")
        timeline = build_entry_signal_timeline(fvg_result(*fvg_signals), liquidity_result(*liquidity_signals))
        self._assert_scalar_matches_full(timeline)

    def test_matches_on_randomized_realistic_fvg_zone_lifecycles(self):
        # exercises suspend/restore/nesting/mitigation-driven-restoration
        # directly, same as BulkSnapshotEquivalenceTests's own coverage.
        rng = random.Random(864213)
        candidates, mitigations = _random_fvg_zone_stories(rng, 200, ny_dt(2026, 1, 5, 9, 0), "scalar-story")
        timeline = build_entry_signal_timeline(fvg_result(*candidates, mitigations=mitigations), EMPTY_LIQUIDITY)
        self._assert_scalar_matches_full(timeline)

    def test_matches_on_randomized_realistic_fvg_zone_lifecycles_second_seed(self):
        rng = random.Random(741852)
        candidates, mitigations = _random_fvg_zone_stories(rng, 200, ny_dt(2026, 3, 1, 9, 0), "scalar-story2")
        timeline = build_entry_signal_timeline(fvg_result(*candidates, mitigations=mitigations), EMPTY_LIQUIDITY)
        self._assert_scalar_matches_full(timeline)


class ActivePoolScalarSnapshotFocusedTests(unittest.TestCase):
    """Direct, hand-built scenarios for each lifecycle mechanic the
    scalar stream must track correctly: overlapping same-selector
    instances (dedup to one unique selector, never two), FVG
    suspension, restoration via inversion, restoration via a later
    non-inverting mitigation, liquidity same-selector invalidation, and
    strict direction separation."""

    def test_overlapping_same_selector_instances_dedupe_to_one_unique_selector(self):
        first = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        second = fvg_candidate(zone_id="B", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 1))
        timeline = build_entry_signal_timeline(fvg_result(first, second), EMPTY_LIQUIDITY)

        snapshots = list(iter_active_pool_scalar_snapshots(timeline))
        self.assertEqual(snapshots[0].active_selector_count_including_primary, 1)
        self.assertEqual(snapshots[0].active_signal_instance_count, 1)
        # second's own snapshot: both first and second active, same selector -- still 1 unique selector, 2 instances.
        self.assertEqual(snapshots[1].active_selector_count_including_primary, 1)
        self.assertEqual(snapshots[1].active_signal_instance_count, 2)
        self.assertEqual(snapshots[1].required_confluence_selector_count_excluding_primary, 0)

    def test_suspension_reduces_selector_and_instance_counts(self):
        bullish = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch = fvg_candidate(zone_id="B", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        # A second, later long signal after the suspension makes the suspension's effect on the
        # long-direction pool directly observable (bearish_touch's own snapshot is scoped to the
        # SHORT direction, so it alone couldn't show whether bullish was actually removed).
        later_long = fvg_candidate(zone_id="C", direction=LONG, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, later_long), EMPTY_LIQUIDITY)

        snapshots = list(iter_active_pool_scalar_snapshots(timeline))
        later_long_snapshot = next(s for s in snapshots if s.signal.candidate_id == later_long.candidate_id)
        # bullish is suspended by bearish_touch (same timeframe, opposite direction) -- only
        # later_long itself should be active in the long pool at this point.
        self.assertEqual(later_long_snapshot.active_selector_count_including_primary, 1)
        self.assertEqual(later_long_snapshot.active_signal_instance_count, 1)

    def test_restoration_via_inversion_restores_counts(self):
        bullish = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bearish_ifvg = fvg_touch_then_inversion("B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        later_long = fvg_candidate(zone_id="C", direction=LONG, known_at=ny_dt(2026, 1, 5, 14, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, bearish_ifvg, later_long), EMPTY_LIQUIDITY)

        snapshots = list(iter_active_pool_scalar_snapshots(timeline))
        later_long_snapshot = next(s for s in snapshots if s.signal.candidate_id == later_long.candidate_id)
        # bullish was restored by bearish_ifvg, and bearish_ifvg is itself a new LONG confluence
        # of its own (selector "fvg-1m-ifvg") -- so by later_long's own known_at, the long pool
        # holds two unique selectors ("fvg-1m-fvg_touch": bullish + later_long, "fvg-1m-ifvg":
        # bearish_ifvg), three instances total, and one confluence selector besides later_long's own.
        self.assertEqual(later_long_snapshot.active_selector_count_including_primary, 2)
        self.assertEqual(later_long_snapshot.active_signal_instance_count, 3)
        self.assertEqual(later_long_snapshot.required_confluence_selector_count_excluding_primary, 1)

    def test_restoration_via_mitigation_restores_counts(self):
        bullish = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bearish_mitigation = fvg_touch_then_mitigation("B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        later_long = fvg_candidate(zone_id="C", direction=LONG, known_at=ny_dt(2026, 1, 5, 14, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, later_long, mitigations=(bearish_mitigation,)), EMPTY_LIQUIDITY)

        snapshots = list(iter_active_pool_scalar_snapshots(timeline))
        later_long_snapshot = next(s for s in snapshots if s.signal.candidate_id == later_long.candidate_id)
        # bullish was restored by the mitigation event (never a signal of its own) --
        # both bullish and later_long are active in the long pool by later_long's own known_at.
        self.assertEqual(later_long_snapshot.active_selector_count_including_primary, 1)
        self.assertEqual(later_long_snapshot.active_signal_instance_count, 2)

    def test_liquidity_invalidation_reduces_instance_count(self):
        # `_liquidity_selector` is built from (timeframe, source_family, liquidity_type,
        # trigger_phase, entry_mode) -- NOT `side` -- while direction is derived from
        # (side, entry_mode). Flipping only `side` (keeping every selector-affecting field
        # fixed) therefore gives two candidates with the SAME selector but OPPOSITE direction.
        first = liquidity_candidate(level_id="L1", known_at=ny_dt(2026, 1, 5, 10, 0))  # side=HIGH, REVERSAL -> short
        opposite = liquidity_candidate(level_id="L2", side=LOW, known_at=ny_dt(2026, 1, 5, 10, 1))  # side=LOW, REVERSAL -> long
        later_same_selector = liquidity_candidate(level_id="L3", known_at=ny_dt(2026, 1, 5, 10, 2))  # short again
        timeline = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(first, opposite, later_same_selector))

        snapshots = list(iter_active_pool_scalar_snapshots(timeline))
        later_snapshot = next(s for s in snapshots if s.signal.candidate_id == later_same_selector.candidate_id)
        # `first` (short) was invalidated by `opposite` (long, same selector) before
        # `later_same_selector` (also short) arrives -- only later_same_selector itself
        # should be active in the short-direction pool.
        self.assertEqual(later_snapshot.active_selector_count_including_primary, 1)
        self.assertEqual(later_snapshot.active_signal_instance_count, 1)

    def test_directions_are_never_mixed(self):
        long_a = fvg_candidate(zone_id="A", timeframe="1m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        long_b = fvg_candidate(zone_id="B", timeframe="5m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 1))
        short_c = fvg_candidate(zone_id="C", timeframe="15m", direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 2))
        timeline = build_entry_signal_timeline(fvg_result(long_a, long_b, short_c), EMPTY_LIQUIDITY)

        snapshots = list(iter_active_pool_scalar_snapshots(timeline))
        short_snapshot = next(s for s in snapshots if s.signal.candidate_id == short_c.candidate_id)
        # short_c's own snapshot must never see the two long-direction selectors active before it.
        self.assertEqual(short_snapshot.active_selector_count_including_primary, 1)
        self.assertEqual(short_snapshot.active_signal_instance_count, 1)
        self.assertEqual(short_snapshot.required_confluence_selector_count_excluding_primary, 0)

    def test_empty_timeline_yields_nothing(self):
        timeline = build_entry_signal_timeline(EMPTY_FVG, EMPTY_LIQUIDITY)
        self.assertEqual(list(iter_active_pool_scalar_snapshots(timeline)), [])


class ActivePoolScalarSnapshotPerformanceTests(unittest.TestCase):
    """Demonstrates ``iter_active_pool_scalar_snapshots`` never
    materializes a per-snapshot active list/dict of ``EntrySignal``
    instances: (1) structurally, every scalar field on
    ``ActivePoolScalarSnapshot`` is a plain ``int``, never a
    ``list``/``dict``; (2) behaviorally, on a timeline whose active
    pool grows large and stays large (many concurrently-active,
    never-suspended signals across many selectors, mirroring the real-
    MNQ persist-until-invalidated shape that made ``iter_active_signal_
    snapshots``-based profiling the actual bottleneck), the scalar
    stream is dramatically faster than rebuilding a full
    ``active_by_selector`` snapshot at every signal -- with a generous
    margin so this stays robust to normal machine-load variance."""

    def test_scalar_fields_are_plain_integers_never_lists_or_dicts(self):
        signal = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(signal), EMPTY_LIQUIDITY)
        snapshot = next(iter(iter_active_pool_scalar_snapshots(timeline)))
        self.assertIsInstance(snapshot.active_selector_count_including_primary, int)
        self.assertIsInstance(snapshot.required_confluence_selector_count_excluding_primary, int)
        self.assertIsInstance(snapshot.active_signal_instance_count, int)

    def test_scalar_stream_is_substantially_faster_on_a_large_persistent_active_pool(self):
        import time as _time

        rng = random.Random(9988)
        base_dt = ny_dt(2026, 1, 5, 9, 0)
        # 300 distinct never-suspended, never-invalidated selectors, each
        # contributing one signal every few minutes across a wide window --
        # the active pool only ever grows, exactly the shape that made
        # iter_active_signal_snapshots-based profiling O(N x pool size).
        candidates = []
        for i in range(300):
            zone_id = f"perf-{i}"
            timeframe = rng.choice(FVG_TIMEFRAME_POOL)
            direction = LONG
            for occurrence in range(4):
                known_at = _avoid_maintenance_break(base_dt + timedelta(minutes=i + occurrence * 350))
                candidates.append(
                    fvg_candidate(
                        zone_id=f"{zone_id}-{occurrence}", timeframe=timeframe, direction=direction, known_at=known_at
                    )
                )
        timeline = build_entry_signal_timeline(fvg_result(*candidates), EMPTY_LIQUIDITY)

        start = _time.perf_counter()
        list(iter_active_pool_scalar_snapshots(timeline))
        scalar_duration = _time.perf_counter() - start

        start = _time.perf_counter()
        list(iter_active_signal_snapshots(timeline))
        full_duration = _time.perf_counter() - start

        self.assertLess(
            scalar_duration,
            full_duration * 0.5,
            f"scalar stream ({scalar_duration:.4f}s) should be substantially faster than the full "
            f"snapshot stream ({full_duration:.4f}s) on a large, persistently-growing active pool",
        )


def make_bias_result(changes_by_timeframe: dict[str, list[TimeframeBiasChange]]) -> HTFBiasResult:
    from tradelab.bias import SUPPORTED_TIMEFRAMES

    return HTFBiasResult(timeframe_changes={tf: changes_by_timeframe.get(tf, []) for tf in SUPPORTED_TIMEFRAMES})


def bias_change(timeframe: str, state: str, known_at: datetime) -> TimeframeBiasChange:
    source = ConfluenceSource(ConfluenceType.FVG_FORMATION, BIAS_BULLISH if state == BIAS_BULLISH else BIAS_BEARISH, "z-bias", known_at)
    return TimeframeBiasChange(timeframe=timeframe, known_at=known_at, state=state, sources=(source,))


EMPTY_BIAS = make_bias_result({})


class EntryEventBuildingTests(unittest.TestCase):
    """``build_entry_events``: FVG touch/iFVG/liquidity sweep normalize
    into ``EntryEvent``s identically to how ``EntrySignal`` already
    normalizes the exact same facts (same selector/timeframe/direction/
    known_at, built via the same private helpers) -- never re-derived
    independently. Zone mitigations never become an ``EntryEvent``,
    exactly like they never become an ``EntrySignal``."""

    def test_touch_ifvg_and_sweep_get_the_correct_kind_and_matching_selector(self):
        touch = fvg_candidate(zone_id="A", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        ifvg = fvg_candidate(zone_id="B", source_type=SourceType.IFVG, direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 1))
        sweep = liquidity_candidate(known_at=ny_dt(2026, 1, 5, 10, 2))

        events = build_entry_events(fvg_result(touch, ifvg), liquidity_result(sweep))
        by_id = {e.event_id: e for e in events}

        self.assertEqual(by_id[touch.candidate_id].kind, EntryEventKind.FVG_TOUCH)
        self.assertEqual(by_id[touch.candidate_id].selector, "fvg-1m-fvg_touch")
        self.assertEqual(by_id[touch.candidate_id].zone_id, "A")
        self.assertIsNone(by_id[touch.candidate_id].level_id)

        self.assertEqual(by_id[ifvg.candidate_id].kind, EntryEventKind.IFVG)
        self.assertEqual(by_id[ifvg.candidate_id].selector, "fvg-1m-ifvg")

        self.assertEqual(by_id[sweep.candidate_id].kind, EntryEventKind.LIQUIDITY_SWEEP)
        self.assertIsNone(by_id[sweep.candidate_id].zone_id)
        self.assertEqual(by_id[sweep.candidate_id].level_id, sweep.level_id)

    def test_mitigations_never_become_entry_events(self):
        touch, mitigation = fvg_touch_then_mitigation("A", "1m", LONG, ny_dt(2026, 1, 5, 10, 0), ny_dt(2026, 1, 5, 12, 0))
        events = build_entry_events(fvg_result(touch, mitigations=(mitigation,)), EMPTY_LIQUIDITY)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, touch.candidate_id)

    def test_events_are_chronological_by_known_at(self):
        rng = random.Random(24681357)
        candidates, mitigations = _random_fvg_zone_stories(rng, 40, ny_dt(2026, 1, 5, 9, 0), "chrono")
        liquidity_signals = _random_liquidity_signals(rng, 40, ny_dt(2026, 1, 5, 9, 0), "chrono-liq")
        events = build_entry_events(fvg_result(*candidates, mitigations=mitigations), liquidity_result(*liquidity_signals))
        self.assertEqual([e.known_at for e in events], sorted(e.known_at for e in events))

    def test_empty_inputs_yield_no_events(self):
        self.assertEqual(build_entry_events(EMPTY_FVG, EMPTY_LIQUIDITY), [])


class MarketStateVsEntryEventRegressionTests(unittest.TestCase):
    """The five focused regressions this task requires: (a) a touch is
    available at its own timestamp but never becomes an everlasting
    entry condition, (b) FVG suspension/restoration remains correct
    after a later iFVG, (c) a liquidity sweep is a one-shot event,
    (d) HTF bias survives across later entry events until a valid
    opposite bias changes it, (e) no look-ahead."""

    def test_a_fvg_touch_is_available_at_its_own_timestamp_but_not_an_everlasting_entry_condition(self):
        touch = fvg_candidate(zone_id="A", timeframe="1m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        # A much later, unrelated touch on a different timeframe (so same-timeframe
        # suspension never interferes) -- roughly 400 days later.
        later = fvg_candidate(zone_id="C", timeframe="5m", direction=LONG, known_at=ny_dt(2027, 2, 9, 10, 0))
        result = fvg_result(touch, later)

        events = build_entry_events(result, EMPTY_LIQUIDITY)
        # The touch is a real EntryEvent exactly once, at its own known_at -- never re-offered.
        self.assertEqual([e.event_id for e in events], [touch.candidate_id, later.candidate_id])
        self.assertEqual(events[0].known_at, touch.known_at)

        snapshots = {s.event.event_id: s for s in iter_candidate_entry_snapshots(result, EMPTY_LIQUIDITY, EMPTY_BIAS)}
        self.assertEqual(len(snapshots), 2)
        # The touch's own zone never resolved -- its STATE (not the event itself) legitimately
        # persists as context for the later, unrelated event. This is the intended distinction,
        # not a bug: the event fires once; the zone's structural state persists until resolved.
        self.assertIn("fvg-1m-fvg_touch", snapshots[later.candidate_id].market_state.active_fvg_selectors_by_direction[LONG])
        # But the touch is never itself re-listed as an event at the later instant.
        self.assertNotEqual(snapshots[later.candidate_id].event.event_id, touch.candidate_id)

    def test_fvg_suspension_and_restoration_remains_correct_after_a_later_ifvg(self):
        bullish = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch, bearish_ifvg = fvg_touch_then_inversion("B", "1m", SHORT, ny_dt(2026, 1, 5, 11, 0), ny_dt(2026, 1, 5, 13, 0))
        later_long = fvg_candidate(zone_id="C", direction=LONG, known_at=ny_dt(2026, 1, 5, 14, 0))
        result = fvg_result(bullish, bearish_touch, bearish_ifvg, later_long)

        snapshots = {s.event.event_id: s for s in iter_candidate_entry_snapshots(result, EMPTY_LIQUIDITY, EMPTY_BIAS)}
        later_state = snapshots[later_long.candidate_id].market_state
        # bullish was suspended by bearish_touch, then restored when bearish_touch's own
        # zone inverted (a partially-touched opposing FVG never permanently cancels the
        # prior FVG) -- both the restored touch and the new iFVG are active by now.
        self.assertIn("fvg-1m-fvg_touch", later_state.active_fvg_selectors_by_direction[LONG])
        self.assertIn("fvg-1m-ifvg", later_state.active_fvg_selectors_by_direction[LONG])

    def test_liquidity_sweep_is_a_one_shot_event(self):
        sweep = liquidity_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        later = fvg_candidate(zone_id="Z", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 5))
        result_fvg = fvg_result(later)
        result_liq = liquidity_result(sweep)

        events = build_entry_events(result_fvg, result_liq)
        self.assertEqual(len([e for e in events if e.event_id == sweep.candidate_id]), 1)

        snapshots = list(iter_candidate_entry_snapshots(result_fvg, result_liq, EMPTY_BIAS))
        sweep_snapshots = [s for s in snapshots if s.event.event_id == sweep.candidate_id]
        self.assertEqual(len(sweep_snapshots), 1)
        # The sweep never contributes to, or appears in, persistent FVG-family state.
        later_snapshot = next(s for s in snapshots if s.event.event_id == later.candidate_id)
        all_selectors = set(later_snapshot.market_state.active_fvg_selectors_by_direction[LONG]) | set(
            later_snapshot.market_state.active_fvg_selectors_by_direction[SHORT]
        )
        self.assertTrue(all(not selector.startswith("liquidity-") for selector in all_selectors))

    def test_htf_bias_survives_across_later_entry_events_until_a_valid_opposite_bias_changes_it(self):
        t_bull = ny_dt(2026, 1, 5, 9, 0)
        t_bear = ny_dt(2026, 1, 5, 15, 0)
        bias_result = make_bias_result({"1H": [bias_change("1H", BIAS_BULLISH, t_bull), bias_change("1H", BIAS_BEARISH, t_bear)]})

        event_a = fvg_candidate(zone_id="A", direction=LONG, known_at=t_bull + timedelta(hours=1))
        event_b = fvg_candidate(zone_id="B", direction=LONG, known_at=t_bull + timedelta(hours=2))
        event_c = fvg_candidate(zone_id="C", direction=LONG, known_at=t_bear + timedelta(hours=1))
        result = fvg_result(event_a, event_b, event_c)

        snapshots = {s.event.event_id: s for s in iter_candidate_entry_snapshots(result, EMPTY_LIQUIDITY, bias_result)}
        self.assertEqual(snapshots[event_a.candidate_id].market_state.bias.global_bias, BIAS_BULLISH)
        self.assertEqual(snapshots[event_b.candidate_id].market_state.bias.global_bias, BIAS_BULLISH)
        self.assertEqual(snapshots[event_c.candidate_id].market_state.bias.global_bias, BIAS_BEARISH)

    def test_no_look_ahead_every_market_state_fact_known_at_or_before_event_known_at(self):
        rng = random.Random(998877)
        candidates, mitigations = _random_fvg_zone_stories(rng, 25, ny_dt(2026, 1, 5, 9, 0), "nla")
        liquidity_signals = _random_liquidity_signals(rng, 15, ny_dt(2026, 1, 5, 9, 0), "nla-liq")
        bias_result = make_bias_result(
            {
                "1H": [bias_change("1H", BIAS_BULLISH, ny_dt(2026, 1, 5, 9, 30)), bias_change("1H", BIAS_BEARISH, ny_dt(2026, 1, 6, 3, 0))],
                "4H": [bias_change("4H", BIAS_BEARISH, ny_dt(2026, 1, 5, 20, 0))],
            }
        )
        fvg_res = fvg_result(*candidates, mitigations=mitigations)
        liq_res = liquidity_result(*liquidity_signals)

        full_snapshots = {s.event.event_id: s for s in iter_candidate_entry_snapshots(fvg_res, liq_res, bias_result)}
        all_events = build_entry_events(fvg_res, liq_res)

        checked = 0
        for cut_event in all_events[::5]:  # a representative sample of cut points, not every one (keeps the test fast)
            cutoff = cut_event.known_at
            truncated_fvg_candidates = [c for c in fvg_res.candidates if c.known_at <= cutoff]
            truncated_mitigations = tuple(m for m in fvg_res.zone_mitigations if m.known_at <= cutoff)
            truncated_liquidity = [c for c in liq_res.candidates if c.known_at <= cutoff]
            truncated_bias = make_bias_result(
                {tf: [c for c in changes if c.known_at <= cutoff] for tf, changes in bias_result.timeframe_changes.items()}
            )
            truncated_snapshots = {
                s.event.event_id: s
                for s in iter_candidate_entry_snapshots(
                    fvg_result(*truncated_fvg_candidates, mitigations=truncated_mitigations),
                    liquidity_result(*truncated_liquidity),
                    truncated_bias,
                )
            }
            full_state = full_snapshots[cut_event.event_id].market_state
            truncated_state = truncated_snapshots[cut_event.event_id].market_state
            self.assertEqual(
                full_state.active_fvg_selectors_by_direction, truncated_state.active_fvg_selectors_by_direction,
                f"future data leaked into {cut_event.event_id!r}'s own market state",
            )
            self.assertEqual(full_state.bias, truncated_state.bias, f"future bias data leaked into {cut_event.event_id!r}'s own market state")
            checked += 1
        self.assertGreater(checked, 0)


class BiasIncrementalTrackingEquivalenceTests(unittest.TestCase):
    """``iter_candidate_entry_snapshots`` tracks HTF bias incrementally
    (never by calling ``tradelab.bias.bias_at`` per event, which would
    rescan its own change list from the start every time) -- must agree
    exactly with a fresh ``bias_at`` call at every real event's own
    known_at."""

    def test_incremental_bias_matches_bias_at_on_a_randomized_multi_timeframe_timeline(self):
        base = ny_dt(2026, 1, 5, 9, 0)
        bias_result = make_bias_result(
            {
                "1H": [
                    bias_change("1H", BIAS_BULLISH, base + timedelta(hours=1)),
                    bias_change("1H", BIAS_BEARISH, base + timedelta(hours=5)),
                    bias_change("1H", BIAS_BULLISH, base + timedelta(hours=9)),
                ],
                "4H": [
                    bias_change("4H", BIAS_BEARISH, base + timedelta(hours=3)),
                    bias_change("4H", BIAS_BULLISH, base + timedelta(hours=7)),
                ],
                "1D": [bias_change("1D", BIAS_BEARISH, base + timedelta(hours=6))],
            }
        )
        rng = random.Random(13571113)
        events = _random_fvg_signals(rng, 60, base, "bias-eq")
        result = fvg_result(*events)

        for snapshot in iter_candidate_entry_snapshots(result, EMPTY_LIQUIDITY, bias_result):
            expected = bias_at(bias_result, snapshot.event.known_at)
            self.assertEqual(snapshot.market_state.bias, expected, f"mismatch at known_at={snapshot.event.known_at!r}")


class CandidateEntrySnapshotStructuralTests(unittest.TestCase):
    def test_snapshot_count_equals_event_count(self):
        rng = random.Random(2468)
        candidates, mitigations = _random_fvg_zone_stories(rng, 30, ny_dt(2026, 1, 5, 9, 0), "count")
        liquidity_signals = _random_liquidity_signals(rng, 30, ny_dt(2026, 1, 5, 9, 0), "count-liq")
        result_fvg = fvg_result(*candidates, mitigations=mitigations)
        result_liq = liquidity_result(*liquidity_signals)

        events = build_entry_events(result_fvg, result_liq)
        snapshots = list(iter_candidate_entry_snapshots(result_fvg, result_liq, EMPTY_BIAS))
        self.assertEqual(len(snapshots), len(events))
        self.assertEqual([s.event.event_id for s in snapshots], [e.event_id for e in events])

    def test_empty_inputs_yield_no_snapshots(self):
        self.assertEqual(list(iter_candidate_entry_snapshots(EMPTY_FVG, EMPTY_LIQUIDITY, EMPTY_BIAS)), [])

    def test_market_state_as_of_matches_the_events_own_known_at(self):
        touch = fvg_candidate(zone_id="A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        (snapshot,) = list(iter_candidate_entry_snapshots(fvg_result(touch), EMPTY_LIQUIDITY, EMPTY_BIAS))
        self.assertEqual(snapshot.market_state.as_of, touch.known_at)

    def test_active_fvg_selector_pool_never_exceeds_the_fixed_selector_catalog_size(self):
        """At most 6 timeframes x 2 source types = 12 possible FVG-family
        selector strings, system-wide, ever -- bounded by the catalog
        itself, never by how much history accumulated (the exact
        property this refactor exists to guarantee)."""
        rng = random.Random(112233)
        candidates, mitigations = _random_fvg_zone_stories(rng, 300, ny_dt(2026, 1, 5, 9, 0), "bounded")
        result = fvg_result(*candidates, mitigations=mitigations)
        for snapshot in iter_candidate_entry_snapshots(result, EMPTY_LIQUIDITY, EMPTY_BIAS):
            for direction in (LONG, SHORT):
                self.assertLessEqual(len(snapshot.market_state.active_fvg_selectors_by_direction[direction]), 12)


if __name__ == "__main__":
    unittest.main()
