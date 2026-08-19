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

from tradelab.entry_signal_timeline import (
    LONG,
    SHORT,
    SourceKind,
    build_entry_signal_timeline,
    iter_active_signal_snapshots,
    query_active_signals,
)
from tradelab.fvg_entry import FVGContactType, FVGEntryCandidate, FVGEntryCandidateResult, SourceType
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


def fvg_result(*candidates: FVGEntryCandidate) -> FVGEntryCandidateResult:
    return FVGEntryCandidateResult(candidates=list(candidates))


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
        # A suspended by B; B (while itself suspended by C) later inverts,
        # restoring A; C later inverts too, restoring B's own touch.
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

        # after B inverts: A restored, C still active, iFVG(B) active LONG; b_touch still suspended by C
        after_b_ifvg_long = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 13, 30))
        self.assertEqual(
            sorted(s.candidate_id for s in after_b_ifvg_long.active_signals),
            sorted([a.candidate_id, c_touch.candidate_id, b_ifvg.candidate_id]),
        )
        after_b_ifvg_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 13, 30))
        self.assertEqual(after_b_ifvg_short.active_signals, [])

        # after C inverts: b_touch restored, iFVG(C) active SHORT
        final_long = query_active_signals(timeline, selectors, LONG, ny_dt(2026, 1, 5, 15, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in final_long.active_signals),
            sorted([a.candidate_id, c_touch.candidate_id, b_ifvg.candidate_id]),
        )
        final_short = query_active_signals(timeline, selectors, SHORT, ny_dt(2026, 1, 5, 15, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in final_short.active_signals),
            sorted([b_touch.candidate_id, c_ifvg.candidate_id]),
        )

    def test_a_suspended_fvg_does_not_reappear_due_to_an_unrelated_same_direction_signal(self):
        bullish = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        bearish_touch = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        unrelated_bullish = fvg_candidate(zone_id="1m-FVG-3", direction=LONG, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(bullish, bearish_touch, unrelated_bullish), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        active_ids = {s.candidate_id for s in result.active_signals}
        self.assertNotIn(bullish.candidate_id, active_ids)
        self.assertIn(unrelated_bullish.candidate_id, active_ids)


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


def _random_fvg_zone_stories(rng: random.Random, zone_count: int, base_dt: datetime, id_prefix: str) -> list[FVGEntryCandidate]:
    """Randomized but semantically-valid FVG zone lifecycles -- each zone
    gets a first touch (partial, full-mitigation, or immediate
    inversion) and, only if the first touch was partial, an optional
    later inversion of that same zone -- mirroring
    ``tradelab.fvg_touch``'s own "at most one touch event per zone,
    ever" contract. Directly stresses suspend/restore/nesting."""
    out: list[FVGEntryCandidate] = []
    for i in range(zone_count):
        zone_id = f"{id_prefix}-{i}"
        timeframe = rng.choice(FVG_TIMEFRAME_POOL)
        direction = rng.choice([LONG, SHORT])
        touch_known_at = _avoid_maintenance_break(base_dt + timedelta(minutes=rng.randint(0, 400)))
        kind = rng.choice(["partial_only", "partial_then_inverted", "full_mitigation", "immediate_inversion"])
        if kind == "immediate_inversion":
            inverted = SHORT if direction == LONG else LONG
            out.append(fvg_candidate(zone_id=zone_id, timeframe=timeframe, source_type=SourceType.IFVG, direction=inverted, known_at=touch_known_at))
        elif kind == "full_mitigation":
            out.append(
                fvg_candidate(
                    zone_id=zone_id, timeframe=timeframe, source_type=SourceType.FVG_TOUCH,
                    contact_type=FVGContactType.FULL_MITIGATION_NO_INVERSION, direction=direction, known_at=touch_known_at,
                )
            )
        else:
            inversion_known_at = _avoid_maintenance_break(touch_known_at + timedelta(minutes=rng.randint(1, 200)))
            touch, ifvg = fvg_touch_then_inversion(zone_id, timeframe, direction, touch_known_at, inversion_known_at)
            out.append(touch)
            if kind == "partial_then_inverted":
                out.append(ifvg)
    return out


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
        # exercises suspend/restore/nesting directly: real touch-then-
        # inversion pairs (same zone_id, opposite directions), immediate
        # inversions, and full-mitigation-without-inversion, interleaved
        # across three timeframes.
        rng = random.Random(112233)
        signals = _random_fvg_zone_stories(rng, 150, ny_dt(2026, 1, 5, 9, 0), "story")
        timeline = build_entry_signal_timeline(fvg_result(*signals), EMPTY_LIQUIDITY)
        self._assert_snapshots_match_query(timeline)

    def test_matches_query_active_signals_on_randomized_realistic_fvg_zone_lifecycles_second_seed(self):
        rng = random.Random(998877)
        signals = _random_fvg_zone_stories(rng, 200, ny_dt(2026, 3, 1, 9, 0), "story2")
        timeline = build_entry_signal_timeline(fvg_result(*signals), EMPTY_LIQUIDITY)
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


if __name__ == "__main__":
    unittest.main()
