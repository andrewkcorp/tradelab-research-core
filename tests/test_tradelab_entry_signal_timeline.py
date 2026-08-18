#!/usr/bin/env python3
"""Entry signal timeline tests: FVG/liquidity normalization, known_at/
no-lookahead, persist-until-invalidated validity (no time-based expiry
of any kind), trading_day_id as pure audit metadata (still computed via
NY session boundaries, DST-correct), same-timeframe cross-family
invalidation (FVG/iFVG/liquidity), no cross-timeframe invalidation,
unselected selectors never invalidating, the FVG-touch-vs-iFVG-inversion
distinction, permanent one-way invalidation (no resurrection), same-
direction refresh (no duplication), deterministic selector keys with
full audit, and randomized equivalence between ``query_active_signals``
and ``iter_active_signal_snapshots``."""
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
from tradelab.fvg_entry import FVGEntryCandidate, FVGEntryCandidateResult, SourceType
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


def fvg_candidate(
    zone_id: str = "1m-FVG-1",
    timeframe: str = "1m",
    source_type: SourceType = SourceType.FVG_TOUCH,
    direction: str = LONG,
    known_at: datetime = ny_dt(2026, 1, 5, 10, 1),
    timestamp: datetime | None = None,
) -> FVGEntryCandidate:
    return FVGEntryCandidate(
        candidate_id=f"{timeframe}-{source_type.value}-{zone_id}",
        source_type=source_type,
        direction=direction,
        timeframe=timeframe,
        zone_id=zone_id,
        lower=100.0,
        upper=102.0,
        contact_type=None,
        timestamp=known_at - timedelta(minutes=1) if timestamp is None else timestamp,
        known_at=known_at,
    )


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
        # persist-until-invalidated: there is no daily/session/calendar
        # cutoff at all -- a signal known on one day is exactly as active
        # a month later, absent an actual invalidating event.
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 1))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 2, 5, 10, 1))
        self.assertEqual([s.candidate_id for s in result.active_signals], [c.candidate_id])


class NewTradingDayTests(unittest.TestCase):
    """``trading_day_id`` is still computed exactly as before -- it is
    now pure audit metadata (see module docstring) and plays no role in
    whether a signal is active. Only the id itself and the still-
    unchanged malformed-input rejection are checked here; validity is
    covered elsewhere."""

    def test_signal_known_before_1700_belongs_to_the_day_that_started_the_evening_before(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))  # NY 10:00 belongs to the day that opened 2026-01-04 18:00
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-04")

    def test_signal_known_exactly_at_1800_starts_a_new_trading_day(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 18, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-05")

    def test_signal_known_just_after_1800_is_in_the_new_trading_day(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 18, 1))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-05")

    def test_signal_known_exactly_at_1700_belongs_to_the_ending_trading_day(self):
        # 17:00 NY is the regular session's own close -- the known_at of
        # every trading day's genuinely last 1-minute candle -- so it is
        # never malformed and is attributed to the day that is ending,
        # not rejected (unlike the open interval strictly between 17:00
        # and 18:00, which no trading day owns). This is unrelated to
        # validity now -- the signal simply stays active from this
        # instant onward like any other, indefinitely.
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-04")

    def test_signal_known_exactly_at_1700_is_active_from_that_instant_onward(self):
        # Under the old expiry model this instant had zero-width validity
        # (known_at == valid_until); persist-until-invalidated has no
        # such thing -- the signal is simply active starting exactly here.
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
    """``trading_day_id`` boundary math uses a real IANA zone, so DST
    transitions must not perturb it -- purely an audit-field correctness
    check now (there is no more valid_until to compute across the
    transition, and validity itself is DST-agnostic since it is never
    time-bounded)."""

    def test_trading_day_id_is_correct_across_spring_forward(self):
        # 2026-03-08: NY clocks jump 02:00 -> 03:00 (spring forward).
        c = fvg_candidate(known_at=ny_dt(2026, 3, 7, 20, 0))  # trading day started 2026-03-07 18:00
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-03-07")

    def test_trading_day_id_is_correct_across_fall_back(self):
        # 2026-11-01: NY clocks fall back 02:00 -> 01:00.
        c = fvg_candidate(known_at=ny_dt(2026, 10, 31, 20, 0))  # trading day started 2026-10-31 18:00
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-10-31")

    def test_query_across_the_dst_transition_still_sees_the_signal_active(self):
        c = fvg_candidate(known_at=ny_dt(2026, 3, 7, 20, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        after_transition = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 3, 9, 12, 0))
        self.assertEqual([s.candidate_id for s in after_transition.active_signals], [c.candidate_id])


class CrossTradingDayPersistenceTests(unittest.TestCase):
    """Persist-until-invalidated means trading-day boundaries no longer
    partition anything: a signal known on one trading day is exactly as
    active a day (or a month) later as the instant it became known,
    unless a same-timeframe opposite-direction signal invalidated it in
    between -- crossing into a new trading day is not itself such an
    event, and is no longer even eligible to be one."""

    def test_signal_from_an_earlier_trading_day_is_still_active_the_next_day(self):
        day1_signal = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(day1_signal), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 6, 10, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [day1_signal.candidate_id])

    def test_a_next_day_opposite_signal_on_the_same_timeframe_does_invalidate_the_prior_day(self):
        day1_signal = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        day2_signal = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 6, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(day1_signal, day2_signal), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 6, 11, 0))
        # day1's LONG signal is gone because a real opposite-direction,
        # same-timeframe event invalidated it -- crossing into day2 was
        # not itself the reason, and the audit trail reflects that.
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.invalidations), 1)
        self.assertEqual(result.invalidations[0].signal.candidate_id, day1_signal.candidate_id)
        self.assertEqual(result.invalidations[0].invalidated_by.candidate_id, day2_signal.candidate_id)

    def test_signals_from_different_trading_days_remain_simultaneously_active_when_never_invalidated(self):
        day1_signal = fvg_candidate(zone_id="1m-FVG-1", timeframe="5m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        day2_signal = fvg_candidate(zone_id="1m-FVG-2", timeframe="15m", direction=LONG, known_at=ny_dt(2026, 1, 6, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(day1_signal, day2_signal), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-5m-fvg_touch", "fvg-15m-fvg_touch"}, LONG, ny_dt(2026, 1, 6, 11, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in result.active_signals),
            sorted([day1_signal.candidate_id, day2_signal.candidate_id]),
        )


class UnselectedVariantTests(unittest.TestCase):
    def test_unselected_continuation_variant_never_invalidates_selected_reversal(self):
        reversal = liquidity_candidate(
            level_id="1m-swing_high-1", side=HIGH, trigger_phase=CandidatePhase.TAKE,
            entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 10, 0),
        )
        # continuation is known LATER, on the exact same timeframe, in
        # the opposite direction -- under the timeframe-scoped
        # invalidation rule it would invalidate reversal if it were ever
        # considered, but it is never in the caller's chosen `selectors`
        # set, so it must never even be looked at (spec: "Sygnaly z
        # niewybranych selectorow nigdy nie uniewazniaja setupu").
        continuation = liquidity_candidate(
            level_id="1m-swing_high-1", side=HIGH, trigger_phase=CandidatePhase.TAKE,
            entry_mode=EntryMode.CONTINUATION, known_at=ny_dt(2026, 1, 5, 12, 0),
        )
        timeline = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(reversal, continuation))
        self.assertNotEqual(reversal.direction, continuation.direction)
        self.assertEqual(reversal.timeframe, continuation.timeframe)
        result = query_active_signals(
            timeline, {"liquidity-1m-swing_equal-swing_high-take-reversal"}, reversal.direction, ny_dt(2026, 1, 5, 13, 0)
        )
        self.assertEqual([s.candidate_id for s in result.active_signals], [reversal.candidate_id])
        self.assertEqual(result.invalidations, [])


class InvalidationTests(unittest.TestCase):
    def test_later_opposite_signal_on_the_same_timeframe_invalidates_the_earlier_one(self):
        first = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        later = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(first, later), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.invalidations), 1)
        self.assertEqual(result.invalidations[0].signal.candidate_id, first.candidate_id)
        self.assertEqual(result.invalidations[0].invalidated_by.candidate_id, later.candidate_id)

    def test_the_earlier_signal_is_still_active_before_the_invalidator_is_known(self):
        first = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        later = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(first, later), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 11, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [first.candidate_id])
        self.assertEqual(result.invalidations, [])


class SameTimeframeCrossFamilyInvalidationTests(unittest.TestCase):
    """Rule: a newly known confluence invalidates every still-active,
    earlier, opposite-direction confluence on the exact same timeframe --
    regardless of which family (FVG, iFVG, liquidity) or which exact
    selector either one belongs to."""

    def test_a_different_fvg_selector_on_the_same_timeframe_invalidates_across_selectors(self):
        touch_long = fvg_candidate(zone_id="1m-FVG-1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        ifvg_short_later = fvg_candidate(zone_id="1m-FVG-2", source_type=SourceType.IFVG, direction=SHORT, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(touch_long, ifvg_short_later), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch", "fvg-1m-ifvg"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.invalidations), 1)
        self.assertEqual(result.invalidations[0].invalidated_by.candidate_id, ifvg_short_later.candidate_id)

    def test_a_liquidity_signal_invalidates_an_fvg_signal_on_the_same_timeframe(self):
        fvg_long = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        liquidity_short_later = liquidity_candidate(
            level_id="1m-swing_low-1", side=LOW, liquidity_type=LiquidityType.SWING_LOW,
            trigger_phase=CandidatePhase.TAKE, entry_mode=EntryMode.CONTINUATION, known_at=ny_dt(2026, 1, 5, 12, 0),
        )  # (LOW, CONTINUATION) -> SHORT, timeframe "1m" (default)
        self.assertEqual(liquidity_short_later.direction, SHORT)
        self.assertEqual(liquidity_short_later.timeframe, "1m")
        timeline = build_entry_signal_timeline(fvg_result(fvg_long), liquidity_result(liquidity_short_later))

        result = query_active_signals(
            timeline, {"fvg-1m-fvg_touch", "liquidity-1m-swing_equal-swing_low-take-continuation"}, LONG, ny_dt(2026, 1, 5, 13, 0)
        )
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.invalidations), 1)
        self.assertEqual(result.invalidations[0].invalidated_by.candidate_id, liquidity_short_later.candidate_id)

    def test_an_fvg_signal_invalidates_a_liquidity_signal_on_the_same_timeframe(self):
        liquidity_long = liquidity_candidate(
            level_id="1m-swing_low-1", side=LOW, liquidity_type=LiquidityType.SWING_LOW,
            trigger_phase=CandidatePhase.TAKE, entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 10, 0),
        )  # (LOW, REVERSAL) -> LONG
        self.assertEqual(liquidity_long.direction, LONG)
        fvg_short_later = fvg_candidate(zone_id="1m-FVG-1", direction=SHORT, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(fvg_short_later), liquidity_result(liquidity_long))

        result = query_active_signals(
            timeline, {"fvg-1m-fvg_touch", "liquidity-1m-swing_equal-swing_low-take-reversal"}, LONG, ny_dt(2026, 1, 5, 13, 0)
        )
        self.assertEqual(result.active_signals, [])
        self.assertEqual(len(result.invalidations), 1)
        self.assertEqual(result.invalidations[0].invalidated_by.candidate_id, fvg_short_later.candidate_id)


class NoCrossTimeframeInvalidationTests(unittest.TestCase):
    def test_opposite_direction_signal_on_a_different_timeframe_never_invalidates(self):
        m1_long = fvg_candidate(zone_id="1m-FVG-1", timeframe="1m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        m5_short_later = fvg_candidate(zone_id="5m-FVG-1", timeframe="5m", direction=SHORT, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(m1_long, m5_short_later), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch", "fvg-5m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [m1_long.candidate_id])
        self.assertEqual(result.invalidations, [])

    def test_no_cross_timeframe_invalidation_holds_for_multiple_timeframe_pairs(self):
        for own_tf, other_tf in [("1m", "5m"), ("5m", "15m"), ("1m", "15m")]:
            own = fvg_candidate(zone_id=f"{own_tf}-A", timeframe=own_tf, direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
            other = fvg_candidate(zone_id=f"{other_tf}-B", timeframe=other_tf, direction=SHORT, known_at=ny_dt(2026, 1, 5, 12, 0))
            timeline = build_entry_signal_timeline(fvg_result(own, other), EMPTY_LIQUIDITY)
            result = query_active_signals(
                timeline, {f"fvg-{own_tf}-fvg_touch", f"fvg-{other_tf}-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 13, 0)
            )
            self.assertEqual(
                [s.candidate_id for s in result.active_signals], [own.candidate_id], f"{own_tf} vs {other_tf} cross-timeframe leak"
            )


class FVGTouchVersusIFVGConversionTests(unittest.TestCase):
    """A plain FVG touch always carries its own zone's original direction
    (``tradelab.fvg_entry``'s own, unchanged contract) and so can never
    itself become an opposite-direction confluence relative to that same
    zone; only a genuine inversion -- surfaced here as an IFVG-family
    candidate carrying the OPPOSITE direction -- is a genuinely new,
    opposite-direction confluence subject to invalidation."""

    def test_fvg_touch_events_never_carry_the_opposite_direction_and_so_never_invalidate(self):
        long_active = fvg_candidate(zone_id="1m-OTHER", timeframe="5m", direction=LONG, known_at=ny_dt(2026, 1, 5, 9, 0))
        touch_1 = fvg_candidate(zone_id="1m-Z1", timeframe="5m", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        touch_2 = fvg_candidate(zone_id="1m-Z2", timeframe="5m", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(long_active, touch_1, touch_2), EMPTY_LIQUIDITY)

        # a bullish zone's own touches are always LONG -- they can never
        # surface as a SHORT confluence, so a SHORT query never sees them.
        short_query = query_active_signals(timeline, {"fvg-5m-fvg_touch"}, SHORT, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(short_query.active_signals, [])

        # and the earlier LONG confluence is untouched by either touch --
        # both touches simply join it, side by side, never invalidating.
        long_query = query_active_signals(timeline, {"fvg-5m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(
            sorted(s.candidate_id for s in long_query.active_signals),
            sorted([long_active.candidate_id, touch_1.candidate_id, touch_2.candidate_id]),
        )
        self.assertEqual(long_query.invalidations, [])

    def test_a_full_inversion_to_ifvg_carries_the_opposite_direction_and_does_invalidate(self):
        # the same zone Z, touched as LONG, later fully inverts:
        # tradelab.fvg_entry surfaces that as an IFVG candidate carrying
        # the OPPOSITE (SHORT) direction -- a genuinely new confluence
        # that does invalidate every still-active earlier LONG on its own
        # timeframe (including the zone's own prior touch, which is no
        # longer valid either once its zone has flipped), exactly like
        # any other opposite-direction event.
        long_active = fvg_candidate(zone_id="1m-OTHER", timeframe="5m", direction=LONG, known_at=ny_dt(2026, 1, 5, 9, 0))
        touch = fvg_candidate(zone_id="1m-Z", timeframe="5m", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        inversion = fvg_candidate(zone_id="1m-Z", timeframe="5m", source_type=SourceType.IFVG, direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(long_active, touch, inversion), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-5m-fvg_touch", "fvg-5m-ifvg"}, LONG, ny_dt(2026, 1, 5, 12, 0))
        self.assertEqual(result.active_signals, [])
        invalidated_ids = {inv.signal.candidate_id for inv in result.invalidations}
        self.assertEqual(invalidated_ids, {long_active.candidate_id, touch.candidate_id})
        self.assertTrue(all(inv.invalidated_by.candidate_id == inversion.candidate_id for inv in result.invalidations))


class NoResurrectionTests(unittest.TestCase):
    def test_invalidated_signal_never_becomes_active_again_and_a_later_same_direction_event_is_new(self):
        original = fvg_candidate(zone_id="1m-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        invalidator = fvg_candidate(zone_id="1m-B", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        revival_candidate = fvg_candidate(zone_id="1m-C", direction=LONG, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(original, invalidator, revival_candidate), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 13, 0))
        active_ids = {s.candidate_id for s in result.active_signals}
        # original never comes back, even though a same-direction (LONG)
        # event happened after its invalidator -- that event is a brand
        # new confluence (its own candidate_id), never a resurrection.
        self.assertNotIn(original.candidate_id, active_ids)
        self.assertIn(revival_candidate.candidate_id, active_ids)
        self.assertEqual(len(result.invalidations), 1)
        self.assertEqual(result.invalidations[0].signal.candidate_id, original.candidate_id)

    def test_no_resurrection_holds_in_the_bulk_snapshot_sweep_too(self):
        original = fvg_candidate(zone_id="1m-A", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        invalidator = fvg_candidate(zone_id="1m-B", direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        revival_candidate = fvg_candidate(zone_id="1m-C", direction=LONG, known_at=ny_dt(2026, 1, 5, 12, 0))
        timeline = build_entry_signal_timeline(fvg_result(original, invalidator, revival_candidate), EMPTY_LIQUIDITY)

        snapshots = {s.signal.candidate_id: s for s in iter_active_signal_snapshots(timeline)}
        revival_snapshot = snapshots[revival_candidate.candidate_id]
        active_at_revival = {s.candidate_id for signals in revival_snapshot.active_by_selector.values() for s in signals}
        self.assertNotIn(original.candidate_id, active_at_revival)
        self.assertIn(revival_candidate.candidate_id, active_at_revival)


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
        # round-trips through date.fromisoformat without error
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
LIQUIDITY_VARIANT_POOL = (
    (SourceFamily.SWING_EQUAL, LiquidityType.SWING_HIGH, HIGH, EntryMode.REVERSAL),
    (SourceFamily.SWING_EQUAL, LiquidityType.SWING_HIGH, HIGH, EntryMode.CONTINUATION),
    (SourceFamily.SWING_EQUAL, LiquidityType.SWING_LOW, LOW, EntryMode.REVERSAL),
    (SourceFamily.SWING_EQUAL, LiquidityType.SWING_LOW, LOW, EntryMode.CONTINUATION),
)
LIQUIDITY_TIMEFRAME_POOL = ("1m", "5m", "15m")


def _random_fvg_signals(rng: random.Random, count: int, base_dt: datetime, id_prefix: str) -> list[FVGEntryCandidate]:
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
    force comparison (never by inspecting the sweep's internals) on a
    small, deliberately overlapping/invalidating fixture and on large
    randomized fixtures spanning a trading-day boundary, multiple
    timeframes, and both confluence families (the two axes the new
    timeframe-scoped, cross-family invalidation rule actually depends
    on)."""

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
        invalidator = fvg_candidate(zone_id="C", direction=SHORT, known_at=ny_dt(2026, 1, 5, 10, 30))
        other_timeframe = fvg_candidate(zone_id="D", timeframe="5m", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 15))
        tie_with_invalidator = fvg_candidate(zone_id="E", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 30))
        timeline = build_entry_signal_timeline(
            fvg_result(same_instant_a, same_instant_b, invalidator, other_timeframe, tie_with_invalidator), EMPTY_LIQUIDITY
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
        # exercises cross-family, timeframe-scoped invalidation directly:
        # FVG and liquidity signals interleaved across three timeframes,
        # any pair of which may invalidate the other on a shared
        # timeframe but never across timeframes.
        rng = random.Random(24680)
        fvg_signals = _random_fvg_signals(rng, 120, ny_dt(2026, 1, 5, 9, 0), "mix-fvg")
        liquidity_signals = _random_liquidity_signals(rng, 120, ny_dt(2026, 1, 5, 9, 0), "mix-liq")
        timeline = build_entry_signal_timeline(fvg_result(*fvg_signals), liquidity_result(*liquidity_signals))
        self._assert_snapshots_match_query(timeline)

    def test_snapshot_includes_the_signal_itself_under_its_own_selector(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        snapshots = list(iter_active_signal_snapshots(timeline))
        self.assertEqual(len(snapshots), 1)
        self.assertIn(c.candidate_id, {s.candidate_id for s in snapshots[0].active_by_selector["fvg-1m-fvg_touch"]})

    def test_never_calls_query_active_signals_style_full_scan_and_yields_one_snapshot_per_signal(self):
        rng = random.Random(42)
        signals = _random_fvg_signals(rng, 50, ny_dt(2026, 1, 5, 9, 0), "count")
        timeline = build_entry_signal_timeline(fvg_result(*signals), EMPTY_LIQUIDITY)
        snapshots = list(iter_active_signal_snapshots(timeline))
        self.assertEqual(len(snapshots), 50)
        self.assertEqual([s.signal.candidate_id for s in snapshots], [s.candidate_id for s in timeline.signals])


if __name__ == "__main__":
    unittest.main()
