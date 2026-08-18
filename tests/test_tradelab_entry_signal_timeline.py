#!/usr/bin/env python3
"""Entry signal timeline tests: FVG/liquidity normalization, known_at/
no-lookahead, expiry exactly at 17:00 NY, new trading day from 18:00 NY,
DST correctness, no cross-trading-day combination, unselected liquidity
variants never invalidating the selected one, same-selector opposite-
direction invalidation, same-direction refresh (no duplication), and
deterministic selector keys with full audit."""
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


class ExpiryTests(unittest.TestCase):
    def test_signal_valid_just_before_1700_ny_same_trading_day(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 16, 59))
        self.assertEqual(len(result.active_signals), 1)

    def test_signal_expired_exactly_at_1700_ny(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 17, 0))
        self.assertEqual(result.active_signals, [])

    def test_signal_expired_well_after_1700_ny(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 20, 0))
        self.assertEqual(result.active_signals, [])


class NewTradingDayTests(unittest.TestCase):
    def test_signal_known_before_1700_belongs_to_the_day_that_started_the_evening_before(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 10, 0))  # NY 10:00 belongs to the day that opened 2026-01-04 18:00
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-04")
        self.assertEqual(timeline.signals[0].valid_until, ny_dt(2026, 1, 5, 17, 0))

    def test_signal_known_exactly_at_1800_starts_a_new_trading_day(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 18, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-05")
        self.assertEqual(timeline.signals[0].valid_until, ny_dt(2026, 1, 6, 17, 0))

    def test_signal_known_just_after_1800_is_in_the_new_trading_day(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 18, 1))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-05")

    def test_signal_known_exactly_at_1700_belongs_to_the_ending_trading_day(self):
        # 17:00 NY is the regular session's own close -- the known_at of
        # every trading day's genuinely last 1-minute candle -- so it is
        # never malformed and is attributed to the day that is ending,
        # not rejected (unlike the open interval strictly between 17:00
        # and 18:00, which no trading day owns).
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        self.assertEqual(timeline.signals[0].trading_day_id, "2026-01-04")
        self.assertEqual(timeline.signals[0].valid_until, ny_dt(2026, 1, 5, 17, 0))

    def test_signal_known_exactly_at_1700_has_zero_width_validity(self):
        # known_at == valid_until for this signal: it becomes known at the
        # exact instant its own trading day ends, so [known_at, valid_until)
        # is empty -- it can never be found active, by construction, not by bug.
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 17, 0))
        self.assertEqual(result.active_signals, [])

    def test_signal_known_inside_the_break_is_rejected(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 30))
        with self.assertRaises(ValueError):
            build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)

    def test_signal_known_just_before_1800_still_belongs_to_the_previous_days_window(self):
        c = fvg_candidate(known_at=ny_dt(2026, 1, 5, 17, 59))
        with self.assertRaises(ValueError):
            build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)


class DSTTests(unittest.TestCase):
    def test_spring_forward_valid_until_is_the_correct_utc_instant(self):
        # 2026-03-08: NY clocks jump 02:00 -> 03:00 (spring forward).
        # A signal known well before that, still on 2026-03-07 EST.
        c = fvg_candidate(known_at=ny_dt(2026, 3, 7, 20, 0))  # trading day started 2026-03-07 18:00
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        signal = timeline.signals[0]
        self.assertEqual(signal.trading_day_id, "2026-03-07")
        # 2026-03-08 17:00 NY is EDT (UTC-4) after the transition -> 21:00 UTC.
        local_valid_until = datetime(2026, 3, 8, 17, 0, tzinfo=NY)
        self.assertEqual(local_valid_until.utcoffset(), timedelta(hours=-4))
        self.assertEqual(signal.valid_until, local_valid_until.astimezone(UTC))
        self.assertEqual(signal.valid_until, datetime(2026, 3, 8, 21, 0, tzinfo=UTC))

    def test_fall_back_valid_until_is_the_correct_utc_instant(self):
        # 2026-11-01: NY clocks fall back 02:00 -> 01:00.
        c = fvg_candidate(known_at=ny_dt(2026, 10, 31, 20, 0))  # trading day started 2026-10-31 18:00
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        signal = timeline.signals[0]
        self.assertEqual(signal.trading_day_id, "2026-10-31")
        # 2026-11-01 17:00 NY is EST (UTC-5) after the transition -> 22:00 UTC.
        local_valid_until = datetime(2026, 11, 1, 17, 0, tzinfo=NY)
        self.assertEqual(local_valid_until.utcoffset(), timedelta(hours=-5))
        self.assertEqual(signal.valid_until, local_valid_until.astimezone(UTC))
        self.assertEqual(signal.valid_until, datetime(2026, 11, 1, 22, 0, tzinfo=UTC))

    def test_query_across_the_dst_transition_still_respects_expiry(self):
        c = fvg_candidate(known_at=ny_dt(2026, 3, 7, 20, 0))
        timeline = build_entry_signal_timeline(fvg_result(c), EMPTY_LIQUIDITY)
        just_before = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 3, 8, 16, 59))
        at_expiry = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 3, 8, 17, 0))
        self.assertEqual(len(just_before.active_signals), 1)
        self.assertEqual(at_expiry.active_signals, [])


class CrossTradingDayTests(unittest.TestCase):
    def test_signals_from_different_trading_days_do_not_combine(self):
        day1_signal = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        day2_signal = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 6, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(day1_signal, day2_signal), EMPTY_LIQUIDITY)
        self.assertNotEqual(timeline.signals[0].trading_day_id, timeline.signals[1].trading_day_id)
        # querying while day1's signal is still valid never sees day2's (not yet known)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 16, 0))
        self.assertEqual([s.candidate_id for s in result.active_signals], [day1_signal.candidate_id])

    def test_a_next_day_opposite_signal_never_invalidates_the_prior_expired_day(self):
        day1_signal = fvg_candidate(zone_id="1m-FVG-1", direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        day2_signal = fvg_candidate(zone_id="1m-FVG-2", direction=SHORT, known_at=ny_dt(2026, 1, 6, 10, 0))
        timeline = build_entry_signal_timeline(fvg_result(day1_signal, day2_signal), EMPTY_LIQUIDITY)
        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 6, 11, 0))
        # day1's LONG signal is gone because it EXPIRED (17:00 on day1), not because it was invalidated.
        self.assertEqual(result.active_signals, [])
        self.assertEqual(result.invalidations, [])


class UnselectedVariantTests(unittest.TestCase):
    def test_unselected_continuation_variant_never_invalidates_selected_reversal(self):
        reversal = liquidity_candidate(
            level_id="1m-swing_high-1", side=HIGH, trigger_phase=CandidatePhase.TAKE,
            entry_mode=EntryMode.REVERSAL, known_at=ny_dt(2026, 1, 5, 10, 0),
        )
        continuation = liquidity_candidate(
            level_id="1m-swing_high-1", side=HIGH, trigger_phase=CandidatePhase.TAKE,
            entry_mode=EntryMode.CONTINUATION, known_at=ny_dt(2026, 1, 5, 10, 0),
        )
        timeline = build_entry_signal_timeline(EMPTY_FVG, liquidity_result(reversal, continuation))
        self.assertNotEqual(reversal.direction, continuation.direction)
        # only the reversal selector is chosen by the strategy
        result = query_active_signals(
            timeline, {"liquidity-1m-swing_equal-swing_high-take-reversal"}, reversal.direction, ny_dt(2026, 1, 5, 12, 0)
        )
        self.assertEqual([s.candidate_id for s in result.active_signals], [reversal.candidate_id])
        self.assertEqual(result.invalidations, [])


class InvalidationTests(unittest.TestCase):
    def test_later_opposite_signal_in_the_same_selector_invalidates_the_earlier_one(self):
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

    def test_selected_opposite_signal_invalidates_the_correct_category_only(self):
        # a same-time opposite signal in a DIFFERENT selector must not interfere
        touch_long = fvg_candidate(zone_id="1m-FVG-1", source_type=SourceType.FVG_TOUCH, direction=LONG, known_at=ny_dt(2026, 1, 5, 10, 0))
        touch_short_later = fvg_candidate(zone_id="1m-FVG-2", source_type=SourceType.FVG_TOUCH, direction=SHORT, known_at=ny_dt(2026, 1, 5, 12, 0))
        ifvg_short_later = fvg_candidate(zone_id="1m-FVG-3", source_type=SourceType.IFVG, direction=SHORT, known_at=ny_dt(2026, 1, 5, 11, 0))
        timeline = build_entry_signal_timeline(fvg_result(touch_long, touch_short_later, ifvg_short_later), EMPTY_LIQUIDITY)

        result = query_active_signals(timeline, {"fvg-1m-fvg_touch"}, LONG, ny_dt(2026, 1, 5, 11, 30))
        # the ifvg-selector short signal (known at 11:00) must NOT invalidate touch_long,
        # since it belongs to a different selector.
        self.assertEqual([s.candidate_id for s in result.active_signals], [touch_long.candidate_id])


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


class BulkSnapshotEquivalenceTests(unittest.TestCase):
    """``iter_active_signal_snapshots`` must agree with ``query_active_
    signals`` at every real signal's own instant -- verified by brute-
    force comparison (never by inspecting the sweep's internals) on a
    small, deliberately overlapping/invalidating fixture and on large
    randomized fixtures spanning a trading-day boundary."""

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
