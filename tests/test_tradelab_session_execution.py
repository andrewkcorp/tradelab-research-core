"""Hard CME session-close execution policy (task 048, backlog item B).

Covers the boundary itself, the forced exit, DST on both sides of both
transitions, an early-close day taken from the calendar, TP/SL still winning
before the forced exit, and the absence of any *unintended* early closing --
plus the three execution paths agreeing with each other under the policy.
"""
from __future__ import annotations

import dataclasses
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.backtest import TradeExitReason, prepare_backtest_context
from tradelab.backtest_v2 import (
    ALL_EXIT_VARIANTS,
    ExecutionSkipReason,
    _simulate_market_execution_reference,
    session_close_index_for,
    simulate_market_execution,
    simulate_market_execution_batched,
)
from tradelab.bar import Bar1m
from tradelab.entry_signal_timeline import EntryEvent, EntryEventKind
from tradelab.entry_strategy_universe import ObservedEntry
from tradelab.fvg_entry import FVGEntryCandidate, SourceType
from tradelab.fvg_touch import FVGContactType
from tradelab.liquidity import HIGH, LOW, LiquidityDetectionResult, LiquidityLevel, LiquidityStatus, LiquidityType
from tradelab.session_calendar import (
    LEGACY_SPECIAL_SCHEDULE_SCHEMA_VERSION,
    ClosedInterval,
    SpecialSessionSchedule,
)
from tradelab.session_execution import (
    ENTRY_BLACKOUT_BEFORE_CLOSE,
    SessionCloseIndex,
    SessionClosePolicy,
    natural_session_close_utc,
)

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
LONG = "long"
SHORT = "short"


def ny(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """A New York wall-clock instant, as UTC. Never UTC arithmetic."""
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


def bar(ts: datetime, o: float, h: float, l: float, c: float) -> Bar1m:
    return Bar1m(timestamp=ts, open=o, high=h, low=l, close=c, volume=1.0, contract="TEST")


def flat_bars(start: datetime, count: int, *, price: float = 100.0, spread: float = 1.0) -> list[Bar1m]:
    """Bars that touch nothing: no TP and no SL can ever fire on them, so any
    exit observed in a test using them is unambiguously the forced close."""
    return [
        bar(start + timedelta(minutes=i), price, price + spread, price - spread, price)
        for i in range(count)
    ]


def liquidity_level(level_id: str, side: str, price: float, formed_at: datetime) -> LiquidityLevel:
    return LiquidityLevel(
        level_id=level_id, timeframe="1m",
        liquidity_type=(LiquidityType.SWING_HIGH if side == HIGH else LiquidityType.SWING_LOW),
        side=side, lower=price, upper=price, member_count=1, member_timestamps=(formed_at,),
        formed_at=formed_at, status=LiquidityStatus.ACTIVE, taken_at=None, absorbed_by_level_id=None,
    )


def observed_long(known_at: datetime, entry_id: str = "sig-1") -> ObservedEntry:
    candidate = FVGEntryCandidate(
        candidate_id=f"1m-{SourceType.FVG_TOUCH.value}-z1", source_type=SourceType.FVG_TOUCH, direction=LONG,
        timeframe="1m", zone_id="z1", lower=99.0, upper=101.0, contact_type=FVGContactType.PARTIAL_TOUCH,
        timestamp=known_at, known_at=known_at,
    )
    event = EntryEvent(
        event_id=entry_id, kind=EntryEventKind.FVG_TOUCH, selector="fvg-1m-fvg_touch", timeframe="1m",
        direction=LONG, timestamp=known_at, known_at=known_at, zone_id="z1", level_id=None, source=candidate,
    )
    return ObservedEntry(entry_id=event.event_id, known_at=event.known_at, direction=event.direction, source=event)


def rigged_context(bars: list[Bar1m], *, tp_price: float, sl_price: float, schedule=None):
    """A context whose liquidity pool holds exactly one TP and one SL level,
    both formed before the first bar, so target resolution is deterministic
    and every test below is about execution timing, not detection."""
    context = prepare_backtest_context(
        bars, [], tick_size=0.25, pivot_strength=2, reclaim_window_bars=3, special_schedule=schedule,
    )
    formed_at = bars[0].timestamp - timedelta(minutes=30)
    return dataclasses.replace(
        context,
        swing_equal_liquidity_results={
            "1m": LiquidityDetectionResult(
                levels=[liquidity_level("tp", HIGH, tp_price, formed_at), liquidity_level("sl", LOW, sl_price, formed_at)],
                takes=[], reclaims=[],
            )
        },
    )


RANK_1 = next(v for v in ALL_EXIT_VARIANTS if v.variant_id == "tp=liquidity_rank_1|sl=liquidity_rank_1")


# ---------------------------------------------------------------------------
# The boundary itself
# ---------------------------------------------------------------------------


class BoundaryTests(unittest.TestCase):
    def test_1559_is_open_and_1600_is_blocked(self):
        policy = SessionClosePolicy()
        self.assertFalse(policy.blocks_new_entry(ny(2024, 5, 6, 15, 59)))
        self.assertTrue(policy.blocks_new_entry(ny(2024, 5, 6, 16, 0)))

    def test_1600_is_inclusive_to_the_exact_minute(self):
        policy = SessionClosePolicy()
        close = policy.session_close_utc(ny(2024, 5, 6, 12, 0))
        self.assertEqual(close, ny(2024, 5, 6, 17, 0))
        self.assertEqual(close - ENTRY_BLACKOUT_BEFORE_CLOSE, ny(2024, 5, 6, 16, 0))

    def test_1659_is_the_forced_exit_bar_and_1658_is_not(self):
        policy = SessionClosePolicy()
        self.assertTrue(policy.is_forced_exit_bar(ny(2024, 5, 6, 16, 59)))
        self.assertFalse(policy.is_forced_exit_bar(ny(2024, 5, 6, 16, 58)))

    def test_a_bar_early_in_the_session_is_neither_blocked_nor_forced(self):
        policy = SessionClosePolicy()
        for hour in (18, 20, 23, 3, 9, 14):
            ts = ny(2024, 5, 7, hour) if hour >= 18 else ny(2024, 5, 8, hour)
            self.assertFalse(policy.blocks_new_entry(ts), hour)
            self.assertFalse(policy.is_forced_exit_bar(ts), hour)


class DSTTests(unittest.TestCase):
    """The boundary is a NY wall-clock fact. Its UTC offset must move by
    itself across both transitions -- never be computed as fixed hours."""

    def test_spring_forward_weekend(self):
        policy = SessionClosePolicy()
        # 2024-03-10 is the spring-forward Sunday.
        before = ny(2024, 3, 8, 16, 59)   # EST, UTC-5
        after = ny(2024, 3, 12, 16, 59)   # EDT, UTC-4
        self.assertEqual(before.hour, 21)
        self.assertEqual(after.hour, 20)
        self.assertTrue(policy.is_forced_exit_bar(before))
        self.assertTrue(policy.is_forced_exit_bar(after))
        self.assertFalse(policy.blocks_new_entry(ny(2024, 3, 8, 15, 59)))
        self.assertFalse(policy.blocks_new_entry(ny(2024, 3, 12, 15, 59)))

    def test_fall_back_weekend(self):
        policy = SessionClosePolicy()
        # 2024-11-03 is the fall-back Sunday.
        before = ny(2024, 11, 1, 16, 59)  # EDT
        after = ny(2024, 11, 5, 16, 59)   # EST
        self.assertEqual(before.hour, 20)
        self.assertEqual(after.hour, 21)
        self.assertTrue(policy.is_forced_exit_bar(before))
        self.assertTrue(policy.is_forced_exit_bar(after))

    def test_natural_close_never_uses_utc_day_arithmetic(self):
        """An 18:00 NY bar on the Friday before spring-forward belongs to the
        session ending 17:00 NY the next calendar day -- which is 23 UTC hours
        later that week, not 24."""
        evening = ny(2024, 3, 8, 18, 0)
        close = natural_session_close_utc(evening)
        self.assertEqual(close, ny(2024, 3, 9, 17, 0))
        self.assertEqual(close - evening, timedelta(hours=23))


class EarlyCloseTests(unittest.TestCase):
    """When the checksummed calendar says the session closes early, both the
    blackout and the forced exit move with it."""

    def setUp(self) -> None:
        # A 13:00 NY early close: everything from 13:00 to the regular 18:00
        # reopen is closed.
        self.schedule = SpecialSessionSchedule(
            schema_version=LEGACY_SPECIAL_SCHEDULE_SCHEMA_VERSION,
            source_url="https://www.cmegroup.com/trading-hours.html",
            retrieved_at=datetime(2026, 8, 22, tzinfo=UTC),
            closed_intervals=(
                ClosedInterval(
                    start_utc=ny(2024, 7, 3, 13, 0), end_utc=ny(2024, 7, 3, 18, 0),
                    reason="Independence Day early close",
                ),
            ),
        )
        self.policy = SessionClosePolicy(self.schedule)

    def test_close_moves_to_the_calendar_early_close(self):
        self.assertEqual(self.policy.session_close_utc(ny(2024, 7, 3, 9, 0)), ny(2024, 7, 3, 13, 0))

    def test_blackout_and_forced_exit_move_with_it(self):
        self.assertFalse(self.policy.blocks_new_entry(ny(2024, 7, 3, 11, 59)))
        self.assertTrue(self.policy.blocks_new_entry(ny(2024, 7, 3, 12, 0)))
        self.assertTrue(self.policy.is_forced_exit_bar(ny(2024, 7, 3, 12, 59)))
        self.assertFalse(self.policy.is_forced_exit_bar(ny(2024, 7, 3, 12, 58)))
        # And 16:00 NY, the regular rule's boundary, is not special that day --
        # the market is already shut.
        self.assertFalse(self.policy.is_forced_exit_bar(ny(2024, 7, 2, 12, 59)))

    def test_an_ordinary_day_is_unaffected_by_the_schedule(self):
        self.assertEqual(self.policy.session_close_utc(ny(2024, 7, 2, 9, 0)), ny(2024, 7, 2, 17, 0))
        self.assertTrue(self.policy.is_forced_exit_bar(ny(2024, 7, 2, 16, 59)))


class IndexTests(unittest.TestCase):
    def test_index_agrees_with_the_policy_bar_for_bar(self):
        policy = SessionClosePolicy()
        bars = flat_bars(ny(2024, 5, 6, 15, 50), 70)
        index = SessionCloseIndex.build(bars, policy)
        for i, b in enumerate(bars):
            self.assertEqual(index.blocks_new_entry(i), policy.blocks_new_entry(b.timestamp), b.timestamp)
            self.assertEqual(index.is_forced_exit_bar(i), policy.is_forced_exit_bar(b.timestamp), b.timestamp)

    def test_next_forced_exit_index(self):
        bars = flat_bars(ny(2024, 5, 6, 15, 50), 70)
        index = SessionCloseIndex.build(bars, SessionClosePolicy())
        first_forced = next(i for i, b in enumerate(bars) if b.timestamp == ny(2024, 5, 6, 16, 59))
        self.assertEqual(index.next_forced_exit_index(0), first_forced)
        self.assertEqual(index.next_forced_exit_index(first_forced), first_forced)
        self.assertIsNone(index.next_forced_exit_index(len(bars)))


# ---------------------------------------------------------------------------
# End to end, through the real execution loops
# ---------------------------------------------------------------------------


class ForcedExitExecutionTests(unittest.TestCase):
    def _run_all_three(self, entries, context, variant=RANK_1):
        accelerated, acc_skipped = simulate_market_execution(entries, LONG, variant, context, "v")
        reference, ref_skipped = _simulate_market_execution_reference(entries, LONG, variant, context, "v")
        batched = simulate_market_execution_batched(entries, LONG, (variant,), context, "cls")
        batch_trades, batch_skipped = batched[variant.variant_id]
        # The three paths must agree on every observable, always.
        self.assertEqual(
            [(t.exit_reason, t.exit_timestamp, t.exit_price) for t in accelerated],
            [(t.exit_reason, t.exit_timestamp, t.exit_price) for t in reference],
            "accelerated vs reference",
        )
        self.assertEqual(
            [(t.exit_reason, t.exit_timestamp, t.exit_price) for t in accelerated],
            [(t.exit_reason, t.exit_timestamp, t.exit_price) for t in batch_trades],
            "accelerated vs batched",
        )
        self.assertEqual({s.reason for s in acc_skipped}, {s.reason for s in ref_skipped})
        self.assertEqual({s.reason for s in acc_skipped}, {s.reason for s in batch_skipped})
        return accelerated, acc_skipped

    def test_open_position_is_force_closed_at_the_1659_close(self):
        start = ny(2024, 5, 6, 15, 30)
        bars = flat_bars(start, 100)
        # TP and SL both far outside every bar's range, so nothing can resolve
        # this trade except the forced close.
        context = rigged_context(bars, tp_price=500.0, sl_price=1.0)
        entry_at = ny(2024, 5, 6, 15, 40)
        trades, skipped = self._run_all_three((observed_long(entry_at),), context)

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.exit_reason, TradeExitReason.SESSION_CLOSE_EXIT)
        self.assertEqual(trade.exit_timestamp, ny(2024, 5, 6, 16, 59))
        forced_bar = next(b for b in bars if b.timestamp == ny(2024, 5, 6, 16, 59))
        self.assertEqual(trade.exit_price, forced_bar.close)

    def test_tp_before_the_forced_exit_still_wins(self):
        start = ny(2024, 5, 6, 15, 30)
        bars = flat_bars(start, 100)
        tp_bar_index = next(i for i, b in enumerate(bars) if b.timestamp == ny(2024, 5, 6, 16, 10))
        bars[tp_bar_index] = bar(bars[tp_bar_index].timestamp, 100.0, 120.0, 99.0, 100.0)
        context = rigged_context(bars, tp_price=110.0, sl_price=1.0)
        trades, _ = self._run_all_three((observed_long(ny(2024, 5, 6, 15, 40)),), context)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_reason, TradeExitReason.TP_HIT)
        self.assertEqual(trades[0].exit_timestamp, ny(2024, 5, 6, 16, 10))

    def test_sl_before_the_forced_exit_still_wins(self):
        start = ny(2024, 5, 6, 15, 30)
        bars = flat_bars(start, 100)
        sl_bar_index = next(i for i, b in enumerate(bars) if b.timestamp == ny(2024, 5, 6, 16, 30))
        bars[sl_bar_index] = bar(bars[sl_bar_index].timestamp, 100.0, 101.0, 80.0, 100.0)
        context = rigged_context(bars, tp_price=500.0, sl_price=90.0)
        trades, _ = self._run_all_three((observed_long(ny(2024, 5, 6, 15, 40)),), context)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_reason, TradeExitReason.SL_HIT)
        self.assertEqual(trades[0].exit_timestamp, ny(2024, 5, 6, 16, 30))

    def test_tp_touched_on_the_forced_exit_bar_itself_wins_over_the_forced_close(self):
        start = ny(2024, 5, 6, 15, 30)
        bars = flat_bars(start, 100)
        forced_index = next(i for i, b in enumerate(bars) if b.timestamp == ny(2024, 5, 6, 16, 59))
        bars[forced_index] = bar(bars[forced_index].timestamp, 100.0, 120.0, 99.0, 100.0)
        context = rigged_context(bars, tp_price=110.0, sl_price=1.0)
        trades, _ = self._run_all_three((observed_long(ny(2024, 5, 6, 15, 40)),), context)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_reason, TradeExitReason.TP_HIT)
        self.assertEqual(trades[0].exit_price, 110.0)

    def test_no_position_is_closed_earlier_than_the_forced_exit_bar(self):
        """The regression this guards: a policy that closed at the start of the
        blackout, or on any bar in it, instead of at the session's last minute."""
        start = ny(2024, 5, 6, 15, 30)
        bars = flat_bars(start, 100)
        context = rigged_context(bars, tp_price=500.0, sl_price=1.0)
        trades, _ = self._run_all_three((observed_long(ny(2024, 5, 6, 15, 40)),), context)
        self.assertEqual(trades[0].exit_timestamp, ny(2024, 5, 6, 16, 59))
        # And the trade genuinely spans the whole blackout hour.
        self.assertLess(trades[0].entry_timestamp, ny(2024, 5, 6, 16, 0))

    def test_a_position_far_from_any_close_is_untouched(self):
        start = ny(2024, 5, 6, 9, 0)
        bars = flat_bars(start, 60)
        context = rigged_context(bars, tp_price=500.0, sl_price=1.0)
        trades, _ = self._run_all_three((observed_long(ny(2024, 5, 6, 9, 10)),), context)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_reason, TradeExitReason.OPEN_UNRESOLVED)


class EntryBlackoutExecutionTests(unittest.TestCase):
    def test_entry_at_1559_is_taken_and_at_1600_is_refused(self):
        start = ny(2024, 5, 6, 15, 30)
        bars = flat_bars(start, 100)
        context = rigged_context(bars, tp_price=500.0, sl_price=1.0)

        taken, taken_skipped = simulate_market_execution((observed_long(ny(2024, 5, 6, 15, 59)),), LONG, RANK_1, context, "v")
        self.assertEqual(taken_skipped, [])
        self.assertEqual(len(taken), 1)

        refused, refused_skipped = simulate_market_execution((observed_long(ny(2024, 5, 6, 16, 0)),), LONG, RANK_1, context, "v")
        self.assertEqual(refused, [])
        self.assertEqual([s.reason for s in refused_skipped], [ExecutionSkipReason.SESSION_CLOSE_BLOCK])

    def test_blackout_is_its_own_skip_reason_not_folded_into_news(self):
        self.assertNotEqual(ExecutionSkipReason.SESSION_CLOSE_BLOCK, ExecutionSkipReason.NEWS_BLOCK)

    def test_every_minute_of_the_blackout_hour_refuses_entry(self):
        start = ny(2024, 5, 6, 15, 30)
        bars = flat_bars(start, 100)
        context = rigged_context(bars, tp_price=500.0, sl_price=1.0)
        for minute in range(0, 60):
            entry_at = ny(2024, 5, 6, 16, minute)
            trades, skipped = simulate_market_execution((observed_long(entry_at, f"sig-{minute}"),), LONG, RANK_1, context, "v")
            self.assertEqual(trades, [], entry_at)
            self.assertEqual([s.reason for s in skipped], [ExecutionSkipReason.SESSION_CLOSE_BLOCK], entry_at)


class ContextWiringTests(unittest.TestCase):
    def test_the_index_is_derived_from_the_context_and_cached(self):
        bars = flat_bars(ny(2024, 5, 6, 15, 30), 20)
        context = rigged_context(bars, tp_price=500.0, sl_price=1.0)
        first = session_close_index_for(context)
        self.assertIs(first, session_close_index_for(context))
        self.assertEqual(len(first), len(bars))

    def test_the_contexts_own_early_close_calendar_is_what_execution_uses(self):
        """Not a default calendar, not a re-loaded one: the schedule pinned on
        the context is what moves the forced exit."""
        early_close_at = ny(2024, 7, 3, 13, 0)
        schedule = SpecialSessionSchedule(
            schema_version=LEGACY_SPECIAL_SCHEDULE_SCHEMA_VERSION,
            source_url="https://www.cmegroup.com/trading-hours.html",
            retrieved_at=datetime(2026, 8, 22, tzinfo=UTC),
            closed_intervals=(ClosedInterval(start_utc=early_close_at, end_utc=ny(2024, 7, 3, 18, 0), reason="early close"),),
        )
        bars = flat_bars(ny(2024, 7, 3, 11, 30), 90)
        bars = [b for b in bars if b.timestamp < early_close_at]
        context = rigged_context(bars, tp_price=500.0, sl_price=1.0, schedule=schedule)
        trades, _ = simulate_market_execution((observed_long(ny(2024, 7, 3, 11, 40)),), LONG, RANK_1, context, "v")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_reason, TradeExitReason.SESSION_CLOSE_EXIT)
        self.assertEqual(trades[0].exit_timestamp, ny(2024, 7, 3, 12, 59))


if __name__ == "__main__":
    unittest.main()


class BundleRoundTripTests(unittest.TestCase):
    """The session-close policy reads the calendar off the context, and in a
    sharded run the worker's context comes out of a bundle. If the bundle
    dropped the schedule, every worker would silently fall back to the plain
    weekly schedule and mis-handle every holiday and early close -- with no
    error anywhere. This asserts it survives.
    """

    def test_the_special_session_schedule_survives_the_shard_bundle(self):
        import tempfile
        from pathlib import Path

        from tradelab.backtest_v2 import ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, build_shard_plan_for_universe
        from tradelab.both_direction_universe import BothDirectionUniverseResult
        from tradelab.shard_bundle import load_context, write_bundle

        schedule = SpecialSessionSchedule(
            schema_version=LEGACY_SPECIAL_SCHEDULE_SCHEMA_VERSION,
            source_url="https://www.cmegroup.com/trading-hours.html",
            retrieved_at=datetime(2026, 8, 22, tzinfo=UTC),
            closed_intervals=(
                ClosedInterval(start_utc=ny(2024, 7, 3, 13, 0), end_utc=ny(2024, 7, 3, 18, 0), reason="early close"),
            ),
        )
        bars = flat_bars(ny(2024, 7, 3, 11, 30), 90)  # 11:30 through 12:59 NY inclusive
        bars = [b for b in bars if b.timestamp < ny(2024, 7, 3, 13, 0)]
        context = rigged_context(bars, tp_price=500.0, sl_price=1.0, schedule=schedule)
        empty_universe = BothDirectionUniverseResult(
            equivalence_classes=(), paired_spec_count=0, equivalence_class_count=0,
        )
        plan = build_shard_plan_for_universe(
            empty_universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, shard_count=1,
            market_artifact_sha256="fixture", news_csv_sha256=None, parameters={}, git_commit="test",
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_bundle(bundle, plan=plan, context=context, equivalence_classes=(), fvg_zone_index={})
            loaded, handles = load_context(bundle)
            try:
                self.assertIsNotNone(loaded.special_session_schedule)
                self.assertEqual(
                    loaded.special_session_schedule.closed_intervals[0].start_utc,
                    ny(2024, 7, 3, 13, 0),
                )
                # And the policy built from the loaded context moves the close.
                index = session_close_index_for(loaded)
                forced = [i for i in range(len(index)) if index.is_forced_exit_bar(i)]
                self.assertEqual(loaded.bars_1m[forced[0]].timestamp, ny(2024, 7, 3, 12, 59))
            finally:
                if handles is not None:
                    handles.close()


class JournalPolicyStampTests(unittest.TestCase):
    """A journal written under a different execution policy must be refused,
    not merged into. This is the mechanism behind "old results are not
    comparable" -- without it, resuming an older run would silently produce a
    single result set containing trades from two different rule sets.
    """

    def test_the_policy_id_is_stamped_into_the_manifest_by_the_engine(self):
        import json
        import sqlite3
        import tempfile
        from pathlib import Path

        from tradelab.accounting import RiskConfig
        from tradelab.backtest_v2 import ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, _open_journal
        from tradelab.equity import EquityConfig
        from tradelab.research_run import RunPhase, create_run
        from tradelab.session_execution import SESSION_CLOSE_POLICY_ID

        with tempfile.TemporaryDirectory() as tmp:
            reporter = create_run(
                runs_dir=Path(tmp) / "runs", market_artifact_path=None, news_csv_path=None,
                parameters={"note": "policy stamp"}, command="test", is_demo=True,
            )
            reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
            journal = _open_journal(
                reporter, market_artifact_path=None, news_csv_path=None,
                parameters={"note": "policy stamp"}, risk_config=RiskConfig(),
                equity_config=EquityConfig(), exit_variants=ALL_EXIT_VARIANTS,
                instruments=DEFAULT_INSTRUMENTS,
            )
            journal.close()
            connection = sqlite3.connect(str(Path(reporter.run_dir) / "backtest_v2_journal.sqlite3"))
            try:
                raw = connection.execute("SELECT parameters_json FROM run_manifest").fetchone()[0]
            finally:
                connection.close()
        stored = json.loads(raw)
        self.assertEqual(stored["session_close_policy"], SESSION_CLOSE_POLICY_ID)
        # The caller's own parameters are preserved alongside it, not replaced.
        self.assertEqual(stored["note"], "policy stamp")

    def test_a_journal_written_under_a_different_policy_refuses_to_resume(self):
        import tempfile
        from pathlib import Path

        from tradelab.backtest_v2_journal import BacktestV2Journal

        def _open(policy: str, db: Path):
            return BacktestV2Journal(
                db, run_id="r1", market_artifact_path=None, market_artifact_sha256=None,
                news_csv_path=None, news_csv_sha256=None, git_commit=None,
                parameters={"session_close_policy": policy}, risk_config={}, equity_config={},
                exit_variant_ids=["v"], instruments=["MNQ"],
            )

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "journal.sqlite3"
            _open("cme-hard-session-close-v1", db).close()
            with self.assertRaises(ValueError) as caught:
                _open("no-session-close-policy", db)
            self.assertIn("parameters", str(caught.exception).lower())
