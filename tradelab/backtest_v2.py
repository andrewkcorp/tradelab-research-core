"""Backtest engine v2: the real, executable backtest on top of the
merged entry-strategy universe (task 039, ``tradelab.entry_strategy_
universe``), with next-bar-open market execution, TP/SL exit-variant
enumeration, MNQ/NQ position sizing, and live-run telemetry via
``tradelab.research_run``.

**Authoritative entry source.** Every trade in this module originates
from ``EntryStrategyEquivalenceClass.observed_entries`` -- the exact,
already-deduplicated, already no-look-ahead-audited entry sequence task
039 already computed. This module never calls ``tradelab.entry_signal_
timeline.query_active_signals``/``iter_active_signal_snapshots`` (the
older, obsolete, persist-until-invalidated model) to determine *when* a
trade fires -- that question is already answered upstream.

**Reused, not reimplemented.** This module deliberately reuses, rather
than re-derives, every piece of already-correct, already-tested
machinery it can:

- ``tradelab.tp_targets``/``tradelab.sl_targets`` for TP/SL candidate
  detection -- both already built around a generic ``MarketEntryContext``
  (entry_id/direction/entry_price/entry_known_at), never the old
  ``EntrySignal`` model, so **no adaptation was needed** here (audited;
  see ``docs/backtest-v2-execution-spec.md``'s own "audit" section for
  the full reasoning).
- ``tradelab.backtest``'s own private ``_check_exit``/``_finalize_
  trade``/``_OpenPosition``/``ExecutedTrade``/``TradeExitReason``/
  ``BacktestResult`` -- the exact, already-tested same-1m-bar SL-first
  tie-break and gross-points computation, imported directly rather than
  re-derived (this package's own established convention for reusing a
  sibling module's private helpers, e.g. ``tradelab.selector_catalog``
  importing ``tradelab.liquidity_entry._DIRECTION``).
- ``tradelab.accounting.apply_accounting`` for MNQ/NQ position sizing
  (nearest to $500 price-risk, never exceeding $550, integer contracts
  only, $20 entry cost charged once per executed trade regardless of
  outcome) and ``tradelab.equity.build_equity_curve`` for the cash-flow
  equity curve, net P&L, max drawdown, and the >$2,000-drawdown episode
  count -- this module never recomputes any of sizing, P&L, or drawdown
  itself.

**What IS new here**: the entry-triggering loop itself (driven by
``ObservedEntry`` instead of a selector query), the exit-variant
grammar (which TP/SL *rule* a given simulation pass uses), and the
telemetry-integrated production runner.

**Market execution, precisely** (see ``docs/backtest-v2-execution-spec.
md`` for the full, standalone writeup):

- Entry fills at the **open of the first 1m bar whose own timestamp is
  ``>=`` the entry event's ``known_at``** -- never the triggering
  candle's own close/high/low. Every ``known_at`` in this whole
  package is already a candle-*close* instant, which is numerically
  identical to the *next* candle's own open instant (bars are
  contiguous 1-minute intervals) -- so "the open of the first bar at or
  after ``known_at``" and "the open of the first bar strictly after the
  candle that revealed the signal" are the same bar. This is exactly
  the rule ``tradelab.backtest.backtest_prepared_strategy`` already
  established (``bar.timestamp in trigger_known_ats`` -> fill at
  ``bar.open``); this module mirrors it exactly, with ``>=`` for
  gap-tolerance (a data gap right at ``known_at`` still fills at the
  next *available* bar, never silently drops the trade).
- TP/SL are evaluated starting with that same entry bar (a violent
  intrabar move can hit either target on the very candle a trade
  opens).
- **Same-1m-bar TP+SL conflict: stop-loss wins**, unconditionally --
  reused verbatim from ``tradelab.backtest._check_exit``.
- One position per (equivalence class, exit variant, instrument) at a
  time; an otherwise-eligible entry while a position is open is
  skipped (``ExecutionSkipReason.POSITION_ALREADY_OPEN``) -- the
  position already open is never touched by it.
- The red-folder news no-entry window (``tradelab.news.is_entry_
  blocked``, ``[scheduled_at-15min, scheduled_at+5min]``) is checked
  against the **actual fill bar's own timestamp**, not the raw signal's
  ``known_at`` -- these can differ when a data gap pushes the fill bar
  later. An already-open position is never force-closed by news.
- **TP/SL source eligibility is likewise evaluated at the actual fill
  instant, not the raw signal's own ``known_at``** -- the
  ``MarketEntryContext`` passed to ``detect_tp_target_candidates``/
  ``detect_sl_target_candidates`` carries ``entry_known_at=fill_bar.
  timestamp``, for exactly the same reason the news check above already
  uses the fill bar: a liquidity level or data wick that becomes known
  strictly *between* the signal's own ``known_at`` and a gap-delayed
  fill is genuinely available by the time the trade actually opens, and
  excluding it would be inconsistent with both the news check and with
  reality. **This is not look-ahead**: the fill bar is, by construction,
  the first available bar at or after the signal's ``known_at`` --
  nothing about this candidate window ever reaches past the instant the
  trade itself opens. The original signal instant is preserved
  separately and only for audit, as ``ExecutedTrade.signal_known_at``
  (``_OpenPosition.signal_known_at=entry.known_at``) -- it is never used
  to filter which TP/SL sources are eligible. In the ordinary, no-gap
  case (the overwhelming majority) the fill bar's timestamp and the
  signal's ``known_at`` are numerically identical, so this only ever
  changes behavior across an actual data gap.

**Exit-variant grammar** (``enumerate_exit_variants``, fixed, 20
combinations, always fully enumerated regardless of a given strategy's
actual data -- an infeasible combination simply produces zero or few
trades, audited via ``skipped_entries_by_reason``, never silently
omitted from the catalog):

- 4 TP kinds: nearest liquidity rank 1/2/3, or nearest data wick
  (exactly one data-wick variant, never ranked -- ``tp_targets.py``
  itself never ranks data wicks).
- 5 SL kinds: nearest opposing liquidity rank 1/2/3, or the entry's own
  **linked** FVG boundary on 5m or 15m (long: lower bound; short: upper
  bound) -- "linked" means the FVG zone that is *this specific entry's
  own* provenance (its own primary FVG/iFVG candidate, or -- for a
  sweep setup entry -- the setup's own underlying FVG/iFVG candidate),
  never an arbitrary unrelated zone merely because its selector was
  part of the strategy's required persistent context.

**Instrument/sizing**: MNQ ($2.00/point) and NQ ($20.00/point), via
``tradelab.accounting`` unchanged -- see that module for the exact
$500-nearest/$550-max/$20-fee algorithm.

**Memory-bounded production orchestration.** ``run_strategy_variant_
backtest`` itself stays rich and unchanged -- for direct/unit use, its
return value (``StrategyVariantResult``) still carries every
``ExecutedTrade``, the full ``AccountBacktestResult``, and the full
``EquityResult`` (equity curve included). The *production runner*
(``run_production_backtest_v2``, and the CLI smoke test, both via the
shared ``_expand_and_simulate``) never accumulates these rich results
across variants, though: a full exhaustive run is, in the worst case,
hundreds of thousands to well over a million (equivalence class, exit
variant, instrument) combinations (see task 040's real discovery
numbers), and retaining every trade/equity point for every one of them
in a single Python list is not something this process could support.
Instead, for every completed variant:

- a small, JSON-friendly ``VariantSummaryRow`` (strategy variant id,
  equivalence class id, direction, instrument, exit variant, net P&L,
  ROI, win rate, trade count, max drawdown, >$2,000-drawdown episode
  count, skipped-entry counts, yearly and monthly summaries -- **no
  trade-level or equity-point detail**) is durably committed
  immediately to ``backtest_v2_journal.sqlite3`` (task 043,
  ``tradelab.backtest_v2_journal.BacktestV2Journal`` -- the
  *authoritative* record and resume source, one row per unique
  ``strategy_variant_id``, a real uniqueness constraint), and, purely
  as a plain-text export/audit mirror never read back for resume
  logic, appended to ``variant_summaries.jsonl`` via the same atomic
  ``tradelab.research_run.append_jsonl`` every other progress event
  already uses;
- the *full* ``StrategyVariantResult`` is offered to a bounded, in-
  memory top-N tracker (``_TopNTracker``, a size-capped min-heap keyed
  by ``(net_pnl_usd, -max_drawdown_usd, strategy_variant_id)`` --
  higher P&L wins, ties broken by lower drawdown, then by the
  variant's own globally-unique id for full determinism) that never
  holds more than ``top_n_leaderboard`` (default 25) rich results at
  once, discarding anything that doesn't make the current cut. Detailed
  trades/equity are therefore retained only *transiently*, per
  evaluated variant, and durably only for whichever handful currently
  sit in the top N -- never for the full population.

The dashboard leaderboard (``reporter.update_leaderboard``) is rebuilt
from this bounded structure at every checkpoint and once more at the
end -- never from a reconstructed giant list. On a resumed run, that
bounded structure additionally starts from the top-N already-completed
summaries read back from the journal (a single bounded SQL query, never
a full-table load) -- see ``_merged_leaderboard``.

**Resumability (task 043).** A crash or interruption is always safe to
resume from: every completed variant's summary is durably committed to
SQLite *before* the run's own progress counters or checkpoint ever
report it as done (see ``BacktestV2Journal.commit_summary``'s own
docstring for the exact ordering guarantee). Calling ``run_production_
backtest_v2`` again with the *same* ``run_id`` (and the same ``runs_
dir``) re-attaches to the existing run directory (never re-creating
``run_manifest.json``, which would erase its original ``created_at``),
opens the same journal, verifies the new call's own market/news
provenance, parameters, risk/equity config, exit variants, instruments,
and code commit against what is already stored (refusing, with a clear
``ValueError`` naming exactly which field(s) differ, on any unsafe
mismatch), and then skips every ``strategy_variant_id`` already
durably completed -- it is never recomputed, not merely re-persisted.
See ``docs/backtest-v2-execution-spec.md`` for the exact resume
command.
"""
from __future__ import annotations

import bisect
import dataclasses
import heapq
import itertools
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from tradelab.accounting import AccountBacktestResult, Instrument, RiskConfig, apply_accounting
from tradelab.backtest import (
    BacktestResult,
    ExecutedTrade,
    PreparedBacktestContext,
    TradeExitReason,
    _check_exit,
    _finalize_trade,
    _OpenPosition,
)
from tradelab.backtest_v2_journal import BacktestV2Journal, JournalVariantSummary
from tradelab.bar import Bar1m
from tradelab.bar_range_index import BarRangeIndex
from tradelab.entry_signal_timeline import LONG, SHORT, EntryEvent, EntryEventKind
from tradelab.entry_strategy_universe import EntryStrategyEquivalenceClass, EntryStrategyUniverseResult, ObservedEntry, discover_entry_strategy_universe
from tradelab.equity import EquityConfig, EquityResult, build_equity_curve
from tradelab.fvg import DEFAULT_TICK_SIZE
from tradelab.fvg_entry import FVGEntryCandidate
from tradelab.news import is_entry_blocked
from tradelab.research_run import LeaderboardEntry, ProgressThrottle, RunPhase, RunReporter, append_jsonl, atomic_write_json, create_run, read_run_manifest
from tradelab.research_run import _git_provenance as _current_git_provenance
from tradelab.research_run import _sha256_of_file as _hash_file
from tradelab.sl_targets import FVG_SL_TIMEFRAMES, SelectedFVGConfluence, SLTargetCandidate, detect_sl_target_candidates
from tradelab.sweep_fvg_setup import SweepFVGSetupEvent
from tradelab.tp_targets import MarketEntryContext, TPTargetCandidate, detect_tp_target_candidates

_UTC = timezone.utc
_DIRECTIONS = (LONG, SHORT)

DEFAULT_RISK_CONFIG = RiskConfig()
DEFAULT_EQUITY_CONFIG = EquityConfig()
DEFAULT_INSTRUMENTS: tuple[Instrument, ...] = (Instrument.MNQ, Instrument.NQ)


# ---------------------------------------------------------------------------
# Exit-variant grammar
# ---------------------------------------------------------------------------

_TP_LIQUIDITY_RANKS: tuple[int, ...] = (1, 2, 3)
_SL_LIQUIDITY_RANKS: tuple[int, ...] = (1, 2, 3)
_SL_FVG_TIMEFRAMES: tuple[str, ...] = tuple(sorted(FVG_SL_TIMEFRAMES))  # ("15m", "5m") -- sorted for determinism


class TPVariantKind(str, Enum):
    LIQUIDITY_RANK = "liquidity_rank"
    DATA_WICK = "data_wick"


class SLVariantKind(str, Enum):
    LIQUIDITY_RANK = "liquidity_rank"
    FVG_BOUNDARY = "fvg_boundary"


@dataclass(frozen=True, order=True)
class ExitVariantSpec:
    """One (TP rule, SL rule) combination -- a *rule*, re-evaluated
    fresh against each individual trade's own entry price/instant, not
    a fixed price chosen once per strategy."""

    variant_id: str
    tp_kind: TPVariantKind
    tp_rank: int | None  # 1..3 for LIQUIDITY_RANK, None for DATA_WICK
    sl_kind: SLVariantKind
    sl_rank: int | None  # 1..3 for LIQUIDITY_RANK, None for FVG_BOUNDARY
    sl_fvg_timeframe: str | None  # "5m"/"15m" for FVG_BOUNDARY, None for LIQUIDITY_RANK

    def __post_init__(self) -> None:
        if self.tp_kind == TPVariantKind.LIQUIDITY_RANK:
            if self.tp_rank not in _TP_LIQUIDITY_RANKS:
                raise ValueError(f"ExitVariantSpec.tp_rank must be one of {_TP_LIQUIDITY_RANKS} for LIQUIDITY_RANK, got {self.tp_rank!r}")
        elif self.tp_rank is not None:
            raise ValueError("ExitVariantSpec.tp_rank must be None for DATA_WICK")
        if self.sl_kind == SLVariantKind.LIQUIDITY_RANK:
            if self.sl_rank not in _SL_LIQUIDITY_RANKS:
                raise ValueError(f"ExitVariantSpec.sl_rank must be one of {_SL_LIQUIDITY_RANKS} for LIQUIDITY_RANK, got {self.sl_rank!r}")
            if self.sl_fvg_timeframe is not None:
                raise ValueError("ExitVariantSpec.sl_fvg_timeframe must be None for LIQUIDITY_RANK")
        else:
            if self.sl_rank is not None:
                raise ValueError("ExitVariantSpec.sl_rank must be None for FVG_BOUNDARY")
            if self.sl_fvg_timeframe not in _SL_FVG_TIMEFRAMES:
                raise ValueError(f"ExitVariantSpec.sl_fvg_timeframe must be one of {_SL_FVG_TIMEFRAMES}, got {self.sl_fvg_timeframe!r}")


def enumerate_exit_variants() -> tuple[ExitVariantSpec, ...]:
    """The full, fixed 4 (TP) x 5 (SL) = 20-combination exit-variant
    catalog -- always enumerated in full, deterministic order,
    regardless of whether a given strategy's own data can actually
    produce trades for every combination (an infeasible one simply
    yields zero or few trades -- see module docstring)."""
    variants: list[ExitVariantSpec] = []
    tp_options: list[tuple[TPVariantKind, int | None, str]] = [(TPVariantKind.LIQUIDITY_RANK, r, f"liquidity_rank_{r}") for r in _TP_LIQUIDITY_RANKS]
    tp_options.append((TPVariantKind.DATA_WICK, None, "data_wick"))
    sl_options: list[tuple[SLVariantKind, int | None, str | None, str]] = [
        (SLVariantKind.LIQUIDITY_RANK, r, None, f"liquidity_rank_{r}") for r in _SL_LIQUIDITY_RANKS
    ]
    sl_options.extend((SLVariantKind.FVG_BOUNDARY, None, tf, f"fvg_boundary_{tf}") for tf in _SL_FVG_TIMEFRAMES)

    for tp_kind, tp_rank, tp_label in tp_options:
        for sl_kind, sl_rank, sl_tf, sl_label in sl_options:
            variants.append(
                ExitVariantSpec(
                    variant_id=f"tp={tp_label}|sl={sl_label}", tp_kind=tp_kind, tp_rank=tp_rank,
                    sl_kind=sl_kind, sl_rank=sl_rank, sl_fvg_timeframe=sl_tf,
                )
            )
    return tuple(variants)


ALL_EXIT_VARIANTS: tuple[ExitVariantSpec, ...] = enumerate_exit_variants()


# ---------------------------------------------------------------------------
# Execution-stage skip auditing
# ---------------------------------------------------------------------------


class ExecutionSkipReason(str, Enum):
    POSITION_ALREADY_OPEN = "skipped_position_already_open"
    NEWS_BLOCK = "news_block"
    NO_BAR_AVAILABLE = "no_bar_available"
    MISSING_TP_TARGET = "missing_tp_target"
    MISSING_SL_TARGET = "missing_sl_target"
    NON_POSITIVE_RISK = "non_positive_risk"


@dataclass(frozen=True)
class SkippedEntryV2:
    entry_id: str
    known_at: datetime
    direction: str
    reason: ExecutionSkipReason


def _linked_fvg_candidate(entry: ObservedEntry) -> FVGEntryCandidate | None:
    """The FVG/iFVG candidate that is *this specific entry's own*
    provenance -- never an arbitrary zone merely active as required
    context (see module docstring). ``None`` if this entry has no FVG
    provenance at all (a pure liquidity-sweep primary with no setup)."""
    source = entry.source
    if isinstance(source, SweepFVGSetupEvent):
        fvg_event = source.fvg_event
        candidate = fvg_event.source
        return candidate if isinstance(candidate, FVGEntryCandidate) else None
    if isinstance(source, EntryEvent):
        if source.kind in (EntryEventKind.FVG_TOUCH, EntryEventKind.IFVG):
            candidate = source.source
            return candidate if isinstance(candidate, FVGEntryCandidate) else None
    return None


def _resolve_tp_candidate(
    variant: ExitVariantSpec,
    market_entry: MarketEntryContext,
    swing_equal_results: Mapping,
    session_results: Mapping,
    data_wick_result,
) -> TPTargetCandidate | None:
    result = detect_tp_target_candidates(market_entry, swing_equal_results, session_results, data_wick_result)
    if variant.tp_kind == TPVariantKind.DATA_WICK:
        return result.data_wick_targets[0] if result.data_wick_targets else None
    for candidate in result.liquidity_targets:
        if candidate.rank == variant.tp_rank:
            return candidate
    return None


def _resolve_sl_candidate(
    variant: ExitVariantSpec,
    entry: ObservedEntry,
    market_entry: MarketEntryContext,
    swing_equal_results: Mapping,
    session_results: Mapping,
) -> SLTargetCandidate | None:
    if variant.sl_kind == SLVariantKind.LIQUIDITY_RANK:
        empty_confluence = SelectedFVGConfluence(entry_id=market_entry.entry_id, fvg_candidates=())
        result = detect_sl_target_candidates(market_entry, empty_confluence, swing_equal_results, session_results)
        for candidate in result.liquidity_targets:
            if candidate.rank == variant.sl_rank:
                return candidate
        return None
    linked = _linked_fvg_candidate(entry)
    if linked is None or linked.timeframe != variant.sl_fvg_timeframe:
        return None
    confluence = SelectedFVGConfluence(entry_id=market_entry.entry_id, fvg_candidates=(linked,))
    result = detect_sl_target_candidates(market_entry, confluence, swing_equal_results, session_results)
    return result.fvg_targets[0] if result.fvg_targets else None


def _find_fill_bar_index(bars: tuple[Bar1m, ...], sorted_timestamps: list[datetime], known_at: datetime, start_from: int) -> int | None:
    """Index of the first bar at or after ``known_at``, searching only
    from ``start_from`` onward (observed entries are processed in
    chronological order, so the search cursor only ever moves forward
    -- O(1) amortized per entry across a whole equivalence class, never
    O(bars) per entry)."""
    idx = bisect.bisect_left(sorted_timestamps, known_at, lo=start_from)
    return idx if idx < len(bars) else None


def _simulate_market_execution_reference(
    observed_entries: tuple[ObservedEntry, ...],
    direction: str,
    variant: ExitVariantSpec,
    context: PreparedBacktestContext,
    strategy_variant_id: str,
    *,
    on_bar_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ExecutedTrade], list[SkippedEntryV2]]:
    """**The slow, unaccelerated reference implementation (task 045).**
    Byte-for-byte the original execution loop -- next-bar-open fill,
    same-1m-bar SL-first tie-break (reused from ``tradelab.backtest``),
    one position at a time, news-blocked at the actual fill instant,
    walking every single 1m bar one at a time regardless of whether a
    position is open. See module docstring for the full rule set.

    Kept, unmodified, **only** as the ground truth for the randomized
    slow-reference-vs-accelerated equivalence test -- ``simulate_market_
    execution`` below is the real, accelerated implementation every
    production/preflight/smoke-test call site actually uses. Never call
    this at real-data scale: it is the exact loop that made a real
    preflight measure 1,091,880 strategy-variant simulations at
    0.0009323/sec (~37 years on one process) -- see ``tradelab.bar_
    range_index``'s own module docstring."""
    bars = context.bars_1m
    total_bars = len(bars)
    sorted_timestamps = [b.timestamp for b in bars]  # already chronological by construction
    trades: list[ExecutedTrade] = []
    skipped: list[SkippedEntryV2] = []
    position: _OpenPosition | None = None
    trade_seq = 0
    bar_cursor = 0

    for entry in observed_entries:
        fill_index = _find_fill_bar_index(bars, sorted_timestamps, entry.known_at, bar_cursor)

        # Advance the position through every bar strictly before the fill
        # bar (or, if no fill bar exists, through every remaining bar) --
        # an already-open position must be checked for exit on all of
        # them, in chronological order, before this entry is even
        # evaluated.
        advance_to = fill_index if fill_index is not None else len(bars)
        while bar_cursor < advance_to:
            if position is not None:
                exit_outcome = _check_exit(position, bars[bar_cursor])
                if exit_outcome is not None:
                    reason, exit_price = exit_outcome
                    trades.append(_finalize_trade(position, exit_timestamp=bars[bar_cursor].timestamp, exit_price=exit_price, reason=reason))
                    position = None
            bar_cursor += 1
            if on_bar_progress is not None:
                on_bar_progress(bar_cursor, total_bars)

        if fill_index is None:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.NO_BAR_AVAILABLE))
            continue

        fill_bar = bars[fill_index]

        if position is not None:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.POSITION_ALREADY_OPEN))
            continue

        blocked, _blocking = is_entry_blocked(fill_bar.timestamp, context.news_events)
        if blocked:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.NEWS_BLOCK))
            continue

        trade_seq += 1
        trade_id = f"{strategy_variant_id}-{trade_seq:06d}-{entry.entry_id}"
        # entry_known_at is the FILL bar's own timestamp, not the raw
        # signal's known_at -- a source that becomes known between the
        # signal and a gap-delayed fill is genuinely available by the
        # time the trade actually opens, and must be eligible (see
        # module docstring's "source eligibility" note; this mirrors the
        # news-window check just above, which already uses fill_bar.
        # timestamp for exactly this reason). The original signal
        # instant is preserved separately, for audit only, as _OpenPosition
        # .signal_known_at below -- never used for target-eligibility filtering.
        market_entry = MarketEntryContext(entry_id=trade_id, direction=direction, entry_price=fill_bar.open, entry_known_at=fill_bar.timestamp)

        tp_candidate = _resolve_tp_candidate(variant, market_entry, context.swing_equal_liquidity_results, context.session_liquidity_results, context.data_wick_result)
        if tp_candidate is None:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.MISSING_TP_TARGET))
            continue

        sl_candidate = _resolve_sl_candidate(variant, entry, market_entry, context.swing_equal_liquidity_results, context.session_liquidity_results)
        if sl_candidate is None:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.MISSING_SL_TARGET))
            continue

        if sl_candidate.distance <= 0 or tp_candidate.distance <= 0:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.NON_POSITIVE_RISK))
            continue

        position = _OpenPosition(
            trade_id=trade_id, strategy_id=strategy_variant_id, direction=direction, signal_known_at=entry.known_at,
            entry_timestamp=fill_bar.timestamp, entry_price=fill_bar.open, tp_candidate=tp_candidate, sl_candidate=sl_candidate,
        )
        # Evaluate TP/SL starting with the entry bar itself (spec).
        exit_outcome = _check_exit(position, fill_bar)
        if exit_outcome is not None:
            reason, exit_price = exit_outcome
            trades.append(_finalize_trade(position, exit_timestamp=fill_bar.timestamp, exit_price=exit_price, reason=reason))
            position = None
        # Deliberately left at fill_index, not fill_index + 1: a later
        # observed entry sharing this exact known_at (a genuine tie) must
        # still be able to resolve to this same bar as its own fill bar
        # -- its own "advance to" search below re-examines this bar
        # (harmless no-op if position already closed) before correctly
        # seeing whatever the real post-processing state is.
        bar_cursor = fill_index

    # Drain any still-open position through the remaining bars.
    while bar_cursor < len(bars) and position is not None:
        exit_outcome = _check_exit(position, bars[bar_cursor])
        if exit_outcome is not None:
            reason, exit_price = exit_outcome
            trades.append(_finalize_trade(position, exit_timestamp=bars[bar_cursor].timestamp, exit_price=exit_price, reason=reason))
            position = None
        bar_cursor += 1
        if on_bar_progress is not None:
            on_bar_progress(bar_cursor, total_bars)

    if position is not None:
        trades.append(_finalize_trade(position, exit_timestamp=None, exit_price=None, reason=TradeExitReason.OPEN_UNRESOLVED))

    return trades, skipped


def _find_first_exit(
    bars: tuple[Bar1m, ...], bar_range_index: BarRangeIndex, position: _OpenPosition, lo: int, hi: int
) -> tuple[int, TradeExitReason, float] | None:
    """The exact bar index in ``[lo, hi)`` where ``position`` first
    exits (TP or SL), or ``None`` if it survives the whole range --
    O(log n) via two independent first-hit range queries (one for the
    long/short-appropriate TP condition, one for SL), reusing ``tradelab.
    backtest._check_exit`` at the single winning bar to get the exact
    same SL-wins-same-bar tie-break the reference implementation applies
    bar by bar.

    **Why this is exact, not approximate**: ``tp_idx``/``sl_idx`` are
    each the true first index in ``[lo, hi)`` satisfying their own
    condition. If they differ, the smaller one is chronologically first
    and ``_check_exit`` at that bar sees only its own condition true
    (if the *other* condition were also true there, the other's own
    first-hit index could not be later -- a direct consequence of both
    being genuine *first*-hit indices). If they're equal (both
    conditions first become true on the identical bar), ``_check_exit``
    applies its own, already-tested SL-first tie-break there exactly as
    the reference bar-by-bar loop would."""
    if position.direction == LONG:
        tp_idx = bar_range_index.first_high_at_least(lo, hi, position.tp_candidate.price)
        sl_idx = bar_range_index.first_low_at_most(lo, hi, position.sl_candidate.price)
    else:
        tp_idx = bar_range_index.first_low_at_most(lo, hi, position.tp_candidate.price)
        sl_idx = bar_range_index.first_high_at_least(lo, hi, position.sl_candidate.price)

    if tp_idx is None and sl_idx is None:
        return None
    if sl_idx is not None and (tp_idx is None or sl_idx <= tp_idx):
        exit_index = sl_idx
    else:
        exit_index = tp_idx

    outcome = _check_exit(position, bars[exit_index])
    assert outcome is not None, "first_high_at_least/first_low_at_most found a qualifying bar, but _check_exit disagreed"
    reason, price = outcome
    return exit_index, reason, price


def simulate_market_execution(
    observed_entries: tuple[ObservedEntry, ...],
    direction: str,
    variant: ExitVariantSpec,
    context: PreparedBacktestContext,
    strategy_variant_id: str,
    *,
    on_bar_progress: Callable[[int, int], None] | None = None,
    bar_range_index: BarRangeIndex | None = None,
) -> tuple[list[ExecutedTrade], list[SkippedEntryV2]]:
    """**The real, accelerated execution loop (task 045).** Identical
    rules to ``_simulate_market_execution_reference`` -- same observed
    entries, fill rules, news filter, target selection, one-position
    rule, TP/SL entry-bar behavior, and SL-wins-same-bar rule -- proven
    identical, not merely believed so, by ``tests/test_tradelab_
    backtest_v2.py``'s randomized slow-reference-vs-accelerated
    equivalence test. Two changes only, both purely mechanical:

    1. **No-position gaps are skipped in O(1)**, not walked bar by bar
       -- when no position is open, the cursor jumps directly to the
       next observed entry's own fill index (or the end of the bar
       sequence), since ``_check_exit`` is a guaranteed no-op with no
       open position.
    2. **While a position is open, its exit is found via ``tradelab.
       bar_range_index.BarRangeIndex`` in O(log n)** instead of an
       O(bars) per-bar scan -- see ``_find_first_exit``.

    ``bar_range_index``, if given, must have been built from this exact
    ``context.bars_1m`` (``BarRangeIndex.build(context.bars_1m)``) --
    callers driving many simulations against the same context should
    build it once and pass it to every call, since the O(n log n) build
    cost is only worth paying once. If omitted, one is built on the fly
    (fine for a single ad-hoc call; wasteful in a loop)."""
    bars = context.bars_1m
    total_bars = len(bars)
    index = bar_range_index if bar_range_index is not None else BarRangeIndex.build(bars)
    trades: list[ExecutedTrade] = []
    skipped: list[SkippedEntryV2] = []
    position: _OpenPosition | None = None
    trade_seq = 0
    bar_cursor = 0

    for entry in observed_entries:
        fill_index = index.find_bar_at_or_after(entry.known_at, bar_cursor)
        advance_to = fill_index if fill_index is not None else len(bars)

        if bar_cursor < advance_to:
            if position is not None:
                hit = _find_first_exit(bars, index, position, bar_cursor, advance_to)
                if hit is not None:
                    exit_index, reason, exit_price = hit
                    trades.append(_finalize_trade(position, exit_timestamp=bars[exit_index].timestamp, exit_price=exit_price, reason=reason))
                    position = None
            bar_cursor = advance_to
            if on_bar_progress is not None:
                on_bar_progress(bar_cursor, total_bars)

        if fill_index is None:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.NO_BAR_AVAILABLE))
            continue

        fill_bar = bars[fill_index]

        if position is not None:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.POSITION_ALREADY_OPEN))
            continue

        blocked, _blocking = is_entry_blocked(fill_bar.timestamp, context.news_events)
        if blocked:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.NEWS_BLOCK))
            continue

        trade_seq += 1
        trade_id = f"{strategy_variant_id}-{trade_seq:06d}-{entry.entry_id}"
        # entry_known_at is the FILL bar's own timestamp, not the raw
        # signal's known_at -- see _simulate_market_execution_reference's
        # own identical comment for the full reasoning (unchanged here).
        market_entry = MarketEntryContext(entry_id=trade_id, direction=direction, entry_price=fill_bar.open, entry_known_at=fill_bar.timestamp)

        tp_candidate = _resolve_tp_candidate(variant, market_entry, context.swing_equal_liquidity_results, context.session_liquidity_results, context.data_wick_result)
        if tp_candidate is None:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.MISSING_TP_TARGET))
            continue

        sl_candidate = _resolve_sl_candidate(variant, entry, market_entry, context.swing_equal_liquidity_results, context.session_liquidity_results)
        if sl_candidate is None:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.MISSING_SL_TARGET))
            continue

        if sl_candidate.distance <= 0 or tp_candidate.distance <= 0:
            skipped.append(SkippedEntryV2(entry_id=entry.entry_id, known_at=entry.known_at, direction=entry.direction, reason=ExecutionSkipReason.NON_POSITIVE_RISK))
            continue

        position = _OpenPosition(
            trade_id=trade_id, strategy_id=strategy_variant_id, direction=direction, signal_known_at=entry.known_at,
            entry_timestamp=fill_bar.timestamp, entry_price=fill_bar.open, tp_candidate=tp_candidate, sl_candidate=sl_candidate,
        )
        # Evaluate TP/SL starting with the entry bar itself (spec) --
        # a single-bar check, same as the reference implementation.
        exit_outcome = _check_exit(position, fill_bar)
        if exit_outcome is not None:
            reason, exit_price = exit_outcome
            trades.append(_finalize_trade(position, exit_timestamp=fill_bar.timestamp, exit_price=exit_price, reason=reason))
            position = None
        # Deliberately left at fill_index, not fill_index + 1 -- see
        # _simulate_market_execution_reference's own identical comment;
        # the next iteration's range query over [fill_index, ...) simply
        # finds nothing new if the position already closed here.
        bar_cursor = fill_index

    # Drain any still-open position through the remaining bars.
    if bar_cursor < len(bars) and position is not None:
        hit = _find_first_exit(bars, index, position, bar_cursor, len(bars))
        if hit is not None:
            exit_index, reason, exit_price = hit
            trades.append(_finalize_trade(position, exit_timestamp=bars[exit_index].timestamp, exit_price=exit_price, reason=reason))
            position = None
            bar_cursor = exit_index + 1
        else:
            bar_cursor = len(bars)
        if on_bar_progress is not None:
            on_bar_progress(bar_cursor, total_bars)

    if position is not None:
        trades.append(_finalize_trade(position, exit_timestamp=None, exit_price=None, reason=TradeExitReason.OPEN_UNRESOLVED))

    return trades, skipped


# ---------------------------------------------------------------------------
# Per-strategy-variant result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyVariantResult:
    strategy_variant_id: str
    equivalence_class_id: str
    exit_variant: ExitVariantSpec
    instrument: Instrument
    direction: str
    trades: tuple[ExecutedTrade, ...]
    account_result: AccountBacktestResult
    equity_result: EquityResult
    net_pnl_usd: float
    roi: float
    win_rate: float | None
    trade_count: int
    max_drawdown_usd: float
    drawdowns_over_2000_count: int
    skipped_entries_by_reason: Mapping[str, int]
    yearly: Mapping[str, Mapping[str, float | int]]
    monthly: Mapping[str, Mapping[str, float | int]]


def _win_rate(account_result: AccountBacktestResult) -> float | None:
    resolved = [t for t in account_result.accounted_trades if t.net_pnl_usd is not None]
    if not resolved:
        return None
    wins = sum(1 for t in resolved if t.net_pnl_usd > 0)
    return wins / len(resolved)


def _yearly_monthly_breakdown(account_result: AccountBacktestResult, equity_result: EquityResult) -> tuple[dict, dict]:
    yearly_pnl: dict[int, float] = {}
    monthly_pnl: dict[str, float] = {}
    for point in equity_result.equity_points:
        year = point.timestamp.year
        month_key = f"{year:04d}-{point.timestamp.month:02d}"
        yearly_pnl[year] = yearly_pnl.get(year, 0.0) + point.cash_delta_usd
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + point.cash_delta_usd

    yearly_count: dict[int, int] = {}
    monthly_count: dict[str, int] = {}
    for trade in account_result.accounted_trades:
        year = trade.entry_timestamp.year
        month_key = f"{year:04d}-{trade.entry_timestamp.month:02d}"
        yearly_count[year] = yearly_count.get(year, 0) + 1
        monthly_count[month_key] = monthly_count.get(month_key, 0) + 1

    yearly = {
        str(year): {"net_pnl_usd": yearly_pnl.get(year, 0.0), "trade_count": yearly_count.get(year, 0)}
        for year in sorted(set(yearly_pnl) | set(yearly_count))
    }
    monthly = {
        key: {"net_pnl_usd": monthly_pnl.get(key, 0.0), "trade_count": monthly_count.get(key, 0)}
        for key in sorted(set(monthly_pnl) | set(monthly_count))
    }
    return yearly, monthly


def strategy_variant_id(equivalence_class_id: str, variant_id: str, instrument_value: str) -> str:
    """The deterministic, globally-unique id of one (equivalence class,
    exit variant, instrument) combination -- a pure function of its own
    three components, computable *before* ever running the (expensive)
    simulation itself. This is what makes skip-already-completed resume
    (task 043) cheap: the production loop can decide whether a variant
    needs to be simulated at all without touching ``run_strategy_
    variant_backtest``."""
    return f"{equivalence_class_id}::{variant_id}::{instrument_value}"


def simulate_variant_execution(
    equivalence_class: EntryStrategyEquivalenceClass,
    variant: ExitVariantSpec,
    context: PreparedBacktestContext,
    *,
    on_bar_progress: Callable[[int, int], None] | None = None,
    bar_range_index: BarRangeIndex | None = None,
) -> tuple[list[ExecutedTrade], list[SkippedEntryV2]]:
    """The **instrument-independent** market execution for one
    (equivalence class, exit variant) -- task 045, requirement 4.
    Observed entries, fill rules, target selection, and exits are all a
    function of price and time alone; MNQ/NQ only ever differ at the
    *accounting* stage (point value, contract sizing) via ``tradelab.
    accounting.apply_accounting`` -- never here. This is what makes it
    safe, and correct, to call this exactly once per (equivalence class,
    exit variant) and reuse its ``(trades, skipped)`` result for every
    instrument, rather than re-running the identical simulation once per
    instrument (``_build_variant_result_for_instrument`` is the cheap,
    O(trade_count) per-instrument step that follows).

    The ``strategy_variant_id`` fed into ``simulate_market_execution``
    (and, from it, into every ``ExecutedTrade.trade_id``/``tp_source_id``/
    ``sl_source_id``) is deliberately the *instrument-neutral*
    ``f"{class_id}::{variant_id}"`` -- those are audit-trail labels, never
    used as lookup keys anywhere downstream (verified: ``tradelab.
    accounting``/``tradelab.equity`` only ever copy ``strategy_id``
    through, never branch on it). The authoritative, instrument-specific
    identifier is ``StrategyVariantResult.strategy_variant_id``, computed
    separately, per instrument, in ``_build_variant_result_for_
    instrument`` below -- unchanged from before this task."""
    direction = equivalence_class.representative_spec.primary.direction
    neutral_id = f"{equivalence_class.class_id}::{variant.variant_id}"
    return simulate_market_execution(
        equivalence_class.observed_entries, direction, variant, context, neutral_id,
        on_bar_progress=on_bar_progress, bar_range_index=bar_range_index,
    )


def _build_variant_result_for_instrument(
    equivalence_class: EntryStrategyEquivalenceClass,
    variant: ExitVariantSpec,
    instrument: Instrument,
    trades: list[ExecutedTrade],
    skipped: list[SkippedEntryV2],
    *,
    risk_config: RiskConfig,
    equity_config: EquityConfig,
) -> StrategyVariantResult:
    """Sizing/P&L (``tradelab.accounting.apply_accounting``, unmodified)
    then equity/drawdown (``tradelab.equity.build_equity_curve``,
    unmodified) for one instrument, from an already-simulated, shared
    ``(trades, skipped)`` pair -- O(trade_count), never re-running
    market execution. ``strategy_variant_id`` here (unlike the neutral
    one fed into the shared simulation) is the real, instrument-specific
    identifier used everywhere downstream (journal, leaderboard,
    ranking) -- unchanged from before this task."""
    direction = equivalence_class.representative_spec.primary.direction
    variant_id = strategy_variant_id(equivalence_class.class_id, variant.variant_id, instrument.value)

    backtest_result = BacktestResult(
        strategy_id=variant_id, trades=trades, skipped_signals=[], trade_count=len(trades),
        tp_hit_count=sum(1 for t in trades if t.exit_reason == TradeExitReason.TP_HIT),
        sl_hit_count=sum(1 for t in trades if t.exit_reason == TradeExitReason.SL_HIT),
        open_unresolved_count=sum(1 for t in trades if t.exit_reason == TradeExitReason.OPEN_UNRESOLVED),
        gross_pnl_points_total=sum(t.gross_pnl_points for t in trades if t.gross_pnl_points is not None),
    )
    account_result = apply_accounting(backtest_result, instrument, risk_config)
    equity_result = build_equity_curve(account_result, equity_config)

    skipped_by_reason: dict[str, int] = {}
    for entry in skipped:
        skipped_by_reason[entry.reason.value] = skipped_by_reason.get(entry.reason.value, 0) + 1
    for untradeable in account_result.untradeable_trades:
        skipped_by_reason[untradeable.reason.value] = skipped_by_reason.get(untradeable.reason.value, 0) + 1

    yearly, monthly = _yearly_monthly_breakdown(account_result, equity_result)

    return StrategyVariantResult(
        strategy_variant_id=variant_id, equivalence_class_id=equivalence_class.class_id, exit_variant=variant,
        instrument=instrument, direction=direction, trades=tuple(trades), account_result=account_result, equity_result=equity_result,
        net_pnl_usd=equity_result.net_cash_pnl_usd, roi=(equity_result.net_cash_pnl_usd / equity_config.initial_balance_usd),
        win_rate=_win_rate(account_result), trade_count=len(account_result.accounted_trades),
        max_drawdown_usd=equity_result.max_drawdown_usd, drawdowns_over_2000_count=equity_result.drawdown_over_threshold_episode_count,
        skipped_entries_by_reason=skipped_by_reason, yearly=yearly, monthly=monthly,
    )


def run_strategy_variant_backtest(
    equivalence_class: EntryStrategyEquivalenceClass,
    variant: ExitVariantSpec,
    instrument: Instrument,
    context: PreparedBacktestContext,
    *,
    risk_config: RiskConfig = DEFAULT_RISK_CONFIG,
    equity_config: EquityConfig = DEFAULT_EQUITY_CONFIG,
    on_bar_progress: Callable[[int, int], None] | None = None,
    bar_range_index: BarRangeIndex | None = None,
) -> StrategyVariantResult:
    """Simulate one full (equivalence class, exit variant, instrument)
    strategy backtest -- a thin, single-instrument convenience wrapper
    around ``simulate_variant_execution`` + ``_build_variant_result_
    for_instrument`` (task 045), preserving this function's own
    pre-existing signature and behavior exactly. Callers evaluating
    *both* MNQ and NQ for the same (equivalence class, exit variant)
    should prefer ``run_strategy_variant_backtest_all_instruments``
    instead, to avoid running market execution twice for an identical
    result.

    ``on_bar_progress``/``bar_range_index`` (task 044/045) are forwarded,
    unmodified, straight through to ``simulate_market_execution`` --
    purely observational/accelerating, never read back into anything
    this function returns."""
    trades, skipped = simulate_variant_execution(equivalence_class, variant, context, on_bar_progress=on_bar_progress, bar_range_index=bar_range_index)
    return _build_variant_result_for_instrument(equivalence_class, variant, instrument, trades, skipped, risk_config=risk_config, equity_config=equity_config)


def run_strategy_variant_backtest_all_instruments(
    equivalence_class: EntryStrategyEquivalenceClass,
    variant: ExitVariantSpec,
    instruments: tuple[Instrument, ...],
    context: PreparedBacktestContext,
    *,
    risk_config: RiskConfig = DEFAULT_RISK_CONFIG,
    equity_config: EquityConfig = DEFAULT_EQUITY_CONFIG,
    on_bar_progress: Callable[[int, int], None] | None = None,
    bar_range_index: BarRangeIndex | None = None,
) -> dict[Instrument, StrategyVariantResult]:
    """Simulate market execution for this (equivalence class, exit
    variant) **exactly once** and apply accounting/equity separately for
    every instrument in ``instruments`` -- task 045, requirement 4. Every
    returned ``StrategyVariantResult`` is exactly what ``run_strategy_
    variant_backtest`` would have returned for that one instrument,
    called individually -- proven by ``tests/test_tradelab_backtest_v2.
    py``'s own equivalence test -- just without repeating the O(entries +
    exits) simulation work per instrument."""
    trades, skipped = simulate_variant_execution(equivalence_class, variant, context, on_bar_progress=on_bar_progress, bar_range_index=bar_range_index)
    return {
        instrument: _build_variant_result_for_instrument(equivalence_class, variant, instrument, trades, skipped, risk_config=risk_config, equity_config=equity_config)
        for instrument in instruments
    }


def _to_leaderboard_entry(result: StrategyVariantResult, rank: int) -> LeaderboardEntry:
    equity_curve = tuple({"ts": p.timestamp.isoformat(), "equity": p.balance_usd} for p in result.equity_result.equity_points)
    trade_pnls = tuple(t.net_pnl_usd for t in result.account_result.accounted_trades if t.net_pnl_usd is not None)
    trades_by_month = {k: v["trade_count"] for k, v in result.monthly.items()}
    return LeaderboardEntry(
        rank=rank, strategy_id=result.strategy_variant_id, pnl=result.net_pnl_usd, roi=result.roi,
        win_rate=(result.win_rate if result.win_rate is not None else 0.0), trade_count=result.trade_count,
        max_drawdown=result.max_drawdown_usd, drawdowns_over_2000_count=result.drawdowns_over_2000_count,
        equity_curve=equity_curve, trade_pnls=trade_pnls, trades_by_month=trades_by_month,
    )


# ---------------------------------------------------------------------------
# Memory-bounded production orchestration: a durable summary row per
# variant (every variant, no trade/equity detail) + a bounded top-N
# tracker of full results (at most top_n_leaderboard at a time) -- see
# module docstring's "Memory-bounded production orchestration" section.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSummaryRow:
    """The durable, all-variant record of one completed (equivalence
    class, exit variant, instrument) backtest -- deliberately lean
    (no ``ExecutedTrade``/``AccountedTrade``/``EquityPoint`` detail) so
    appending one of these per variant, for the entire run, stays cheap
    regardless of how many variants that turns out to be."""

    strategy_variant_id: str
    equivalence_class_id: str
    direction: str
    instrument: str
    exit_variant_id: str
    tp_kind: str
    tp_rank: int | None
    sl_kind: str
    sl_rank: int | None
    sl_fvg_timeframe: str | None
    net_pnl_usd: float
    roi: float
    win_rate: float | None
    trade_count: int
    max_drawdown_usd: float
    drawdowns_over_2000_count: int
    skipped_entries_by_reason: Mapping[str, int]
    yearly: Mapping[str, Mapping[str, float | int]]
    monthly: Mapping[str, Mapping[str, float | int]]

    @staticmethod
    def from_result(result: StrategyVariantResult) -> "VariantSummaryRow":
        v = result.exit_variant
        return VariantSummaryRow(
            strategy_variant_id=result.strategy_variant_id, equivalence_class_id=result.equivalence_class_id,
            direction=result.direction, instrument=result.instrument.value, exit_variant_id=v.variant_id,
            tp_kind=v.tp_kind.value, tp_rank=v.tp_rank, sl_kind=v.sl_kind.value, sl_rank=v.sl_rank, sl_fvg_timeframe=v.sl_fvg_timeframe,
            net_pnl_usd=result.net_pnl_usd, roi=result.roi, win_rate=result.win_rate, trade_count=result.trade_count,
            max_drawdown_usd=result.max_drawdown_usd, drawdowns_over_2000_count=result.drawdowns_over_2000_count,
            skipped_entries_by_reason=dict(result.skipped_entries_by_reason), yearly=dict(result.yearly), monthly=dict(result.monthly),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_variant_id": self.strategy_variant_id, "equivalence_class_id": self.equivalence_class_id,
            "direction": self.direction, "instrument": self.instrument, "exit_variant_id": self.exit_variant_id,
            "tp_kind": self.tp_kind, "tp_rank": self.tp_rank, "sl_kind": self.sl_kind, "sl_rank": self.sl_rank,
            "sl_fvg_timeframe": self.sl_fvg_timeframe, "net_pnl_usd": self.net_pnl_usd, "roi": self.roi,
            "win_rate": self.win_rate, "trade_count": self.trade_count, "max_drawdown_usd": self.max_drawdown_usd,
            "drawdowns_over_2000_count": self.drawdowns_over_2000_count,
            "skipped_entries_by_reason": dict(self.skipped_entries_by_reason), "yearly": dict(self.yearly), "monthly": dict(self.monthly),
        }


def _leaderboard_sort_key(result: StrategyVariantResult) -> tuple[float, float, str]:
    """Higher net P&L wins; ties broken by lower max drawdown, then by
    the variant's own globally-unique id -- fully deterministic (no two
    distinct variants can ever compare exactly equal, since strategy_
    variant_id is unique per (class, exit variant, instrument))."""
    return (result.net_pnl_usd, -result.max_drawdown_usd, result.strategy_variant_id)


class _TopNTracker:
    """A size-capped min-heap holding at most ``n`` full
    ``StrategyVariantResult`` objects at any time -- the current top-N
    by :func:`_leaderboard_sort_key`, regardless of how many total
    variants are ever offered to it. Never grows past ``n`` entries;
    everything that doesn't make the current cut is discarded
    immediately, not merely deferred."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"_TopNTracker size must be >= 1, got {n!r}")
        self._n = n
        self._heap: list[tuple[float, float, str, StrategyVariantResult]] = []

    def offer(self, result: StrategyVariantResult) -> None:
        key = _leaderboard_sort_key(result)
        if len(self._heap) < self._n:
            heapq.heappush(self._heap, (*key, result))
        elif key > self._heap[0][:3]:
            heapq.heappushpop(self._heap, (*key, result))
        # else: not in the current top N -- discarded, never retained.

    def top_results(self) -> list[StrategyVariantResult]:
        """Best-first (rank 1 first) -- an O(n log n) sort over at most
        ``n`` entries, never over the full population ever offered."""
        return [entry[-1] for entry in sorted(self._heap, key=lambda entry: entry[:3], reverse=True)]

    def __len__(self) -> int:
        return len(self._heap)


# ---------------------------------------------------------------------------
# Telemetry-integrated production runner
# ---------------------------------------------------------------------------

_CHECKPOINT_RESUMABLE_NOTE = (
    "Resumable: tradelab.backtest_v2_journal.BacktestV2Journal durably commits "
    "every completed strategy_variant_id to <run_dir>/backtest_v2_journal.sqlite3 "
    "before it is ever reported as completed here. Re-run tradelab.backtest_v2 "
    "with the same --run-id (and the same runs-dir) to continue from this "
    "checkpoint -- already-completed variants are skipped, never recomputed; an "
    "incompatible configuration (market/news provenance, parameters, risk/equity "
    "config, exit variants, instruments, code commit) is refused with a clear error."
)

_EARLY_PHASE_CHECKPOINT_RESUMABLE_NOTE = (
    "Resumable, but not by skipping straight to this exact point: a resumed run "
    "always restarts from preparing_data and deterministically recomputes the "
    "same context and entry-strategy universe (verified, not merely assumed, via "
    "the journal's own manifest compatibility check) before consulting "
    "tradelab.backtest_v2_journal.BacktestV2Journal for which strategy_variant_id "
    "values are already durably completed -- only the simulating_backtests phase "
    "actually skips prior work. This checkpoint still means the run is safe to "
    "interrupt and resume from this point onward, at no cost beyond redoing this "
    "already-cheap phase."
)


def _leaderboard_entry_from_journal_summary(summary: JournalVariantSummary, rank: int = 1) -> LeaderboardEntry:
    """A durably-completed (in a *previous* process) variant, rendered
    as a ``LeaderboardEntry`` for merging with this process's own
    freshly-computed top-N -- deliberately lean: ``equity_curve``/
    ``trade_pnls`` are empty (never durably persisted, see
    ``VariantSummaryRow``'s own docstring), but ``trades_by_month`` is
    still populated from the summary's own durable ``monthly`` trade
    counts, so the dashboard's results-by-month chart still has real
    data for a resumed run's older completions."""
    return LeaderboardEntry(
        rank=rank, strategy_id=summary.strategy_variant_id, pnl=summary.net_pnl_usd, roi=summary.roi,
        win_rate=(summary.win_rate if summary.win_rate is not None else 0.0), trade_count=summary.trade_count,
        max_drawdown=summary.max_drawdown_usd, drawdowns_over_2000_count=summary.drawdowns_over_2000_count,
        equity_curve=(), trade_pnls=(), trades_by_month={k: v.get("trade_count", 0) for k, v in summary.monthly.items()},
    )


def _merged_leaderboard(durable_top_n: list[LeaderboardEntry], tracker: _TopNTracker, n: int) -> list[LeaderboardEntry]:
    """The true top-N across two disjoint populations: variants
    completed in a *previous* process before this run was resumed
    (``durable_top_n`` -- itself already the correct top-N of that
    population, from a single bounded SQL query, never a full-table
    scan) and variants completed *in this process* (``tracker``, the
    existing in-memory bounded structure, unchanged). The two
    populations can never overlap (a variant is completed exactly once,
    ever -- either before this process attached, or by it, never both),
    so concatenating both already-correct top-N lists, re-sorting by
    the identical ranking key, and truncating to ``n`` is exact, not an
    approximation."""
    fresh = [_to_leaderboard_entry(r, 1) for r in tracker.top_results()]  # rank is a placeholder -- re-ranked below
    combined = list(durable_top_n) + fresh
    combined.sort(key=lambda e: (e.pnl, -e.max_drawdown, e.strategy_id), reverse=True)
    return [dataclasses.replace(e, rank=i) for i, e in enumerate(combined[:n], start=1)]


def _expand_and_simulate(
    reporter: RunReporter,
    journal: BacktestV2Journal,
    universe: EntryStrategyUniverseResult,
    exit_variants: tuple[ExitVariantSpec, ...],
    instruments: tuple[Instrument, ...],
    context: PreparedBacktestContext,
    *,
    risk_config: RiskConfig,
    equity_config: EquityConfig,
    top_n_leaderboard: int,
    checkpoint_every: int,
) -> tuple[int, list[LeaderboardEntry]]:
    """Shared by :func:`run_production_backtest_v2` and the CLI smoke
    test: reports ``expanding_exit_variants`` then ``simulating_
    backtests`` then ``ranking_results``, and is the sole place that
    implements memory-bounded, resumable orchestration (see module
    docstring's "Memory-bounded production orchestration" and
    "Resumability" sections). Every completed variant's lean summary is
    durably committed to ``journal`` *before* it is offered to the
    in-memory top-N tracker or counted toward ``completed`` -- crash
    safety (spec, task 043): a process dying at any point still leaves
    the journal exactly reflecting real, durable progress, never more,
    never less. Already-completed variants are never recomputed --
    checked one at a time via ``journal.is_completed()`` (a single bounded
    SQL lookup by primary key), never by loading every completed id into
    a Python set: at real-data scale (millions of variants) that set
    would itself be a significant, unbounded-with-run-size memory
    allocation, which this loop deliberately avoids. Returns
    ``(completed_count, final_top_n_leaderboard)``.

    **Live, second-by-second telemetry (task 044).** Before simulating
    each variant, and repeatedly while ``simulate_market_execution``
    scans through that variant's own bars, this reports a heartbeat --
    current strategy_variant_id, its 1-based position among ``total_
    variant_runs``, and bars-scanned/bars-total -- via ``reporter.
    heartbeat()``, itself throttled to at most once per second and never
    logged as its own durable event (see that method's own docstring).
    This is purely observational: it never affects ``result``, the
    journal, the leaderboard, or which variant is simulated next.

    **Execution acceleration (task 045).** A single ``tradelab.bar_
    range_index.BarRangeIndex`` is built once, here, from ``context.
    bars_1m`` and reused for every simulation this function runs -- see
    that module's own docstring for why this alone turns an O(bars)
    per-variant scan into O(log bars). Market execution itself is run
    **once per (equivalence class, exit variant)**, never once per
    instrument: ``run_strategy_variant_backtest_all_instruments`` shares
    the identical ``(trades, skipped)`` result across every instrument
    in ``instruments``, applying accounting/equity (the only genuinely
    instrument-dependent step) separately for each -- see that
    function's own docstring. Resume correctness is preserved exactly:
    an instrument already durably completed in a previous process is
    still individually skipped (never re-committed to the journal), and
    market execution for a given (equivalence class, exit variant) is
    only ever (re-)run when at least one of its instruments is still
    pending.
    """
    total_variant_runs = universe.equivalence_class_count * len(exit_variants) * len(instruments)
    reporter.phase_started(RunPhase.EXPANDING_EXIT_VARIANTS, items_total=max(total_variant_runs, 1))
    reporter.progress(items_completed=max(total_variant_runs, 1), exit_variants_generated=total_variant_runs)

    reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=max(total_variant_runs, 1))
    summary_path = reporter.run_dir / "variant_summaries.jsonl"
    bar_range_index = BarRangeIndex.build(context.bars_1m)

    completed = journal.completed_count()
    durable_top_n = [_leaderboard_entry_from_journal_summary(s) for s in journal.top_n_summaries(top_n_leaderboard)]

    top_n = _TopNTracker(top_n_leaderboard)
    if completed:
        reporter.progress(items_completed=completed, backtests_completed=completed)
        reporter.update_leaderboard(_merged_leaderboard(durable_top_n, top_n, top_n_leaderboard))

    for equivalence_class in universe.equivalence_classes:
        for variant in exit_variants:
            variant_ids = {instrument: strategy_variant_id(equivalence_class.class_id, variant.variant_id, instrument.value) for instrument in instruments}
            pending_instruments = [instrument for instrument in instruments if not journal.is_completed(variant_ids[instrument])]
            if not pending_instruments:
                continue  # every instrument for this (class, variant) already durably completed

            # task 044: live, second-by-second telemetry -- purely
            # observational (never read back into result/ranking).
            # reporter.heartbeat() is itself throttled to at most
            # once/second; ProgressThrottle additionally keeps the
            # per-bar call site inside simulate_market_execution cheap
            # for a variant that scans millions of bars.
            current_unit_index = completed + 1
            lead_variant_id = variant_ids[pending_instruments[0]]
            reporter.heartbeat(
                operation="simulating_backtests", variant_id=lead_variant_id, unit_index=current_unit_index, unit_total=total_variant_runs,
                bars_scanned=0, bars_total=len(context.bars_1m),
            )
            bar_progress = ProgressThrottle(
                lambda scanned, total, _vid=lead_variant_id, _idx=current_unit_index: reporter.heartbeat(
                    operation="simulating_backtests", variant_id=_vid, unit_index=_idx, unit_total=total_variant_runs,
                    bars_scanned=scanned, bars_total=total,
                )
            )
            # task 045: market execution runs exactly once for this
            # (equivalence class, exit variant), regardless of how many
            # instruments are pending -- accounting/equity (the only
            # genuinely instrument-dependent step) is applied separately
            # for each below.
            results_by_instrument = run_strategy_variant_backtest_all_instruments(
                equivalence_class, variant, tuple(pending_instruments), context, risk_config=risk_config, equity_config=equity_config,
                on_bar_progress=bar_progress.maybe, bar_range_index=bar_range_index,
            )

            for instrument in pending_instruments:
                result = results_by_instrument[instrument]
                row = VariantSummaryRow.from_result(result)
                journal.commit_summary(row.to_dict())  # durable, authoritative -- BEFORE anything below reports it as done
                append_jsonl(summary_path, row.to_dict())  # plain-text export/audit mirror only, never read back for resume
                top_n.offer(result)
                completed += 1
                if completed % checkpoint_every == 0 or completed == total_variant_runs:
                    reporter.progress(items_completed=completed, backtests_completed=completed)
                    reporter.record_resource_usage()
                    reporter.checkpoint_written(
                        cursor={"stage": "simulating_backtests", "completed": completed, "total": total_variant_runs},
                        resumable=True, resumable_note=_CHECKPOINT_RESUMABLE_NOTE,
                    )
                    reporter.update_leaderboard(_merged_leaderboard(durable_top_n, top_n, top_n_leaderboard))

    reporter.phase_started(RunPhase.RANKING_RESULTS, items_total=1)
    final_leaderboard = _merged_leaderboard(durable_top_n, top_n, top_n_leaderboard)
    reporter.update_leaderboard(final_leaderboard)
    reporter.progress(items_completed=1)
    return completed, final_leaderboard


def _open_journal(
    reporter: RunReporter,
    *,
    market_artifact_path: Path | str | None,
    news_csv_path: Path | str | None,
    parameters: Mapping[str, Any],
    risk_config: RiskConfig,
    equity_config: EquityConfig,
    exit_variants: tuple[ExitVariantSpec, ...],
    instruments: tuple[Instrument, ...],
) -> BacktestV2Journal:
    """Open (creating on first use, verifying on every later use) the
    SQLite journal for ``reporter``'s own run directory -- always
    re-hashes ``market_artifact_path``/``news_csv_path`` fresh (never
    trusting a previously-stored hash), so even a file silently changed
    on disk between runs at the same path is caught, not just a changed
    path. Raises ``ValueError`` immediately on any unsafe mismatch
    against an already-existing journal -- see ``BacktestV2Journal``'s
    own docstring.

    **``git_commit`` is independently discovered here, never read from
    ``reporter.manifest.git_commit``.** On resume, ``reporter`` is
    re-attached from the *original* run's own ``run_manifest.json`` (see
    ``run_production_backtest_v2``), so its ``git_commit`` field is
    whatever commit produced that original run -- comparing it against
    itself would trivially always match, even if the code actually
    executing right now has since changed. This function instead calls
    ``tradelab.research_run._git_provenance`` fresh, exactly as
    ``create_run`` itself does for a brand-new run, so the compatibility
    check always reflects the commit genuinely running this call."""
    market_artifact_path = Path(market_artifact_path) if market_artifact_path is not None else None
    news_csv_path = Path(news_csv_path) if news_csv_path is not None else None
    repo_root = Path(__file__).resolve().parents[1]
    current_git_commit, _dirty = _current_git_provenance(repo_root)
    return BacktestV2Journal(
        reporter.run_dir / "backtest_v2_journal.sqlite3",
        run_id=reporter.run_id,
        market_artifact_path=(str(market_artifact_path) if market_artifact_path is not None else None),
        market_artifact_sha256=(_hash_file(market_artifact_path) if market_artifact_path is not None else None),
        news_csv_path=(str(news_csv_path) if news_csv_path is not None else None),
        news_csv_sha256=(_hash_file(news_csv_path) if news_csv_path is not None else None),
        git_commit=current_git_commit,
        parameters=parameters,
        risk_config={"target_risk_usd": risk_config.target_risk_usd, "max_stop_risk_usd": risk_config.max_stop_risk_usd, "entry_cost_usd": risk_config.entry_cost_usd},
        equity_config={"initial_balance_usd": equity_config.initial_balance_usd, "drawdown_threshold_usd": equity_config.drawdown_threshold_usd},
        exit_variant_ids=[v.variant_id for v in exit_variants],
        instruments=[i.value for i in instruments],
    )


def _prepare_data_phase_callback(reporter: RunReporter) -> Callable[[str, float, int | None], None]:
    """Adapts ``prepare_backtest_context``'s own existing, already-tested
    per-phase diagnostic hook (task-independent -- already used by
    ``tradelab.diagnose_context_preparation``) into a task-044 live
    heartbeat -- reused, never reimplemented: this changes nothing about
    what that function computes or in what order, it purely observes
    each phase completing (``phase_callback`` itself fires once per
    detector phase -- 1m validation, timeframe aggregation, one FVG
    detection per timeframe, etc. -- never per-bar, so no additional
    throttling wrapper is needed here beyond ``heartbeat()``'s own)."""
    def _callback(phase_name: str, _duration_seconds: float, _peak_rss_bytes: int | None) -> None:
        reporter.heartbeat(operation=f"preparing_data:{phase_name}")
    return _callback


def _discovery_progress_callback(reporter: RunReporter) -> Callable[[int, int], None]:
    """A ``ProgressThrottle``-wrapped adapter from ``discover_entry_
    strategy_universe``'s own ``on_progress(primaries_processed,
    primaries_total)`` hook (task 044) to a live heartbeat -- throttled
    here (rather than relying solely on ``heartbeat()``'s own throttle)
    since a real dataset's ``items_by_primary`` can itself be a large
    loop, and this keeps the per-primary call site cheap."""
    throttle = ProgressThrottle(
        lambda processed, total: reporter.heartbeat(operation="discovering_entries", unit_index=processed, unit_total=total)
    )
    return throttle.maybe


def _require_clean_working_tree() -> None:
    """Refuses to proceed with a dirty (uncommitted-changes) working tree.

    A matching git commit hash alone is not sufficient for research/
    production reproducibility: two runs against the identical commit can
    still have executed genuinely different code if one of them carried
    uncommitted local edits. ``_open_journal``'s own commit check (see its
    docstring) only ever compares the commit hash, so this is a separate,
    additional guard -- called once, up front, by ``run_production_
    backtest_v2`` itself, before that function does anything else (no data
    preparation, no entry discovery, no simulation, not even attaching to
    or creating a run directory).

    Never fabricates: if git is unavailable or this isn't a checkout at
    all, ``tradelab.research_run._git_provenance`` returns ``dirty=None``
    -- "unknown", not "known clean" -- and this deliberately does not
    block on that (mirrors that function's own best-effort, never-
    fabricate contract); only a definite ``dirty=True`` is refused.
    """
    repo_root = Path(__file__).resolve().parents[1]
    _commit, dirty = _current_git_provenance(repo_root)
    if dirty:
        raise ValueError(
            "tradelab.backtest_v2: refusing to start or resume a production backtest run -- the TradeLab "
            "working tree has uncommitted changes. A production/research run's results must be reproducible "
            "from an exact, committed code commit, so this is refused before any data preparation, entry "
            "discovery, or simulation work begins. Commit, stash, or revert your changes, or start again from "
            "a clean checkout, then re-run."
        )


def run_production_backtest_v2(
    *,
    market_artifact_path: Path | str,
    news_csv_path: Path | str,
    runs_dir: Path | str,
    pivot_strength: int,
    reclaim_window_bars: int,
    tick_size: float = DEFAULT_TICK_SIZE,
    min_entries_per_year: int = 100,
    instruments: tuple[Instrument, ...] = DEFAULT_INSTRUMENTS,
    exit_variants: tuple[ExitVariantSpec, ...] = ALL_EXIT_VARIANTS,
    risk_config: RiskConfig = DEFAULT_RISK_CONFIG,
    equity_config: EquityConfig = DEFAULT_EQUITY_CONFIG,
    top_n_leaderboard: int = 25,
    checkpoint_every: int = 25,
    run_id: str | None = None,
    is_demo: bool = False,
    command: str | None = None,
) -> RunReporter:
    """The future production entry point: load a real market artifact +
    news CSV, discover the entry-strategy universe (task 039), expand
    every exit variant, and simulate every (equivalence class, exit
    variant, instrument) combination -- reporting every phase, periodic
    checkpoints, resource usage, and leaderboard updates through the
    real ``tradelab.research_run`` reporter API, so ``tradelab.research_
    dashboard`` can watch it live in a second terminal.

    **Memory-bounded.** This function itself never holds more than a
    handful of ``StrategyVariantResult`` objects at once -- see module
    docstring. The durable, all-variant record of the run is
    ``runs/<run_id>/backtest_v2_journal.sqlite3`` (the authoritative
    resume source), mirrored, purely as plain-text export/audit, to
    ``variant_summaries.jsonl``.

    **Resumable (task 043).** Pass the *same* ``run_id`` (and the same
    ``runs_dir``) as a previous, interrupted call to resume it: this
    function detects an existing journal at that path, re-attaches to
    the existing run directory (never re-creating ``run_manifest.json``,
    which would erase its original ``created_at``), verifies the
    current call's own configuration against what is already durably
    stored, and skips every already-completed variant -- see module
    docstring's "Resumability" section for the exact fields checked and
    ``docs/backtest-v2-execution-spec.md`` for the exact command. A
    ``run_id`` that has no existing journal (including ``None``, which
    always generates a fresh one) always starts a brand-new run.

    Never called against the full 3-year artifact in this task (see
    ``docs/backtest-v2-execution-spec.md``); intended for a caller-
    chosen, typically much smaller, dataset -- including this module's
    own tiny synthetic smoke test (see ``main(["smoke-test", ...])``).

    **Refuses a dirty working tree (task 043).** Before anything else --
    no data preparation, no entry discovery, no simulation, not even
    attaching to or creating a run directory -- this function requires a
    clean, committed working tree (``_require_clean_working_tree()``),
    on both a fresh start *and* a resume: a matching commit hash alone
    does not guarantee the code that actually ran was reproducible if
    the tree carried uncommitted changes. ``run_preflight_v2`` has no
    such restriction -- it remains usable on a dirty tree, and simply
    records the dirty flag it already captures via its own
    ``run_manifest.json`` (``RunManifest.git_commit_dirty``).
    """
    _require_clean_working_tree()
    runs_dir = Path(runs_dir)
    resuming = run_id is not None and (runs_dir / run_id / "backtest_v2_journal.sqlite3").is_file()

    parameters = {
        "tick_size": tick_size, "pivot_strength": pivot_strength, "reclaim_window_bars": reclaim_window_bars,
        "min_entries_per_year": min_entries_per_year,
    }

    if resuming:
        reporter = RunReporter(runs_dir / run_id, read_run_manifest(runs_dir / run_id))
    else:
        reporter = create_run(
            runs_dir=runs_dir, market_artifact_path=market_artifact_path, news_csv_path=news_csv_path,
            parameters={
                **parameters, "instruments": [i.value for i in instruments], "exit_variant_count": len(exit_variants),
                "risk_config": {"target_risk_usd": risk_config.target_risk_usd, "max_stop_risk_usd": risk_config.max_stop_risk_usd, "entry_cost_usd": risk_config.entry_cost_usd},
                "equity_config": {"initial_balance_usd": equity_config.initial_balance_usd, "drawdown_threshold_usd": equity_config.drawdown_threshold_usd},
            },
            run_id=run_id, command=command, is_demo=is_demo,
        )

    journal = _open_journal(
        reporter, market_artifact_path=market_artifact_path, news_csv_path=news_csv_path, parameters=parameters,
        risk_config=risk_config, equity_config=equity_config, exit_variants=exit_variants, instruments=instruments,
    )
    try:
        from tradelab.backtest import prepare_backtest_context
        from tradelab.market_data_artifact import load_market_data_artifact, load_market_data_series
        from tradelab.news_loaders import read_news_events_csv

        reporter.phase_started(RunPhase.PREPARING_DATA, items_total=4)
        artifact_path = Path(market_artifact_path)
        artifact = load_market_data_artifact(artifact_path)
        reporter.progress(items_completed=1)
        series_result = load_market_data_series(artifact, base_dir=artifact_path.parent)
        reporter.progress(items_completed=2)
        news_events = read_news_events_csv(Path(news_csv_path))
        reporter.progress(items_completed=3)
        context = prepare_backtest_context(
            list(series_result.bars), list(news_events), tick_size=tick_size, pivot_strength=pivot_strength, reclaim_window_bars=reclaim_window_bars,
            phase_callback=_prepare_data_phase_callback(reporter),
        )
        reporter.progress(items_completed=4)
        reporter.record_resource_usage()
        reporter.checkpoint_written(
            cursor={"stage": "preparing_data", "bar_count": len(context.bars_1m)}, resumable=True, resumable_note=_EARLY_PHASE_CHECKPOINT_RESUMABLE_NOTE,
        )

        reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=1)
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=min_entries_per_year,
            on_progress=_discovery_progress_callback(reporter),
        )
        reporter.progress(
            items_completed=1, discovered_specs=universe.syntactically_enumerated_spec_count,
            rejected_specs=universe.specs_rejected_by_yearly_eligibility_count, eligible_specs=universe.eligible_spec_count,
        )

        reporter.phase_started(RunPhase.DEDUPLICATING_ENTRIES, items_total=1)
        reporter.progress(items_completed=1, unique_entry_classes=universe.equivalence_class_count)
        reporter.checkpoint_written(
            cursor={"stage": "deduplicating_entries", "equivalence_class_count": universe.equivalence_class_count},
            resumable=True, resumable_note=_EARLY_PHASE_CHECKPOINT_RESUMABLE_NOTE,
        )

        completed, final_leaderboard = _expand_and_simulate(
            reporter, journal, universe, exit_variants, instruments, context, risk_config=risk_config, equity_config=equity_config,
            top_n_leaderboard=top_n_leaderboard, checkpoint_every=checkpoint_every,
        )

        reporter.completed(
            summary={
                "equivalence_class_count": universe.equivalence_class_count, "exit_variant_count": len(exit_variants),
                "strategy_variant_result_count": completed, "top_net_pnl_usd": (final_leaderboard[0].pnl if final_leaderboard else None),
            }
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure must be reported, never silently crash the run directory into an inconsistent state
        reporter.failed(error=f"{type(exc).__name__}: {exc}", traceback_text=traceback.format_exc())
        raise
    finally:
        journal.close()
    return reporter


# ---------------------------------------------------------------------------
# Real-data preflight (task 043): prepare the real artifact and discover the
# entry universe WITHOUT simulating every variant.
# ---------------------------------------------------------------------------


_TEMPLATE_SUMMARY_ROW = VariantSummaryRow(
    strategy_variant_id="EXAMPLE_CLASS_ID_0000000000000000::tp_first_swing_high_r1__sl_first_swing_low_r1::MNQ",
    equivalence_class_id="EXAMPLE_CLASS_ID_0000000000000000", direction="long", instrument="MNQ",
    exit_variant_id="tp_first_swing_high_r1__sl_first_swing_low_r1", tp_kind="first_swing_high", tp_rank=1,
    sl_kind="first_swing_low", sl_rank=1, sl_fvg_timeframe=None, net_pnl_usd=1234.56, roi=0.1234, win_rate=0.55,
    trade_count=42, max_drawdown_usd=567.89, drawdowns_over_2000_count=1,
    skipped_entries_by_reason={"news_block": 3, "infeasible_size": 1},
    yearly={"2024": {"trade_count": 20, "net_pnl_usd": 600.0}, "2025": {"trade_count": 22, "net_pnl_usd": 634.56}},
    monthly={f"2024-{m:02d}": {"trade_count": 2, "net_pnl_usd": 50.0} for m in range(1, 13)},
)
"""A representative (never persisted, never simulated) ``VariantSummaryRow``
used only to size the *structural* (unmeasured) disk-space estimate the
preflight command reports by default -- see ``_estimate_summary_disk_usage``.
"""


def _estimate_summary_row_bytes() -> int:
    return len(json.dumps(_TEMPLATE_SUMMARY_ROW.to_dict()).encode("utf-8"))


def _iter_variant_combinations(
    universe: EntryStrategyUniverseResult, exit_variants: tuple[ExitVariantSpec, ...], instruments: tuple[Instrument, ...]
):
    """The exact same nested enumeration order ``_expand_and_simulate`` uses
    for production -- reused here (never re-derived) so a preflight sample
    walks a genuine, order-faithful prefix of the real work, and so the
    preflight's own total-simulation count is trivially guaranteed to match
    what production would actually run."""
    for equivalence_class in universe.equivalence_classes:
        for variant in exit_variants:
            for instrument in instruments:
                yield equivalence_class, variant, instrument


def _measure_sample_throughput(
    universe: EntryStrategyUniverseResult, exit_variants: tuple[ExitVariantSpec, ...], instruments: tuple[Instrument, ...],
    context: PreparedBacktestContext, *, risk_config: RiskConfig, equity_config: EquityConfig, sample_size: int,
    reporter: RunReporter | None = None,
) -> dict[str, Any]:
    """Actually simulates up to ``sample_size`` variants (a small,
    caller-chosen prefix of the real enumeration order) purely to *measure*
    (never estimate blind) real per-variant wall-clock cost and real
    per-row summary size. Never simulates the full universe -- ``sample_
    size`` is the caller's own explicit, bounded budget.

    **Live telemetry (task 044).** When ``reporter`` is given, this
    reports a heartbeat before and during each sample (current sample
    number of ``len(combos)``, current strategy_variant_id, and bars-
    scanned/bars-total from inside that sample's own simulation), and
    once more the instant each sample finishes -- so, e.g., ``--
    throughput-sample-size 25`` visibly advances 0/25 -> 1/25 -> 2/25 ->
    ... rather than jumping straight from 0 to 25 at the very end.
    ``RunReporter.heartbeat()`` itself enforces a hard, non-bypassable
    cap of at most one actual write to ``live_status.json`` per second
    (see that method's own docstring): several samples completing within
    the same second still only ever produce one write for that second,
    carrying whichever sample's state was most recent when the write
    became due -- coalesced, never a burst of writes proportional to
    however many samples happened to finish quickly. The very next
    reporter call after this loop (``run_preflight_v2``'s own ``
    progress()``) flushes whatever the last heartbeat call updated in
    memory, so the final sample's own completion is always visible in
    ``preflight.json``'s neighboring ``live_status.json`` immediately
    after this function returns, even if its own heartbeat call was
    itself throttled away.

    **Execution acceleration (task 045).** Builds one ``tradelab.bar_
    range_index.BarRangeIndex`` and reuses it for every sample, and
    groups consecutive combos sharing the same (equivalence class, exit
    variant) -- which, since ``_iter_variant_combinations`` enumerates
    instruments innermost, means every instrument the sample includes
    for a given (equivalence class, exit variant) -- to run market
    execution exactly once per group via ``run_strategy_variant_
    backtest_all_instruments``, exactly mirroring ``_expand_and_
    simulate``'s own dedup. This makes the measured throughput honestly
    reflect production's real, deduplicated cost, not an inflated
    per-instrument-redone estimate."""
    combos = list(itertools.islice(_iter_variant_combinations(universe, exit_variants, instruments), sample_size))
    row_byte_sizes: list[int] = []
    bar_range_index = BarRangeIndex.build(context.bars_1m)
    completed = 0
    start = time.perf_counter()
    for (_class_id, _variant_id), group in itertools.groupby(combos, key=lambda c: (c[0].class_id, c[1].variant_id)):
        group = list(group)
        equivalence_class, variant = group[0][0], group[0][1]
        group_instruments = tuple(c[2] for c in group)
        lead_variant_id = strategy_variant_id(equivalence_class.class_id, variant.variant_id, group_instruments[0].value)
        on_bar_progress = None
        if reporter is not None:
            reporter.heartbeat(
                operation="sampling_throughput", variant_id=lead_variant_id, unit_index=completed + 1, unit_total=len(combos),
                bars_scanned=0, bars_total=len(context.bars_1m),
            )
            bar_progress = ProgressThrottle(
                lambda scanned, total, _vid=lead_variant_id, _idx=completed + 1: reporter.heartbeat(
                    operation="sampling_throughput", variant_id=_vid, unit_index=_idx, unit_total=len(combos),
                    bars_scanned=scanned, bars_total=total,
                )
            )
            on_bar_progress = bar_progress.maybe
        results_by_instrument = run_strategy_variant_backtest_all_instruments(
            equivalence_class, variant, group_instruments, context, risk_config=risk_config, equity_config=equity_config,
            on_bar_progress=on_bar_progress, bar_range_index=bar_range_index,
        )
        for instrument in group_instruments:
            result = results_by_instrument[instrument]
            row = VariantSummaryRow.from_result(result)
            row_byte_sizes.append(len(json.dumps(row.to_dict()).encode("utf-8")))
            completed += 1
            if reporter is not None:
                variant_id = strategy_variant_id(equivalence_class.class_id, variant.variant_id, instrument.value)
                reporter.heartbeat(
                    operation="sampling_throughput", variant_id=variant_id, unit_index=completed, unit_total=len(combos),
                    bars_scanned=len(context.bars_1m), bars_total=len(context.bars_1m),
                )
    elapsed_seconds = time.perf_counter() - start
    variants_per_second = (len(combos) / elapsed_seconds) if elapsed_seconds > 0 else None
    return {
        "sample_size_requested": sample_size,
        "sample_size_measured": len(combos),
        "elapsed_seconds": elapsed_seconds,
        "variants_per_second": variants_per_second,
        "avg_summary_row_bytes": (sum(row_byte_sizes) / len(row_byte_sizes)) if row_byte_sizes else None,
        "note": (
            "MEASURED from a small, caller-configured sample of real simulations -- "
            "not a projection from unmeasured assumptions. Still only an estimate of "
            "full-run duration: real throughput can vary across the full dataset."
        ),
    }


def _estimate_summary_disk_usage(total_variant_simulations: int, sample: dict[str, Any] | None) -> dict[str, Any]:
    if sample is not None and sample.get("avg_summary_row_bytes") is not None:
        bytes_per_row = sample["avg_summary_row_bytes"]
        basis = "measured: average serialized size of the throughput sample's own real summary rows."
    else:
        bytes_per_row = _estimate_summary_row_bytes()
        basis = "structural: serialized size of a representative (unsimulated) template row -- no sample was requested."
    return {
        "estimated_total_bytes": bytes_per_row * total_variant_simulations,
        "estimated_bytes_per_row": bytes_per_row,
        "basis": basis,
        "note": "Estimate only -- real per-row size varies with trade_count and the number of distinct months/years touched.",
    }


def run_preflight_v2(
    *,
    market_artifact_path: Path | str,
    news_csv_path: Path | str,
    runs_dir: Path | str,
    pivot_strength: int,
    reclaim_window_bars: int,
    tick_size: float = DEFAULT_TICK_SIZE,
    min_entries_per_year: int = 100,
    instruments: tuple[Instrument, ...] = DEFAULT_INSTRUMENTS,
    exit_variants: tuple[ExitVariantSpec, ...] = ALL_EXIT_VARIANTS,
    risk_config: RiskConfig = DEFAULT_RISK_CONFIG,
    equity_config: EquityConfig = DEFAULT_EQUITY_CONFIG,
    throughput_sample_size: int = 0,
    run_id: str | None = None,
    is_demo: bool = False,
    command: str | None = None,
) -> RunReporter:
    """Prepares the real market artifact + news CSV and discovers the real
    entry-strategy universe (task 039) -- exactly the same ``prepare_
    backtest_context``/``discover_entry_strategy_universe`` calls ``run_
    production_backtest_v2`` itself makes -- but never expands or simulates
    a single strategy variant (unless ``throughput_sample_size`` opts into a
    small, explicitly bounded, clearly-labelled measured sample). Because
    the universe discovery step is identical code, run identically, the
    preflight's own counts can never drift from what a real run would
    later use, and it can never omit or alter the final strategy universe.

    Writes a durable ``preflight.json`` artifact into the run directory
    (via the same ``tradelab.research_run`` telemetry model ``tradelab.
    research_dashboard`` already understands -- ``run_manifest.json``,
    phase/progress events, and a normal ``completed`` summary are all
    written exactly as a production run would write them), containing:
    bar count + covered time range, syntactically enumerated specs,
    eligible specs, unique equivalence classes, exit variant count,
    instruments, the *exact* total strategy-variant simulation count,
    a disk-space estimate for the eventual ``variant_summaries.jsonl``/
    journal output, and -- only if ``throughput_sample_size > 0`` -- a
    measured throughput estimate from that small sample, clearly labelled
    as an estimate.
    """
    from tradelab.backtest import prepare_backtest_context
    from tradelab.market_data_artifact import load_market_data_artifact, load_market_data_series
    from tradelab.news_loaders import read_news_events_csv

    reporter = create_run(
        runs_dir=runs_dir, market_artifact_path=market_artifact_path, news_csv_path=news_csv_path,
        parameters={
            "mode": "preflight", "tick_size": tick_size, "pivot_strength": pivot_strength,
            "reclaim_window_bars": reclaim_window_bars, "min_entries_per_year": min_entries_per_year,
            "instruments": [i.value for i in instruments], "exit_variant_count": len(exit_variants),
            "throughput_sample_size": throughput_sample_size,
        },
        run_id=run_id, command=command, is_demo=is_demo,
    )
    try:
        reporter.phase_started(RunPhase.PREPARING_DATA, items_total=4)
        artifact_path = Path(market_artifact_path)
        artifact = load_market_data_artifact(artifact_path)
        reporter.progress(items_completed=1)
        series_result = load_market_data_series(artifact, base_dir=artifact_path.parent)
        reporter.progress(items_completed=2)
        news_events = read_news_events_csv(Path(news_csv_path))
        reporter.progress(items_completed=3)
        context = prepare_backtest_context(
            list(series_result.bars), list(news_events), tick_size=tick_size, pivot_strength=pivot_strength, reclaim_window_bars=reclaim_window_bars,
            phase_callback=_prepare_data_phase_callback(reporter),
        )
        reporter.progress(items_completed=4)
        reporter.record_resource_usage()
        reporter.checkpoint_written(
            cursor={"stage": "preparing_data", "bar_count": len(context.bars_1m)}, resumable=False,
            resumable_note="Preflight is a read-only, single-pass diagnostic -- it never simulates the full universe, so there is nothing to resume.",
        )

        reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=1)
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=min_entries_per_year,
            on_progress=_discovery_progress_callback(reporter),
        )
        reporter.progress(
            items_completed=1, discovered_specs=universe.syntactically_enumerated_spec_count,
            rejected_specs=universe.specs_rejected_by_yearly_eligibility_count, eligible_specs=universe.eligible_spec_count,
        )

        reporter.phase_started(RunPhase.DEDUPLICATING_ENTRIES, items_total=1)
        reporter.progress(items_completed=1, unique_entry_classes=universe.equivalence_class_count)

        total_variant_simulations = universe.equivalence_class_count * len(exit_variants) * len(instruments)

        throughput_sample = None
        if throughput_sample_size > 0:
            reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=min(throughput_sample_size, total_variant_simulations))
            throughput_sample = _measure_sample_throughput(
                universe, exit_variants, instruments, context, risk_config=risk_config, equity_config=equity_config,
                sample_size=throughput_sample_size, reporter=reporter,
            )
            reporter.progress(items_completed=throughput_sample["sample_size_measured"])

        bars = context.bars_1m
        disk_estimate = _estimate_summary_disk_usage(total_variant_simulations, throughput_sample)
        preflight = {
            "market_artifact_path": str(artifact_path),
            "news_csv_path": str(Path(news_csv_path)),
            "bar_count": len(bars),
            "covered_time_range": {
                "start": bars[0].timestamp.isoformat() if bars else None,
                "end": bars[-1].timestamp.isoformat() if bars else None,
            },
            "syntactically_enumerated_spec_count": universe.syntactically_enumerated_spec_count,
            "eligible_spec_count": universe.eligible_spec_count,
            "equivalence_class_count": universe.equivalence_class_count,
            "exit_variant_count": len(exit_variants),
            "instruments": [i.value for i in instruments],
            "total_strategy_variant_simulations": total_variant_simulations,
            "disk_space_estimate": disk_estimate,
            "throughput_estimate": throughput_sample,
        }
        reporter.phase_started(RunPhase.RANKING_RESULTS, items_total=1)
        atomic_write_json(reporter.run_dir / "preflight.json", preflight)
        reporter.progress(items_completed=1)

        reporter.completed(
            summary={
                "mode": "preflight", "equivalence_class_count": universe.equivalence_class_count,
                "exit_variant_count": len(exit_variants), "total_strategy_variant_simulations": total_variant_simulations,
            }
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure must be reported, never silently crash the run directory into an inconsistent state
        reporter.failed(error=f"{type(exc).__name__}: {exc}", traceback_text=traceback.format_exc())
        raise
    return reporter


def read_preflight(run_dir: Path | str) -> dict[str, Any]:
    """Reads back a durable ``preflight.json`` artifact written by ``run_
    preflight_v2`` -- the same read pattern ``tradelab.research_run.read_
    run_manifest``/``read_leaderboard`` already use for this run directory's
    other durable artifacts."""
    with open(Path(run_dir) / "preflight.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI + tiny synthetic smoke test
# ---------------------------------------------------------------------------


def build_synthetic_smoke_context() -> PreparedBacktestContext:
    """A tiny (900 bars, 15 hours), fully synthetic, deterministic 1m
    bar series -- never real market data. Deliberately straddles the
    2024/2025 UTC year boundary, since ``tradelab.entry_strategy_
    universe``'s eligibility windows are the real, fixed calendar years
    2024/2025 -- a synthetic run still needs at least one entry landing
    in each to demonstrate the full pipeline (even with ``--min-
    entries-per-year 1``). A slow, regular zigzag (20 points every
    minute, for 30-minute legs, versus a 0.25 tick size) reliably forms
    real FVGs and swing highs/lows on 1m *and* on the 5m/15m aggregates
    (each leg is long enough to still look cleanly directional after
    aggregation) in so few bars, without needing a realistic price
    process -- this is a mechanism smoke test, not a market simulation.
    Starts just past the daily NY 22:00-23:00 UTC maintenance-break
    window and stays clear of the next one entirely.
    """
    from tradelab.backtest import prepare_backtest_context

    start = datetime(2024, 12, 31, 23, 5, tzinfo=_UTC)
    bars: list[Bar1m] = []
    price = 20000.0
    leg_length = 30
    for i in range(900):
        ts = start + timedelta(minutes=i)
        cycle_bar = i % (2 * leg_length)
        cycle_index = i // (2 * leg_length)
        # A widening zigzag: each full cycle returns near the same base
        # price, but with a larger amplitude than the last -- so every
        # later swing high/low genuinely exceeds ("takes") the previous
        # one, rather than merely retracing the exact same range.
        amplitude = 20.0 + cycle_index * 5.0
        move = amplitude if cycle_bar < leg_length else -amplitude
        open_p = price
        close_p = price + move
        high_p = max(open_p, close_p) + 3.0
        low_p = min(open_p, close_p) - 3.0
        bars.append(Bar1m(timestamp=ts, open=open_p, high=high_p, low=low_p, close=close_p, volume=100.0, contract="SMOKE"))
        price = close_p

    return prepare_backtest_context(bars, [], tick_size=0.25, pivot_strength=2, reclaim_window_bars=3)


def run_execution_acceleration_benchmark(
    context: PreparedBacktestContext,
    universe: EntryStrategyUniverseResult,
    exit_variants: tuple[ExitVariantSpec, ...],
    *,
    sample_size: int,
) -> dict[str, Any]:
    """Benchmarks ``_simulate_market_execution_reference`` (slow,
    unaccelerated) against ``simulate_market_execution`` (accelerated)
    over an identical, bounded sample of up to ``sample_size`` real
    (equivalence class, exit variant) combinations from ``universe`` --
    task 045, requirement 5. Reports *measured* wall-clock timings for
    both and confirms exact result equality on the sample; never claims
    a throughput number without measuring it here (the sample is real
    execution, not a projection).

    ``context`` may be the deterministic synthetic smoke fixture
    (``build_synthetic_smoke_context()``) or a real market artifact's
    own prepared context (optionally truncated to a bar prefix) -- see
    ``main(["benchmark", ...])``."""
    combos: list[tuple[EntryStrategyEquivalenceClass, ExitVariantSpec]] = []
    for equivalence_class in universe.equivalence_classes:
        for variant in exit_variants:
            combos.append((equivalence_class, variant))
            if len(combos) >= sample_size:
                break
        if len(combos) >= sample_size:
            break

    bar_range_index = BarRangeIndex.build(context.bars_1m)

    reference_start = time.perf_counter()
    reference_results = [
        _simulate_market_execution_reference(
            equivalence_class.observed_entries, equivalence_class.representative_spec.primary.direction, variant, context,
            f"{equivalence_class.class_id}::{variant.variant_id}",
        )
        for equivalence_class, variant in combos
    ]
    reference_elapsed_seconds = time.perf_counter() - reference_start

    accelerated_start = time.perf_counter()
    accelerated_results = [
        simulate_market_execution(
            equivalence_class.observed_entries, equivalence_class.representative_spec.primary.direction, variant, context,
            f"{equivalence_class.class_id}::{variant.variant_id}", bar_range_index=bar_range_index,
        )
        for equivalence_class, variant in combos
    ]
    accelerated_elapsed_seconds = time.perf_counter() - accelerated_start

    return {
        "sample_size": len(combos),
        "bar_count": len(context.bars_1m),
        "equivalence_class_count": universe.equivalence_class_count,
        "exit_variant_count": len(exit_variants),
        "reference_elapsed_seconds": reference_elapsed_seconds,
        "accelerated_elapsed_seconds": accelerated_elapsed_seconds,
        "reference_variants_per_second": (len(combos) / reference_elapsed_seconds) if reference_elapsed_seconds > 0 else None,
        "accelerated_variants_per_second": (len(combos) / accelerated_elapsed_seconds) if accelerated_elapsed_seconds > 0 else None,
        "speedup_factor": (reference_elapsed_seconds / accelerated_elapsed_seconds) if accelerated_elapsed_seconds > 0 else None,
        "exact_result_equality": reference_results == accelerated_results,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="tradelab.backtest_v2: real backtest engine on the entry-strategy universe.")
    sub = parser.add_subparsers(dest="mode", required=True)

    real = sub.add_parser("run", help="Run against a real market artifact + news CSV (never the full 3-year set from this CLI without care).")
    real.add_argument("--market-artifact", required=True, type=Path)
    real.add_argument("--news-csv", required=True, type=Path)
    real.add_argument("--runs-dir", default="runs")
    real.add_argument("--pivot-strength", type=int, required=True)
    real.add_argument("--reclaim-window-bars", type=int, required=True)
    real.add_argument("--min-entries-per-year", type=int, default=100)
    real.add_argument(
        "--run-id", default=None,
        help="Resume an interrupted run: pass the SAME --run-id used on the original invocation "
        "(printed as 'run_id: ...' when that run started). Requires identical market artifact, "
        "news CSV, parameters, risk/equity config, exit variants, and instruments -- verified "
        "against runs/<run-id>/backtest_v2_journal.sqlite3 and refused with a clear error on mismatch. "
        "Omit to start a brand-new run.",
    )

    preflight = sub.add_parser(
        "preflight", help="Prepare the real artifact + discover the entry universe WITHOUT simulating every variant; writes a durable preflight.json."
    )
    preflight.add_argument("--market-artifact", required=True, type=Path)
    preflight.add_argument("--news-csv", required=True, type=Path)
    preflight.add_argument("--runs-dir", default="runs")
    preflight.add_argument("--pivot-strength", type=int, required=True)
    preflight.add_argument("--reclaim-window-bars", type=int, required=True)
    preflight.add_argument("--min-entries-per-year", type=int, default=100)
    preflight.add_argument(
        "--throughput-sample-size", type=int, default=0,
        help="Optional: actually simulate this many variants (a small, bounded prefix) to measure real "
        "throughput and real per-row disk usage, clearly labelled as an estimate in preflight.json. "
        "Default 0 simulates nothing at all.",
    )

    smoke = sub.add_parser("smoke-test", help="Run a tiny, deterministic, fully synthetic end-to-end smoke test (writes a real monitored run directory).")
    smoke.add_argument("--runs-dir", default="runs")
    smoke.add_argument("--min-entries-per-year", type=int, default=1)

    benchmark = sub.add_parser(
        "benchmark",
        help="Benchmark the accelerated execution loop (task 045) against the slow reference implementation, on the "
        "deterministic synthetic fixture by default, or a real market artifact (optionally truncated to a bar prefix) "
        "-- reports measured baseline/accelerated timings, variants/sec, speedup, and exact-result equality.",
    )
    benchmark.add_argument("--market-artifact", type=Path, default=None, help="Optional real market artifact manifest; omit to use the deterministic synthetic fixture.")
    benchmark.add_argument("--news-csv", type=Path, default=None)
    benchmark.add_argument("--prefix-bars", type=int, default=None, help="Truncate the loaded real artifact to this many leading bars (ignored without --market-artifact).")
    benchmark.add_argument("--pivot-strength", type=int, default=2)
    benchmark.add_argument("--reclaim-window-bars", type=int, default=3)
    benchmark.add_argument("--min-entries-per-year", type=int, default=1)
    benchmark.add_argument("--sample-size", type=int, default=200, help="Number of (equivalence class, exit variant) combinations to benchmark.")

    args = parser.parse_args(argv)

    if args.mode == "run":
        reporter = run_production_backtest_v2(
            market_artifact_path=args.market_artifact, news_csv_path=args.news_csv, runs_dir=args.runs_dir,
            pivot_strength=args.pivot_strength, reclaim_window_bars=args.reclaim_window_bars, min_entries_per_year=args.min_entries_per_year,
            run_id=args.run_id, command="python3 -m tradelab.backtest_v2 run",
        )
    elif args.mode == "preflight":
        reporter = run_preflight_v2(
            market_artifact_path=args.market_artifact, news_csv_path=args.news_csv, runs_dir=args.runs_dir,
            pivot_strength=args.pivot_strength, reclaim_window_bars=args.reclaim_window_bars, min_entries_per_year=args.min_entries_per_year,
            throughput_sample_size=args.throughput_sample_size, command="python3 -m tradelab.backtest_v2 preflight",
        )
    elif args.mode == "benchmark":
        if args.market_artifact is not None:
            from tradelab.backtest import prepare_backtest_context
            from tradelab.market_data_artifact import load_market_data_artifact, load_market_data_series
            from tradelab.news_loaders import read_news_events_csv

            artifact = load_market_data_artifact(args.market_artifact)
            series_result = load_market_data_series(artifact, base_dir=args.market_artifact.parent)
            bars = list(series_result.bars)
            if args.prefix_bars is not None:
                bars = bars[: args.prefix_bars]
            news_events = list(read_news_events_csv(args.news_csv)) if args.news_csv is not None else []
            context = prepare_backtest_context(bars, news_events, pivot_strength=args.pivot_strength, reclaim_window_bars=args.reclaim_window_bars)
        else:
            context = build_synthetic_smoke_context()
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=args.min_entries_per_year
        )
        report = run_execution_acceleration_benchmark(context, universe, ALL_EXIT_VARIANTS, sample_size=args.sample_size)
        print(json.dumps(report, indent=2))
        return 0
    else:
        context = build_synthetic_smoke_context()
        reporter = create_run(
            runs_dir=args.runs_dir, market_artifact_path=None, news_csv_path=None,
            parameters={"note": "tiny synthetic smoke test", "min_entries_per_year": args.min_entries_per_year},
            command="python3 -m tradelab.backtest_v2 smoke-test", is_demo=True,
        )
        _run_smoke_pipeline(reporter, context, min_entries_per_year=args.min_entries_per_year)

    print(f"Run written to {reporter.run_dir}")
    print(f"run_id: {reporter.run_id}")
    return 0


def _run_smoke_pipeline(reporter: RunReporter, context: PreparedBacktestContext, *, min_entries_per_year: int) -> None:
    """Drives the same memory-bounded ``_expand_and_simulate`` the
    production runner uses -- the smoke test's own scale is tiny, but
    it exercises exactly the same durable-summary-stream + bounded-top-N
    orchestration path, not a separate, divergent implementation."""
    try:
        reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
        reporter.progress(items_completed=1)

        reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=1)
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=min_entries_per_year
        )
        reporter.progress(items_completed=1, discovered_specs=universe.syntactically_enumerated_spec_count, eligible_specs=universe.eligible_spec_count)

        reporter.phase_started(RunPhase.DEDUPLICATING_ENTRIES, items_total=1)
        reporter.progress(items_completed=1, unique_entry_classes=universe.equivalence_class_count)

        journal = _open_journal(
            reporter, market_artifact_path=None, news_csv_path=None,
            parameters={"note": "tiny synthetic smoke test", "min_entries_per_year": min_entries_per_year},
            risk_config=DEFAULT_RISK_CONFIG, equity_config=DEFAULT_EQUITY_CONFIG,
            exit_variants=ALL_EXIT_VARIANTS, instruments=DEFAULT_INSTRUMENTS,
        )
        try:
            completed, final_leaderboard = _expand_and_simulate(
                reporter, journal, universe, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, context, risk_config=DEFAULT_RISK_CONFIG,
                equity_config=DEFAULT_EQUITY_CONFIG, top_n_leaderboard=10, checkpoint_every=25,
            )
        finally:
            journal.close()
        reporter.completed(
            summary={"strategy_variant_result_count": completed, "top_net_pnl_usd": (final_leaderboard[0].pnl if final_leaderboard else None)}
        )
    except Exception as exc:  # noqa: BLE001
        reporter.failed(error=f"{type(exc).__name__}: {exc}", traceback_text=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
