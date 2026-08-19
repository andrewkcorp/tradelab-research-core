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
"""
from __future__ import annotations

import bisect
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Mapping

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
from tradelab.bar import Bar1m
from tradelab.entry_signal_timeline import LONG, SHORT, EntryEvent, EntryEventKind
from tradelab.entry_strategy_universe import EntryStrategyEquivalenceClass, ObservedEntry, discover_entry_strategy_universe
from tradelab.equity import EquityConfig, EquityResult, build_equity_curve
from tradelab.fvg import DEFAULT_TICK_SIZE
from tradelab.fvg_entry import FVGEntryCandidate
from tradelab.news import is_entry_blocked
from tradelab.research_run import LeaderboardEntry, RunPhase, RunReporter, create_run
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


def simulate_market_execution(
    observed_entries: tuple[ObservedEntry, ...],
    direction: str,
    variant: ExitVariantSpec,
    context: PreparedBacktestContext,
    strategy_variant_id: str,
) -> tuple[list[ExecutedTrade], list[SkippedEntryV2]]:
    """The core, new execution loop -- next-bar-open fill, same-1m-bar
    SL-first tie-break (reused from ``tradelab.backtest``), one
    position at a time, news-blocked at the actual fill instant. See
    module docstring for the full rule set."""
    bars = context.bars_1m
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
        market_entry = MarketEntryContext(entry_id=trade_id, direction=direction, entry_price=fill_bar.open, entry_known_at=entry.known_at)

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


def run_strategy_variant_backtest(
    equivalence_class: EntryStrategyEquivalenceClass,
    variant: ExitVariantSpec,
    instrument: Instrument,
    context: PreparedBacktestContext,
    *,
    risk_config: RiskConfig = DEFAULT_RISK_CONFIG,
    equity_config: EquityConfig = DEFAULT_EQUITY_CONFIG,
) -> StrategyVariantResult:
    """Simulate one full (equivalence class, exit variant, instrument)
    strategy backtest -- market execution (this module), then sizing/
    P&L (``tradelab.accounting.apply_accounting``, unmodified), then
    equity/drawdown (``tradelab.equity.build_equity_curve``,
    unmodified)."""
    direction = equivalence_class.representative_spec.primary.direction
    strategy_variant_id = f"{equivalence_class.class_id}::{variant.variant_id}::{instrument.value}"

    trades, skipped = simulate_market_execution(equivalence_class.observed_entries, direction, variant, context, strategy_variant_id)

    backtest_result = BacktestResult(
        strategy_id=strategy_variant_id, trades=trades, skipped_signals=[], trade_count=len(trades),
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
        strategy_variant_id=strategy_variant_id, equivalence_class_id=equivalence_class.class_id, exit_variant=variant,
        instrument=instrument, direction=direction, trades=tuple(trades), account_result=account_result, equity_result=equity_result,
        net_pnl_usd=equity_result.net_cash_pnl_usd, roi=(equity_result.net_cash_pnl_usd / equity_config.initial_balance_usd),
        win_rate=_win_rate(account_result), trade_count=len(account_result.accounted_trades),
        max_drawdown_usd=equity_result.max_drawdown_usd, drawdowns_over_2000_count=equity_result.drawdown_over_threshold_episode_count,
        skipped_entries_by_reason=skipped_by_reason, yearly=yearly, monthly=monthly,
    )


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
# Telemetry-integrated production runner
# ---------------------------------------------------------------------------


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

    **Resumability is honestly not implemented here.** Every checkpoint
    is written with ``resumable=False`` and an explicit note -- real,
    SQLite-backed resumability exists in ``tradelab.research_journal``,
    but that module wraps ``tradelab.research_runner.iter_research_
    evaluations`` (built on the *older* ``StrategySpec``/``EntrySignal``
    entry model this module deliberately does not use), so it is not
    yet compatible with this engine's own entry source without further
    adaptation -- a future task, not silently pretended here.

    Never called against the full 3-year artifact in this task (see
    ``docs/backtest-v2-execution-spec.md``); intended for a caller-
    chosen, typically much smaller, dataset -- including this module's
    own tiny synthetic smoke test (see ``main(["--smoke-test", ...])``).
    """
    reporter = create_run(
        runs_dir=runs_dir, market_artifact_path=market_artifact_path, news_csv_path=news_csv_path,
        parameters={
            "tick_size": tick_size, "pivot_strength": pivot_strength, "reclaim_window_bars": reclaim_window_bars,
            "min_entries_per_year": min_entries_per_year, "instruments": [i.value for i in instruments],
            "exit_variant_count": len(exit_variants), "risk_config": {"target_risk_usd": risk_config.target_risk_usd, "max_stop_risk_usd": risk_config.max_stop_risk_usd, "entry_cost_usd": risk_config.entry_cost_usd},
            "equity_config": {"initial_balance_usd": equity_config.initial_balance_usd, "drawdown_threshold_usd": equity_config.drawdown_threshold_usd},
        },
        run_id=run_id, command=command, is_demo=is_demo,
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
            list(series_result.bars), list(news_events), tick_size=tick_size, pivot_strength=pivot_strength, reclaim_window_bars=reclaim_window_bars
        )
        reporter.progress(items_completed=4)
        reporter.record_resource_usage()
        reporter.checkpoint_written(cursor={"stage": "preparing_data", "bar_count": len(context.bars_1m)})

        reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=1)
        universe = discover_entry_strategy_universe(
            context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result, context.htf_bias_result, min_entries_per_year=min_entries_per_year
        )
        reporter.progress(
            items_completed=1, discovered_specs=universe.syntactically_enumerated_spec_count,
            rejected_specs=universe.specs_rejected_by_yearly_eligibility_count, eligible_specs=universe.eligible_spec_count,
        )

        reporter.phase_started(RunPhase.DEDUPLICATING_ENTRIES, items_total=1)
        reporter.progress(items_completed=1, unique_entry_classes=universe.equivalence_class_count)
        reporter.checkpoint_written(cursor={"stage": "deduplicating_entries", "equivalence_class_count": universe.equivalence_class_count})

        total_variant_runs = universe.equivalence_class_count * len(exit_variants) * len(instruments)
        reporter.phase_started(RunPhase.EXPANDING_EXIT_VARIANTS, items_total=max(total_variant_runs, 1))
        reporter.progress(items_completed=max(total_variant_runs, 1), exit_variants_generated=total_variant_runs)

        reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=max(total_variant_runs, 1))
        results: list[StrategyVariantResult] = []
        completed = 0
        for equivalence_class in universe.equivalence_classes:
            for variant in exit_variants:
                for instrument in instruments:
                    result = run_strategy_variant_backtest(equivalence_class, variant, instrument, context, risk_config=risk_config, equity_config=equity_config)
                    results.append(result)
                    completed += 1
                    if completed % checkpoint_every == 0 or completed == total_variant_runs:
                        reporter.progress(items_completed=completed, backtests_completed=completed)
                        reporter.record_resource_usage()
                        reporter.checkpoint_written(
                            cursor={"stage": "simulating_backtests", "completed": completed, "total": total_variant_runs},
                            resumable=False,
                            resumable_note=(
                                "Not yet resumable: tradelab.research_journal provides real, SQLite-backed "
                                "resumability, but only for the older StrategySpec/EntrySignal evaluation "
                                "stream (tradelab.research_runner.iter_research_evaluations) -- this engine's "
                                "own entry source (EntryStrategyEquivalenceClass.observed_entries) is not yet "
                                "wired into it. This checkpoint records progress position for the live "
                                "dashboard only."
                            ),
                        )
                        top = sorted(results, key=lambda r: r.net_pnl_usd, reverse=True)[:top_n_leaderboard]
                        reporter.update_leaderboard([_to_leaderboard_entry(r, rank) for rank, r in enumerate(top, start=1)])

        reporter.phase_started(RunPhase.RANKING_RESULTS, items_total=1)
        ranked = sorted(results, key=lambda r: r.net_pnl_usd, reverse=True)
        top = ranked[:top_n_leaderboard]
        reporter.update_leaderboard([_to_leaderboard_entry(r, rank) for rank, r in enumerate(top, start=1)])
        reporter.progress(items_completed=1)

        reporter.completed(
            summary={
                "equivalence_class_count": universe.equivalence_class_count, "exit_variant_count": len(exit_variants),
                "strategy_variant_result_count": len(results), "top_net_pnl_usd": (ranked[0].net_pnl_usd if ranked else None),
            }
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure must be reported, never silently crash the run directory into an inconsistent state
        reporter.failed(error=f"{type(exc).__name__}: {exc}", traceback_text=traceback.format_exc())
        raise
    return reporter


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

    smoke = sub.add_parser("smoke-test", help="Run a tiny, deterministic, fully synthetic end-to-end smoke test (writes a real monitored run directory).")
    smoke.add_argument("--runs-dir", default="runs")
    smoke.add_argument("--min-entries-per-year", type=int, default=1)

    args = parser.parse_args(argv)

    if args.mode == "run":
        reporter = run_production_backtest_v2(
            market_artifact_path=args.market_artifact, news_csv_path=args.news_csv, runs_dir=args.runs_dir,
            pivot_strength=args.pivot_strength, reclaim_window_bars=args.reclaim_window_bars, min_entries_per_year=args.min_entries_per_year,
            command="python3 -m tradelab.backtest_v2 run",
        )
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

        exit_variants = ALL_EXIT_VARIANTS
        total = universe.equivalence_class_count * len(exit_variants) * len(DEFAULT_INSTRUMENTS)
        reporter.phase_started(RunPhase.EXPANDING_EXIT_VARIANTS, items_total=max(total, 1))
        reporter.progress(items_completed=max(total, 1), exit_variants_generated=total)

        reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=max(total, 1))
        results = []
        completed = 0
        for equivalence_class in universe.equivalence_classes:
            for variant in exit_variants:
                for instrument in DEFAULT_INSTRUMENTS:
                    results.append(run_strategy_variant_backtest(equivalence_class, variant, instrument, context))
                    completed += 1
        if completed:
            reporter.progress(items_completed=completed, backtests_completed=completed)
            reporter.checkpoint_written(cursor={"stage": "simulating_backtests", "completed": completed, "total": total})
            top = sorted(results, key=lambda r: r.net_pnl_usd, reverse=True)[:10]
            reporter.update_leaderboard([_to_leaderboard_entry(r, rank) for rank, r in enumerate(top, start=1)])

        reporter.phase_started(RunPhase.RANKING_RESULTS, items_total=1)
        reporter.progress(items_completed=1)
        reporter.completed(summary={"strategy_variant_result_count": len(results)})
    except Exception as exc:  # noqa: BLE001
        reporter.failed(error=f"{type(exc).__name__}: {exc}", traceback_text=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
