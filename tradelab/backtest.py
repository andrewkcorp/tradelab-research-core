"""Single-strategy backtester: the first module that actually runs one
already-declared ``StrategySpec`` (spec section 3.2, "pełna siatka
wyjść" per strategy) end-to-end against 1-minute OHLCV data and every
existing confluence/TP/SL module, and reports what happened.

Not the full researchengine: this module backtests exactly **one**
``StrategySpec`` at a time, never enumerates the space
``tradelab.strategy_spec.iter_strategy_specs`` can produce, and never
runs multiple trades concurrently. Also explicitly out of scope, on
purpose, for this first version: USD P&L, the flat $20 round-trip cost,
position sizing/contract count, an equity curve, drawdown, and any
cross-strategy ranking -- ``BacktestResult`` reports gross P&L in
**points** only; everything account-level is a later layer built on top
of this one.

**Data flow** (spec 2.2/2.1/3.x, reusing every existing detector
unchanged -- this module defines no new confluence and re-detects
nothing that already has a canonical detector):

1. ``tradelab.validation.validate_bars`` on the raw 1-minute input (hard
   invariants only -- out-of-order/duplicate timestamps, an impossible
   OHLC range, negative volume all raise immediately).
2. ``tradelab.aggregate.aggregate_bars`` builds 5m/15m/1H/4H/1D from the
   *same* 1-minute series -- these higher timeframes are never sourced
   any other way (spec 2.2: "Świece 5m, 15m, 1H, 4H i 1D muszą być
   agregowane przez TradeLab z tej samej serii 1m").
3. FVG/iFVG (``tradelab.fvg``) on all six native timeframes; FVG touch
   (``tradelab.fvg_touch``) on 1m/5m/15m; FVG/iFVG entry candidates
   (``tradelab.fvg_entry``) from those two. Swing/equal liquidity
   (``tradelab.liquidity``) on all six timeframes; session liquidity
   (``tradelab.session_liquidity``) on 1m/5m/15m; liquidity entry
   candidates (``tradelab.liquidity_entry``) from the 1m/5m/15m subset
   of both. Data wicks (``tradelab.data_wicks``) on native 1m against
   ``news_events``. HTF bias (``tradelab.bias``) from the 1H/4H/1D FVG
   results. The unified entry timeline (``tradelab.entry_signal_
   timeline``) from the FVG and liquidity entry candidates.
4. A single forward walk over the 1-minute bars drives execution --
   see "1-minute OHLCV execution adapter" below.

**Signal-driven entry, not bar-polling.** A candidate entry is
evaluated exactly once, at the earliest bar it can possibly fire: the
1-minute bar whose own ``timestamp`` equals a ``primary_entry_selector``
``EntrySignal.known_at`` of the strategy's own ``direction`` (spec 2:
"sygnał jest znany dopiero po zamknięciu świecy; market entry następuje
najwcześniej na otwarciu następnej świecy 1m" -- and every ``known_at``
this package produces already falls exactly on a 1-minute boundary, so
that next bar's own ``timestamp`` *is* that instant). This is a
deliberate, documented simplification for this first backtester: a
signal that is still technically "active" per ``tradelab.entry_signal_
timeline``'s own persistence window is *not* re-evaluated on every later
bar it remains valid -- each underlying signal fact gets exactly one
entry attempt, which keeps the audit trail one record per real event
rather than one per minute of persistence. Required confluence and bias
are evaluated as of that same bar (their own persistence/invalidation
rules already live entirely in ``tradelab.entry_signal_timeline``/
``tradelab.bias`` -- this module only queries them, never reimplements
them).

**1-minute OHLCV execution adapter.** ``tradelab.trade_execution`` is
deliberately tick-level (its own module docstring: entry is the first
*tick* strictly after signal_known_at, exit is the first *tick* to hit
TP or SL) and is *not* reused here for 1-minute bars -- doing so would
mean pretending an OHLC bar is a single price point, which is exactly
the kind of same-candle TP/SL ambiguity spec 2.2/3.2 explicitly calls
out and gives its own, different resolution rule for. Instead:

- **Entry** fills at the trigger bar's own ``open`` -- "otwarcie
  następnej świecy 1m" (spec 2).
- **Exit is checked against that very same entry bar's own high/low
  first**, then against every following bar's high/low in order, never
  skipping the entry bar the way ``trade_execution``'s tick model
  deliberately skips the entry tick -- an OHLC bar has no finer-grained
  path information, so "entered at the open, then this same candle's
  own range plays out" is the natural, standard bar-based reading, and
  it is exactly the case spec's "w tej samej minucie po entry" tie-break
  is written for.
- **Same-candle TP+SL conflict: SL always wins** (spec, this task) --
  checked in that literal order (``sl_hit`` before ``tp_hit``) on every
  candle, entry candle included. This is the only piece of genuinely new
  execution logic in this module; everything else is orchestration of
  existing detectors.
- Both TP and SL fill at their exact chosen candidate price (the
  ``TPTargetCandidate``/``SLTargetCandidate`` object's own ``price``) --
  unlike ``trade_execution``'s tick-level stop-market slippage
  modelling, an OHLC bar carries no information about *how far* price
  moved past a level intra-bar, only that it did, so there is nothing
  more precise to fill at.
- If the data ends with a position still open, the trade is recorded
  ``OPEN_UNRESOLVED`` (mirrors ``tradelab.trade_execution.TradeStatus.
  OPEN_UNRESOLVED``) -- never an invented close.

**Single active position**: while a position is open, any further
primary-selector signal that becomes known is recorded as a
``SkippedSignal`` with reason ``ACTIVE_POSITION`` and never evaluated
further (spec, this task: "Sygnały podczas aktywnej pozycji zapisujemy
jako pominięte").

**Red-folder news** (spec 4.2): a candidate entry whose trigger bar
falls inside ``tradelab.news.is_entry_blocked``'s ``[T-15min, T+5min]``
window (inclusive both ends, already enforced there) is skipped with
reason ``NEWS_BLOCK``. An already-open position is never closed by a
news window opening -- this module never manufactures an exit.

**No lookahead**: every fact this module's own decisions depend on --
the primary/confluence signal sets, the HTF bias snapshot, the red-
folder check, and every TP/SL candidate -- is queried as of the trigger
bar's own ``timestamp`` (the entry fill instant), and each underlying
detector already guarantees its own output never reveals a fact before
its true ``known_at``. This module adds no lookahead of its own on top
of that.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from tradelab.aggregate import SUPPORTED_TIMEFRAMES as _AGGREGATE_TIMEFRAMES
from tradelab.aggregate import aggregate_bars
from tradelab.bar import Bar1m
from tradelab.bias import BEARISH, BULLISH, NEUTRAL, bias_at, compute_htf_bias
from tradelab.bias import SUPPORTED_TIMEFRAMES as _BIAS_TIMEFRAMES
from tradelab.data_wicks import DataWickDetectionResult, detect_data_wicks
from tradelab.entry_signal_timeline import EntrySignal, build_entry_signal_timeline, query_active_signals
from tradelab.fvg import DEFAULT_TICK_SIZE, FVGDetectionResult, detect_fvgs, from_aggregated_bar, from_bar1m
from tradelab.fvg import SUPPORTED_TIMEFRAMES as _FVG_TIMEFRAMES
from tradelab.fvg_entry import SUPPORTED_TIMEFRAMES as _FVG_ENTRY_TIMEFRAMES
from tradelab.fvg_entry import detect_fvg_entry_candidates
from tradelab.fvg_touch import detect_fvg_touch_events
from tradelab.liquidity import LiquidityDetectionResult, detect_liquidity
from tradelab.liquidity import SUPPORTED_TIMEFRAMES as _LIQUIDITY_TIMEFRAMES
from tradelab.liquidity_entry import SUPPORTED_TIMEFRAMES as _LIQUIDITY_ENTRY_TIMEFRAMES
from tradelab.liquidity_entry import detect_liquidity_entry_candidates
from tradelab.news import NewsEvent, is_entry_blocked
from tradelab.session_liquidity import SUPPORTED_TIMEFRAMES as _SESSION_TIMEFRAMES
from tradelab.session_liquidity import detect_session_liquidity
from tradelab.sl_targets import SLSourceType, SLTargetCandidate, SelectedFVGConfluence, detect_sl_target_candidates
from tradelab.strategy_spec import LONG, BiasPolicy, SLChoice, StrategySpec, TPChoice
from tradelab.tp_targets import MarketEntryContext, TPSourceType, TPTargetCandidate, detect_tp_target_candidates
from tradelab.validation import validate_bars

class SkipReason(str, Enum):
    NEWS_BLOCK = "news_block"
    ACTIVE_POSITION = "active_position"
    MISSING_TARGET = "missing_target"
    BIAS_MISMATCH = "bias_mismatch"
    MISSING_REQUIRED_CONFLUENCE = "missing_required_confluence"


class TradeExitReason(str, Enum):
    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"
    OPEN_UNRESOLVED = "open_unresolved"


@dataclass(frozen=True)
class SkippedSignal:
    """One primary-selector signal that became known but did not turn
    into a trade. ``timestamp`` is that signal's own ``known_at`` (the
    earliest bar it could have entered on)."""

    timestamp: datetime
    reason: SkipReason


@dataclass(frozen=True)
class ExecutedTrade:
    """One fully-audited simulated trade for one ``StrategySpec``.

    ``exit_timestamp``/``exit_price``/``gross_pnl_points`` are ``None``
    only for ``TradeExitReason.OPEN_UNRESOLVED`` -- data ran out with the
    position still open, never an invented close.
    """

    trade_id: str
    strategy_id: str
    direction: str
    signal_known_at: datetime
    entry_timestamp: datetime
    entry_price: float
    tp_price: float
    tp_source_type: str
    tp_source_id: str
    sl_price: float
    sl_source_type: str
    sl_source_id: str
    exit_timestamp: datetime | None
    exit_price: float | None
    exit_reason: TradeExitReason
    gross_pnl_points: float | None


@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    trades: list[ExecutedTrade]
    skipped_signals: list[SkippedSignal]
    trade_count: int
    tp_hit_count: int
    sl_hit_count: int
    open_unresolved_count: int
    gross_pnl_points_total: float


@dataclass(frozen=True)
class _OpenPosition:
    trade_id: str
    strategy_id: str
    direction: str
    signal_known_at: datetime
    entry_timestamp: datetime
    entry_price: float
    tp_candidate: TPTargetCandidate
    sl_candidate: SLTargetCandidate


def backtest_strategy(
    strategy: StrategySpec,
    bars_1m: list[Bar1m],
    news_events: list[NewsEvent],
    *,
    tick_size: float = DEFAULT_TICK_SIZE,
    pivot_strength: int,
    reclaim_window_bars: int,
) -> BacktestResult:
    """Backtest exactly one ``StrategySpec`` against a 1-minute OHLCV
    series and a set of red-folder news events.

    ``tick_size`` (default ``tradelab.fvg.DEFAULT_TICK_SIZE``, 0.25 for
    NQ/MNQ), ``pivot_strength``, and ``reclaim_window_bars`` (both
    mandatory, no default -- the same convention ``tradelab.liquidity``/
    ``tradelab.session_liquidity`` already enforce, since both are meant
    to be swept by a future parameter grid) are passed straight through
    to every detector that needs them.
    """
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size!r}")
    if not isinstance(pivot_strength, int) or pivot_strength < 1:
        raise ValueError(f"pivot_strength must be a positive integer, got {pivot_strength!r}")
    if not isinstance(reclaim_window_bars, int) or reclaim_window_bars < 1:
        raise ValueError(f"reclaim_window_bars must be a positive integer, got {reclaim_window_bars!r}")

    validate_bars(bars_1m)
    candles = _build_candles(bars_1m)

    fvg_results: dict[str, FVGDetectionResult] = {
        tf: detect_fvgs(candles[tf], tf, tick_size=tick_size) for tf in _FVG_TIMEFRAMES
    }
    touch_results = {
        tf: detect_fvg_touch_events(candles[tf], fvg_results[tf], tf) for tf in _FVG_ENTRY_TIMEFRAMES
    }
    fvg_entry_result = detect_fvg_entry_candidates(
        {tf: fvg_results[tf] for tf in _FVG_ENTRY_TIMEFRAMES}, touch_results
    )

    swing_equal_results: dict[str, LiquidityDetectionResult] = {
        tf: detect_liquidity(candles[tf], tf, pivot_strength=pivot_strength, reclaim_window_bars=reclaim_window_bars, tick_size=tick_size)
        for tf in _LIQUIDITY_TIMEFRAMES
    }
    session_results: dict[str, LiquidityDetectionResult] = {
        tf: detect_session_liquidity(candles[tf], tf, reclaim_window_bars=reclaim_window_bars) for tf in _SESSION_TIMEFRAMES
    }
    liquidity_entry_result = detect_liquidity_entry_candidates(
        {tf: swing_equal_results[tf] for tf in _LIQUIDITY_ENTRY_TIMEFRAMES}, session_results
    )

    data_wick_result: DataWickDetectionResult = detect_data_wicks(candles["1m"], news_events, tick_size=tick_size)

    htf_bias_result = compute_htf_bias({tf: fvg_results[tf] for tf in _BIAS_TIMEFRAMES})

    timeline = build_entry_signal_timeline(fvg_entry_result, liquidity_entry_result)

    trigger_known_ats = {
        signal.known_at
        for signal in timeline.signals_by_selector.get(strategy.primary_entry_selector, [])
        if signal.direction == strategy.direction
    }

    trades: list[ExecutedTrade] = []
    skipped_signals: list[SkippedSignal] = []
    position: _OpenPosition | None = None

    for bar in bars_1m:
        if bar.timestamp in trigger_known_ats:
            if position is not None:
                skipped_signals.append(SkippedSignal(timestamp=bar.timestamp, reason=SkipReason.ACTIVE_POSITION))
            else:
                outcome = _evaluate_entry(
                    strategy=strategy,
                    bar=bar,
                    timeline=timeline,
                    htf_bias_result=htf_bias_result,
                    news_events=news_events,
                    swing_equal_results=swing_equal_results,
                    session_results=session_results,
                    data_wick_result=data_wick_result,
                )
                if isinstance(outcome, SkippedSignal):
                    skipped_signals.append(outcome)
                else:
                    position = outcome

        if position is not None:
            exit_outcome = _check_exit(position, bar)
            if exit_outcome is not None:
                reason, exit_price = exit_outcome
                trades.append(_finalize_trade(position, exit_timestamp=bar.timestamp, exit_price=exit_price, reason=reason))
                position = None

    if position is not None:
        trades.append(_finalize_trade(position, exit_timestamp=None, exit_price=None, reason=TradeExitReason.OPEN_UNRESOLVED))

    tp_hit_count = sum(1 for t in trades if t.exit_reason == TradeExitReason.TP_HIT)
    sl_hit_count = sum(1 for t in trades if t.exit_reason == TradeExitReason.SL_HIT)
    open_unresolved_count = sum(1 for t in trades if t.exit_reason == TradeExitReason.OPEN_UNRESOLVED)
    gross_pnl_points_total = sum(t.gross_pnl_points for t in trades if t.gross_pnl_points is not None)

    return BacktestResult(
        strategy_id=strategy.strategy_id,
        trades=trades,
        skipped_signals=skipped_signals,
        trade_count=len(trades),
        tp_hit_count=tp_hit_count,
        sl_hit_count=sl_hit_count,
        open_unresolved_count=open_unresolved_count,
        gross_pnl_points_total=gross_pnl_points_total,
    )


def _build_candles(bars_1m: list[Bar1m]) -> dict[str, list]:
    # tradelab.aggregate.SUPPORTED_TIMEFRAMES is exactly {"5m","15m","1H",
    # "4H","1D"} -- every native FVG/liquidity timeframe except 1m, which
    # aggregate_bars does not (and cannot) produce from itself.
    candles = {"1m": [from_bar1m(b) for b in bars_1m]}
    for tf in _AGGREGATE_TIMEFRAMES:
        candles[tf] = [from_aggregated_bar(b) for b in aggregate_bars(bars_1m, tf)]
    return candles


def _evaluate_entry(
    *,
    strategy: StrategySpec,
    bar: Bar1m,
    timeline,
    htf_bias_result,
    news_events: list[NewsEvent],
    swing_equal_results: dict[str, LiquidityDetectionResult],
    session_results: dict[str, LiquidityDetectionResult],
    data_wick_result: DataWickDetectionResult,
) -> _OpenPosition | SkippedSignal:
    selectors = {strategy.primary_entry_selector, *strategy.required_confluence_selectors}
    query_result = query_active_signals(timeline, selectors, strategy.direction, bar.timestamp)

    active_by_selector: dict[str, list[EntrySignal]] = {}
    for signal in query_result.active_signals:
        active_by_selector.setdefault(signal.selector, []).append(signal)

    for confluence_selector in strategy.required_confluence_selectors:
        if not active_by_selector.get(confluence_selector):
            return SkippedSignal(timestamp=bar.timestamp, reason=SkipReason.MISSING_REQUIRED_CONFLUENCE)

    bias_reject = _check_bias(strategy.bias_policy, strategy.direction, htf_bias_result, bar.timestamp)
    if bias_reject:
        return SkippedSignal(timestamp=bar.timestamp, reason=SkipReason.BIAS_MISMATCH)

    blocked, _ = is_entry_blocked(bar.timestamp, news_events)
    if blocked:
        return SkippedSignal(timestamp=bar.timestamp, reason=SkipReason.NEWS_BLOCK)

    trade_id = f"{strategy.strategy_id}-entry-{bar.timestamp.isoformat()}"
    entry = MarketEntryContext(entry_id=trade_id, direction=strategy.direction, entry_price=bar.open, entry_known_at=bar.timestamp)

    tp_result = detect_tp_target_candidates(entry, swing_equal_results, session_results, data_wick_result)
    tp_candidate = _select_tp_candidate(strategy.tp_choice, tp_result)
    if tp_candidate is None:
        return SkippedSignal(timestamp=bar.timestamp, reason=SkipReason.MISSING_TARGET)

    selected_fvg = _build_selected_fvg_confluence(strategy.sl_choice, active_by_selector, trade_id)
    sl_result = detect_sl_target_candidates(entry, selected_fvg, swing_equal_results, session_results)
    sl_candidate = _select_sl_candidate(strategy.sl_choice, sl_result)
    if sl_candidate is None:
        return SkippedSignal(timestamp=bar.timestamp, reason=SkipReason.MISSING_TARGET)

    return _OpenPosition(
        trade_id=trade_id,
        strategy_id=strategy.strategy_id,
        direction=strategy.direction,
        signal_known_at=bar.timestamp,
        entry_timestamp=bar.timestamp,
        entry_price=bar.open,
        tp_candidate=tp_candidate,
        sl_candidate=sl_candidate,
    )


def _check_bias(bias_policy: BiasPolicy, direction: str, htf_bias_result, ts_utc: datetime) -> bool:
    """True iff this bias_policy rejects the entry at ts_utc."""
    if bias_policy == BiasPolicy.IGNORE:
        return False
    snapshot = bias_at(htf_bias_result, ts_utc)
    if bias_policy == BiasPolicy.REQUIRE_ALIGNED:
        wanted = BULLISH if direction == LONG else BEARISH
        return snapshot.global_bias != wanted
    return snapshot.global_bias != NEUTRAL  # REQUIRE_NEUTRAL


def _select_tp_candidate(tp_choice: TPChoice, tp_result) -> TPTargetCandidate | None:
    if tp_choice.source_type == TPSourceType.LIQUIDITY:
        return _pick_by_rank(tp_result.liquidity_targets, tp_choice.rank)
    return tp_result.data_wick_targets[0] if tp_result.data_wick_targets else None  # nearest-first already


def _select_sl_candidate(sl_choice: SLChoice, sl_result) -> SLTargetCandidate | None:
    if sl_choice.source_type == SLSourceType.LIQUIDITY:
        return _pick_by_rank(sl_result.liquidity_targets, sl_choice.rank)
    return sl_result.fvg_targets[0] if sl_result.fvg_targets else None  # nearest-first already


def _pick_by_rank(candidates: list, rank: int | None):
    for candidate in candidates:
        if candidate.rank == rank:
            return candidate
    return None


def _build_selected_fvg_confluence(
    sl_choice: SLChoice, active_by_selector: dict[str, list[EntrySignal]], trade_id: str
) -> SelectedFVGConfluence:
    if sl_choice.source_type != SLSourceType.FVG_BOUNDARY:
        return SelectedFVGConfluence(entry_id=trade_id, fvg_candidates=())
    linked_signals = active_by_selector.get(sl_choice.fvg_link_selector, [])
    return SelectedFVGConfluence(entry_id=trade_id, fvg_candidates=tuple(signal.source for signal in linked_signals))


def _check_exit(position: _OpenPosition, bar: Bar1m) -> tuple[TradeExitReason, float] | None:
    """SL-first same-candle tie-break (spec, this task): checked in this
    literal order on every candle, including the entry candle itself."""
    tp_price = position.tp_candidate.price
    sl_price = position.sl_candidate.price
    if position.direction == LONG:
        sl_hit = bar.low <= sl_price
        tp_hit = bar.high >= tp_price
    else:
        sl_hit = bar.high >= sl_price
        tp_hit = bar.low <= tp_price

    if sl_hit:
        return TradeExitReason.SL_HIT, sl_price
    if tp_hit:
        return TradeExitReason.TP_HIT, tp_price
    return None


def _finalize_trade(
    position: _OpenPosition,
    *,
    exit_timestamp: datetime | None,
    exit_price: float | None,
    reason: TradeExitReason,
) -> ExecutedTrade:
    gross_pnl_points = None
    if exit_price is not None:
        gross_pnl_points = exit_price - position.entry_price if position.direction == LONG else position.entry_price - exit_price

    return ExecutedTrade(
        trade_id=position.trade_id,
        strategy_id=position.strategy_id,
        direction=position.direction,
        signal_known_at=position.signal_known_at,
        entry_timestamp=position.entry_timestamp,
        entry_price=position.entry_price,
        tp_price=position.tp_candidate.price,
        tp_source_type=position.tp_candidate.source_type.value,
        tp_source_id=position.tp_candidate.source_id,
        sl_price=position.sl_candidate.price,
        sl_source_type=position.sl_candidate.source_type.value,
        sl_source_id=position.sl_candidate.source_id,
        exit_timestamp=exit_timestamp,
        exit_price=exit_price,
        exit_reason=reason,
        gross_pnl_points=gross_pnl_points,
    )
