"""Single-strategy backtester: the first module that actually runs one
already-declared ``StrategySpec`` (spec section 3.2, "pełna siatka
wyjść" per strategy) end-to-end against 1-minute OHLCV data and every
existing confluence/TP/SL module, and reports what happened.

Not the full researchengine: this module backtests exactly **one**
``StrategySpec`` at a time, never enumerates the space
``tradelab.strategy_spec.iter_strategy_specs`` can produce, and never
runs multiple trades concurrently. Also explicitly out of scope, on
purpose, for this first version: the minimum-100-trades-per-year filter,
strategy ranking, result persistence, and parallel execution -- those
are later layers built on top of this one.

**Two phases, deliberately split** (spec, this task): everything that
depends only on ``bars_1m``/``news_events``/``tick_size``/
``pivot_strength``/``reclaim_window_bars`` -- validation, aggregation,
every confluence detector, HTF bias, the unified entry timeline -- is
expensive and, crucially, does not depend on any particular
``StrategySpec`` at all. ``prepare_backtest_context`` runs that work
exactly once and returns it as an auditable ``PreparedBacktestContext``;
``backtest_prepared_strategy`` then backtests one ``StrategySpec``
against an already-prepared context, touching none of the expensive
detectors again. This is what lets a future exhaustive research pass
(enumerating many ``StrategySpec``s over the same market data) prepare
the data once and reuse it for every strategy, instead of paying the
full detection cost per strategy -- see the module-level "Preparation
vs. execution" section below for the exact contract. ``backtest_
strategy`` remains the simple, one-call public entry point: it is now a
thin wrapper that prepares a context and immediately backtests one
strategy against it, with an unchanged signature and identical results
to before this split.

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

   All of step 3 (and validation/aggregation) is exactly the work
   ``prepare_backtest_context`` does, once, regardless of how many
   strategies will later be tested against its result.
4. A single forward walk over the 1-minute bars drives execution --
   see "1-minute OHLCV execution adapter" below. This is exactly
   ``backtest_prepared_strategy``'s own work: nothing above it is
   touched again.

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

**Preparation vs. execution.** ``PreparedBacktestContext`` is a plain,
already-frozen bundle of every fact ``prepare_backtest_context`` computed
-- ``fvg_results``, ``fvg_touch_results``, ``fvg_entry_result``,
``swing_equal_liquidity_results``, ``session_liquidity_results``,
``liquidity_entry_result``, ``data_wick_result``, ``htf_bias_result``,
``timeline``, plus the exact inputs it was built from (``bars_1m``,
``news_events``, ``tick_size``, ``pivot_strength``,
``reclaim_window_bars``) for audit/reproducibility. Every one of its
fields is itself an immutable value (frozen dataclasses and plain
dict/list containers that nothing in this module ever mutates in
place) -- ``backtest_prepared_strategy`` only ever *reads* from a
context, whether called once or a thousand times, for one strategy or
many, in any order: none of those calls can affect any other, and a
context is never mutated by the act of backtesting a strategy against
it. This is exactly the property a future exhaustive research pass
needs to prepare one dataset once and safely reuse it for every
``StrategySpec`` in a large parameter sweep (and, later, to apply a
minimum-100-trades-per-year filter across that sweep's results) without
re-running a single detector per strategy.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from tradelab.aggregate import SUPPORTED_TIMEFRAMES as _AGGREGATE_TIMEFRAMES
from tradelab.aggregate import aggregate_bars
from tradelab.bar import Bar1m
from tradelab.bias import BEARISH, BULLISH, NEUTRAL, HTFBiasResult, bias_at, compute_htf_bias
from tradelab.bias import SUPPORTED_TIMEFRAMES as _BIAS_TIMEFRAMES
from tradelab.data_wicks import DataWickDetectionResult, detect_data_wicks
from tradelab.entry_signal_timeline import (
    EntrySignal,
    EntrySignalTimeline,
    build_entry_signal_timeline,
    query_active_signals,
)
from tradelab.fvg import DEFAULT_TICK_SIZE, FVGDetectionResult, detect_fvgs, from_aggregated_bar, from_bar1m
from tradelab.fvg import SUPPORTED_TIMEFRAMES as _FVG_TIMEFRAMES
from tradelab.fvg_entry import SUPPORTED_TIMEFRAMES as _FVG_ENTRY_TIMEFRAMES
from tradelab.fvg_entry import FVGEntryCandidateResult, detect_fvg_entry_candidates
from tradelab.fvg_touch import FVGTouchDetectionResult, detect_fvg_touch_events
from tradelab.liquidity import LiquidityDetectionResult, detect_liquidity
from tradelab.liquidity import SUPPORTED_TIMEFRAMES as _LIQUIDITY_TIMEFRAMES
from tradelab.liquidity_entry import SUPPORTED_TIMEFRAMES as _LIQUIDITY_ENTRY_TIMEFRAMES
from tradelab.liquidity_entry import LiquidityEntryCandidateResult, detect_liquidity_entry_candidates
from tradelab.news import NewsEvent, is_entry_blocked
from tradelab.session_calendar import SpecialSessionSchedule
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
class PreparedBacktestContext:
    """Every fact that depends only on ``bars_1m``/``news_events``/
    ``tick_size``/``pivot_strength``/``reclaim_window_bars`` -- never on
    any particular ``StrategySpec`` -- computed exactly once by
    ``prepare_backtest_context``. See the module docstring's
    "Preparation vs. execution" section for the full contract:
    ``backtest_prepared_strategy`` only ever reads from a context, so
    the same one may be reused, safely and in any order, for as many
    strategies as needed.

    ``bars_1m``/``news_events``/``tick_size``/``pivot_strength``/
    ``reclaim_window_bars``/``special_session_schedule`` are the exact inputs this context was built
    from -- kept here for audit/reproducibility, not recomputed from
    anything else. ``fvg_results``/``fvg_touch_results`` are keyed by
    every native FVG timeframe (fvg_touch_results only 1m/5m/15m,
    fvg_entry.SUPPORTED_TIMEFRAMES); ``swing_equal_liquidity_results``
    is keyed by every native liquidity timeframe;
    ``session_liquidity_results`` by session_liquidity.
    SUPPORTED_TIMEFRAMES (1m/5m/15m) -- the same two-dict convention
    every detector-adapter module in this package already uses.

    **What "frozen" actually covers here.** ``@dataclass(frozen=True)``
    only stops *this dataclass's own fields* from being reassigned --
    it says nothing about the mutability of the objects those fields
    point to, and a plain ``list``/``dict`` handed straight through from
    a caller-supplied argument would still alias that caller's own,
    still-mutable object. Two deliberate steps close that gap without
    pretending every detection model in this package is now deeply
    immutable (it isn't, and this module doesn't need it to be, since it
    never mutates any of it):

    - ``bars_1m``/``news_events`` are stored as **tuples**, built once
      at the start of ``prepare_backtest_context`` -- a snapshot taken
      before validation/detection ever runs, so a caller later mutating
      the original list they passed in (appending a bar, clearing it,
      whatever) can never retroactively change what this context
      describes or desynchronize it from the FVG/liquidity/bias facts
      already computed from the original data.
    - ``fvg_results``/``fvg_touch_results``/``swing_equal_liquidity_
      results``/``session_liquidity_results`` are exposed as
      ``types.MappingProxyType`` -- a genuinely read-only view (``d[k]
      = v`` raises ``TypeError``) over the dict this module built
      internally, not the dict itself.

    ``fvg_entry_result``/``liquidity_entry_result``/``data_wick_
    result``/``htf_bias_result``/``timeline`` are already-frozen
    dataclasses from their own detector modules (as they were before
    this context existed) -- this module does not additionally deep-
    freeze the plain lists nested *inside* them (e.g. ``FVGDetectionResult
    .zones``); it simply never mutates any of it, which is the only
    guarantee ``backtest_prepared_strategy`` actually relies on.
    """

    bars_1m: tuple[Bar1m, ...]
    news_events: tuple[NewsEvent, ...]
    tick_size: float
    pivot_strength: int
    reclaim_window_bars: int
    fvg_results: Mapping[str, FVGDetectionResult]
    fvg_touch_results: Mapping[str, FVGTouchDetectionResult]
    fvg_entry_result: FVGEntryCandidateResult
    swing_equal_liquidity_results: Mapping[str, LiquidityDetectionResult]
    session_liquidity_results: Mapping[str, LiquidityDetectionResult]
    liquidity_entry_result: LiquidityEntryCandidateResult
    data_wick_result: DataWickDetectionResult
    htf_bias_result: HTFBiasResult
    timeline: EntrySignalTimeline
    special_session_schedule: SpecialSessionSchedule | None = None


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


def _peak_rss_bytes() -> int | None:
    """Best-effort peak (not current) resident set size of this process,
    in bytes, since process start -- ``None`` if unavailable (``resource``
    is POSIX-only; never fabricated on platforms without it). This is
    ``resource.getrusage(...).ru_maxrss``, a monotonically non-decreasing
    high-water mark for the *whole process*, not a per-phase delta --
    sampling it after a phase completes reports "the peak RSS observed by
    the end of this phase," which still pinpoints which phase pushes
    memory up, but two phases both reporting a similar number does not
    mean either phase is itself cheap; it may mean an earlier phase's
    peak was never released. ``ru_maxrss`` is kibibytes on Linux but
    already bytes on macOS (a well-known platform inconsistency in the
    POSIX ``getrusage`` API) -- normalized to bytes here so every caller
    gets one consistent unit regardless of platform.
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def prepare_backtest_context(
    bars_1m: list[Bar1m],
    news_events: list[NewsEvent],
    *,
    tick_size: float = DEFAULT_TICK_SIZE,
    pivot_strength: int,
    reclaim_window_bars: int,
    special_schedule: SpecialSessionSchedule | None = None,
    phase_callback: Callable[[str, float, int | None], None] | None = None,
) -> PreparedBacktestContext:
    """Validate ``bars_1m`` and run every strategy-independent detector
    exactly once, returning the result as an auditable, reusable
    ``PreparedBacktestContext`` -- see the module docstring.

    ``tick_size`` (default ``tradelab.fvg.DEFAULT_TICK_SIZE``, 0.25 for
    NQ/MNQ), ``pivot_strength``, and ``reclaim_window_bars`` (both
    mandatory, no default -- the same convention ``tradelab.liquidity``/
    ``tradelab.session_liquidity`` already enforce, since both are meant
    to be swept by a future parameter grid) are passed straight through
    to every detector that needs them.

    ``special_schedule`` is forwarded unchanged to both 1-minute gap
    validation and every higher-timeframe aggregate. This guarantees an
    official early close cannot pass artifact audit and then be silently
    treated as missing data while constructing 5m/15m/1H/4H/1D candles.

    ``phase_callback``, if given, is called with ``(phase_name,
    duration_seconds, peak_rss_bytes)`` as each diagnostic phase
    completes, in this exact order: ``"1m_validation_loading"``,
    ``"timeframe_aggregation"``, one ``"fvg_detection:{timeframe}"`` per
    native FVG timeframe, ``"fvg_entry_candidate_construction"``, one
    ``"liquidity_detection:swing_equal:{timeframe}"`` per native
    liquidity timeframe, one ``"liquidity_detection:session:{timeframe}"``
    per session-liquidity timeframe,
    ``"liquidity_entry_candidate_construction"``, ``"data_wick_
    detection"``, ``"htf_bias_construction"``,
    ``"entry_signal_timeline_construction"``, then ``"total"``. This is
    purely a diagnostic, side-effecting hook (see ``tradelab.diagnose_
    context_preparation``) -- it changes no detector's inputs, outputs,
    or call order, and every phase below runs exactly the same
    computation, in exactly the same order, whether or not a callback is
    given; ``peak_rss_bytes`` is ``None`` where unavailable (see
    ``_peak_rss_bytes``).
    """
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size!r}")
    if not isinstance(pivot_strength, int) or pivot_strength < 1:
        raise ValueError(f"pivot_strength must be a positive integer, got {pivot_strength!r}")
    if not isinstance(reclaim_window_bars, int) or reclaim_window_bars < 1:
        raise ValueError(f"reclaim_window_bars must be a positive integer, got {reclaim_window_bars!r}")

    def _report(phase: str, phase_start: float) -> None:
        if phase_callback is not None:
            phase_callback(phase, time.perf_counter() - phase_start, _peak_rss_bytes())

    total_start = time.perf_counter()

    # Snapshot the inputs into tuples *before* anything else runs, so a
    # caller mutating the list they originally passed in can never
    # retroactively change what this context describes or desynchronize
    # it from the facts already computed from the original data (see
    # PreparedBacktestContext's own docstring).
    validation_start = time.perf_counter()
    bars_1m = tuple(bars_1m)
    news_events = tuple(news_events)

    validate_bars(bars_1m, special_schedule=special_schedule)
    candles: dict[str, list] = {"1m": [from_bar1m(b) for b in bars_1m]}
    _report("1m_validation_loading", validation_start)

    # tradelab.aggregate.SUPPORTED_TIMEFRAMES is exactly {"5m","15m","1H",
    # "4H","1D"} -- every native FVG/liquidity timeframe except 1m, which
    # aggregate_bars does not (and cannot) produce from itself.
    aggregation_start = time.perf_counter()
    for tf in _AGGREGATE_TIMEFRAMES:
        candles[tf] = [
            from_aggregated_bar(b)
            for b in aggregate_bars(
                bars_1m,
                tf,
                special_schedule=special_schedule,
            )
        ]
    _report("timeframe_aggregation", aggregation_start)

    fvg_results: dict[str, FVGDetectionResult] = {}
    for tf in sorted(_FVG_TIMEFRAMES):
        tf_start = time.perf_counter()
        fvg_results[tf] = detect_fvgs(candles[tf], tf, tick_size=tick_size)
        _report(f"fvg_detection:{tf}", tf_start)

    fvg_entry_start = time.perf_counter()
    touch_results = {
        tf: detect_fvg_touch_events(candles[tf], fvg_results[tf], tf) for tf in _FVG_ENTRY_TIMEFRAMES
    }
    fvg_entry_result = detect_fvg_entry_candidates(
        {tf: fvg_results[tf] for tf in _FVG_ENTRY_TIMEFRAMES}, touch_results
    )
    _report("fvg_entry_candidate_construction", fvg_entry_start)

    swing_equal_results: dict[str, LiquidityDetectionResult] = {}
    for tf in sorted(_LIQUIDITY_TIMEFRAMES):
        tf_start = time.perf_counter()
        swing_equal_results[tf] = detect_liquidity(
            candles[tf], tf, pivot_strength=pivot_strength, reclaim_window_bars=reclaim_window_bars, tick_size=tick_size
        )
        _report(f"liquidity_detection:swing_equal:{tf}", tf_start)

    session_results: dict[str, LiquidityDetectionResult] = {}
    for tf in sorted(_SESSION_TIMEFRAMES):
        tf_start = time.perf_counter()
        session_results[tf] = detect_session_liquidity(candles[tf], tf, reclaim_window_bars=reclaim_window_bars)
        _report(f"liquidity_detection:session:{tf}", tf_start)

    liquidity_entry_start = time.perf_counter()
    liquidity_entry_result = detect_liquidity_entry_candidates(
        {tf: swing_equal_results[tf] for tf in _LIQUIDITY_ENTRY_TIMEFRAMES}, session_results
    )
    _report("liquidity_entry_candidate_construction", liquidity_entry_start)

    data_wick_start = time.perf_counter()
    data_wick_result: DataWickDetectionResult = detect_data_wicks(candles["1m"], news_events, tick_size=tick_size)
    _report("data_wick_detection", data_wick_start)

    htf_bias_start = time.perf_counter()
    htf_bias_result = compute_htf_bias({tf: fvg_results[tf] for tf in _BIAS_TIMEFRAMES})
    _report("htf_bias_construction", htf_bias_start)

    timeline_start = time.perf_counter()
    timeline = build_entry_signal_timeline(fvg_entry_result, liquidity_entry_result)
    _report("entry_signal_timeline_construction", timeline_start)

    _report("total", total_start)

    return PreparedBacktestContext(
        bars_1m=bars_1m,
        news_events=news_events,
        tick_size=tick_size,
        pivot_strength=pivot_strength,
        reclaim_window_bars=reclaim_window_bars,
        special_session_schedule=special_schedule,
        fvg_results=MappingProxyType(fvg_results),
        fvg_touch_results=MappingProxyType(touch_results),
        fvg_entry_result=fvg_entry_result,
        swing_equal_liquidity_results=MappingProxyType(swing_equal_results),
        session_liquidity_results=MappingProxyType(session_results),
        liquidity_entry_result=liquidity_entry_result,
        data_wick_result=data_wick_result,
        htf_bias_result=htf_bias_result,
        timeline=timeline,
    )


def backtest_strategy(
    strategy: StrategySpec,
    bars_1m: list[Bar1m],
    news_events: list[NewsEvent],
    *,
    tick_size: float = DEFAULT_TICK_SIZE,
    pivot_strength: int,
    reclaim_window_bars: int,
    special_schedule: SpecialSessionSchedule | None = None,
) -> BacktestResult:
    """Backtest exactly one ``StrategySpec`` against a 1-minute OHLCV
    series and a set of red-folder news events -- a thin convenience
    wrapper around ``prepare_backtest_context`` + ``backtest_prepared_
    strategy`` for the common one-strategy case. Preparing a context and
    calling ``backtest_prepared_strategy`` directly is the way to
    backtest *several* strategies against the same market data without
    paying the detection cost more than once (see the module docstring).
    """
    context = prepare_backtest_context(
        bars_1m,
        news_events,
        tick_size=tick_size,
        pivot_strength=pivot_strength,
        reclaim_window_bars=reclaim_window_bars,
        special_schedule=special_schedule,
    )
    return backtest_prepared_strategy(strategy, context)


def backtest_prepared_strategy(strategy: StrategySpec, context: PreparedBacktestContext) -> BacktestResult:
    """Backtest exactly one ``StrategySpec`` against an already-prepared
    ``PreparedBacktestContext`` -- never touches any of the detectors
    ``prepare_backtest_context`` already ran, and never mutates
    ``context`` itself, so the same context may be reused for any number
    of further strategies, in any order, with no cross-contamination.
    """
    trigger_known_ats = {
        signal.known_at
        for signal in context.timeline.signals_by_selector.get(strategy.primary_entry_selector, [])
        if signal.direction == strategy.direction
    }

    trades: list[ExecutedTrade] = []
    skipped_signals: list[SkippedSignal] = []
    position: _OpenPosition | None = None

    for bar in context.bars_1m:
        if bar.timestamp in trigger_known_ats:
            if position is not None:
                skipped_signals.append(SkippedSignal(timestamp=bar.timestamp, reason=SkipReason.ACTIVE_POSITION))
            else:
                outcome = _evaluate_entry(
                    strategy=strategy,
                    bar=bar,
                    timeline=context.timeline,
                    htf_bias_result=context.htf_bias_result,
                    news_events=context.news_events,
                    swing_equal_results=context.swing_equal_liquidity_results,
                    session_results=context.session_liquidity_results,
                    data_wick_result=context.data_wick_result,
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


def _evaluate_entry(
    *,
    strategy: StrategySpec,
    bar: Bar1m,
    timeline,
    htf_bias_result,
    news_events: tuple[NewsEvent, ...],
    swing_equal_results: Mapping[str, LiquidityDetectionResult],
    session_results: Mapping[str, LiquidityDetectionResult],
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
    # FVG_BOUNDARY: sl_targets never ranks these -- fvg_targets is already
    # sorted in sl_targets' own deterministic order (ascending distance,
    # then known_at, then sl_target_id), and fvg_variant_number picks
    # exactly one position in that order (1-based); this module never
    # picks "nearest" on its own (see SLChoice's own docstring). A
    # variant number with no matching position -- fewer qualifying
    # boundaries exist for this entry than the strategy asked for -- is
    # a MISSING_TARGET, same as any other absent candidate.
    return _pick_by_position(sl_result.fvg_targets, sl_choice.fvg_variant_number)


def _pick_by_rank(candidates: list, rank: int | None):
    for candidate in candidates:
        if candidate.rank == rank:
            return candidate
    return None


def _pick_by_position(candidates: list, position: int | None):
    if position is None or position < 1 or position > len(candidates):
        return None
    return candidates[position - 1]


def _build_selected_fvg_confluence(
    sl_choice: SLChoice, active_by_selector: dict[str, list[EntrySignal]], trade_id: str
) -> SelectedFVGConfluence:
    if sl_choice.source_type != SLSourceType.FVG_BOUNDARY:
        return SelectedFVGConfluence(entry_id=trade_id, fvg_candidates=())
    linked_signals = active_by_selector.get(sl_choice.fvg_link_selector, [])
    return SelectedFVGConfluence(entry_id=trade_id, fvg_candidates=tuple(signal.source for signal in linked_signals))


def _check_exit(position: _OpenPosition, bar: Bar1m) -> tuple[TradeExitReason, float] | None:
    """Resolve one OHLC exit with conservative, executable fill rules.

    SL wins an ambiguous same-candle TP/SL tie.  A stop is a market order:
    when the candle opens through it, the executable fill is the worse open;
    an intrabar touch without an opening gap fills at the stop level.  TP is
    a limit order and therefore fills at its own level.
    """
    tp_price = position.tp_candidate.price
    sl_price = position.sl_candidate.price
    if position.direction == LONG:
        sl_hit = bar.low <= sl_price
        tp_hit = bar.high >= tp_price
    else:
        sl_hit = bar.high >= sl_price
        tp_hit = bar.low <= tp_price

    if sl_hit:
        if position.direction == LONG and bar.open < sl_price:
            return TradeExitReason.SL_HIT, bar.open
        if position.direction != LONG and bar.open > sl_price:
            return TradeExitReason.SL_HIT, bar.open
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
