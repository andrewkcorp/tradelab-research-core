"""TradeLab data foundation: normalized 1m bars, loaders, validation, aggregation.

Research-only. No broker connectivity, no live execution, no network calls.
"""
from tradelab.bar import Bar1m
from tradelab.aggregate import AggregatedBar, aggregate_bars
from tradelab.bias import (
    BiasSnapshot,
    ConfluenceSource,
    ConfluenceType,
    HTFBiasResult,
    TimeframeBiasChange,
    bias_at,
    compute_htf_bias,
)
from tradelab.data_wicks import DataWickDetectionResult, DataWickEvent, DataWickDirection, detect_data_wicks
from tradelab.entry_signal_timeline import (
    EntrySignal,
    EntrySignalQueryResult,
    EntrySignalTimeline,
    SignalInvalidation,
    SourceKind,
    build_entry_signal_timeline,
    query_active_signals,
)
from tradelab.fvg import Candle, FVGDetectionResult, FVGEvent, FVGStatus, FVGZone, detect_fvgs, from_aggregated_bar, from_bar1m
from tradelab.fvg_entry import FVGEntryCandidate, FVGEntryCandidateResult, SourceType, detect_fvg_entry_candidates
from tradelab.fvg_touch import FVGContactType, FVGTouchDetectionResult, FVGTouchEvent, detect_fvg_touch_events
from tradelab.liquidity import (
    LiquidityDetectionResult,
    LiquidityLevel,
    LiquidityStatus,
    LiquidityTakeEvent,
    LiquidityType,
    ReclaimOutcome,
    ReclaimResult,
    detect_liquidity,
)
from tradelab.liquidity_entry import (
    CandidatePhase,
    EntryMode,
    LiquidityEntryCandidate,
    LiquidityEntryCandidateResult,
    SourceFamily,
    detect_liquidity_entry_candidates,
)
from tradelab.loaders import read_csv_bars, read_parquet_bars
from tradelab.news import NewsEvent, is_entry_blocked, validate_news_events
from tradelab.news_loaders import read_news_events_csv, read_news_events_parquet
from tradelab.session_calendar import expected_open_minutes, is_regular_session_open
from tradelab.session_liquidity import detect_session_liquidity
from tradelab.sl_targets import (
    FVG_SL_TIMEFRAMES,
    SLSourceType,
    SLTargetCandidate,
    SLTargetCandidateResult,
    SelectedFVGConfluence,
    detect_sl_target_candidates,
)
from tradelab.strategy_spec import (
    BiasPolicy,
    SLChoice,
    StrategySpec,
    TPChoice,
    iter_strategy_specs,
    selector_fixed_direction,
    selector_timeframe,
)
from tradelab.tp_targets import (
    LiquiditySourceFamily,
    MarketEntryContext,
    TPSourceType,
    TPTargetCandidate,
    TPTargetCandidateResult,
    detect_tp_target_candidates,
)
from tradelab.validation import MissingMinutesGap, ValidationResult, validate_bars

__all__ = [
    "Bar1m",
    "AggregatedBar",
    "aggregate_bars",
    "MissingMinutesGap",
    "ValidationResult",
    "validate_bars",
    "read_csv_bars",
    "read_parquet_bars",
    "is_regular_session_open",
    "expected_open_minutes",
    "NewsEvent",
    "validate_news_events",
    "is_entry_blocked",
    "read_news_events_csv",
    "read_news_events_parquet",
    "Candle",
    "FVGZone",
    "FVGEvent",
    "FVGStatus",
    "FVGDetectionResult",
    "detect_fvgs",
    "from_bar1m",
    "from_aggregated_bar",
    "LiquidityLevel",
    "LiquidityTakeEvent",
    "LiquidityType",
    "LiquidityStatus",
    "ReclaimResult",
    "ReclaimOutcome",
    "LiquidityDetectionResult",
    "detect_liquidity",
    "detect_session_liquidity",
    "DataWickEvent",
    "DataWickDirection",
    "DataWickDetectionResult",
    "detect_data_wicks",
    "ConfluenceType",
    "ConfluenceSource",
    "TimeframeBiasChange",
    "HTFBiasResult",
    "BiasSnapshot",
    "compute_htf_bias",
    "bias_at",
    "FVGContactType",
    "FVGTouchEvent",
    "FVGTouchDetectionResult",
    "detect_fvg_touch_events",
    "SourceFamily",
    "CandidatePhase",
    "EntryMode",
    "LiquidityEntryCandidate",
    "LiquidityEntryCandidateResult",
    "detect_liquidity_entry_candidates",
    "SourceType",
    "FVGEntryCandidate",
    "FVGEntryCandidateResult",
    "detect_fvg_entry_candidates",
    "SourceKind",
    "EntrySignal",
    "EntrySignalTimeline",
    "SignalInvalidation",
    "EntrySignalQueryResult",
    "build_entry_signal_timeline",
    "query_active_signals",
    "TPSourceType",
    "LiquiditySourceFamily",
    "MarketEntryContext",
    "TPTargetCandidate",
    "TPTargetCandidateResult",
    "detect_tp_target_candidates",
    "FVG_SL_TIMEFRAMES",
    "SLSourceType",
    "SelectedFVGConfluence",
    "SLTargetCandidate",
    "SLTargetCandidateResult",
    "detect_sl_target_candidates",
    "BiasPolicy",
    "TPChoice",
    "SLChoice",
    "StrategySpec",
    "iter_strategy_specs",
    "selector_timeframe",
    "selector_fixed_direction",
]
