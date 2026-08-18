"""CLI: prepare a real ``PreparedBacktestContext`` from a continuous MNQ
market artifact and a high-impact USD news CSV, and write a small,
deterministic diagnostic profile describing it.

.. code-block:: text

    python3 -m tradelab.profile_research_context \\
        --market-artifact /path/to/continuous_artifact_manifest.json \\
        --news-csv /path/to/us_high_impact_v1.csv \\
        --output-json /path/to/context_profile.json

**This module never fetches, downloads, or commits data.** It reads
exactly one already-local ``tradelab.market_data_artifact.MarketDataArtifact``
manifest (``--market-artifact``, resolving its own ``data_path`` next to
the manifest, exactly like every other consumer of that module) and one
already-local news CSV (``--news-csv``) -- both caller-supplied paths,
never hard-coded. Nothing this module reads or writes is meant to enter
this git repository; see ``.gitignore``.

**Does not enumerate strategies and does not run research.** This is
strictly a context-preparation and diagnostics tool: it calls
``tradelab.backtest.prepare_backtest_context`` exactly once (the same
"preparation vs. execution" split every other research module in this
package already relies on) and reports counts of what that preparation
produced -- it never calls ``tradelab.strategy_spec.iter_strategy_specs``
or ``tradelab.research_runner.iter_research_evaluations``, and it never
opens a trade. A future runner is expected to reuse the same
``PreparedBacktestContext`` this module builds; this module only proves
that context is soundly built and reports its shape, then discards the
context itself.

**Every validation step reuses an already-existing, already-tested
module -- nothing here re-implements checksum, chronology, OHLCV, or
news validation:**

- ``tradelab.market_data_artifact.load_market_data_artifact`` +
  ``load_market_data_series`` load and audit the market artifact --
  checksum, UTC timezone (enforced by ``MarketDataArtifact`` construction
  and ``tradelab.bar.Bar1m``'s own unconditional UTC normalization),
  chronology, and OHLCV (via ``tradelab.validation.validate_bars``,
  reused inside the audit). Any *critical* audit finding (checksum
  mismatch, coverage exceeding the declared window, an invalid bar
  series) raises ``ValueError`` and this module writes nothing.
- ``tradelab.news_loaders.read_news_events_csv`` loads and validates the
  news CSV (schema, whitelist, exact-minute timestamps, duplicate
  detection via ``tradelab.news.validate_news_events``). Any invalid row
  raises ``ValueError`` and this module writes nothing.

**News coverage limitations are reported, never fabricated.** This
module never invents a synthetic news event to "fill" a gap -- it only
reports, as plain counts, how the loaded news events' ``scheduled_at``
range relates to the market data's own observed coverage (before /
within / after), so a reader can judge for themselves whether the news
dataset actually covers the period being profiled.

**Context preparation parameters are explicit, never guessed** (this
task's own instruction): ``tick_size=0.25`` (MNQ), ``pivot_strength=3``,
``reclaim_window_bars=5`` -- passed straight through to ``prepare_
backtest_context`` unchanged, recorded verbatim in the profile's own
``context_preparation_params``. Every FVG/liquidity/data-wick/bias/red-
folder-news rule this invokes is entirely ``tradelab.backtest``'s own,
already-existing, unmodified behavior.

**The profile is small and audit-only -- never the context itself**
(this task's own instruction): ``context_profile.json`` never
serializes ``PreparedBacktestContext`` or any of its trade-level detail
(no bar rows, no zone/level objects, no candidate objects) -- only
counts, derived from it: market artifact + audit metadata, news
metadata + coverage notes, 1m/aggregated bar counts, FVG/iFVG counts by
timeframe and direction, liquidity level and entry-candidate counts by
timeframe/family/type/phase/mode/direction, data wick counts by
direction, HTF bias change counts and state distribution, and entry
signal counts by canonical selector and direction -- both for the whole
available period and, separately, raw signal counts for the two
explicit, half-open, non-guessed research windows this task itself
specifies (``2024``: ``[2024-01-01T00:00:00Z, 2025-01-01T00:00:00Z)``,
``2025``: ``[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)``) -- the same
``tradelab.research_runner.ResearchWindow`` half-open contract used
everywhere else in this package. P&L/drawdown are not computed here at
all (no research is run); these two windows exist in this profile only
to make each window's *entry signal* volume auditable ahead of a future
research run.

**Written atomically.** ``context_profile.json`` is first written to a
sibling temp file, then moved into place with ``os.replace`` in one
step -- a failure at any point leaves no partial file at ``--output-json``.

**Deterministic.** Every dict in the profile is serialized with
``sort_keys=True`` (so key order can never depend on Python's per-process
hash-randomized ``dict``/``frozenset`` iteration order), and every
list-of-records structure (liquidity entry candidate counts, entry
signal counts) is explicitly sorted by its own field tuple before
serialization -- two runs against identical input always produce a
byte-identical file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from tradelab.aggregate import aggregate_bars
from tradelab.backtest import PreparedBacktestContext, prepare_backtest_context
from tradelab.bar import Bar1m
from tradelab.data_wicks import DataWickDetectionResult, DataWickDirection
from tradelab.entry_signal_timeline import EntrySignalTimeline
from tradelab.fvg import BEARISH, BULLISH, FVGDetectionResult
from tradelab.bias import HTFBiasResult
from tradelab.liquidity import LiquidityDetectionResult
from tradelab.liquidity_entry import LiquidityEntryCandidateResult
from tradelab.market_data_artifact import (
    MarketDataArtifact,
    MarketDataArtifactAuditReport,
    load_market_data_artifact,
    load_market_data_series,
)
from tradelab.news import NewsEvent
from tradelab.news_loaders import read_news_events_csv
from tradelab.research_runner import ResearchWindow

PROFILE_SCHEMA_VERSION = "1.0"

# MNQ, explicit -- never DEFAULT_TICK_SIZE relied upon implicitly (this
# task's own instruction: pivot_strength/reclaim_window_bars/tick_size
# are always stated explicitly, matching the existing convention
# tradelab.liquidity/tradelab.backtest already enforce).
TICK_SIZE_MNQ = 0.25
PIVOT_STRENGTH = 3
RECLAIM_WINDOW_BARS = 5

_AGGREGATE_TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "1H", "4H", "1D")
_UTC = timezone.utc

# The two explicit, full, half-open calendar-year windows this task
# itself specifies -- never guessed, never derived from the data.
PROFILE_WINDOWS: tuple[ResearchWindow, ...] = (
    ResearchWindow("2024", datetime(2024, 1, 1, tzinfo=_UTC), datetime(2025, 1, 1, tzinfo=_UTC)),
    ResearchWindow("2025", datetime(2025, 1, 1, tzinfo=_UTC), datetime(2026, 1, 1, tzinfo=_UTC)),
)


@dataclass(frozen=True)
class ContextProfileResult:
    """Top-level counts from a successful ``profile_research_context``
    run -- a quick, human-scannable overview; the full detail lives in
    ``context_profile.json`` itself at ``output_json``."""

    output_json: Path
    bar_count_1m: int
    news_event_count: int
    fvg_zone_count_total: int
    fvg_event_count_total: int
    liquidity_level_count_total: int
    liquidity_entry_candidate_count_total: int
    data_wick_count_total: int
    bias_event_count_total: int
    entry_signal_count_total: int


def profile_research_context(market_artifact_path: Path | str, news_csv_path: Path | str, output_json_path: Path | str) -> ContextProfileResult:
    """Load and audit ``market_artifact_path``, load and validate
    ``news_csv_path``, call ``prepare_backtest_context`` with this
    module's explicit parameters, and atomically write a deterministic
    diagnostic profile to ``output_json_path``. Raises ``ValueError`` --
    and writes nothing -- if the market artifact fails its audit with a
    critical finding, or if the news CSV is invalid. Never runs strategy
    enumeration or research; never serializes the resulting context
    itself, only counts derived from it.
    """
    market_artifact_path = Path(market_artifact_path)
    news_csv_path = Path(news_csv_path)
    output_json_path = Path(output_json_path)

    artifact = load_market_data_artifact(market_artifact_path)
    series_result = load_market_data_series(artifact, base_dir=market_artifact_path.parent)
    bars = list(series_result.bars)

    news_events = read_news_events_csv(news_csv_path)
    news_sha256 = hashlib.sha256(news_csv_path.read_bytes()).hexdigest()

    context = prepare_backtest_context(
        bars, news_events, tick_size=TICK_SIZE_MNQ, pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS
    )

    profile = _build_profile(
        market_artifact_path=market_artifact_path,
        artifact=artifact,
        audit_report=series_result.audit_report,
        news_csv_path=news_csv_path,
        news_sha256=news_sha256,
        news_events=news_events,
        context=context,
    )
    _write_atomic_json(output_json_path, profile)

    return ContextProfileResult(
        output_json=output_json_path,
        bar_count_1m=len(context.bars_1m),
        news_event_count=len(news_events),
        fvg_zone_count_total=sum(len(result.zones) for result in context.fvg_results.values()),
        fvg_event_count_total=sum(len(result.events) for result in context.fvg_results.values()),
        liquidity_level_count_total=(
            sum(len(result.levels) for result in context.swing_equal_liquidity_results.values())
            + sum(len(result.levels) for result in context.session_liquidity_results.values())
        ),
        liquidity_entry_candidate_count_total=len(context.liquidity_entry_result.candidates),
        data_wick_count_total=len(context.data_wick_result.events),
        bias_event_count_total=sum(len(changes) for changes in context.htf_bias_result.timeframe_changes.values()),
        entry_signal_count_total=len(context.timeline.signals),
    )


def _build_profile(
    *,
    market_artifact_path: Path,
    artifact: MarketDataArtifact,
    audit_report: MarketDataArtifactAuditReport,
    news_csv_path: Path,
    news_sha256: str,
    news_events: list[NewsEvent],
    context: PreparedBacktestContext,
) -> dict:
    market_artifact_section = artifact.to_dict()
    market_artifact_section["manifest_path"] = str(market_artifact_path)
    market_artifact_section["audit"] = audit_report.to_dict()

    bars = context.bars_1m
    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "context_preparation_params": {
            "tick_size": TICK_SIZE_MNQ,
            "pivot_strength": PIVOT_STRENGTH,
            "reclaim_window_bars": RECLAIM_WINDOW_BARS,
        },
        "market_artifact": market_artifact_section,
        "news": _news_section(news_csv_path, news_sha256, news_events, audit_report),
        "market_data": {
            "bar_count_1m": len(bars),
            "coverage_start": bars[0].timestamp.isoformat() if bars else None,
            "coverage_end": bars[-1].timestamp.isoformat() if bars else None,
            "aggregated_bar_counts": _aggregated_bar_counts(bars),
        },
        "fvg": {
            "zone_counts_by_timeframe_direction": _fvg_zone_counts(context.fvg_results),
            "ifvg_event_counts_by_timeframe_direction": _fvg_event_counts(context.fvg_results),
        },
        "liquidity": {
            "level_counts": _liquidity_level_counts(context.swing_equal_liquidity_results, context.session_liquidity_results),
            "entry_candidate_counts": _liquidity_entry_candidate_counts(context.liquidity_entry_result),
        },
        "data_wicks": _data_wick_counts(context.data_wick_result),
        "bias": _bias_counts(context.htf_bias_result),
        "entry_signals": {
            "by_selector_direction": _selector_direction_counts(context.timeline),
            "raw_by_window": {window.window_id: _selector_direction_counts_in_window(context.timeline, window) for window in PROFILE_WINDOWS},
        },
    }


def _news_section(news_csv_path: Path, news_sha256: str, news_events: list[NewsEvent], audit_report: MarketDataArtifactAuditReport) -> dict:
    scheduled_ats = [event.scheduled_at for event in news_events]
    return {
        "csv_path": str(news_csv_path),
        "sha256": news_sha256,
        "event_count": len(news_events),
        "schema_versions": sorted({event.schema_version for event in news_events}),
        "scheduled_at_min": min(scheduled_ats).isoformat() if scheduled_ats else None,
        "scheduled_at_max": max(scheduled_ats).isoformat() if scheduled_ats else None,
        "event_type_counts": _news_event_type_counts(news_events),
        "coverage_notes": _news_coverage_notes(news_events, audit_report.observed_coverage_start, audit_report.observed_coverage_end),
    }


def _news_event_type_counts(news_events: list[NewsEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in news_events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    return counts


def _news_coverage_notes(news_events: list[NewsEvent], observed_start: datetime | None, observed_end: datetime | None) -> dict:
    """Reports, never fabricates: how many loaded news events fall
    before / within / after the market data's own observed coverage.
    ``None`` for the count fields when the market data has no observed
    coverage at all (an empty bar series) -- there is nothing to compare
    against, so this never guesses "within" or "before" in that case."""
    if observed_start is None or observed_end is None:
        return {
            "market_observed_coverage_start": None,
            "market_observed_coverage_end": None,
            "news_events_before_market_coverage": None,
            "news_events_within_market_coverage": None,
            "news_events_after_market_coverage": None,
        }
    before = sum(1 for event in news_events if event.scheduled_at < observed_start)
    after = sum(1 for event in news_events if event.scheduled_at > observed_end)
    within = len(news_events) - before - after
    return {
        "market_observed_coverage_start": observed_start.isoformat(),
        "market_observed_coverage_end": observed_end.isoformat(),
        "news_events_before_market_coverage": before,
        "news_events_within_market_coverage": within,
        "news_events_after_market_coverage": after,
    }


def _aggregated_bar_counts(bars_1m: tuple[Bar1m, ...]) -> dict[str, int]:
    bars_list = list(bars_1m)
    return {timeframe: len(aggregate_bars(bars_list, timeframe)) for timeframe in _AGGREGATE_TIMEFRAMES}


def _fvg_zone_counts(fvg_results: Mapping[str, FVGDetectionResult]) -> dict[str, dict[str, int]]:
    return {
        timeframe: {
            "bullish": sum(1 for zone in result.zones if zone.direction == BULLISH),
            "bearish": sum(1 for zone in result.zones if zone.direction == BEARISH),
        }
        for timeframe, result in fvg_results.items()
    }


def _fvg_event_counts(fvg_results: Mapping[str, FVGDetectionResult]) -> dict[str, dict[str, int]]:
    return {
        timeframe: {
            "bullish": sum(1 for event in result.events if event.direction == BULLISH),
            "bearish": sum(1 for event in result.events if event.direction == BEARISH),
        }
        for timeframe, result in fvg_results.items()
    }


def _liquidity_level_counts(
    swing_equal_results: Mapping[str, LiquidityDetectionResult], session_results: Mapping[str, LiquidityDetectionResult]
) -> dict[str, dict[str, dict[str, int]]]:
    def _counts(results: Mapping[str, LiquidityDetectionResult]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for timeframe, result in results.items():
            per_type: dict[str, int] = {}
            for level in result.levels:
                per_type[level.liquidity_type.value] = per_type.get(level.liquidity_type.value, 0) + 1
            out[timeframe] = per_type
        return out

    return {"swing_equal": _counts(swing_equal_results), "session": _counts(session_results)}


def _liquidity_entry_candidate_counts(liquidity_entry_result: LiquidityEntryCandidateResult) -> list[dict]:
    tally: dict[tuple[str, str, str, str, str, str], int] = {}
    for candidate in liquidity_entry_result.candidates:
        key = (
            candidate.timeframe,
            candidate.source_family.value,
            candidate.liquidity_type.value,
            candidate.trigger_phase.value,
            candidate.entry_mode.value,
            candidate.direction,
        )
        tally[key] = tally.get(key, 0) + 1
    return [
        {
            "timeframe": timeframe,
            "source_family": source_family,
            "liquidity_type": liquidity_type,
            "trigger_phase": trigger_phase,
            "entry_mode": entry_mode,
            "direction": direction,
            "count": count,
        }
        for (timeframe, source_family, liquidity_type, trigger_phase, entry_mode, direction), count in sorted(tally.items())
    ]


def _data_wick_counts(data_wick_result: DataWickDetectionResult) -> dict[str, int]:
    upper = sum(1 for event in data_wick_result.events if event.direction == DataWickDirection.UPPER)
    lower = sum(1 for event in data_wick_result.events if event.direction == DataWickDirection.LOWER)
    return {"upper": upper, "lower": lower, "total": upper + lower}


def _bias_counts(htf_bias_result: HTFBiasResult) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for timeframe, changes in htf_bias_result.timeframe_changes.items():
        distribution = {"bullish": 0, "bearish": 0, "neutral": 0}
        for change in changes:
            distribution[change.state] += 1
        out[timeframe] = {"change_count": len(changes), "state_distribution": distribution}
    return out


def _selector_direction_counts(timeline: EntrySignalTimeline) -> list[dict]:
    tally: dict[tuple[str, str], int] = {}
    for selector, signals in timeline.signals_by_selector.items():
        for signal in signals:
            key = (selector, signal.direction)
            tally[key] = tally.get(key, 0) + 1
    return [{"selector": selector, "direction": direction, "count": count} for (selector, direction), count in sorted(tally.items())]


def _selector_direction_counts_in_window(timeline: EntrySignalTimeline, window: ResearchWindow) -> list[dict]:
    """Raw signal counts (by ``known_at``, matching ``tradelab.
    research_runner``'s own safe-prefilter convention), restricted to
    ``window``'s explicit half-open ``[start_utc, end_utc)`` range --
    never a real accounted-trade count, since no backtest runs here."""
    tally: dict[tuple[str, str], int] = {}
    for selector, signals in timeline.signals_by_selector.items():
        for signal in signals:
            if window.start_utc <= signal.known_at < window.end_utc:
                key = (selector, signal.direction)
                tally[key] = tally.get(key, 0) + 1
    return [{"selector": selector, "direction": direction, "count": count} for (selector, direction), count in sorted(tally.items())]


def _write_atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m tradelab.profile_research_context",
        description=(
            "Prepare a real PreparedBacktestContext from a continuous MNQ market artifact and a high-impact "
            "USD news CSV, and write a small, deterministic diagnostic profile. Never enumerates strategies "
            "and never runs research."
        ),
    )
    parser.add_argument("--market-artifact", required=True, type=Path, help="Path to a continuous market artifact manifest JSON.")
    parser.add_argument("--news-csv", required=True, type=Path, help="Path to a validated high-impact USD news CSV.")
    parser.add_argument("--output-json", required=True, type=Path, help="Path to atomically write the diagnostic context_profile.json to.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        result = profile_research_context(args.market_artifact, args.news_csv, args.output_json)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote context profile to {result.output_json}")
    print(f"  1m bars:                    {result.bar_count_1m}")
    print(f"  news events:                {result.news_event_count}")
    print(f"  FVG zones:                  {result.fvg_zone_count_total}")
    print(f"  iFVG events:                {result.fvg_event_count_total}")
    print(f"  liquidity levels:           {result.liquidity_level_count_total}")
    print(f"  liquidity entry candidates: {result.liquidity_entry_candidate_count_total}")
    print(f"  data wicks:                 {result.data_wick_count_total}")
    print(f"  bias events:                {result.bias_event_count_total}")
    print(f"  entry signals:              {result.entry_signal_count_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
