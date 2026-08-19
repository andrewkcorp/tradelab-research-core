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
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from tradelab.aggregate import aggregate_bars
from tradelab.backtest import PreparedBacktestContext, prepare_backtest_context
from tradelab.bar import Bar1m
from tradelab.data_wicks import DataWickDetectionResult, DataWickDirection
from tradelab.entry_signal_timeline import (
    EntrySignal,
    EntrySignalTimeline,
    SignalSuspension,
    iter_fvg_lifecycle_transitions,
)
from tradelab.fvg import BEARISH, BULLISH, FVGDetectionResult, FVGMitigationEvent
from tradelab.bias import HTFBiasResult
from tradelab.fvg_entry import FVGEntryCandidateResult
from tradelab.fvg_entry import SourceType as FVGCandidateSourceType
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

# v2.0 (this task): adds FVG-entry-candidate touch/iFVG/mitigation counts,
# entry_signal_timeline suspension/restoration counts and active-FVG-
# confluence duration statistics (all downstream of the confluence-
# validity rework -- persist-until-invalidated, reversible FVG/iFVG
# suspension/restoration, FVGMitigationEvent), a provenance section
# (git commit, python version, exact command, checksums restated at the
# top level), and an optional comparison against a prior profile JSON.
# v1.0 profiles do not have any of these fields -- see `comparison`
# below for exactly what a v1.0-vs-v2.0 comparison can and cannot show.
PROFILE_SCHEMA_VERSION = "2.0"

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
    fvg_mitigation_count_total: int
    suspension_count_total: int
    restoration_count_total: int


def profile_research_context(
    market_artifact_path: Path | str,
    news_csv_path: Path | str,
    output_json_path: Path | str,
    *,
    previous_profile_json_path: Path | str | None = None,
    command: str | None = None,
) -> ContextProfileResult:
    """Load and audit ``market_artifact_path``, load and validate
    ``news_csv_path``, call ``prepare_backtest_context`` with this
    module's explicit parameters, and atomically write a deterministic
    diagnostic profile to ``output_json_path``. Raises ``ValueError`` --
    and writes nothing -- if the market artifact fails its audit with a
    critical finding, or if the news CSV is invalid. Never runs strategy
    enumeration or research; never serializes the resulting context
    itself, only counts derived from it.

    ``previous_profile_json_path``, if given, is read (no validation
    beyond JSON parsing -- it is never trusted as a source of truth
    about the *current* run) and reported under ``comparison`` -- see
    ``_comparison_section`` for exactly what is and is not comparable.
    ``command``, if given, is recorded verbatim under
    ``provenance.command`` -- this function never reconstructs it from
    ``sys.argv`` itself (only ``main()`` does that), so calling this
    function directly (e.g. from a test, or another script) never
    silently records that *caller's own* argv as if it were this
    profile's invocation.
    """
    market_artifact_path = Path(market_artifact_path)
    news_csv_path = Path(news_csv_path)
    output_json_path = Path(output_json_path)
    previous_profile_json_path = Path(previous_profile_json_path) if previous_profile_json_path is not None else None

    artifact = load_market_data_artifact(market_artifact_path)
    series_result = load_market_data_series(artifact, base_dir=market_artifact_path.parent)
    bars = list(series_result.bars)

    news_events = read_news_events_csv(news_csv_path)
    news_sha256 = hashlib.sha256(news_csv_path.read_bytes()).hexdigest()

    context = prepare_backtest_context(
        bars, news_events, tick_size=TICK_SIZE_MNQ, pivot_strength=PIVOT_STRENGTH, reclaim_window_bars=RECLAIM_WINDOW_BARS
    )

    timeline_stats = _entry_signal_timeline_stats(context.timeline)
    entry_signals_by_selector_direction = _selector_direction_counts(context.timeline)

    previous_profile = None
    if previous_profile_json_path is not None:
        previous_profile = json.loads(previous_profile_json_path.read_text(encoding="utf-8"))

    profile = _build_profile(
        market_artifact_path=market_artifact_path,
        artifact=artifact,
        audit_report=series_result.audit_report,
        news_csv_path=news_csv_path,
        news_sha256=news_sha256,
        news_events=news_events,
        context=context,
        timeline_stats=timeline_stats,
        entry_signals_by_selector_direction=entry_signals_by_selector_direction,
        previous_profile_json_path=previous_profile_json_path,
        previous_profile=previous_profile,
        command=command,
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
        fvg_mitigation_count_total=len(context.fvg_entry_result.zone_mitigations),
        suspension_count_total=timeline_stats["suspension_count_total"],
        restoration_count_total=timeline_stats["restoration_count_total"],
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
    timeline_stats: dict,
    entry_signals_by_selector_direction: list[dict],
    previous_profile_json_path: Path | None,
    previous_profile: dict | None,
    command: str | None,
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
        "fvg_entry_candidates": {
            "note": (
                "Derived from PreparedBacktestContext.fvg_entry_result (the "
                "same candidates/mitigations tradelab.entry_signal_timeline "
                "consumes), not from the raw fvg_results zones/events above -- "
                "ifvg_counts_by_timeframe_direction is expected to exactly "
                "match fvg.ifvg_event_counts_by_timeframe_direction (same "
                "underlying facts, two different source objects)."
            ),
            "touch_counts_by_timeframe_direction_contact_type": _fvg_touch_candidate_counts(context.fvg_entry_result),
            "ifvg_counts_by_timeframe_direction": _fvg_ifvg_candidate_counts(context.fvg_entry_result),
            "mitigation_counts_by_timeframe_direction": _fvg_mitigation_counts(context.fvg_entry_result),
        },
        "liquidity": {
            "level_counts": _liquidity_level_counts(context.swing_equal_liquidity_results, context.session_liquidity_results),
            "entry_candidate_counts": _liquidity_entry_candidate_counts(context.liquidity_entry_result),
        },
        "data_wicks": _data_wick_counts(context.data_wick_result),
        "bias": _bias_counts(context.htf_bias_result),
        "entry_signals": {
            "by_selector_direction": entry_signals_by_selector_direction,
            "raw_by_window": {window.window_id: _selector_direction_counts_in_window(context.timeline, window) for window in PROFILE_WINDOWS},
        },
        "entry_signal_timeline": timeline_stats,
        "provenance": _provenance_section(
            market_artifact_sha256=artifact.sha256, news_csv_sha256=news_sha256, command=command
        ),
        "comparison": _comparison_section(
            previous_profile_json_path=previous_profile_json_path,
            previous_profile=previous_profile,
            current_entry_signals_by_selector_direction=entry_signals_by_selector_direction,
        ),
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


def _fvg_touch_candidate_counts(fvg_entry_result: FVGEntryCandidateResult) -> dict[str, dict[str, dict[str, int]]]:
    """FVG_TOUCH entry candidates by timeframe/direction/contact_type --
    ``partial_touch`` and ``full_mitigation_no_inversion`` only
    (``inversion_ifvg`` never produces an FVG_TOUCH candidate, see
    ``tradelab.fvg_entry``)."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for candidate in fvg_entry_result.candidates:
        if candidate.source_type != FVGCandidateSourceType.FVG_TOUCH:
            continue
        direction_bucket = out.setdefault(candidate.timeframe, {}).setdefault(candidate.direction, {})
        contact_type = candidate.contact_type.value
        direction_bucket[contact_type] = direction_bucket.get(contact_type, 0) + 1
    return out


def _fvg_ifvg_candidate_counts(fvg_entry_result: FVGEntryCandidateResult) -> dict[str, dict[str, int]]:
    """IFVG entry candidates by timeframe/direction -- expected to
    exactly match ``_fvg_event_counts`` (same underlying facts, derived
    from ``fvg_entry_result.candidates`` instead of the raw ``fvg_
    results`` events)."""
    out: dict[str, dict[str, int]] = {}
    for candidate in fvg_entry_result.candidates:
        if candidate.source_type != FVGCandidateSourceType.IFVG:
            continue
        bucket = out.setdefault(candidate.timeframe, {})
        bucket[candidate.direction] = bucket.get(candidate.direction, 0) + 1
    return out


def _fvg_mitigation_counts(fvg_entry_result: FVGEntryCandidateResult) -> dict[str, dict[str, int]]:
    """``tradelab.fvg.FVGMitigationEvent`` counts by timeframe/direction
    -- a zone becoming ``MITIGATED`` without inverting, whether that was
    its first contact or a later, separate one after an earlier partial
    touch. Never a new entry candidate itself (see ``fvg_entry_
    candidates.note`` in the profile)."""
    out: dict[str, dict[str, int]] = {}
    for mitigation in fvg_entry_result.zone_mitigations:
        bucket = out.setdefault(mitigation.timeframe, {})
        bucket[mitigation.direction] = bucket.get(mitigation.direction, 0) + 1
    return out


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear interpolation between closest ranks -- deterministic,
    dependency-free (never relies on ``statistics.quantiles``' own
    interpolation choice or its >=2-points requirement)."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = fraction * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower)


def _is_immediate_resolution_restoration(restoration) -> bool:
    """True for the one ``SignalRestoration`` shape that does NOT undo a
    prior ``SignalSuspension``: ``tradelab.entry_signal_timeline._apply_
    fvg_batch`` step 2 emits a ``SignalRestoration`` (never a
    ``SignalSuspension``) when a ``FULL_MITIGATION_NO_INVERSION``-typed
    touch "challenges" the opposite-direction pool -- since that
    challenger's own zone is already, immediately resolved the instant
    it is known, the target is never actually removed from ``active``
    (see that function's own step-2 docstring); the restoration exists
    purely for audit transparency ("the competing FVG became inactive
    without becoming iFVG"). Every *other* ``SignalRestoration`` shape
    (``restored_by`` is either an ``IFVG``-typed signal or a
    ``tradelab.fvg.FVGMitigationEvent``) does undo a real, prior
    suspension -- ``suspension_count_total`` and this excluded subset
    are therefore not directly comparable, which is exactly why
    ``still_suspended_at_end_count`` below is computed from the non-
    immediate-resolution subset only, never from the raw restoration
    total."""
    restored_by = restoration.restored_by
    return (
        isinstance(restored_by, EntrySignal)
        and restored_by.source.source_type == FVGCandidateSourceType.FVG_TOUCH
    )


def _entry_signal_timeline_stats(timeline: EntrySignalTimeline) -> dict:
    """Suspension/restoration totals and active-FVG-confluence duration
    statistics -- entirely downstream of the confluence-validity rework
    (persist-until-invalidated, reversible FVG/iFVG suspension/
    restoration). Liquidity has no suspension/restoration concept (only
    permanent, same-selector invalidation), so it is not represented
    here at all. One linear pass over ``iter_fvg_lifecycle_transitions``
    (``O(N log N)``, never re-implements the suspend/restore rules).

    ``restoration_count_total`` counts every ``SignalRestoration``
    ``iter_fvg_lifecycle_transitions`` yields, including a documented,
    intentional shape that does not undo any suspension at all -- see
    ``_is_immediate_resolution_restoration``. It is therefore NOT
    directly comparable to ``suspension_count_total`` 1:1;
    ``still_suspended_at_end_count`` uses only the subset of
    restorations that actually did undo a suspension."""
    known_at_by_candidate_id = {s.candidate_id: s.known_at for s in timeline.signals if s.source_kind.value == "fvg"}
    first_suspension_known_at: dict[str, datetime] = {}
    suspension_count = 0
    restoration_count = 0
    immediate_resolution_restoration_count = 0
    for transition in iter_fvg_lifecycle_transitions(timeline):
        if isinstance(transition, SignalSuspension):
            suspension_count += 1
            first_suspension_known_at.setdefault(transition.signal.candidate_id, transition.suspended_by.known_at)
        else:
            restoration_count += 1
            if _is_immediate_resolution_restoration(transition):
                immediate_resolution_restoration_count += 1
    restoration_of_suspended_signal_count = restoration_count - immediate_resolution_restoration_count

    durations_minutes = sorted(
        (first_suspension_known_at[candidate_id] - known_at).total_seconds() / 60.0
        for candidate_id, known_at in known_at_by_candidate_id.items()
        if candidate_id in first_suspension_known_at
    )
    never_suspended_count = len(known_at_by_candidate_id) - len(durations_minutes)

    return {
        "suspension_count_total": suspension_count,
        "restoration_count_total": restoration_count,
        "restoration_note": (
            "Includes a documented, intentional SignalRestoration shape "
            "that does not undo any prior suspension (a FULL_MITIGATION_"
            "NO_INVERSION-typed touch challenges the opposite pool but "
            "its own zone is already resolved the instant it is known, "
            "so the target was never actually removed from the active "
            "pool -- see tradelab.entry_signal_timeline._apply_fvg_batch "
            "step 2). restoration_count_total is therefore NOT directly "
            "comparable to suspension_count_total 1:1; see "
            "restoration_of_suspended_signal_count_total and "
            "immediate_resolution_restoration_count_total for the split."
        ),
        "restoration_of_suspended_signal_count_total": restoration_of_suspended_signal_count,
        "immediate_resolution_restoration_count_total": immediate_resolution_restoration_count,
        "still_suspended_at_end_count": suspension_count - restoration_of_suspended_signal_count,
        "active_fvg_confluence_duration_stats_minutes": {
            "note": (
                "Measures time from each FVG-family confluence's own known_at "
                "to its FIRST suspension only. A confluence that is later "
                "restored and then suspended again by a different, later "
                "challenger is not reflected as a second, separate interval "
                "here -- this is a deliberate simplification, not a full "
                "multi-interval active-time accounting."
            ),
            "completed_count": len(durations_minutes),
            "never_suspended_count": never_suspended_count,
            "min": durations_minutes[0] if durations_minutes else None,
            "max": durations_minutes[-1] if durations_minutes else None,
            "mean": sum(durations_minutes) / len(durations_minutes) if durations_minutes else None,
            "median": _percentile(durations_minutes, 0.5) if durations_minutes else None,
            "p90": _percentile(durations_minutes, 0.9) if durations_minutes else None,
        },
    }


def _git_provenance() -> dict[str, str | bool | None]:
    """Best-effort git commit/dirty-state of the checkout this module
    itself is running from -- ``None``/``None`` (never fabricated) if
    this is not a git checkout or ``git`` is unavailable."""
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True, timeout=10
        ).stdout
        return {"git_commit": commit, "git_commit_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"git_commit": None, "git_commit_dirty": None}


def _provenance_section(*, market_artifact_sha256: str, news_csv_sha256: str, command: str | None) -> dict:
    git_info = _git_provenance()
    return {
        "python_version": sys.version,
        "command": command,
        "market_artifact_sha256": market_artifact_sha256,
        "news_csv_sha256": news_csv_sha256,
        **git_info,
    }


def _compare_raw_signal_counts(previous_counts: list[dict], current_counts: list[dict]) -> dict:
    """Compares two ``entry_signals.by_selector_direction`` lists --
    these are raw signal *generation* counts, unaffected by the
    confluence-validity model (FVG/liquidity detection and candidate
    generation did not change; only which signals are *active* at a
    given instant did). A mismatch here would indicate the underlying
    data or code changed too (e.g. more bars collected since the prior
    profile), not a validity-model regression."""
    previous_map = {(row["selector"], row["direction"]): row["count"] for row in previous_counts}
    current_map = {(row["selector"], row["direction"]): row["count"] for row in current_counts}
    only_in_previous = sorted(f"{selector}/{direction}" for selector, direction in previous_map if (selector, direction) not in current_map)
    only_in_current = sorted(f"{selector}/{direction}" for selector, direction in current_map if (selector, direction) not in previous_map)
    changed = sorted(
        f"{selector}/{direction}: {previous_map[(selector, direction)]} -> {current_map[(selector, direction)]}"
        for selector, direction in set(previous_map) & set(current_map)
        if previous_map[(selector, direction)] != current_map[(selector, direction)]
    )
    return {
        "identical": not only_in_previous and not only_in_current and not changed,
        "only_in_previous": only_in_previous,
        "only_in_current": only_in_current,
        "changed": changed,
    }


_OBSOLETE_VALIDITY_MODEL_WARNING = (
    "The referenced previous profile was generated under the OBSOLETE "
    "end-of-NY-trading-day entry confluence validity model (a confluence "
    "expired at 17:00 NY on its own trading day, and never combined "
    "across trading days). That model has been fully replaced -- persist-"
    "until-invalidated validity, liquidity invalidation scoped to the "
    "identical selector, and reversible FVG/iFVG same-timeframe "
    "suspension/restoration (including resolution via a later, non-"
    "inverting FVGMitigationEvent). This previous profile's own counts "
    "must NOT be used for strategy-universe estimation, combo mining, or "
    "any other purpose that depends on which confluences were active at "
    "a given instant -- only this current profile reflects that."
)

_ACTIVE_POOL_COMPARISON_NOTE = (
    "An old-vs-new comparison of active-pool composition at real primary-"
    "signal timestamps is NOT possible from the referenced previous "
    "profile: its schema (see previous_profile_schema_version) stores "
    "only aggregate signal counts by selector/direction, never per-"
    "instant active-pool snapshots or suspension/restoration events. "
    "Raw signal *generation* counts are comparable (see "
    "raw_signal_count_comparison) and are expected to match, since "
    "generation is unaffected by the validity-model change -- but "
    "active-pool membership over time is not recoverable from a stored "
    "aggregate profile. A true comparison would require re-running the "
    "now-removed historical (end-of-day-expiry) entry_signal_timeline "
    "implementation side-by-side against the current one on identical "
    "input; that historical implementation was fully replaced, not "
    "merely deprecated, and reconstructing it solely for this comparison "
    "is out of scope here. This profile instead reports the current "
    "model's own active-pool/duration statistics in full (see "
    "entry_signal_timeline), establishing a clean baseline for any "
    "future comparison."
)


def _comparison_section(
    *,
    previous_profile_json_path: Path | None,
    previous_profile: dict | None,
    current_entry_signals_by_selector_direction: list[dict],
) -> dict:
    if previous_profile_json_path is None or previous_profile is None:
        return {
            "previous_profile_path": None,
            "previous_profile_schema_version": None,
            "previous_profile_obsolete_validity_model_warning": None,
            "raw_signal_count_comparison": None,
            "active_pool_comparison_possible": None,
            "active_pool_comparison_note": None,
        }
    previous_entry_signals = previous_profile.get("entry_signals", {}).get("by_selector_direction", [])
    return {
        "previous_profile_path": str(previous_profile_json_path),
        "previous_profile_schema_version": previous_profile.get("profile_schema_version"),
        "previous_profile_obsolete_validity_model_warning": _OBSOLETE_VALIDITY_MODEL_WARNING,
        "raw_signal_count_comparison": _compare_raw_signal_counts(previous_entry_signals, current_entry_signals_by_selector_direction),
        "active_pool_comparison_possible": False,
        "active_pool_comparison_note": _ACTIVE_POOL_COMPARISON_NOTE,
    }


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
    parser.add_argument(
        "--previous-profile-json",
        required=False,
        type=Path,
        default=None,
        help=(
            "Optional path to a prior context_profile.json to report a comparison against under `comparison` -- "
            "see that section's own note for exactly what is and is not comparable. Never required."
        ),
    )
    return parser


def _reconstruct_command(args: argparse.Namespace) -> str:
    parts = [
        "python3 -m tradelab.profile_research_context",
        f"--market-artifact {args.market_artifact}",
        f"--news-csv {args.news_csv}",
        f"--output-json {args.output_json}",
    ]
    if args.previous_profile_json is not None:
        parts.append(f"--previous-profile-json {args.previous_profile_json}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    command = _reconstruct_command(args)
    try:
        result = profile_research_context(
            args.market_artifact,
            args.news_csv,
            args.output_json,
            previous_profile_json_path=args.previous_profile_json,
            command=command,
        )
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote context profile to {result.output_json}")
    print(f"  1m bars:                    {result.bar_count_1m}")
    print(f"  news events:                {result.news_event_count}")
    print(f"  FVG zones:                  {result.fvg_zone_count_total}")
    print(f"  iFVG events:                {result.fvg_event_count_total}")
    print(f"  FVG mitigation events:      {result.fvg_mitigation_count_total}")
    print(f"  liquidity levels:           {result.liquidity_level_count_total}")
    print(f"  liquidity entry candidates: {result.liquidity_entry_candidate_count_total}")
    print(f"  data wicks:                 {result.data_wick_count_total}")
    print(f"  bias events:                {result.bias_event_count_total}")
    print(f"  entry signals:              {result.entry_signal_count_total}")
    print(f"  FVG suspensions (total):    {result.suspension_count_total}")
    print(f"  FVG restorations (total):   {result.restoration_count_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
