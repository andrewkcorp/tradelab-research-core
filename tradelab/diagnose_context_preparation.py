"""Diagnostic-only CLI: measure where ``tradelab.backtest.prepare_
backtest_context`` actually spends time and memory on a deterministic,
contiguous prefix of the real MNQ continuous artifact.

.. code-block:: text

    python3 -m tradelab.diagnose_context_preparation \\
        --market-artifact /path/to/continuous_artifact_manifest.json \\
        --news-csv /path/to/us_high_impact_v1.csv \\
        --prefix-bars 10000,50000,100000

**Evidence only -- this module changes nothing about how a context is
built.** It calls ``tradelab.backtest.prepare_backtest_context`` with
its new, purely diagnostic ``phase_callback`` hook (added alongside
this module, in the same PR) and prints what that hook reports --
duration and best-effort peak RSS for each of ``prepare_backtest_
context``'s own internal phases (1m validation/loading, timeframe
aggregation, FVG detection per timeframe, FVG entry-candidate
construction, liquidity-level detection per timeframe/family, liquidity
entry-candidate construction, data-wick detection, HTF bias
construction, entry-signal timeline construction), plus this script's
own outer phase (loading the artifact/news off disk). No detector, no
validity rule, and no strategy-generation code is touched or reordered
beyond what ``prepare_backtest_context`` itself already does (see that
function's own docstring for the exact, unchanged phase order this
script merely observes).

**Never writes anything.** No ``--output-json``, no generated file of
any kind -- everything is printed to stdout. **Never runs combo mining
or a backtest** -- this script calls ``prepare_backtest_context`` and
stops; it never calls ``tradelab.observed_strategy_universe.discover_
observed_universe``, ``tradelab.research_runner.iter_research_
evaluations``, or ``tradelab.backtest.backtest_prepared_strategy``.

**Deterministic, contiguous prefix selection.** ``--prefix-bars N``
always selects exactly the *first* ``N`` bars of the artifact's own
loaded, already-chronological series (``all_bars[:N]``) -- never a
random sample, never bars from the middle or end. The same artifact and
the same ``N`` always produce the exact same prefix, byte-for-byte
reproducible, so repeated runs (and runs across different ``N``) are
directly comparable. Raises ``ValueError`` (never silently truncates)
if ``N`` exceeds the artifact's own available bar count -- a prefix
shorter than requested would silently invalidate any scaling comparison
across multiple ``N`` values.

**One artifact/news load, many prefixes.** ``--prefix-bars`` accepts a
comma-separated list (e.g. ``10000,50000,100000``) so the (potentially
slow, multi-minute) full artifact load happens exactly once and every
requested prefix is diagnosed against that same in-memory series --
never reloading the file per prefix.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tradelab.backtest import PreparedBacktestContext, _peak_rss_bytes, prepare_backtest_context
from tradelab.bar import Bar1m
from tradelab.market_data_artifact import load_market_data_artifact, load_market_data_series
from tradelab.news import NewsEvent
from tradelab.news_loaders import read_news_events_csv

# Identical to tradelab.profile_research_context's own explicit MNQ
# parameters (never guessed, never a different value here) -- this
# diagnostic must measure the exact same detector configuration a real
# profiling run would use, or its timings would not be representative.
TICK_SIZE_MNQ = 0.25
PIVOT_STRENGTH = 3
RECLAIM_WINDOW_BARS = 5


def select_deterministic_prefix(bars: tuple[Bar1m, ...], prefix_bars: int) -> tuple[Bar1m, ...]:
    """The first ``prefix_bars`` bars of ``bars``, in their own existing
    chronological order -- deterministic and contiguous (never a random
    sample, never re-sorted). Raises ``ValueError`` if ``prefix_bars`` is
    not a positive integer, or exceeds ``len(bars)`` -- never silently
    truncates to less than what was asked for, which would invalidate a
    scaling comparison across multiple prefix sizes."""
    if prefix_bars < 1:
        raise ValueError(f"prefix_bars must be a positive integer, got {prefix_bars!r}")
    if prefix_bars > len(bars):
        raise ValueError(f"prefix_bars={prefix_bars} exceeds the artifact's own available bar count ({len(bars)})")
    return bars[:prefix_bars]


def _format_rss(peak_rss_bytes: int | None) -> str:
    if peak_rss_bytes is None:
        return "n/a"
    return f"{peak_rss_bytes / (1024 * 1024):.1f} MB"


def _print_phase(phase: str, duration_seconds: float, peak_rss_bytes: int | None) -> None:
    print(f"    {phase:<45} {duration_seconds:9.3f}s   peak_rss={_format_rss(peak_rss_bytes)}")


def _print_context_summary(context: PreparedBacktestContext) -> None:
    fvg_zone_count = sum(len(result.zones) for result in context.fvg_results.values())
    fvg_event_count = sum(len(result.events) for result in context.fvg_results.values())
    fvg_mitigation_count = len(context.fvg_entry_result.zone_mitigations)
    liquidity_level_count = sum(len(result.levels) for result in context.swing_equal_liquidity_results.values()) + sum(
        len(result.levels) for result in context.session_liquidity_results.values()
    )
    liquidity_entry_candidate_count = len(context.liquidity_entry_result.candidates)
    data_wick_count = len(context.data_wick_result.events)
    bias_event_count = sum(len(changes) for changes in context.htf_bias_result.timeframe_changes.values())
    entry_signal_count = len(context.timeline.signals)

    print(f"    1m bars (prefix):            {len(context.bars_1m)}")
    print(f"    FVG zones (all timeframes):  {fvg_zone_count}")
    print(f"    iFVG events:                 {fvg_event_count}")
    print(f"    FVG mitigation events:       {fvg_mitigation_count}")
    print(f"    liquidity levels:            {liquidity_level_count}")
    print(f"    liquidity entry candidates:  {liquidity_entry_candidate_count}")
    print(f"    data wicks:                  {data_wick_count}")
    print(f"    bias events:                 {bias_event_count}")
    print(f"    entry signals:               {entry_signal_count}")


def diagnose_one_prefix(
    all_bars: tuple[Bar1m, ...], news_events: tuple[NewsEvent, ...], prefix_bars: int
) -> PreparedBacktestContext:
    """Run ``prepare_backtest_context`` on the deterministic first
    ``prefix_bars`` bars of ``all_bars`` (paired with the full,
    unfiltered ``news_events`` -- news volume is comparatively tiny and
    irrelevant to the bar-count scaling this script exists to measure),
    printing every phase's timing/peak-RSS as it completes, then a
    summary of what was actually detected. Returns the built context
    only for tests; the CLI itself never serializes it."""
    prefix = select_deterministic_prefix(all_bars, prefix_bars)
    print(f"--- prefix_bars={prefix_bars} ({prefix[0].timestamp.isoformat()} -> {prefix[-1].timestamp.isoformat()}) ---")
    context = prepare_backtest_context(
        list(prefix),
        list(news_events),
        tick_size=TICK_SIZE_MNQ,
        pivot_strength=PIVOT_STRENGTH,
        reclaim_window_bars=RECLAIM_WINDOW_BARS,
        phase_callback=_print_phase,
    )
    _print_context_summary(context)
    print()
    return context


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m tradelab.diagnose_context_preparation",
        description=(
            "Diagnostic-only: measure per-phase timing and peak RSS of prepare_backtest_context on a "
            "deterministic, contiguous prefix of the real MNQ artifact. Never writes anything, never runs "
            "combo mining or a backtest."
        ),
    )
    parser.add_argument("--market-artifact", required=True, type=Path, help="Path to a continuous market artifact manifest JSON.")
    parser.add_argument("--news-csv", required=True, type=Path, help="Path to a validated high-impact USD news CSV.")
    parser.add_argument(
        "--prefix-bars",
        required=True,
        type=str,
        help="Comma-separated list of deterministic contiguous 1m-bar prefix sizes to diagnose, e.g. 10000,50000,100000.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        prefix_sizes = [int(part.strip()) for part in args.prefix_bars.split(",")]
    except ValueError as error:
        print(f"error: --prefix-bars must be a comma-separated list of integers: {error}", file=sys.stderr)
        return 1

    try:
        load_start = time.perf_counter()
        artifact = load_market_data_artifact(args.market_artifact)
        series_result = load_market_data_series(artifact, base_dir=args.market_artifact.parent)
        all_bars = series_result.bars
        news_events = read_news_events_csv(args.news_csv)
        load_duration = time.perf_counter() - load_start

        print(
            f"artifact load: {load_duration:.3f}s -- {len(all_bars)} total 1m bars available, "
            f"{len(news_events)} news events, peak_rss={_format_rss(_peak_rss_bytes())}"
        )
        print()

        for prefix_bars in prefix_sizes:
            diagnose_one_prefix(all_bars, news_events, prefix_bars)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
