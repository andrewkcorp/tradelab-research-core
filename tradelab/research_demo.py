"""Deterministic DEMO run generator for ``tradelab.research_dashboard``.

**This module never runs real research.** It does not call ``tradelab.
entry_strategy_universe``, ``tradelab.backtest``, or any other real
detection/discovery/backtest code -- every number it reports (specs,
classes, leaderboard P&L, equity curves, drawdowns) is a deterministic,
synthetic value produced by a seeded RNG for the sole purpose of
exercising ``tradelab.research_run``'s reporter API and ``tradelab.
research_dashboard``'s rendering end to end. ``RunManifest.is_demo`` is
always ``True`` for a demo run, ``run_id`` always starts with ``demo-``
(see ``tradelab.research_run.create_run``), and the dashboard renders a
persistent, impossible-to-miss "DEMO -- NOT REAL RESEARCH RESULTS"
banner whenever it is looking at one -- see ``docs/research-dashboard-
v1.md`` for the full "how to tell demo from real" explanation.

**Deterministic.** ``run_demo(seed=...)`` with the same seed always
produces the exact same sequence of phases, progress counts, and
leaderboard values -- a fixed ``random.Random(seed)`` instance drives
every synthetic number, so the demo is itself reproducible and testable
(no wall-clock-dependent assertions needed in its own tests beyond
monotonicity/shape checks).
"""
from __future__ import annotations

import random
import time
from pathlib import Path

from tradelab.research_run import LeaderboardEntry, RunPhase, RunReporter, create_run

_DEMO_STRATEGY_PREFIXES = ("fvg_touch", "ifvg", "liquidity_sweep", "sweep_fvg_touch_setup", "sweep_ifvg_setup")


def _synthetic_leaderboard(rng: random.Random, count: int) -> list[LeaderboardEntry]:
    entries = []
    for i in range(count):
        trade_count = rng.randint(100, 400)
        win_rate = rng.uniform(0.35, 0.62)
        pnl = rng.uniform(-5000, 45000)
        roi = pnl / 25000.0
        max_drawdown = rng.uniform(500, 9000)
        drawdowns_over_2000 = rng.randint(0, 6) if max_drawdown > 2000 else 0

        equity = 25000.0
        equity_curve = []
        trade_pnls = []
        trades_by_month: dict[str, int] = {}
        # Spread trades evenly over a fixed, realistic 24-month span
        # (2024-01 .. 2025-12) -- matching the real 2024/2025 eligibility
        # window elsewhere in this package, purely for a clean demo chart.
        for t in range(trade_count):
            step_pnl = rng.gauss(pnl / trade_count, abs(pnl / trade_count) * 2 + 50)
            equity += step_pnl
            trade_pnls.append(round(step_pnl, 2))
            month_index = min(23, (t * 24) // trade_count)
            year = 2024 + (month_index // 12)
            month = (month_index % 12) + 1
            key = f"{year:04d}-{month:02d}"
            trades_by_month[key] = trades_by_month.get(key, 0) + 1
            equity_curve.append({"ts": f"{year:04d}-{month:02d}-{(t % 27) + 1:02d}T00:00:00+00:00", "equity": round(equity, 2)})

        prefix = _DEMO_STRATEGY_PREFIXES[i % len(_DEMO_STRATEGY_PREFIXES)]
        entries.append(
            LeaderboardEntry(
                rank=i + 1, strategy_id=f"demo-{prefix}-{i:04d}", pnl=round(pnl, 2), roi=round(roi, 4),
                win_rate=round(win_rate, 4), trade_count=trade_count, max_drawdown=round(max_drawdown, 2),
                drawdowns_over_2000_count=drawdowns_over_2000, equity_curve=tuple(equity_curve),
                trade_pnls=tuple(trade_pnls), trades_by_month=trades_by_month,
            )
        )
    entries.sort(key=lambda e: e.pnl, reverse=True)
    return [
        LeaderboardEntry(
            rank=i + 1, strategy_id=e.strategy_id, pnl=e.pnl, roi=e.roi, win_rate=e.win_rate, trade_count=e.trade_count,
            max_drawdown=e.max_drawdown, drawdowns_over_2000_count=e.drawdowns_over_2000_count,
            equity_curve=e.equity_curve, trade_pnls=e.trade_pnls, trades_by_month=e.trades_by_month,
        )
        for i, e in enumerate(entries)
    ]


def run_demo(
    *,
    runs_dir: Path | str,
    seed: int = 20260819,
    real_time: bool = True,
    step_delay_seconds: float = 0.15,
    run_id: str | None = None,
) -> RunReporter:
    """Drive one full, deterministic demo run through every phase in
    order, reporting realistic-shaped (but entirely synthetic) progress,
    checkpoints, resource samples, and leaderboard updates via the real
    :class:`tradelab.research_run.RunReporter` API -- exactly the calls
    a future real runner would make, just with fabricated numbers.

    ``real_time=True`` (default, for an actual live dashboard demo)
    sleeps ``step_delay_seconds`` between progress updates, so a
    dashboard polling ``live_status.json`` visibly sees it move.
    ``real_time=False`` (for fast tests) skips the sleeps -- the
    reported *numbers* are identical either way (seeded RNG), only wall
    -clock pacing differs, so tests can assert on values without being
    slow.
    """
    rng = random.Random(seed)
    reporter = create_run(
        runs_dir=runs_dir, market_artifact_path=None, news_csv_path=None,
        parameters={"seed": seed, "note": "DEMO run -- synthetic data, not real MNQ research"},
        run_id=run_id, command="python3 -m tradelab.research_demo", is_demo=True,
    )

    def _sleep() -> None:
        if real_time:
            time.sleep(step_delay_seconds)

    reporter.phase_started(RunPhase.PREPARING_DATA, items_total=4)
    prep_steps = ("load market artifact", "validate bars", "load news csv", "prepare backtest context")
    for i, _step in enumerate(prep_steps, start=1):
        _sleep()
        reporter.progress(items_completed=i)
    reporter.record_resource_usage()
    reporter.checkpoint_written(cursor={"stage": "preparing_data", "step": len(prep_steps)})

    total_primaries = 3036
    reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=total_primaries)
    discovered = 0
    rejected = 0
    eligible = 0
    steps = 12
    for i in range(1, steps + 1):
        _sleep()
        completed = int(total_primaries * i / steps)
        newly = completed - int(total_primaries * (i - 1) / steps)
        discovered += newly * rng.randint(0, 20)
        newly_eligible = rng.randint(0, max(1, newly // 30))
        eligible += newly_eligible
        rejected = discovered - eligible
        reporter.progress(
            items_completed=completed, discovered_specs=discovered, rejected_specs=rejected, eligible_specs=eligible,
        )
        if i % 4 == 0:
            reporter.record_resource_usage()
            reporter.checkpoint_written(cursor={"stage": "discovering_entries", "primary_index": completed, "primary_total": total_primaries})

    reporter.phase_started(RunPhase.DEDUPLICATING_ENTRIES, items_total=eligible or 1)
    unique_classes = 0
    dedup_steps = 6
    for i in range(1, dedup_steps + 1):
        _sleep()
        completed = int((eligible or 1) * i / dedup_steps)
        unique_classes = max(1, int(completed * rng.uniform(0.55, 0.95)))
        reporter.progress(items_completed=completed, unique_entry_classes=unique_classes)
    reporter.checkpoint_written(cursor={"stage": "deduplicating_entries", "classes_so_far": unique_classes})

    exit_variants_per_class = rng.randint(6, 14)
    total_exit_variants = unique_classes * exit_variants_per_class
    reporter.phase_started(RunPhase.EXPANDING_EXIT_VARIANTS, items_total=total_exit_variants)
    variants_steps = 6
    exit_variants_generated = 0
    for i in range(1, variants_steps + 1):
        _sleep()
        exit_variants_generated = int(total_exit_variants * i / variants_steps)
        reporter.progress(items_completed=exit_variants_generated, exit_variants_generated=exit_variants_generated)

    reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=total_exit_variants)
    backtest_steps = 10
    backtests_completed = 0
    leaderboard: list[LeaderboardEntry] = []
    for i in range(1, backtest_steps + 1):
        _sleep()
        backtests_completed = int(total_exit_variants * i / backtest_steps)
        reporter.progress(items_completed=backtests_completed, backtests_completed=backtests_completed)
        if i % 3 == 0 or i == backtest_steps:
            leaderboard = _synthetic_leaderboard(rng, min(25, max(3, backtests_completed // max(1, exit_variants_per_class))))
            reporter.update_leaderboard(leaderboard)
        if i % 3 == 0:
            reporter.record_resource_usage()
            reporter.checkpoint_written(cursor={"stage": "simulating_backtests", "variant_index": backtests_completed, "variant_total": total_exit_variants})
    if not leaderboard:
        leaderboard = _synthetic_leaderboard(rng, 5)
        reporter.update_leaderboard(leaderboard)

    reporter.phase_started(RunPhase.RANKING_RESULTS, items_total=len(leaderboard))
    _sleep()
    reporter.progress(items_completed=len(leaderboard))
    reporter.update_leaderboard(leaderboard)  # already ranked by _synthetic_leaderboard

    reporter.completed(
        summary={
            "discovered_specs": discovered, "eligible_specs": eligible, "unique_entry_classes": unique_classes,
            "exit_variants_generated": exit_variants_generated, "backtests_completed": backtests_completed,
            "leaderboard_size": len(leaderboard),
        }
    )
    return reporter


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a deterministic DEMO run artifact stream for tradelab.research_dashboard.")
    parser.add_argument("--runs-dir", default="runs", help="Directory to create the demo run under (default: ./runs)")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--fast", action="store_true", help="Skip real-time pacing (writes all artifacts immediately)")
    parser.add_argument("--step-delay-seconds", type=float, default=0.15)
    args = parser.parse_args(argv)

    reporter = run_demo(runs_dir=args.runs_dir, seed=args.seed, real_time=not args.fast, step_delay_seconds=args.step_delay_seconds)
    print(f"Demo run written to {reporter.run_dir}")
    print(f"run_id: {reporter.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
