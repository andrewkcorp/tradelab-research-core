"""Opt-in, measured per-stage wall-clock timing (task 047, requirement 1).

Not a profiler and not always-on: every call site that can record into
a ``StageTimings`` takes it as an optional keyword defaulting to
``None``, and the entire measurement is skipped (one ``is not None``
check, no ``time.perf_counter()`` call) when the caller doesn't pass
one -- so production/preflight runs pay nothing for this existing.
Only ``tradelab.backtest_v2.run_stage_diagnostic`` (and tests) ever
construct one.

Deliberately raw ``time.perf_counter()`` deltas at each call site,
never a context manager -- a ``with`` block's ``__enter__``/``__exit__``
overhead, paid twice per observed entry inside the hot execution loop,
would itself measurably distort the very numbers being measured.

Five stages are used, matching task 047's own required breakdown
exactly (see ``tradelab.backtest_v2.run_stage_diagnostic``'s own
docstring for which call site records into which):
``bar_range_index_build``, ``entry_target_resolution``,
``exit_resolution``, ``accounting_equity``, ``persistence``."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageTimings:
    totals_seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, stage: str, elapsed_seconds: float) -> None:
        self.totals_seconds[stage] = self.totals_seconds.get(stage, 0.0) + elapsed_seconds
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def to_dict(self) -> dict:
        total = sum(self.totals_seconds.values())
        return {
            "totals_seconds": dict(self.totals_seconds),
            "counts": dict(self.counts),
            "share": {stage: (seconds / total if total > 0 else 0.0) for stage, seconds in self.totals_seconds.items()},
        }
