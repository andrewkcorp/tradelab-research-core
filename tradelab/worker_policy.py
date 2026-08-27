"""The production worker count, and the measurement that chose it.

**One source of truth.** The number lives here, not in an argparse default and
not in a string in the dashboard template. ``tradelab.production_pool_run``
reads it for its ``--workers`` default and ``tradelab.research_dashboard``
serves it to the UI, so the CLI, the dashboard and the documentation cannot
drift apart -- a number shown next to a justification has to be the number the
code actually uses, or the justification is worse than useless.

**Set by measurement and an explicit operating policy, never by core count.**
``tradelab.worker_count_benchmark`` runs each candidate several times on one
fixed bundle. A later policy choice may weigh wall-clock time against compute
efficiency, but it must be documented with the measured gate and never inferred
from spare cores or a momentary CPU reading.

This module deliberately imports nothing from the package. It is read by a
read-only dashboard request handler, and a policy constant must never be able
to drag the simulation engine into that process.
"""
from __future__ import annotations

from typing import Any

#: Worker count a production run uses unless the operator overrides it.
PRODUCTION_WORKER_COUNT = 8

#: One sentence, shown in the dashboard next to the live worker count.
WORKER_COUNT_RATIONALE = (
    "Set by measurement, not by core count. See the benchmark for the numbers and the gate."
)

#: The report that decided it, relative to the repository root.
WORKER_COUNT_BENCHMARK_DOC = "docs/worker-count-benchmark-v3.md"

#: The measured figures behind the choice, so the dashboard can show the basis
#: rather than only the conclusion.
WORKER_COUNT_EVIDENCE: dict[str, Any] = {
    "decided_by": "docs/worker-count-benchmark-v3.md",
    "candidates_measured": [5, 8],
    "baseline_workers": 5,
    "candidate_workers": 8,
    "required_speedup_pct": 5.0,
    "measured_speedup_pct": 31.2,
    "repeats": 3,
    "gate_met": True,
    "max_worker_seconds_per_shard_growth_pct": 30.0,
    "summary": (
        "Six paired runs on one fixed bundle: 8 workers held a median +31.2% throughput over 5, "
        "with no swap growth, normal memory pressure, at least 4.435 GB available RAM and no worker failures. "
        "All six runs produced byte-identical journal rows and leaderboards. The production policy permits "
        "up to 30% worker-seconds-per-shard growth on this dedicated research machine; measured growth was 27.5%."
    ),
}


def worker_policy() -> dict[str, Any]:
    """The policy as a plain dict, for the dashboard's own endpoint."""
    return {
        "production_worker_count": PRODUCTION_WORKER_COUNT,
        "rationale": WORKER_COUNT_RATIONALE,
        "benchmark_doc": WORKER_COUNT_BENCHMARK_DOC,
        "evidence": dict(WORKER_COUNT_EVIDENCE),
    }
