"""The production worker count, and the measurement that chose it.

**One source of truth.** The number lives here, not in an argparse default and
not in a string in the dashboard template. ``tradelab.production_pool_run``
reads it for its ``--workers`` default and ``tradelab.research_dashboard``
serves it to the UI, so the CLI, the dashboard and the documentation cannot
drift apart -- a number shown next to a justification has to be the number the
code actually uses, or the justification is worse than useless.

**Set by measurement, never by core count.** ``tradelab.worker_count_
benchmark`` runs each candidate several times on one fixed bundle and applies a
gate fixed before the measurement. Do not raise this because the machine has
spare cores or because a run feels slow: re-run the benchmark.

This module deliberately imports nothing from the package. It is read by a
read-only dashboard request handler, and a policy constant must never be able
to drag the simulation engine into that process.
"""
from __future__ import annotations

from typing import Any

#: Worker count a production run uses unless the operator overrides it.
PRODUCTION_WORKER_COUNT = 5

#: One sentence, shown in the dashboard next to the live worker count.
WORKER_COUNT_RATIONALE = (
    "Set by measurement, not by core count. See the benchmark for the numbers and the gate."
)

#: The report that decided it, relative to the repository root.
WORKER_COUNT_BENCHMARK_DOC = "docs/worker-count-benchmark-v2.md"

#: The measured figures behind the choice, so the dashboard can show the basis
#: rather than only the conclusion.
WORKER_COUNT_EVIDENCE: dict[str, Any] = {
    "decided_by": "docs/worker-count-benchmark-v2.md",
    "candidates_measured": [3, 4, 5],
    "baseline_workers": 4,
    "candidate_workers": 5,
    "required_speedup_pct": 5.0,
    "measured_speedup_pct": 7.8,
    "repeats": 3,
    "gate_met": True,
    "summary": (
        "9 runs on one fixed bundle: 5 workers held a median +7.8% over 4 (and 4 held +14.3% over 3), "
        "with no swap growth, memory pressure normal throughout and no worker failures. "
        "All nine runs produced byte-identical journal rows and leaderboards."
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
