# Roadmap and useful first contributions

These are proposed work items, not claims that a community or user base already
exists. Open an issue to agree the scope before implementing a model change.

## Small, contributor-friendly tasks

- Add a small synthetic causality regression around a missing higher-timeframe
  member and explain the expected `known_at` behaviour.
- Improve the public API examples with a short/long symmetry test.
- Audit documentation references left from the private baseline and replace
  them with self-contained descriptions, without changing code semantics.
- Document setup on a system without an IANA time-zone database; do not claim
  Windows portability before testing it.

## Research correctness and reliability

- Publish reproducible, synthetic optimized-vs-reference benchmarks with machine,
  interpreter, memory and timing methodology. Never publish invented speedups.
- Extend adversarial calendar/DST, gap, fill-order and accounting regressions.
- Reduce implicit coupling between context preparation and result persistence.
- Review local dashboard/cache trust boundaries; keep loopback-only defaults.

## Community readiness

- Collect feedback from real users and record what they found useful.
- Add external reproductions and reviewed contributions with attribution.
- Maintain a changelog and small, reviewable releases.
- Treat funding/program applications as descriptions of verified work, not a
  reason to inflate adoption, test counts or ecosystem impact.
