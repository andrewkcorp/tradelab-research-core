# Contributing to TradeLab

Thank you for helping make research assumptions easier to inspect.

## First contribution

1. Run `python -m tradelab.synthetic_demo` from the source root.
2. Run `python -m unittest tests.test_public_examples -v`.
3. Pick a small item from [the roadmap](docs/roadmap.md), or open an issue before
   proposing a model, calendar or execution-policy change.
4. Make a focused branch and add a synthetic failing test for a bug fix.
5. Run the full test suite and `python tools/check_public_tree.py --tree .`.
6. Open a pull request with exact commands, results and limitations.

Use Python's standard library where possible. A new dependency requires maintainer
agreement. We do not accept broker execution or credentials in this research core.

## Research correctness

- Preserve next-eligible-open semantics and `known_at` causality.
- Cover both directions, boundaries, empty/missing data and adverse ambiguity.
- Do not change a strategy rule, metric definition or fill assumption silently.
- Compare optimized paths against their reference implementations. Faster code
  without equivalence evidence is not an accepted optimization.
- Do not hide failing or data-dependent tests behind broad skip conditions.
- Report the Python/OS version and any optional tests skipped.

## Data and privacy

Fixtures must be synthetic or have documented redistribution rights. Use invented
contract/account identifiers, relative paths and minimal reproductions. Never attach
real account statements, API credentials, paid feeds, private research archives or
user-home paths. Generated databases and run directories are not contributions.

## AI-assisted contributions

AI assistance is welcome, but the contributor remains responsible for correctness,
rights and review. Disclose material assistance in the pull request. Do not invent
benchmark numbers or label a model-generated explanation as an executed test.
Agents follow [AGENTS.md](AGENTS.md); they cannot approve their own model changes.

## Review and licensing

Small changes may be reviewed by the maintainer; changes to simulation semantics
need an explicit methodology note and regression tests. Submitting a contribution
means you have the right to offer it under this repository's MIT License. We do not
require transferring ownership of your contribution.
