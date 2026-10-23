# TradeLab Research Core

**Inspect the assumptions, not just the equity curve.**

TradeLab is a local-first Python toolkit for reproducible, event-driven research
on one-minute OHLCV data. It separates when a feature becomes knowable, when a
signal is eligible, how an order is simulated, and how costs and risk are accounted
for. It is intended for researchers, developers and people reviewing backtests.

**Research only. No broker connection, live order routing, investment advice or
claim of profitable trading.** Synthetic examples demonstrate software behaviour,
not market performance. No proprietary market data is included.

## Try it without an account or API key

Use Python 3.11+ on a system with the IANA time-zone database (macOS/Linux normally
provide it). The core and the examples use the Python standard library.

From the repository root:

```sh
python -m tradelab.synthetic_demo
python -m unittest tests.test_public_examples -v
```

The first command runs the **actual backtest engine** against a tiny, hand-built
synthetic price path. It shows a next-bar entry and a stop-first ambiguity case.
No downloads, registration, external service or real market data are needed.

### Explore the local dashboard

```sh
python -m tradelab.research_demo --fast --runs-dir runs
python -m tradelab.research_dashboard --runs-dir runs --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. Stop the server with Ctrl+C. Use another port if it
is already occupied. Keep this unauthenticated research server on loopback.

**The dashboard demo is different from the engine example:** its leaderboard,
equity and performance metrics are fabricated with a seeded RNG to exercise the
reporting interface. The dashboard displays a DEMO banner. Do not use those
numbers as evidence of a backtest or a profitable strategy.

## What is here

- Validated UTC one-minute bars; local CSV import and optional Parquet support.
- Complete-bar aggregation into higher timeframes, with New York session/DST
  handling and explicit feature-availability timestamps.
- Auditable FVG/iFVG, swing/equal/session-liquidity, context and bias primitives.
- Declarative strategy specifications, reusable prepared contexts and a
  signal-driven single-position simulator.
- Accounting, sizing, costs, equity, mark-to-market and outlier analysis.
- SQLite journals, resumable research components and a self-contained local
  dashboard. JSON telemetry checkpoints are **not** by themselves a resume engine.
- Synthetic regression fixtures, including adverse and boundary cases.

This first public-core candidate intentionally excludes the private project's
selected strategies, research runs, optimization results, vendor data and
TradingView/NinjaTrader exports. It is not the latest private research branch.
`SOURCE_MANIFEST.json` records the exact original source baseline and file hashes.

## Research contracts

1. A signal based on a candle close cannot fill before the next eligible open.
2. Facts have explicit `known_at` times. Unconfirmed pivots/HTF candles must not
   be treated as already known.
3. OHLC candles do not reveal intrabar order. In the documented ambiguity case,
   the simulator resolves stop before target rather than assuming the favourable
   sequence. This does not make idealized OHLC fills equivalent to broker fills.
4. Missing bars, warm-up, calendar coverage, costs and unresolved positions must
   stay visible instead of being silently converted into favourable results.
5. Post-hoc filters and multiple testing must be disclosed. A high in-sample
   score is not an out-of-sample validation.

See [methodology and limitations](docs/methodology.md),
[architecture and API walkthrough](docs/architecture.md) and
[data policy](docs/data-policy.md).

## Development

```sh
python -m unittest discover -s tests -t . -q
python tools/check_public_tree.py --tree .
```

Some optional Parquet tests require `pyarrow`; they are explicitly skipped when
it is not installed. No third-party dependency is mandatory for the core examples.
On Windows or other systems without IANA data, install/provide that database
before using `America/New_York`; Windows support is not certified by this release.

Optional installation into your own virtual environment:

```sh
python -m venv .venv
# Activate .venv with the command appropriate for your shell.
python -m pip install .
```

The build uses setuptools; this is a build-time dependency, not an API service.
The source-checkout quickstart above needs neither installation nor setuptools.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md), [governance](GOVERNANCE.md),
[security reporting](SECURITY.md) and the [roadmap](docs/roadmap.md).
Small synthetic bug reproductions, clearer assumptions and adversarial tests are
particularly welcome. Please do not submit paid datasets, account logs or secrets.

The project is early-stage and independently maintained. We do not claim
established adoption, external contributors, audited profitability or production
readiness. AI-assisted changes receive the same review and evidence requirements
as any other contribution.

## License

The released code is offered under the [MIT License](LICENSE). This does not grant
rights to third-party market data, exchange content or broker services.
