# Architecture and first API example

The pipeline separates data, features, decisions, execution and accounting:

```text
Local CSV / Bar1m
  -> validation and complete aggregation
  -> features with explicit known_at timestamps
  -> selector timeline and declared StrategySpec
  -> prepared context -> signal-driven simulation
  -> accounting / equity / audit
  -> journal + run telemetry -> local dashboard
```

Start with `tradelab.synthetic_demo`: it constructs bars, declares a deliberately
simple educational strategy and calls `backtest_strategy`. It compares a target
case with a candle crossing both barriers. It does not import test fixtures or a
private selected strategy. Read its source before adapting it to your own data.

For larger studies, `prepare_backtest_context` separates data-dependent feature
work from per-strategy simulation. Never assume context caching permits changing
inputs, selectors or calendars without invalidating the relevant cache.

Local CSV example (invented values only):

```text
timestamp,open,high,low,close,volume,contract
2026-01-05T14:30:00Z,100,101,99,100.5,10,SYNTHETIC
```

Load it with `read_csv_bars(Path(...))`, then call `validate_bars`. Naive timestamps,
duplicate/out-of-order bars and invalid ranges should be investigated, not patched
to improve results. Parquet is optional and needs a separately installed reader.

`research_journal` provides durable evaluation state. `research_run` provides
manifests, status, telemetry and checkpoints for reporting; its JSON checkpoints
alone are not an implementation of resuming a backtest. `research_demo` generates
fabricated reporter data and is not the real-engine example.

The dashboard reads local artifacts; it is not a public server. It should remain
bound to loopback. No external chart library, account or API key is needed.

Source modules retain some references to historical private task/spec documents.
Those references explain development provenance, not additional shipped features
or required files. The supported public entrypoints are documented in README.
