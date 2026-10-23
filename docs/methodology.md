# Methodology and limitations

## What the software can establish

TradeLab makes a declared OHLCV simulation auditable. Unit tests demonstrate
contracts on synthetic cases; they do not establish an investment edge. A full
evaluation also needs trustworthy data, explicit costs, calendar coverage,
statistical controls and untouched out-of-sample evidence.

## Time and causality

`Bar1m.timestamp` is the UTC bar-open time, not the moment its OHLC was known.
Feature objects expose availability times (`known_at`). Higher-timeframe candles
are built from the same one-minute input, requiring complete aligned members.
Pivot confirmation must wait for its confirming bars. New York sessions use an
IANA time zone rather than a hard-coded UTC offset.

Signals are evaluated at their eligible decision time. A signal derived from a
close can fill no earlier than the next eligible open. Filtering post-hoc results
is not the same as resimulating entry/exit decisions under that filter.

## Execution assumptions

The simple OHLC backtester assumes fills at its documented reference prices and
cannot observe the true intrabar path, queue position or broker latency. When
both stop and target are crossed by one candle, the documented reference policy
chooses stop first. This can be conservative for that ambiguity while still
being optimistic about other real-world effects, including stop slippage.

Only one active position is allowed in the reference single-strategy simulator;
signals while it is occupied are recorded as skipped. An unresolved final
position is not fabricated into a favourable exit. Tick simulation is a separate
adapter with its own conventions; do not silently compare it as an identical model.

## Accounting and risk

Points are not dollars. Instrument point value, contract count, commissions and
slippage assumptions must be explicit. Account-equity and trade-level drawdowns
answer different questions. A historical drawdown statistic is not a guarantee
about future losses. Legacy account-rule examples are models, not statements of
any firm's current rules or certification of compliance.

## Research validity

Predeclare the hypothesis, universe, selection objective, data period and budget.
Separate discovery, validation and final holdout. Account for multiple comparisons,
selection bias, parameter instability, small samples and regime changes. Report
failed and unresolved cases, not only selected winners. Do not tune on a forward
period and continue calling it untouched forward validation.

No historical market performance claim is part of this public-core release.
Calendar and news algorithms are included, but proprietary calendars/news files
are not. Synthetic fixtures are designed for correctness, not realism.

The packaged `_resources/research_period_policy.json` contains **illustrative**
dates for regression compatibility, not an approved period split for your data.
`load_research_period_policy(path)` accepts an explicitly supplied policy. Some
legacy runner entrypoints load the default policy; do not use those defaults as
an implicit scientific justification. A holdout duration alone is not evidence
that the data were untouched, and late data are not guaranteed to remain unused.
