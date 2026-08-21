"""Sequential, lazy research batch runner: streams a ``ResearchEvaluation``
for every ``(StrategySpec, Instrument)`` pair, reusing one already-
prepared ``PreparedBacktestContext`` across the whole sweep instead of
re-running any detector per strategy.

Deliberately out of scope for this first version (later, separate
steps): a confluence-combination generator, result persistence /
checkpoint-resume, parallel execution, market data loading, a final
cross-strategy ranking pass, and any change to what "confluence" means
anywhere in this package. This module only sequences three already-
existing, already-tested layers -- ``tradelab.backtest.backtest_
prepared_strategy``, ``tradelab.accounting.apply_accounting``,
``tradelab.equity.build_equity_curve`` -- against a caller-supplied
stream of ``StrategySpec``s, and reports, per strategy per instrument,
whether it clears a minimum-trade-frequency bar.

**Lazy, streaming, one strategy at a time.** ``iter_research_evaluations``
is a generator function: calling it runs none of its body, and it never
accumulates the *entire* ``strategy_specs`` input, nor its own output,
into a list -- each ``ResearchEvaluation`` is produced only as the
caller asks for the next one. It also never calls ``prepare_backtest_
context`` -- only ``backtest_prepared_strategy`` against an already-
built context, so every expensive detector in ``tradelab.backtest``'s
own "preparation vs. execution" split (FVG/liquidity/bias/entry-
timeline detection) runs exactly once for the whole sweep, not once per
strategy.

**Frequency rule** (this task's own spec): a strategy qualifies only if
*every one* of ``research_config.windows`` independently sees at least
``research_config.min_trades_per_window`` *accounted* trades whose own
``entry_timestamp`` falls in that window's own half-open ``[start_utc,
end_utc)``. "Accounted" means ``tradelab.accounting.AccountBacktestResult
.accounted_trades`` specifically -- an untradeable trade (stop risk
already exceeds ``max_stop_risk_usd`` at a single contract, sizing
rejected it before any position ever opened) never counts, while an
``OPEN_UNRESOLVED`` trade (data ran out with the position still open)
does count, since a position genuinely opened either way. Production
defaults to MNQ only. A caller may explicitly supply a different instrument
tuple for an isolated comparison; one ``ResearchEvaluation`` is emitted per
requested instrument from a single shared ``BacktestResult``.

**Non-qualifying strategies are never candidates for a later ranking
pass, but are never silently dropped either.** Every ``StrategySpec``
this generator is given produces exactly ``len(instruments)``
``ResearchEvaluation``s, each carrying its own per-window trade counts
and, if rejected, a ``RejectionReason`` -- whether it ultimately
qualified or not.

**Safe prefilter, never a false rejection.** Before running the
backtest at all, this module can already look at how many raw
``EntrySignal``s of the strategy's own ``primary_entry_selector`` and
``direction`` fall in each window (``context.timeline.signals_by_
selector``) -- since every accounted trade traces back to exactly one
such signal (a strategy's own ``entry_timestamp`` always equals the
winning trigger signal's ``known_at``, see ``tradelab.backtest``), the
number of accounted trades landing in a window can never exceed the
number of raw signals of the right direction landing in that same
window. If even one window's raw-signal count already falls short of
``min_trades_per_window``, the real backtest is *guaranteed* to also
fall short there (the true count can only be lower, never higher), so
the whole strategy is guaranteed to fail the "every window" requirement
-- safe to reject without paying for a backtest that cannot possibly
qualify. The reverse is never done: a window whose raw signal count
already clears the bar says nothing about whether the *real* trade
count will (skipped signals, missing TP/SL targets, active-position
overlap, and untradeable sizing can all only reduce the real count), so
such a strategy always still runs the real backtest. A prefiltered
``ResearchEvaluation`` is marked ``prefiltered=True`` and its own
``WindowTradeCount.trade_count`` values are this raw upper bound, not a
real accounted-trade count -- audit-visible, never disguised as the
real figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Iterator

from tradelab.accounting import AccountBacktestResult, Instrument, RiskConfig, apply_accounting
from tradelab.backtest import BacktestResult, PreparedBacktestContext, backtest_prepared_strategy
from tradelab.equity import EquityConfig, EquityResult, build_equity_curve
from tradelab.strategy_spec import StrategySpec

# Production research is MNQ-only. ``Instrument.NQ`` remains available to
# isolated accounting fixtures, but must be passed explicitly for any
# historical comparison.
DEFAULT_INSTRUMENTS: tuple[Instrument, ...] = (Instrument.MNQ,)


@dataclass(frozen=True)
class ResearchWindow:
    """One caller-defined, timezone-aware, half-open ``[start_utc,
    end_utc)`` window a strategy's trade frequency is measured against.
    This module never guesses at window boundaries (e.g. whether a
    partial calendar year should count) -- windows are always exactly
    what the caller passes."""

    window_id: str
    start_utc: datetime
    end_utc: datetime

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("ResearchWindow.window_id must be non-empty")
        if self.start_utc.tzinfo is None:
            raise ValueError(f"ResearchWindow.start_utc must be timezone-aware, got {self.start_utc!r}")
        if self.end_utc.tzinfo is None:
            raise ValueError(f"ResearchWindow.end_utc must be timezone-aware, got {self.end_utc!r}")
        if not self.start_utc < self.end_utc:
            raise ValueError(f"ResearchWindow.start_utc ({self.start_utc!r}) must be < end_utc ({self.end_utc!r})")


@dataclass(frozen=True)
class ResearchConfig:
    """``windows`` must be non-empty, have unique ``window_id``s, and be
    pairwise non-overlapping (two windows may touch exactly at a shared
    boundary -- ``[start, end)`` makes that unambiguous, never a double
    count). ``min_trades_per_window`` (default 100, this task's own
    spec) is the same minimum applied independently to every window."""

    windows: tuple[ResearchWindow, ...]
    risk_config: RiskConfig
    equity_config: EquityConfig
    min_trades_per_window: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.min_trades_per_window, int) or isinstance(self.min_trades_per_window, bool) or self.min_trades_per_window < 1:
            raise ValueError(f"ResearchConfig.min_trades_per_window must be a positive integer, got {self.min_trades_per_window!r}")
        if not isinstance(self.windows, tuple):
            raise ValueError(f"ResearchConfig.windows must be a tuple, got {type(self.windows).__name__}")
        if not self.windows:
            raise ValueError("ResearchConfig.windows must not be empty")
        ids = [w.window_id for w in self.windows]
        if len(ids) != len(set(ids)):
            dupes = sorted({window_id for window_id in ids if ids.count(window_id) > 1})
            raise ValueError(f"Duplicate ResearchWindow.window_id values: {dupes}")
        by_start = sorted(self.windows, key=lambda w: w.start_utc)
        for earlier, later in zip(by_start, by_start[1:]):
            if earlier.end_utc > later.start_utc:
                raise ValueError(
                    f"ResearchWindow {earlier.window_id!r} [{earlier.start_utc!r}, {earlier.end_utc!r}) overlaps "
                    f"{later.window_id!r} [{later.start_utc!r}, {later.end_utc!r}) -- windows must be non-overlapping"
                )
        if not isinstance(self.risk_config, RiskConfig):
            raise ValueError(f"ResearchConfig.risk_config must be a RiskConfig, got {self.risk_config!r}")
        if not isinstance(self.equity_config, EquityConfig):
            raise ValueError(f"ResearchConfig.equity_config must be an EquityConfig, got {self.equity_config!r}")


@dataclass(frozen=True)
class WindowTradeCount:
    """One ``ResearchWindow``'s own trade tally for one
    ``ResearchEvaluation``. ``trade_count`` is a real accounted-trade
    count unless the owning evaluation is prefiltered (``ResearchEvaluation
    .prefiltered``), in which case it is the safe raw-signal upper bound
    instead -- see the module docstring's "Safe prefilter" section."""

    window_id: str
    trade_count: int
    meets_minimum: bool


class RejectionReason(str, Enum):
    INSUFFICIENT_TRADE_FREQUENCY = "insufficient_trade_frequency"


@dataclass(frozen=True)
class ResearchEvaluation:
    """One ``(StrategySpec, Instrument)`` audit record from ``iter_
    research_evaluations``. ``backtest_result``/``account_result``/
    ``equity_result`` are populated only when ``qualifies_frequency`` is
    ``True`` -- a non-qualifying strategy can never become a later
    ranking candidate, so this module does not keep its full trade-level
    detail around for it; ``window_trade_counts`` and ``rejection_
    reason`` remain the audit trail either way."""

    strategy_id: str
    strategy: StrategySpec
    instrument: Instrument
    window_trade_counts: tuple[WindowTradeCount, ...]
    qualifies_frequency: bool
    rejection_reason: RejectionReason | None
    prefiltered: bool
    backtest_result: BacktestResult | None
    account_result: AccountBacktestResult | None
    equity_result: EquityResult | None


def iter_research_evaluations(
    strategy_specs: Iterable[StrategySpec],
    context: PreparedBacktestContext,
    research_config: ResearchConfig,
    instruments: Iterable[Instrument] = DEFAULT_INSTRUMENTS,
) -> Iterator[ResearchEvaluation]:
    """Lazily yield one ``ResearchEvaluation`` per ``(strategy, instrument)``
    pair for every ``StrategySpec`` in ``strategy_specs`` -- a generator
    function, so calling this runs none of its body up front. ``context``
    is read-only and reused unchanged for every strategy (this function
    never calls ``prepare_backtest_context`` and never mutates ``context``);
    ``instruments`` is small and bounded (default NQ + MNQ) so it alone is
    safely materialized into a tuple, unlike ``strategy_specs``, which may
    be arbitrarily large or itself lazy and is iterated exactly once, in
    order, never converted to a list.

    See the module docstring for the frequency rule and the safe
    prefilter this function applies before running each strategy's
    backtest.
    """
    if not isinstance(context, PreparedBacktestContext):
        raise ValueError(f"context must be a PreparedBacktestContext, got {context!r}")
    if not isinstance(research_config, ResearchConfig):
        raise ValueError(f"research_config must be a ResearchConfig, got {research_config!r}")
    instruments = tuple(instruments)
    if not instruments:
        raise ValueError("instruments must not be empty")

    for strategy in strategy_specs:
        raw_counts = _count_per_window(_raw_signal_known_ats(strategy, context), research_config.windows)

        if any(raw_counts[window.window_id] < research_config.min_trades_per_window for window in research_config.windows):
            window_trade_counts = _window_trade_counts(raw_counts, research_config)
            for instrument in instruments:
                yield ResearchEvaluation(
                    strategy_id=strategy.strategy_id,
                    strategy=strategy,
                    instrument=instrument,
                    window_trade_counts=window_trade_counts,
                    qualifies_frequency=False,
                    rejection_reason=RejectionReason.INSUFFICIENT_TRADE_FREQUENCY,
                    prefiltered=True,
                    backtest_result=None,
                    account_result=None,
                    equity_result=None,
                )
            continue

        backtest_result = backtest_prepared_strategy(strategy, context)
        for instrument in instruments:
            yield _evaluate_instrument(strategy, backtest_result, instrument, research_config)


def _raw_signal_known_ats(strategy: StrategySpec, context: PreparedBacktestContext) -> Iterator[datetime]:
    signals = context.timeline.signals_by_selector.get(strategy.primary_entry_selector, [])
    return (signal.known_at for signal in signals if signal.direction == strategy.direction)


def _evaluate_instrument(
    strategy: StrategySpec, backtest_result: BacktestResult, instrument: Instrument, research_config: ResearchConfig
) -> ResearchEvaluation:
    account_result = apply_accounting(backtest_result, instrument, research_config.risk_config)
    entry_timestamps = (trade.entry_timestamp for trade in account_result.accounted_trades)
    real_counts = _count_per_window(entry_timestamps, research_config.windows)
    window_trade_counts = _window_trade_counts(real_counts, research_config)
    qualifies = all(count.meets_minimum for count in window_trade_counts)

    if not qualifies:
        return ResearchEvaluation(
            strategy_id=strategy.strategy_id,
            strategy=strategy,
            instrument=instrument,
            window_trade_counts=window_trade_counts,
            qualifies_frequency=False,
            rejection_reason=RejectionReason.INSUFFICIENT_TRADE_FREQUENCY,
            prefiltered=False,
            backtest_result=None,
            account_result=None,
            equity_result=None,
        )

    equity_result = build_equity_curve(account_result, research_config.equity_config)
    return ResearchEvaluation(
        strategy_id=strategy.strategy_id,
        strategy=strategy,
        instrument=instrument,
        window_trade_counts=window_trade_counts,
        qualifies_frequency=True,
        rejection_reason=None,
        prefiltered=False,
        backtest_result=backtest_result,
        account_result=account_result,
        equity_result=equity_result,
    )


def _count_per_window(timestamps: Iterable[datetime], windows: tuple[ResearchWindow, ...]) -> dict[str, int]:
    counts = {window.window_id: 0 for window in windows}
    for ts in timestamps:
        for window in windows:
            if window.start_utc <= ts < window.end_utc:
                counts[window.window_id] += 1
    return counts


def _window_trade_counts(counts: dict[str, int], research_config: ResearchConfig) -> tuple[WindowTradeCount, ...]:
    return tuple(
        WindowTradeCount(
            window_id=window.window_id,
            trade_count=counts[window.window_id],
            meets_minimum=counts[window.window_id] >= research_config.min_trades_per_window,
        )
        for window in research_config.windows
    )
