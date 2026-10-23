"""Actual engine on invented bars; not a market-performance demonstration.

Unlike research_demo (fabricated dashboard telemetry), this example invokes the
backtester. Prices are deliberately hand-built to make assumptions inspectable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from tradelab.backtest import BacktestResult, backtest_strategy
from tradelab.bar import Bar1m
from tradelab.sl_targets import SLSourceType
from tradelab.strategy_spec import BiasPolicy, LONG, SLChoice, StrategySpec, TPChoice
from tradelab.tp_targets import TPSourceType

START = datetime(2026,1,5,14,30,tzinfo=timezone.utc)


def synthetic_bars(*, ambiguous: bool = False) -> list[Bar1m]:
    """Invent a stop anchor, target anchor, taken pivot and next-open fill."""
    flat = (100.0,100.0,100.0,100.0)
    values = [flat,flat,(90.5,91,90,90.5),flat,flat,(119.5,120,119,119.5),
              flat,flat,(95.5,96,95,95.5),flat,flat,(99,100.5,94,94.5),
              (99,100,98,99.5),
              (99,121,89,100) if ambiguous else (115,121,114,120)]
    return [Bar1m(START+timedelta(minutes=i),o,h,l,c,1.0,'SYNTHETIC')
            for i,(o,h,l,c) in enumerate(values)]


def run_example(*, ambiguous: bool = False) -> BacktestResult:
    spec = StrategySpec(
        strategy_id='synthetic-educational-example',direction=LONG,
        primary_entry_selector='liquidity-1m-swing_equal-swing_low-take-reversal',
        required_confluence_selectors=(),bias_policy=BiasPolicy.IGNORE,
        tp_choice=TPChoice(source_type=TPSourceType.LIQUIDITY,rank=1),
        sl_choice=SLChoice(source_type=SLSourceType.LIQUIDITY,rank=1),
    )
    return backtest_strategy(spec,synthetic_bars(ambiguous=ambiguous),[],
                             pivot_strength=1,reclaim_window_bars=3)


def summary(result: BacktestResult) -> dict:
    return {'trades':result.trade_count,'gross_points':result.gross_pnl_points_total,
            'executions':[{'signal_known_at':t.signal_known_at.isoformat(),
                           'entry_at':t.entry_timestamp.isoformat(),'entry':t.entry_price,
                           'stop':t.sl_price,'target':t.tp_price,
                           'exit_reason':t.exit_reason.value,'exit':t.exit_price}
                          for t in result.trades]}


def main() -> int:
    report = {'synthetic':True,'actual_engine_invoked':True,
              'warning':'Invented bars; not real market performance. Gross points exclude accounting costs.',
              'target_case':summary(run_example()),
              'both_barriers_case':summary(run_example(ambiguous=True))}
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0


if __name__ == '__main__': raise SystemExit(main())
