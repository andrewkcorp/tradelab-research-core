"""TradeLab data foundation: normalized 1m bars, loaders, validation, aggregation.

Research-only. No broker connectivity, no live execution, no network calls.
"""
from tradelab.bar import Bar1m
from tradelab.aggregate import AggregatedBar, aggregate_bars
from tradelab.loaders import read_csv_bars, read_parquet_bars
from tradelab.validation import MissingMinutesGap, ValidationResult, validate_bars

__all__ = [
    "Bar1m",
    "AggregatedBar",
    "aggregate_bars",
    "MissingMinutesGap",
    "ValidationResult",
    "validate_bars",
    "read_csv_bars",
    "read_parquet_bars",
]
