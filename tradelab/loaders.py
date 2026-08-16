"""Local-file loaders for 1-minute bars: CSV and Parquet only.

No network access, no data-vendor client, no API keys. Databento (or any
other vendor) integration is explicitly out of scope for this module.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from tradelab.bar import Bar1m

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def read_csv_bars(path: Path, *, default_contract: str | None = None) -> list[Bar1m]:
    """Read 1-minute bars from a local CSV file.

    Expected columns (case-insensitive): timestamp, open, high, low, close,
    volume, and optionally contract. ``timestamp`` must carry an explicit
    UTC offset (e.g. ``2026-01-01T09:00:00Z`` or ``...+00:00``); naive
    timestamps are rejected rather than silently assumed to be in some
    timezone. If the file has no ``contract`` column, pass
    ``default_contract`` to stamp every row with it.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row")
        columns = {name.lower(): name for name in reader.fieldnames}
        missing = REQUIRED_COLUMNS - columns.keys()
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        has_contract_column = "contract" in columns
        if not has_contract_column and default_contract is None:
            raise ValueError("CSV has no 'contract' column; pass default_contract=...")

        bars = []
        for row_number, row in enumerate(reader, start=2):
            lowered = {key.lower(): value for key, value in row.items() if key}
            try:
                bar = Bar1m(
                    timestamp=_parse_timestamp(lowered["timestamp"]),
                    open=float(lowered["open"]),
                    high=float(lowered["high"]),
                    low=float(lowered["low"]),
                    close=float(lowered["close"]),
                    volume=float(lowered["volume"]),
                    contract=lowered["contract"] if has_contract_column else default_contract,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid row {row_number}: {error}") from error
            bars.append(bar)
    return bars


def read_parquet_bars(path: Path, *, default_contract: str | None = None) -> list[Bar1m]:
    """Read 1-minute bars from a local Parquet file.

    Requires the optional ``pyarrow`` dependency, imported lazily so it is
    only needed when Parquet support is actually used (``read_csv_bars``
    works with the standard library alone).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Parquet support requires the optional 'pyarrow' package. Install it with: pip install pyarrow"
        ) from error

    table = pq.read_table(str(path))
    columns = {name.lower(): name for name in table.column_names}
    missing = REQUIRED_COLUMNS - columns.keys()
    if missing:
        raise ValueError(f"Parquet file missing required columns: {sorted(missing)}")
    has_contract_column = "contract" in columns
    if not has_contract_column and default_contract is None:
        raise ValueError("Parquet file has no 'contract' column; pass default_contract=...")

    timestamp_column = table.column(columns["timestamp"])
    is_native_timestamp = pa.types.is_timestamp(timestamp_column.type)
    timestamps = timestamp_column.to_pylist()
    opens = table.column(columns["open"]).to_pylist()
    highs = table.column(columns["high"]).to_pylist()
    lows = table.column(columns["low"]).to_pylist()
    closes = table.column(columns["close"]).to_pylist()
    volumes = table.column(columns["volume"]).to_pylist()
    contracts = table.column(columns["contract"]).to_pylist() if has_contract_column else None

    bars = []
    for index, raw_timestamp in enumerate(timestamps):
        try:
            timestamp = raw_timestamp if is_native_timestamp else _parse_timestamp(str(raw_timestamp))
            if timestamp.tzinfo is None:
                raise ValueError(f"timestamp {raw_timestamp!r} has no timezone offset; UTC offset required")
            bar = Bar1m(
                timestamp=timestamp,
                open=float(opens[index]),
                high=float(highs[index]),
                low=float(lows[index]),
                close=float(closes[index]),
                volume=float(volumes[index]),
                contract=contracts[index] if contracts is not None else default_contract,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid row {index}: {error}") from error
        bars.append(bar)
    return bars


def _parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp {value!r} has no timezone offset; UTC offset required")
    return parsed
