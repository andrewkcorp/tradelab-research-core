"""Strict parser for a local Databento parent-symbology CSV export.

Databento's ``stype`` `parent` resolution for an input like ``MNQ.FUT``
(dataset ``GLBX.MDP3``, schema ``ohlcv-1m``) resolves to **every**
currently/recently listed instrument under that root at once -- every
quarterly outright contract (``MNQZ3``, ``MNQH4``, ...) and any
calendar spread, all interleaved chronologically in one CSV. That
means the exact same ``ts_event`` minute legitimately appears many
times, once per instrument still trading in it -- this is expected,
not a data error, and it is exactly why this file can never be handed
directly to ``tradelab.loaders.read_csv_bars`` (which assumes one
series with no duplicate timestamps). This module only parses that raw
shape into a plain, ordered list of records -- one per (contract,
minute) -- and stops there. Turning many parallel per-minute records
into one single contract-at-a-time series is a completely separate
decision (which contract "wins" each session), made by
``tradelab.volume_roll_continuous``, never here.

**The exact expected column layout** (case-insensitive header
matching, mirroring ``tradelab.loaders``' own convention):

.. code-block:: text

    ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume,symbol
    2023-08-18T00:00:00.000000000Z,33,1,34070,14938.000000000,14938.000000000,14930.000000000,14930.000000000,9,MNQZ3

``ts_event`` must be exactly Databento's own UTC nanosecond ISO format
(``YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ``) -- always UTC (the trailing
``Z``), always minute-aligned for this ``ohlcv-1m`` schema (sub-minute
precision beyond that is truncated, never rounded, since it is always
exactly zero for real ``ohlcv-1m`` data and this parser does not try
to guess otherwise). ``rtype`` must literally be ``33`` -- Databento's
own record-type code for the ``ohlcv-1m`` schema -- so that a
differently-schemaed export (e.g. ``ohlcv-1h``) is rejected immediately
with a clear error rather than silently misread as 1-minute bars.

This parser validates only CSV mechanics and per-row data integrity
(required columns, timestamp format, numeric fields, non-negative
volume, a sane OHLC range, a non-empty symbol, and that the file is
non-decreasing in ``ts_event`` overall). It does **not** filter to
quarterly outright contracts, does not know about calendar spreads,
and does not decide which contract is the front month -- all of that
is ``tradelab.volume_roll_continuous``'s job, working from the raw
``DatabentoParentRecord`` list this module returns.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_COLUMNS = {"ts_event", "rtype", "publisher_id", "instrument_id", "open", "high", "low", "close", "volume", "symbol"}

# Databento DBN record-type code for the ohlcv-1m schema -- the only
# schema this parser accepts (dataset GLBX.MDP3, per this task's own spec).
EXPECTED_RTYPE = 33

_TS_EVENT_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{9})Z$")


@dataclass(frozen=True)
class DatabentoParentRecord:
    """One raw (contract, minute) row from a Databento parent-symbology
    CSV -- not yet a ``Bar1m``, and not yet filtered to quarterly
    outright contracts. Many records legitimately share the same
    ``ts_event`` (one per contract still trading in that minute)."""

    ts_event: datetime
    rtype: int
    publisher_id: int
    instrument_id: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

    def __post_init__(self) -> None:
        if self.ts_event.tzinfo is None:
            raise ValueError(f"DatabentoParentRecord.ts_event must be timezone-aware: {self.ts_event!r}")
        object.__setattr__(self, "ts_event", self.ts_event.astimezone(timezone.utc))
        if self.ts_event.second != 0 or self.ts_event.microsecond != 0:
            raise ValueError(f"DatabentoParentRecord.ts_event must fall exactly on a minute boundary: {self.ts_event!r}")
        if self.rtype != EXPECTED_RTYPE:
            raise ValueError(f"DatabentoParentRecord.rtype must be {EXPECTED_RTYPE} (ohlcv-1m), got {self.rtype!r}")
        if not isinstance(self.publisher_id, int) or isinstance(self.publisher_id, bool):
            raise ValueError(f"DatabentoParentRecord.publisher_id must be an int, got {self.publisher_id!r}")
        if not isinstance(self.instrument_id, int) or isinstance(self.instrument_id, bool):
            raise ValueError(f"DatabentoParentRecord.instrument_id must be an int, got {self.instrument_id!r}")
        if self.low > self.high:
            raise ValueError(f"DatabentoParentRecord {self.symbol!r} at {self.ts_event!r}: low {self.low} > high {self.high}")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError(f"DatabentoParentRecord {self.symbol!r} at {self.ts_event!r}: OHLC range impossible")
        if self.volume < 0:
            raise ValueError(f"DatabentoParentRecord {self.symbol!r} at {self.ts_event!r}: negative volume {self.volume!r}")
        if not self.symbol:
            raise ValueError(f"DatabentoParentRecord.symbol must be non-empty (ts_event {self.ts_event!r})")


def _parse_ts_event(value: str) -> datetime:
    match = _TS_EVENT_PATTERN.match(value.strip())
    if not match:
        raise ValueError(
            f"ts_event {value!r} does not match the expected Databento UTC nanosecond format "
            "'YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ'"
        )
    base, nanos = match.groups()
    dt = datetime.fromisoformat(base).replace(tzinfo=timezone.utc)
    microseconds = int(nanos) // 1000  # this schema is always minute-aligned; sub-microsecond precision is never meaningful here
    if microseconds:
        dt = dt.replace(microsecond=microseconds)
    return dt


def read_databento_parent_csv(path: Path | str) -> list[DatabentoParentRecord]:
    """Parse a local Databento parent-symbology ``ohlcv-1m`` CSV export
    into an ordered list of ``DatabentoParentRecord`` -- one per row,
    duplicate ``ts_event`` values across different ``symbol``s
    preserved exactly as they appear (never deduplicated, never
    reordered by contract). Raises ``ValueError`` with the offending
    row number on any malformed row, missing column, or a ``ts_event``
    that goes backwards relative to the previous row.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Databento parent CSV has no header row")
        columns = {name.lower(): name for name in reader.fieldnames}
        missing = REQUIRED_COLUMNS - columns.keys()
        if missing:
            raise ValueError(f"Databento parent CSV missing required columns: {sorted(missing)}")

        records: list[DatabentoParentRecord] = []
        previous_ts: datetime | None = None
        for row_number, row in enumerate(reader, start=2):
            lowered = {key.lower(): value for key, value in row.items() if key}
            try:
                ts_event = _parse_ts_event(lowered["ts_event"])
                record = DatabentoParentRecord(
                    ts_event=ts_event,
                    rtype=int(lowered["rtype"]),
                    publisher_id=int(lowered["publisher_id"]),
                    instrument_id=int(lowered["instrument_id"]),
                    open=float(lowered["open"]),
                    high=float(lowered["high"]),
                    low=float(lowered["low"]),
                    close=float(lowered["close"]),
                    volume=float(lowered["volume"]),
                    symbol=lowered["symbol"],
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid row {row_number}: {error}") from error

            if previous_ts is not None and ts_event < previous_ts:
                raise ValueError(
                    f"Row {row_number}: ts_event {ts_event!r} is before the previous row's ts_event {previous_ts!r} "
                    "-- the file must be non-decreasing in time (ties across different symbols are expected)"
                )
            previous_ts = ts_event
            records.append(record)

    return records
