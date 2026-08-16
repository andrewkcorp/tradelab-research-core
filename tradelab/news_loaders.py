"""Local-file loaders for NewsEvent records: CSV and Parquet only.

No network access, no data-vendor client, no API keys, no scrapers. These
loaders only ever read a local file the caller already has on disk -- see
``tradelab.news`` for where that file is expected to come from (a manually
collected, frozen, version-controlled dataset).
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from tradelab.news import NewsEvent, validate_news_events

REQUIRED_COLUMNS = {
    "event_id",
    "scheduled_at",
    "country",
    "currency",
    "event_type",
    "source_agency",
    "source_url",
    "source_reference",
    "time_is_exact",
    "schema_version",
}

_TRUE_VALUES = {"true", "1", "yes"}
_FALSE_VALUES = {"false", "0", "no"}


def read_news_events_csv(path: Path) -> list[NewsEvent]:
    """Read NewsEvent records from a local CSV file.

    Required columns (case-insensitive): event_id, scheduled_at, country,
    currency, event_type, source_agency, source_url, source_reference,
    time_is_exact, schema_version. ``scheduled_at`` must carry an explicit
    UTC offset (e.g. ``2026-01-01T13:30:00Z`` or ``...+00:00``); naive
    timestamps are rejected. Runs ``validate_news_events`` over the whole
    file before returning, so duplicate event_id / duplicate
    (event_type, scheduled_at) also raise a clear error here.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row")
        columns = {name.lower(): name for name in reader.fieldnames}
        missing = REQUIRED_COLUMNS - columns.keys()
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        events = []
        for row_number, row in enumerate(reader, start=2):
            lowered = {key.lower(): value for key, value in row.items() if key}
            try:
                event = NewsEvent(
                    event_id=lowered["event_id"],
                    scheduled_at=_parse_timestamp(lowered["scheduled_at"]),
                    country=lowered["country"],
                    currency=lowered["currency"],
                    event_type=lowered["event_type"],
                    source_agency=lowered["source_agency"],
                    source_url=lowered["source_url"],
                    source_reference=lowered["source_reference"],
                    time_is_exact=_parse_bool(lowered["time_is_exact"]),
                    schema_version=lowered["schema_version"],
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid row {row_number}: {error}") from error
            events.append(event)

    validate_news_events(events)
    return events


def read_news_events_parquet(path: Path) -> list[NewsEvent]:
    """Read NewsEvent records from a local Parquet file.

    Requires the optional ``pyarrow`` dependency, imported lazily so it is
    only needed when Parquet support is actually used (``read_news_events_csv``
    works with the standard library alone). Same column contract and
    validation as ``read_news_events_csv``.
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

    def column(name: str) -> list:
        return table.column(columns[name]).to_pylist()

    scheduled_at_col = table.column(columns["scheduled_at"])
    is_native_timestamp = pa.types.is_timestamp(scheduled_at_col.type)
    scheduled_ats = scheduled_at_col.to_pylist()

    event_ids = column("event_id")
    countries = column("country")
    currencies = column("currency")
    event_types = column("event_type")
    source_agencies = column("source_agency")
    source_urls = column("source_url")
    source_references = column("source_reference")
    time_is_exacts = column("time_is_exact")
    schema_versions = column("schema_version")

    events = []
    for index, raw_timestamp in enumerate(scheduled_ats):
        try:
            timestamp = raw_timestamp if is_native_timestamp else _parse_timestamp(str(raw_timestamp))
            if timestamp.tzinfo is None:
                raise ValueError(f"scheduled_at {raw_timestamp!r} has no timezone offset; UTC offset required")
            time_is_exact = time_is_exacts[index]
            if not isinstance(time_is_exact, bool):
                time_is_exact = _parse_bool(str(time_is_exact))
            event = NewsEvent(
                event_id=event_ids[index],
                scheduled_at=timestamp,
                country=countries[index],
                currency=currencies[index],
                event_type=event_types[index],
                source_agency=source_agencies[index],
                source_url=source_urls[index],
                source_reference=source_references[index],
                time_is_exact=time_is_exact,
                schema_version=schema_versions[index],
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid row {index}: {error}") from error
        events.append(event)

    validate_news_events(events)
    return events


def _parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"scheduled_at {value!r} has no timezone offset; UTC offset required")
    return parsed


def _parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"time_is_exact {value!r} is not a recognized boolean (use true/false)")
