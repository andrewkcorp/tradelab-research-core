"""High-impact USD macro news events: model, validation, and the red-folder
entry blackout window.

v1 scope and provenance: this module works entirely off a manually
collected, frozen, version-controlled local dataset of official releases
from BLS, BEA, the Federal Reserve, and Census -- gathered and reviewed
separately from this code and shipped as a versioned artifact. There is no
paid API, no scraper, and no network call anywhere in this module or its
tests; ``tradelab.news_loaders`` only ever reads local CSV/Parquet files.

v1 deliberately does not model the actual/forecast values a release
publishes. The red-folder blackout decision (``is_entry_blocked``) must be
computable before the number is even published, so it is driven purely by
``scheduled_at`` -- ``NewsEvent`` has no actual/forecast fields to be
misused for that decision, or at all.

Out of scope for this module, on purpose:
- Data wicks (unusually large 1-minute candles around a release) are a
  separate module, ``tradelab.data_wicks``, that *consumes* ``NewsEvent``
  as an input; nothing here detects or scores price reaction.
- CME holidays / early closes are a market-session concept
  (``tradelab.session_calendar``), not a property of a news event. A
  release can be scheduled on a day the market calendar doesn't recognize;
  reconciling the two is the caller's problem, not this module's.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SCHEMA_VERSION = "1.0"

# v1 whitelist: only these event types may be constructed. Each is a
# distinct release even when several are published in the same report (the
# Employment Situation report publishes NFP, the unemployment rate, and
# average hourly earnings together, but they are tracked as separate
# events here because the spec enumerates them separately).
EVENT_TYPES = frozenset(
    {
        "FOMC_RATE_DECISION",
        "FOMC_PRESS_CONFERENCE",
        "CPI",
        "CORE_CPI",
        "PPI",
        "CORE_PPI",
        "NFP",
        "UNEMPLOYMENT_RATE",
        "AVERAGE_HOURLY_EARNINGS",
        "PCE",
        "CORE_PCE",
        "GDP_ADVANCE",
        "RETAIL_SALES",
        "ISM_MANUFACTURING_PMI",
        "ISM_SERVICES_PMI",
        "JOLTS",
        "DURABLE_GOODS",
        "INITIAL_JOBLESS_CLAIMS",
    }
)

EVENT_TYPE_LABELS: dict[str, str] = {
    "FOMC_RATE_DECISION": "FOMC Rate Decision",
    "FOMC_PRESS_CONFERENCE": "FOMC Press Conference",
    "CPI": "CPI",
    "CORE_CPI": "Core CPI",
    "PPI": "PPI",
    "CORE_PPI": "Core PPI",
    "NFP": "Employment Situation / Non-Farm Payrolls",
    "UNEMPLOYMENT_RATE": "Unemployment Rate",
    "AVERAGE_HOURLY_EARNINGS": "Average Hourly Earnings",
    "PCE": "PCE",
    "CORE_PCE": "Core PCE",
    "GDP_ADVANCE": "GDP Advance",
    "RETAIL_SALES": "Retail Sales",
    "ISM_MANUFACTURING_PMI": "ISM Manufacturing PMI",
    "ISM_SERVICES_PMI": "ISM Services PMI",
    "JOLTS": "JOLTS",
    "DURABLE_GOODS": "Durable Goods",
    "INITIAL_JOBLESS_CLAIMS": "Initial Jobless Claims",
}

REQUIRED_COUNTRY = "United States"
REQUIRED_CURRENCY = "USD"

# Red-folder blackout window for entry T: [T - 15min, T + 5min], inclusive.
RED_FOLDER_BEFORE = timedelta(minutes=15)
RED_FOLDER_AFTER = timedelta(minutes=5)


@dataclass(frozen=True)
class NewsBlackoutIndex:
    """Immutable O(log events) boolean lookup for the entry blackout.

    ``is_entry_blocked`` intentionally returns every overlapping source
    event and therefore remains the authoritative audit API.  Market
    execution only needs the boolean decision, however, and previously
    rebuilt that full list by scanning every event for every candidate
    entry.  For a timestamp ``ts`` the equivalent predicate is simply:

    ``event.scheduled_at in [ts - RED_FOLDER_AFTER, ts + RED_FOLDER_BEFORE]``.

    A binary search over sorted scheduled timestamps answers that predicate
    exactly, including both inclusive boundaries.  It changes no news rule
    and stores no derived trading state.
    """

    scheduled_at: tuple[datetime, ...]

    @classmethod
    def build(cls, events: list["NewsEvent"] | tuple["NewsEvent", ...]) -> "NewsBlackoutIndex":
        return cls(tuple(sorted(event.scheduled_at for event in events)))

    def is_blocked(self, ts_utc: datetime) -> bool:
        if ts_utc.tzinfo is None:
            raise ValueError(f"NewsBlackoutIndex.is_blocked: ts_utc must be timezone-aware: {ts_utc!r}")
        ts_utc = ts_utc.astimezone(timezone.utc)
        earliest = ts_utc - RED_FOLDER_AFTER
        latest = ts_utc + RED_FOLDER_BEFORE
        index = bisect.bisect_left(self.scheduled_at, earliest)
        return index < len(self.scheduled_at) and self.scheduled_at[index] <= latest


@dataclass(frozen=True)
class NewsEvent:
    """A single scheduled high-impact USD macro release.

    ``scheduled_at`` is normalized to UTC and must be exact to the minute
    (second=0, microsecond=0) -- v1 only accepts events whose publication
    time is precisely known (``time_is_exact=True``); anything else,
    including a timestamp with leftover seconds, is rejected outright
    rather than stored with an approximate time. TradeLab fills entries on
    1-minute bars, so the T-15/T+5 blackout boundary must land exactly on a
    minute or it is ambiguous which bar it applies to.

    ``schema_version`` must equal ``SCHEMA_VERSION`` exactly (currently
    ``"1.0"``) -- there is no cross-version compatibility in v1, so any
    other value, including a plausible-looking one like ``"1.1"``, is
    rejected rather than silently accepted.
    """

    event_id: str
    scheduled_at: datetime
    country: str
    currency: str
    event_type: str
    source_agency: str
    source_url: str
    source_reference: str
    time_is_exact: bool
    schema_version: str

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("NewsEvent.event_id must be non-empty")

        if self.scheduled_at.tzinfo is None:
            raise ValueError(f"NewsEvent.scheduled_at must be timezone-aware: {self.scheduled_at!r}")
        object.__setattr__(self, "scheduled_at", self.scheduled_at.astimezone(timezone.utc))
        if self.scheduled_at.second != 0 or self.scheduled_at.microsecond != 0:
            raise ValueError(
                f"NewsEvent.scheduled_at must be exact to the minute (second=0, microsecond=0): "
                f"{self.scheduled_at!r} -- entries fill on 1m bars, so the T-15/T+5 blackout boundary "
                "must be unambiguous"
            )

        if self.country != REQUIRED_COUNTRY:
            raise ValueError(f"NewsEvent.country must be {REQUIRED_COUNTRY!r}, got {self.country!r}")
        if self.currency != REQUIRED_CURRENCY:
            raise ValueError(f"NewsEvent.currency must be {REQUIRED_CURRENCY!r}, got {self.currency!r}")
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"NewsEvent.event_type {self.event_type!r} is not in the v1 whitelist: {sorted(EVENT_TYPES)}")

        if not self.time_is_exact:
            raise ValueError(
                f"NewsEvent {self.event_id!r}: time_is_exact=False is rejected -- v1 requires a precisely "
                "known publication time to drive the red-folder blackout window"
            )

        if not self.source_agency:
            raise ValueError(f"NewsEvent {self.event_id!r}: source_agency must be non-empty (required for audit)")
        if not self.source_url:
            raise ValueError(f"NewsEvent {self.event_id!r}: source_url must be non-empty (required for audit)")
        if not self.source_reference:
            raise ValueError(f"NewsEvent {self.event_id!r}: source_reference must be non-empty (required for audit)")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"NewsEvent {self.event_id!r}: schema_version must be exactly {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )


def validate_news_events(events: list[NewsEvent]) -> None:
    """Batch-validate a set of NewsEvent records already built via NewsEvent().

    Raises ``ValueError`` on:
    - a duplicate ``event_id``;
    - two events with the same ``(event_type, scheduled_at)`` pair (the same
      release reported twice, e.g. from overlapping source files).

    Per-record invariants (country/currency/whitelist/exact-time/etc.) are
    already enforced by ``NewsEvent.__post_init__`` at construction time, so
    this function only checks properties of the *set*.
    """
    seen_ids: dict[str, NewsEvent] = {}
    seen_type_and_time: dict[tuple[str, datetime], NewsEvent] = {}
    for event in events:
        if event.event_id in seen_ids:
            raise ValueError(f"Duplicate event_id {event.event_id!r}")
        seen_ids[event.event_id] = event

        key = (event.event_type, event.scheduled_at)
        if key in seen_type_and_time:
            other = seen_type_and_time[key]
            raise ValueError(
                f"Duplicate event: {event.event_type!r} already scheduled at {event.scheduled_at} "
                f"(event_id {other.event_id!r} and {event.event_id!r})"
            )
        seen_type_and_time[key] = event


def is_entry_blocked(ts_utc: datetime, events: list[NewsEvent]) -> tuple[bool, list[NewsEvent]]:
    """Whether a new entry at ``ts_utc`` falls inside any event's red-folder
    blackout window ``[event.scheduled_at - 15min, event.scheduled_at + 5min]``
    (inclusive on both ends).

    Returns ``(blocked, blocking_events)``. ``blocking_events`` lists every
    event whose window covers ``ts_utc`` -- usually zero or one, but more
    than one when two releases' windows overlap; the caller decides what to
    do with an overlap, this function just reports all of them rather than
    picking one arbitrarily.

    Pure UTC arithmetic throughout: no timezone conversion happens here, so
    a DST transition anywhere cannot change which side of the window
    ``ts_utc`` falls on.
    """
    if ts_utc.tzinfo is None:
        raise ValueError(f"is_entry_blocked: ts_utc must be timezone-aware: {ts_utc!r}")
    ts_utc = ts_utc.astimezone(timezone.utc)

    blocking = [
        event
        for event in events
        if event.scheduled_at - RED_FOLDER_BEFORE <= ts_utc <= event.scheduled_at + RED_FOLDER_AFTER
    ]
    return (bool(blocking), blocking)
