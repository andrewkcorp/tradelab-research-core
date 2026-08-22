"""CME Globex equity-index session availability.

The regular weekly schedule is represented directly: Sunday 18:00 NY to
Friday 17:00 NY, with the daily 17:00-18:00 maintenance break. Holiday and
early-close exceptions are deliberately data, not code. A versioned
``SpecialSessionSchedule`` closes explicit UTC intervals whose provenance
and checksum can be linked from a market-data manifest.

Using UTC for exceptions avoids DST ambiguity. An exception file never opens
a minute that the weekly schedule closes; it can only remove minutes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
ONE_MINUTE = timedelta(minutes=1)

# The daily scheduled maintenance break: 17:00-18:00 NY. Encoded as "the NY
# local hour equal to this value is fully within the break" -- i.e. minutes
# 17:00 through 17:59 NY are closed, and 18:00 NY is open again.
DAILY_BREAK_START_HOUR = 17
SPECIAL_SCHEDULE_SCHEMA_VERSION = "cme-special-session-schedule-v1"


@dataclass(frozen=True)
class ClosedInterval:
    start_utc: datetime
    end_utc: datetime
    reason: str

    def __post_init__(self) -> None:
        if self.start_utc.tzinfo is None or self.end_utc.tzinfo is None:
            raise ValueError("special-session interval timestamps must be timezone-aware")
        if self.start_utc.utcoffset() != timedelta(0) or self.end_utc.utcoffset() != timedelta(0):
            raise ValueError("special-session interval timestamps must be UTC")
        if not self.start_utc < self.end_utc:
            raise ValueError("special-session interval start_utc must precede end_utc")
        if not self.reason.strip():
            raise ValueError("special-session interval reason must be non-empty")


@dataclass(frozen=True)
class SpecialSessionSchedule:
    source_url: str
    retrieved_at: datetime
    closed_intervals: tuple[ClosedInterval, ...]
    schema_version: str = SPECIAL_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPECIAL_SCHEDULE_SCHEMA_VERSION:
            raise ValueError(f"unsupported special-session schedule schema {self.schema_version!r}")
        if not self.source_url.strip():
            raise ValueError("special-session schedule source_url must be non-empty")
        if self.retrieved_at.tzinfo is None:
            raise ValueError("special-session schedule retrieved_at must be timezone-aware")
        ordered = tuple(sorted(self.closed_intervals, key=lambda item: item.start_utc))
        if ordered != self.closed_intervals:
            raise ValueError("special-session intervals must be sorted by start_utc")
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_utc < previous.end_utc:
                raise ValueError("special-session closed intervals must not overlap")

    def is_closed(self, minute_start_utc: datetime) -> bool:
        return any(item.start_utc <= minute_start_utc < item.end_utc for item in self.closed_intervals)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp must be explicit UTC, got {value!r}")
    return parsed.astimezone(timezone.utc)


def load_special_session_schedule(path: Path | str) -> SpecialSessionSchedule:
    """Load a provenance-bearing holiday/early-close schedule JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("closed_intervals"), list):
        raise ValueError("special-session schedule must be an object with closed_intervals array")
    try:
        intervals = tuple(
            ClosedInterval(
                start_utc=_parse_utc(item["start_utc"]),
                end_utc=_parse_utc(item["end_utc"]),
                reason=item["reason"],
            )
            for item in raw["closed_intervals"]
        )
        return SpecialSessionSchedule(
            schema_version=raw["schema_version"],
            source_url=raw["source_url"],
            retrieved_at=datetime.fromisoformat(raw["retrieved_at"].replace("Z", "+00:00")),
            closed_intervals=intervals,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid special-session schedule {path}: {error}") from error


def is_regular_session_open(minute_start_utc: datetime) -> bool:
    """Whether the regular NQ/MNQ session is expected to be open for the
    1-minute bar starting at ``minute_start_utc``.

    Includes the regular weekend closure but not one-off holiday exceptions.
    """
    local = minute_start_utc.astimezone(NY)
    weekday = local.weekday()  # Monday=0, Sunday=6
    if weekday == 5:
        return False
    if weekday == 6:
        return local.hour >= 18
    if weekday == 4:
        return local.hour < 17
    return local.hour != DAILY_BREAK_START_HOUR


def expected_open_minutes(
    start_utc: datetime,
    end_utc: datetime,
    *,
    special_schedule: SpecialSessionSchedule | None = None,
) -> list[datetime]:
    """All 1-minute slot start-times (UTC) in ``[start_utc, end_utc)`` at
    which the regular session is expected to be open.

    ``start_utc`` and ``end_utc`` must fall exactly on minute boundaries;
    this is only ever called with the UTC-converted bucket boundaries
    computed by ``tradelab.aggregate``, which already guarantees that.
    """
    minutes = []
    current = start_utc
    while current < end_utc:
        if is_regular_session_open(current) and not (
            special_schedule is not None and special_schedule.is_closed(current)
        ):
            minutes.append(current)
        current += ONE_MINUTE
    return minutes
