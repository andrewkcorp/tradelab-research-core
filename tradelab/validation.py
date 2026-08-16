"""Input validation for 1-minute bar series.

Hard invariants (timestamp order/uniqueness, OHLC range, non-negative volume)
raise ``ValueError`` immediately: a violation means the series cannot be
trusted for aggregation. Missing minutes are not an error by themselves
(overnight halts, weekends and data-vendor gaps are all legitimate) — they
are recorded in a registry so callers/aggregation can reason about them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from tradelab.bar import Bar1m

ONE_MINUTE = timedelta(minutes=1)


@dataclass(frozen=True)
class MissingMinutesGap:
    """A run of one or more missing 1-minute timestamps between two bars."""

    after: datetime
    before: datetime
    missing_count: int


@dataclass(frozen=True)
class ValidationResult:
    gaps: tuple[MissingMinutesGap, ...]


def validate_bars(bars: list[Bar1m]) -> ValidationResult:
    """Validate a 1-minute bar series and build its missing-minutes registry.

    Raises ``ValueError`` on: out-of-order or duplicate timestamps, an
    impossible OHLC range, or negative volume. Returns a ``ValidationResult``
    whose ``gaps`` list every run of missing 1-minute timestamps between
    consecutive bars.
    """
    if not bars:
        return ValidationResult(gaps=())

    gaps: list[MissingMinutesGap] = []
    previous: Bar1m | None = None
    for index, bar in enumerate(bars):
        _validate_ohlc_range(bar, index)
        if bar.volume < 0:
            raise ValueError(f"Row {index}: negative volume {bar.volume!r} at {bar.timestamp}")

        if previous is not None:
            if bar.timestamp <= previous.timestamp:
                raise ValueError(
                    f"Row {index}: timestamp {bar.timestamp} is not strictly after "
                    f"previous timestamp {previous.timestamp} (unordered or duplicate)"
                )
            delta = bar.timestamp - previous.timestamp
            missing = int(delta / ONE_MINUTE) - 1
            if missing > 0:
                gaps.append(MissingMinutesGap(previous.timestamp, bar.timestamp, missing))
        previous = bar

    return ValidationResult(gaps=tuple(gaps))


def _validate_ohlc_range(bar: Bar1m, index: int) -> None:
    if bar.low > bar.high:
        raise ValueError(f"Row {index}: low {bar.low} > high {bar.high} at {bar.timestamp}")
    if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
        raise ValueError(f"Row {index}: OHLC range impossible at {bar.timestamp}")
