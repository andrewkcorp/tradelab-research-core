"""Input validation for 1-minute bar series.

Hard invariants (timestamp order/uniqueness, OHLC range, non-negative volume)
raise ``ValueError`` immediately: a violation means the series cannot be
trusted for aggregation.

Missing-minutes registry: a caller might mean three different things by "the
market wasn't there", and this module is careful to only report the one that
is actually a data problem:

- **Planned market closure.** The daily CME Globex maintenance break for
  NQ/MNQ, 17:00-18:00 America/New_York (``tradelab.session_calendar``). This
  is expected and is never recorded as a gap.
- **A genuine data gap.** Any missing 1-minute bar during a period the
  regular session calendar expects to be open. Always recorded. If a single
  raw gap in the input straddles the daily break, it is split into separate
  ``MissingMinutesGap`` entries -- one per contiguous run of missing,
  expected-open minutes -- rather than reported as one gap that includes the
  break.
- **Special schedules.** When supplied, a checksum-linked holiday/early-close
  schedule removes official closed intervals from the expected-minute set.
  Without it those exceptions remain visible as gaps rather than being guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from collections.abc import Sequence

from tradelab.bar import Bar1m
from tradelab.session_calendar import SpecialSessionSchedule, expected_open_minutes

ONE_MINUTE = timedelta(minutes=1)


@dataclass(frozen=True)
class MissingMinutesGap:
    """A maximal contiguous run of missing 1-minute bars during a period the
    regular session calendar expects to be open (never spans the daily break)."""

    first_missing: datetime
    last_missing: datetime
    missing_count: int


@dataclass(frozen=True)
class ValidationResult:
    gaps: tuple[MissingMinutesGap, ...]


def validate_bars(
    bars: Sequence[Bar1m], *, special_schedule: SpecialSessionSchedule | None = None
) -> ValidationResult:
    """Validate a 1-minute bar series and build its missing-minutes registry.

    Raises ``ValueError`` on: out-of-order or duplicate timestamps, an
    impossible OHLC range, or negative volume. Returns a ``ValidationResult``
    whose ``gaps`` list every contiguous run of missing 1-minute bars that
    the regular NQ/MNQ session calendar expects to be present -- the daily
    17:00-18:00 NY break is excluded, never counted as missing.
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
            missing_open = expected_open_minutes(
                previous.timestamp + ONE_MINUTE,
                bar.timestamp,
                special_schedule=special_schedule,
            )
            gaps.extend(_group_contiguous_runs(missing_open))
        previous = bar

    return ValidationResult(gaps=tuple(gaps))


def validate_price_grid(bars: Sequence[Bar1m], tick_size: float) -> None:
    """Require every OHLC price to lie on the instrument tick grid.

    This is a hard execution invariant, separate from missing-minute quality
    reporting.  It catches malformed vendor/adjustment output before a
    detector can create an untradeable fractional-tick entry, TP or SL.
    """
    if not math.isfinite(tick_size) or tick_size <= 0:
        raise ValueError(f"tick_size must be finite and positive, got {tick_size!r}")
    for index, bar in enumerate(bars):
        for field in ("open", "high", "low", "close"):
            price = getattr(bar, field)
            ticks = price / tick_size
            if not math.isclose(ticks, round(ticks), rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(
                    f"Row {index}: {field} price {price!r} at {bar.timestamp} is not aligned "
                    f"to tick_size={tick_size!r}"
                )


def _group_contiguous_runs(minutes: list[datetime]) -> list[MissingMinutesGap]:
    """Split a sorted list of missing-minute timestamps into one
    MissingMinutesGap per maximal run of minute-adjacent entries. A jump
    larger than one minute -- which is exactly what happens when the daily
    break has been filtered out of the middle of an otherwise-contiguous
    range -- starts a new run."""
    if not minutes:
        return []
    runs: list[MissingMinutesGap] = []
    run_start = run_end = minutes[0]
    run_length = 1
    for minute in minutes[1:]:
        if minute - run_end == ONE_MINUTE:
            run_end = minute
            run_length += 1
        else:
            runs.append(MissingMinutesGap(run_start, run_end, run_length))
            run_start = run_end = minute
            run_length = 1
    runs.append(MissingMinutesGap(run_start, run_end, run_length))
    return runs


def _validate_ohlc_range(bar: Bar1m, index: int) -> None:
    if bar.low > bar.high:
        raise ValueError(f"Row {index}: low {bar.low} > high {bar.high} at {bar.timestamp}")
    if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
        raise ValueError(f"Row {index}: OHLC range impossible at {bar.timestamp}")
