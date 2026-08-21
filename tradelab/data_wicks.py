"""Data wicks: unusually large 1-minute candles around a high-impact USD
news release (spec section 4.3).

A neutral, auditable feature module. It knows nothing about entries,
bias, TP/SL, the backtester, or the strategy enumerator -- those are
separate, later steps. It consumes ``tradelab.news.NewsEvent`` as an
input (spec 4.1/4.2) but does not modify it and is not the red-folder
entry blackout (``tradelab.news.is_entry_blocked``) -- a data wick is a
price-reaction fact about a specific candle, not an entry rule.

**One fixed definition, no parameter grid.** Every threshold below is a
constant of the strategy definition itself, not something a future
parameter sweep varies -- there is deliberately no "try 1.5x/2x/3x range"
or "try 40%/50%/60% wick share" grid here, unlike the swing/equal-highs
``pivot_strength``/``tick_size`` parameters elsewhere in this package.
``tick_size`` remains a call parameter (default ``DEFAULT_TICK_SIZE``)
only because it is instrument-specific, never because the 10-tick
threshold itself is negotiable.

Candidates (spec 4.3): only 1-minute candles whose *open* time falls in
``[T, T + 5 minutes]`` inclusive, where ``T`` is a high-impact USD
``NewsEvent.scheduled_at`` -- six candles per event. A candle inside more
than one event's window (overlapping red-folder windows) is still only
ever evaluated once; if it qualifies, the single resulting event links
every matching ``event_id`` rather than being duplicated per event.

For each candidate:
- ``range = candle.high - candle.low``.
- ``upper_wick = candle.high - max(candle.open, candle.close)``.
- ``lower_wick = min(candle.open, candle.close) - candle.low``.
- ``baseline_median`` = median ``range`` of the ``BASELINE_LENGTH`` (60)
  1-minute candles immediately preceding the candidate -- never the
  candidate itself, never any later candle. If the candidate does not
  have a full, gapless run of exactly 60 preceding 1-minute candles
  (fewer than 60 available, or any timestamp gap within that run or
  between it and the candidate), the candidate is skipped entirely: no
  data wick, no fallback, no partial baseline.
- An **upper** data wick (bearish confluence at the candle's high) forms
  when ``range >= RANGE_MULTIPLIER * baseline_median`` AND
  ``upper_wick >= WICK_SHARE_OF_RANGE * range`` AND
  ``upper_wick >= MIN_WICK_TICKS * tick_size``.
- A **lower** data wick (bullish confluence at the candle's low) forms
  symmetrically from ``lower_wick``. The two directions are evaluated
  independently -- a candle can produce one, both, or neither.

A data wick is known only at the close of the qualifying candle
(``known_at``); entries are separately blocked through ``T + 5 minutes``
regardless by ``tradelab.news.is_entry_blocked`` (spec 4.2) -- this
module does not duplicate or depend on that blackout logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from statistics import median

from tradelab.fvg import DEFAULT_TICK_SIZE, Candle, from_aggregated_bar, from_bar1m  # noqa: F401  (re-exported)
from tradelab.liquidity import _validate_chronology
from tradelab.news import NewsEvent, validate_news_events

# Fixed strategy definition (spec 4.3) -- never swept, never parameters.
BASELINE_LENGTH = 60
CANDIDATE_WINDOW_AFTER = timedelta(minutes=5)  # candidate window is [T, T + 5min] inclusive
RANGE_MULTIPLIER = 2.0
WICK_SHARE_OF_RANGE = 0.5
MIN_WICK_TICKS = 10

_ONE_MINUTE = timedelta(minutes=1)


class DataWickDirection(str, Enum):
    UPPER = "upper"  # bearish confluence at candle.high
    LOWER = "lower"  # bullish confluence at candle.low


@dataclass(frozen=True)
class DataWickEvent:
    """A single qualifying data wick's full audit record.

    ``event_ids`` holds every ``NewsEvent.event_id`` whose ``[T, T+5min]``
    window this candle falls in, sorted by ``(scheduled_at, event_id)`` --
    more than one when red-folder windows overlap; the wick itself is
    still reported exactly once. ``timestamp`` is the qualifying candle's
    own open time (identifies *which* candle); ``known_at`` is its close
    (when the wick actually became knowable) -- the two are deliberately
    kept separate rather than collapsed into one, unlike take/reclaim
    events elsewhere in this package, because both are independently
    useful for audit here.
    """

    event_ids: tuple[str, ...]
    direction: DataWickDirection
    timestamp: datetime
    known_at: datetime
    level: float  # candle.high for UPPER, candle.low for LOWER
    candle_range: float
    baseline_median: float
    wick_length: float
    # First later candle that reaches/passes the wick extreme. ``None`` means
    # it remained uncollected in the supplied history.
    taken_at: datetime | None = None


@dataclass(frozen=True)
class DataWickDetectionResult:
    events: list[DataWickEvent]


def _candidate_map(events: list[NewsEvent]) -> dict[datetime, list[NewsEvent]]:
    """Every 1-minute candle timestamp that is a data-wick candidate for
    at least one event, mapped to the (deterministically ordered) list of
    events whose window it falls in."""
    ordered_events = sorted(events, key=lambda e: (e.scheduled_at, e.event_id))
    by_minute: dict[datetime, list[NewsEvent]] = {}
    for event in ordered_events:
        minute = event.scheduled_at
        end = event.scheduled_at + CANDIDATE_WINDOW_AFTER
        while minute <= end:
            by_minute.setdefault(minute, []).append(event)
            minute += _ONE_MINUTE
    return by_minute


def _is_contiguous_1m_chain(candles: list[Candle]) -> bool:
    return all(candles[i].timestamp - candles[i - 1].timestamp == _ONE_MINUTE for i in range(1, len(candles)))


def detect_data_wicks(
    candles: list[Candle],
    events: list[NewsEvent],
    *,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> DataWickDetectionResult:
    """Detect data wick events on a chronological series of 1-minute
    candles against a set of high-impact USD news events.

    ``candles`` must be native 1-minute candles (e.g. via ``from_bar1m``):
    the baseline/candidate windows are defined in exact 1-minute steps,
    so feeding coarser candles will simply never find a qualifying,
    gapless 60-candle run and silently produce no events, never wrong
    ones.
    """
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size!r}")
    validate_news_events(events)
    _validate_chronology(candles)

    by_minute = _candidate_map(events)
    wick_events: list[DataWickEvent] = []

    for index, candle in enumerate(candles):
        matched_events = by_minute.get(candle.timestamp)
        if not matched_events:
            continue
        if index < BASELINE_LENGTH:
            continue
        window = candles[index - BASELINE_LENGTH : index + 1]
        if not _is_contiguous_1m_chain(window):
            continue

        baseline_candles = window[:-1]
        baseline_median = median(c.high - c.low for c in baseline_candles)
        candle_range = candle.high - candle.low
        if candle_range < RANGE_MULTIPLIER * baseline_median:
            continue

        event_ids = tuple(e.event_id for e in matched_events)
        min_wick = MIN_WICK_TICKS * tick_size
        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low

        if upper_wick >= WICK_SHARE_OF_RANGE * candle_range and upper_wick >= min_wick:
            wick_events.append(
                DataWickEvent(
                    event_ids=event_ids,
                    direction=DataWickDirection.UPPER,
                    timestamp=candle.timestamp,
                    known_at=candle.known_at,
                    level=candle.high,
                    candle_range=candle_range,
                    baseline_median=baseline_median,
                    wick_length=upper_wick,
                )
            )
        if lower_wick >= WICK_SHARE_OF_RANGE * candle_range and lower_wick >= min_wick:
            wick_events.append(
                DataWickEvent(
                    event_ids=event_ids,
                    direction=DataWickDirection.LOWER,
                    timestamp=candle.timestamp,
                    known_at=candle.known_at,
                    level=candle.low,
                    candle_range=candle_range,
                    baseline_median=baseline_median,
                    wick_length=lower_wick,
                )
            )

    # A data wick is a target only until price later revisits its extreme.
    # The forming candle cannot consume itself: it becomes knowable only at
    # ``known_at`` (its close), so scanning starts strictly later.
    resolved: list[DataWickEvent] = []
    for event in wick_events:
        taken_at = None
        for candle in candles:
            if candle.timestamp < event.known_at:
                continue
            if event.direction == DataWickDirection.UPPER and candle.high >= event.level:
                taken_at = candle.timestamp
                break
            if event.direction == DataWickDirection.LOWER and candle.low <= event.level:
                taken_at = candle.timestamp
                break
        resolved.append(DataWickEvent(
            event.event_ids, event.direction, event.timestamp, event.known_at,
            event.level, event.candle_range, event.baseline_median, event.wick_length, taken_at,
        ))
    return DataWickDetectionResult(events=resolved)
