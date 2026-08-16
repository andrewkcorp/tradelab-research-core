"""Aggregation of 1-minute bars into 5m/15m/1H/4H/1D bars.

All bucket boundaries are wall-clock ``America/New_York`` (per
docs/strategy-spec-v0.1.md section 2), computed via ``zoneinfo`` so DST
transitions are handled automatically: a bucket that nominally spans N
wall-clock hours may span N-1 or N+1 real hours across a DST change, but
its boundaries stay pinned to the correct local clock time either way.

The 1D bucket is the CME trading day: it starts at 18:00 NY (the prior
calendar day when the trading day is still "in progress" locally) and
runs 24 wall-clock hours to the next 18:00 NY, per spec section 2.

An aggregated bar is only ever returned once it is fully closed. Every
bucket except the very last one in the input is closed by construction:
seeing a later bar proves wall-clock time has moved past that bucket's
end boundary, even if the bucket has internal missing minutes (a data
coverage issue tracked separately by ``tradelab.validation``, not a
"not yet known" issue). The trailing bucket of the whole series is the
only one that can still be genuinely in progress, so it is dropped
unless the last supplied 1-minute bar exactly reaches its bucket's end
boundary. This keeps the API from ever exposing a partially-formed,
not-yet-known aggregate as if it were historical (spec section 2, point 6).

4H buckets are anchored to NY midnight (00:00, 04:00, ..., 20:00). The
spec does not pin an explicit 4H anchor; this is a documented assumption
tracked as an open item for owner confirmation, not a redefinition of
anything the spec does state.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.bar import Bar1m

NY = ZoneInfo("America/New_York")
ONE_MINUTE = timedelta(minutes=1)
TRADING_DAY_START_HOUR = 18

_INTRADAY_MINUTES = {"5m": 5, "15m": 15, "1H": 60, "4H": 240}
SUPPORTED_TIMEFRAMES = frozenset(_INTRADAY_MINUTES) | {"1D"}


@dataclass(frozen=True)
class AggregatedBar:
    """A closed higher-timeframe bar aggregated from 1-minute bars."""

    timestamp: datetime  # UTC, start of the bucket
    open: float
    high: float
    low: float
    close: float
    volume: float
    contract: str
    timeframe: str
    known_at: datetime  # UTC instant at/after which this bar may be treated as known


def aggregate_bars(bars: list[Bar1m], timeframe: str) -> list[AggregatedBar]:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {sorted(SUPPORTED_TIMEFRAMES)}")
    if not bars:
        return []

    results: list[AggregatedBar] = []
    bucket_start_ny: datetime | None = None
    bucket_end_ny: datetime | None = None
    acc: list[Bar1m] = []
    previous_ts: datetime | None = None

    def finalize(require_full_coverage: bool) -> None:
        if not acc:
            return
        bucket_end_utc = bucket_end_ny.astimezone(timezone.utc)
        if require_full_coverage:
            # Bar1m.timestamp is already a fixed-offset UTC instant, so this
            # comparison is a plain UTC-to-UTC check. Comparing two zoneinfo
            # datetimes here instead would be a trap: CPython's aware-datetime
            # subtraction/comparison takes an identity-based fast path when
            # both operands share the exact same ZoneInfo object, which skips
            # UTC offset reconciliation and can silently misjudge magnitude
            # across a DST transition.
            last_bar_end_utc = acc[-1].timestamp + ONE_MINUTE
            if last_bar_end_utc < bucket_end_utc:
                return  # trailing bucket still in progress: never expose it
        results.append(
            AggregatedBar(
                timestamp=bucket_start_ny.astimezone(timezone.utc),
                open=acc[0].open,
                high=max(b.high for b in acc),
                low=min(b.low for b in acc),
                close=acc[-1].close,
                volume=sum(b.volume for b in acc),
                contract=acc[0].contract,
                timeframe=timeframe,
                known_at=bucket_end_utc,
            )
        )

    for bar in bars:
        if previous_ts is not None and bar.timestamp <= previous_ts:
            raise ValueError(f"Bars must be strictly ordered by timestamp; got {bar.timestamp} after {previous_ts}")
        previous_ts = bar.timestamp

        ts_ny = bar.timestamp.astimezone(NY)
        this_bucket_start = _bucket_start(ts_ny, timeframe)

        if bucket_start_ny is not None and this_bucket_start != bucket_start_ny:
            # A bar in a later bucket proves the previous bucket's window has
            # fully elapsed, regardless of how many minutes it actually contains.
            finalize(require_full_coverage=False)
            acc = []

        if not acc:
            bucket_start_ny = this_bucket_start
            bucket_end_ny = _bucket_end(bucket_start_ny, timeframe)
        elif bar.contract != acc[0].contract:
            raise ValueError(
                f"Contract changed mid-bucket ({acc[0].contract!r} -> {bar.contract!r}) at {bar.timestamp}; "
                "continuous-contract roll handling is out of scope for this loader"
            )
        acc.append(bar)

    finalize(require_full_coverage=True)
    return results


def _bucket_start(ts_ny: datetime, timeframe: str) -> datetime:
    if timeframe == "1D":
        return _trading_day_start(ts_ny)
    minutes = _INTRADAY_MINUTES[timeframe]
    midnight = ts_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes_since_midnight = ts_ny.hour * 60 + ts_ny.minute
    bucket_offset = (minutes_since_midnight // minutes) * minutes
    return midnight + timedelta(minutes=bucket_offset)


def _bucket_end(bucket_start_ny: datetime, timeframe: str) -> datetime:
    if timeframe == "1D":
        return bucket_start_ny + timedelta(hours=24)
    return bucket_start_ny + timedelta(minutes=_INTRADAY_MINUTES[timeframe])


def _trading_day_start(ts_ny: datetime) -> datetime:
    """CME trading day: begins 18:00 NY, ends 18:00 NY the next calendar day."""
    anchor_date = ts_ny.date() if ts_ny.hour >= TRADING_DAY_START_HOUR else ts_ny.date() - timedelta(days=1)
    return datetime(anchor_date.year, anchor_date.month, anchor_date.day, TRADING_DAY_START_HOUR, tzinfo=NY)
