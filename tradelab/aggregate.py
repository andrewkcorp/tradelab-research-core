"""Aggregation of 1-minute bars into 5m/15m/1H/4H/1D bars.

All bucket boundaries are wall-clock ``America/New_York`` (per
docs/strategy-spec-v0.1.md section 2), computed via ``zoneinfo`` so DST
transitions are handled automatically: a bucket that nominally spans N
wall-clock hours may span N-1 or N+1 real hours across a DST change, but
its boundaries stay pinned to the correct local clock time either way.

The 1D bucket is the CME trading day: it starts at 18:00 NY (the prior
calendar day when the trading day is still "in progress" locally) and
runs 24 wall-clock hours to the next 18:00 NY, per spec section 2.

4H buckets are anchored to the same 18:00 NY trading-day open, sliced
into six fixed segments: 18:00, 22:00, 02:00, 06:00, 10:00, 14:00 NY.
This is a fixed TradeLab business rule (not derived from the spec
document), chosen to line up with the CME trading day rather than NY
midnight.

Completeness policy: an aggregated bar is only ever returned when its
bucket's 1-minute bars exactly match ``tradelab.session_calendar``'s set of
expected-open minutes for that bucket's window -- no exceptions for the
first bucket of the input, for a gap in the middle of a bucket, or for the
daily 17:00-18:00 NY maintenance break (which is expected to have *no*
bars, not treated as a gap). Any mismatch -- a missing open-session minute,
an extra bar landing inside the break, or data that otherwise doesn't match
the regular schedule -- drops the whole bucket rather than aggregating it
from partial or non-standard data: rolling up incomplete OHLC would
fabricate highs/lows/closes that were never actually observed, which can
manufacture spurious FVGs, liquidity sweeps and bias in any downstream
research that reads this data.

Because ``tradelab.session_calendar`` has no notion of CME holidays or
early closes (see its module docstring), a bucket touching such a day will
also be rejected here -- there is no separate "unsupported schedule" marker,
it just never appears in the output, same as a bucket with a real data gap.
Callers who need to know *why* a period has no aggregate should cross
reference ``tradelab.validation.validate_bars``' missing-minutes registry
against the input.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradelab.bar import Bar1m
from tradelab.session_calendar import expected_open_minutes

NY = ZoneInfo("America/New_York")
TRADING_DAY_START_HOUR = 18
FOUR_HOUR_SEGMENT_HOURS = 4

_INTRADAY_MINUTES = {"5m": 5, "15m": 15, "1H": 60, "4H": 240}
SUPPORTED_TIMEFRAMES = frozenset(_INTRADAY_MINUTES) | {"1D"}


@dataclass(frozen=True)
class AggregatedBar:
    """A closed, fully-covered higher-timeframe bar aggregated from 1-minute bars."""

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

    def finalize() -> None:
        if not acc:
            return
        # Bar1m.timestamp is always a fixed-offset UTC instant (see tradelab.bar),
        # so these are plain UTC-to-UTC computations. Deliberately not done on the
        # NY-zoneinfo bucket_start_ny/bucket_end_ny values: CPython's aware-datetime
        # subtraction takes an identity-based fast path when both operands share the
        # exact same ZoneInfo object, which skips UTC offset reconciliation and can
        # silently misjudge magnitude across a DST transition.
        bucket_start_utc = bucket_start_ny.astimezone(timezone.utc)
        bucket_end_utc = bucket_end_ny.astimezone(timezone.utc)
        expected = expected_open_minutes(bucket_start_utc, bucket_end_utc)
        actual = [b.timestamp for b in acc]
        if actual != expected:
            return  # missing an expected-open minute, or data outside the regular schedule
        results.append(
            AggregatedBar(
                timestamp=bucket_start_utc,
                open=acc[0].open,
                high=max(b.high for b in acc),
                low=min(b.low for b in acc),
                close=acc[-1].close,
                volume=sum(b.volume for b in acc),
                contract=acc[0].contract,
                timeframe=timeframe,
                # The aggregate is knowable when its final expected *open*
                # minute closes.  For 1D and the 14:00-18:00 4H bucket the
                # nominal wall-clock tail 17:00-18:00 is a scheduled CME
                # closure, so waiting until 18:00 would delay HTF state by an
                # artificial hour even though no further member bar can occur.
                known_at=acc[-1].timestamp + timedelta(minutes=1),
            )
        )

    for bar in bars:
        if previous_ts is not None and bar.timestamp <= previous_ts:
            raise ValueError(f"Bars must be strictly ordered by timestamp; got {bar.timestamp} after {previous_ts}")
        previous_ts = bar.timestamp

        ts_ny = bar.timestamp.astimezone(NY)
        this_bucket_start = _bucket_start(ts_ny, timeframe)

        if bucket_start_ny is not None and this_bucket_start != bucket_start_ny:
            finalize()
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

    finalize()
    return results


def _bucket_start(ts_ny: datetime, timeframe: str) -> datetime:
    if timeframe == "1D":
        return _trading_day_start(ts_ny)
    if timeframe == "4H":
        return _four_hour_bucket_start(ts_ny)
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


def _four_hour_bucket_start(ts_ny: datetime) -> datetime:
    """4H segments anchored to the trading day open: 18/22/02/06/10/14 NY.

    Fixed TradeLab business rule. Computed as an offset from the same
    18:00 trading-day anchor used for 1D, so a 4H bucket never straddles
    the trading-day boundary.
    """
    trading_day_start_ny = _trading_day_start(ts_ny)
    hours_since_open = (ts_ny.hour - TRADING_DAY_START_HOUR) % 24
    segment_index = hours_since_open // FOUR_HOUR_SEGMENT_HOURS
    return trading_day_start_ny + timedelta(hours=segment_index * FOUR_HOUR_SEGMENT_HOURS)
