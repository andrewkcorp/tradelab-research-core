"""Regular trading-session calendar for NQ/MNQ on CME Globex.

This is a TradeLab execution/data-availability rule, not a strategy
definition. Two other modules build on it, for two different purposes:

- ``tradelab.aggregate`` uses it to know which 1-minute slots a bucket must
  contain before it can be aggregated at all.
- ``tradelab.validation`` uses it to know which missing 1-minute slots are
  an actual data gap worth reporting, as opposed to the market simply being
  closed on schedule.

Regular session, every calendar day: opens 18:00 NY (previous calendar day)
and runs to 17:00 NY (current day) -- a 23-hour active window -- followed by
a one-hour scheduled maintenance break from 17:00 to 18:00 NY, after which
the next day's session opens. This module applies that same daily shape
uniformly to every day; it has no per-date exceptions.

Deliberately out of scope (tracked as an explicit limitation, not a silent
assumption): CME holidays, early-close days, and any other one-off special
schedule. Because those aren't modeled, this calendar cannot distinguish a
normal trading day from a holiday or an early close -- it will claim the
regular 23h/1h shape applies on those days too, which is simply wrong. In
practice this stays safe rather than silently wrong on the aggregation side:
real holiday/early-close data has fewer 1-minute bars than this calendar
expects, so ``tradelab.aggregate.aggregate_bars``' completeness check (which
requires an exact match against ``expected_open_minutes``) will reject any
bucket touching such a day instead of fabricating an aggregate from partial
data. A full weekend closure is handled the same way, by the same accident
of conservatism, not because weekends are modeled here. On the validation
side, the same blind spot means a holiday/early-close day's missing minutes
get reported by ``tradelab.validation.validate_bars`` as an ordinary data
gap -- there is no way, from this module alone, to tell "the market was
closed for a reason we don't model" apart from "the data is missing".
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
ONE_MINUTE = timedelta(minutes=1)

# The daily scheduled maintenance break: 17:00-18:00 NY. Encoded as "the NY
# local hour equal to this value is fully within the break" -- i.e. minutes
# 17:00 through 17:59 NY are closed, and 18:00 NY is open again.
DAILY_BREAK_START_HOUR = 17


def is_regular_session_open(minute_start_utc: datetime) -> bool:
    """Whether the regular NQ/MNQ session is expected to be open for the
    1-minute bar starting at ``minute_start_utc``.

    True for every minute of every day except the daily 17:00-18:00 NY
    maintenance break. Has no holiday/early-close awareness -- see the
    module docstring.
    """
    return minute_start_utc.astimezone(NY).hour != DAILY_BREAK_START_HOUR


def expected_open_minutes(start_utc: datetime, end_utc: datetime) -> list[datetime]:
    """All 1-minute slot start-times (UTC) in ``[start_utc, end_utc)`` at
    which the regular session is expected to be open.

    ``start_utc`` and ``end_utc`` must fall exactly on minute boundaries;
    this is only ever called with the UTC-converted bucket boundaries
    computed by ``tradelab.aggregate``, which already guarantees that.
    """
    minutes = []
    current = start_utc
    while current < end_utc:
        if is_regular_session_open(current):
            minutes.append(current)
        current += ONE_MINUTE
    return minutes
