"""Trading-session labels for post-hoc analysis of already-executed trades.

**Post-hoc, never a filter.** Nothing here participates in a backtest. No
detector consults it, no entry rule reads it, and no strategy variant is
enumerated from it. It exists so an *already finished* run's trades can be
sliced by time of day after the fact -- which is a hypothesis-generating view
of one existing result, not a new strategy. A session that looks best in
discovery has to be confirmed on a separate, unused holdout before it may
become a production filter; that is a research-process rule, not something
this module can enforce, so it is stated here where the labels are defined.

**The frame is New York local time**, with DST handled by ``zoneinfo``,
because that is the frame the CME session itself is defined in (see
``tradelab.session_calendar``) and the frame the hard session-close execution
policy uses (``tradelab.session_execution``). Labelling in UTC would make a
"session" drift by an hour twice a year relative to the market it describes.

**Boundaries are a convention, and a changeable one.** They are ordinary
market-hours conventions, not facts derived from the data:

    asia          18:00 -> 03:00   (NY evening through the Asian day)
    london        03:00 -> 09:30
    new_york_am   09:30 -> 12:00   (the RTH open through midday)
    new_york_pm   12:00 -> 16:00   (midday to the entry blackout)
    close         16:00 -> 17:00   (the hard-close blackout hour)

Because the label is computed from a stored UTC timestamp rather than baked
into the trade at execution time, changing these boundaries re-labels every
historical trade immediately and needs no rerun. That is deliberate: the
boundaries are the part most likely to be argued about.

The ``close`` bucket should hold no trade *entries* at all under the hard
session-close policy, which blocks new positions for that whole hour. Seeing
entries there means the policy did not run, which is exactly why it is a
visible bucket rather than being folded into ``new_york_pm``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SessionWindow:
    """One named window in New York local time.

    ``start`` may be later than ``end``, which means the window wraps past
    local midnight (``asia`` does). Membership is half-open: ``start`` is in
    the window, ``end`` is not, so the windows tile the day with no bar
    belonging to two of them and none belonging to none.
    """

    key: str
    label: str
    start: time
    end: time

    def contains(self, local_time: time) -> bool:
        if self.start <= self.end:
            return self.start <= local_time < self.end
        return local_time >= self.start or local_time < self.end


SESSION_WINDOWS: tuple[SessionWindow, ...] = (
    SessionWindow("asia", "Asia", time(18, 0), time(3, 0)),
    SessionWindow("london", "London", time(3, 0), time(9, 30)),
    SessionWindow("new_york_am", "New York AM", time(9, 30), time(12, 0)),
    SessionWindow("new_york_pm", "New York PM", time(12, 0), time(16, 0)),
    SessionWindow("close", "Close (blackout)", time(16, 0), time(17, 0)),
)

SESSION_KEYS: tuple[str, ...] = tuple(window.key for window in SESSION_WINDOWS)

#: Minutes 17:00-17:59 NY are the CME maintenance break: no session owns them.
#: A trade timestamped there means the bar series contains a minute the
#: calendar says cannot exist, so it is labelled rather than silently bucketed.
MAINTENANCE_BREAK_KEY = "maintenance_break"


def session_for(timestamp_utc: datetime) -> str:
    """The session key owning ``timestamp_utc``.

    Returns ``MAINTENANCE_BREAK_KEY`` for the 17:00-17:59 NY hour, which no
    window covers.
    """
    if timestamp_utc.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    local = timestamp_utc.astimezone(NY).timetz().replace(tzinfo=None)
    for window in SESSION_WINDOWS:
        if window.contains(local):
            return window.key
    return MAINTENANCE_BREAK_KEY


def ny_hour(timestamp_utc: datetime) -> int:
    """The New York local hour, 0-23 -- the unit the dashboard's hour-range
    filter works in, kept here so exactly one module owns the conversion."""
    if timestamp_utc.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return timestamp_utc.astimezone(NY).hour


def session_catalog() -> list[dict[str, str]]:
    """The windows as plain dicts, for the dashboard's own filter UI."""
    return [
        {
            "key": window.key,
            "label": window.label,
            "start_ny": window.start.strftime("%H:%M"),
            "end_ny": window.end.strftime("%H:%M"),
        }
        for window in SESSION_WINDOWS
    ]
