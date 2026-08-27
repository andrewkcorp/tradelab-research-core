"""Hard CME session-close execution policy for MNQ.

**An execution rule, not a generator axis.** Nothing here is a strategy
parameter, nothing here is enumerated, and nothing here can be switched off
per variant: every MNQ strategy this package simulates is subject to it. It
answers exactly two questions about one 1-minute bar:

1. may a *new* position be opened on it, and
2. must an *already open* position be closed on it.

**The rule** (spec, task 048):

- CME Globex runs a daily maintenance break at ``17:00-18:00`` America/
  New_York. ``tradelab.session_calendar`` already owns that fact, together
  with the weekend closure and the checksummed special-session schedule of
  holidays and early closes; this module reuses it rather than restating it.
- From ``16:00`` NY inclusive, **no new position may be opened** for the
  remainder of that session.
- Any position still open at the ``16:59`` NY bar is **force-closed at that
  bar's close** -- it does not wait for TP, for SL, or for the market to
  reopen at 18:00. The exit reason is ``TradeExitReason.SESSION_CLOSE_EXIT``.
- TP and SL before that instant are unaffected, including on the 16:59 bar
  itself: an intrabar TP/SL touch there is a genuine fill and wins over the
  forced close, which is only what happens to a position that survives the
  whole bar.

**Why the rule is expressed relative to each session's own close, not to
the literal wall-clock 16:00.** The spec names 16:00/16:59 because that is
what a regular session's 17:00 close implies. On a CME early-close day the
session ends at 13:00 NY, and a literal 16:00 rule would do nothing at all
there -- no blackout, no forced exit -- leaving a position to ride an
overnight or multi-day closure, which is the precise outcome the rule
exists to prevent. This module therefore computes each session's own close
from the same calendar the resampling and data-quality checks use, and
applies:

    entry blackout   = [session_close - 60 min, session_close)
    forced exit bar  = the bar whose minute ends at session_close

On a regular session those are exactly 16:00-16:59 NY and the 16:59 NY bar,
which is the literal reading of the spec. On a 13:00 NY early close they are
12:00-12:59 and the 12:59 bar. The generalization is stated here rather than
buried, because it is a decision, not a derivation.

**DST is never computed, only converted.** Every boundary is built as a
17:00 America/New_York wall-clock instant on a specific NY calendar date and
then converted to UTC, so the UTC offset moves by itself across a DST
transition. No arithmetic ever adds "24 hours" to a UTC instant to find the
next session close -- that is exactly the bug a DST weekend produces.

**Old results are not comparable.** Any run produced before this policy
existed executed a different rule set. Its trades are not a subset of what
this policy produces and must never be merged with, resumed into, or
compared against a run that applies it; ``tradelab.backtest_v2_journal``'s
manifest carries the policy id so a mismatched resume is refused loudly.
"""
from __future__ import annotations

import array
import bisect
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from tradelab.session_calendar import DAILY_BREAK_START_HOUR, NY, SpecialSessionSchedule

ONE_MINUTE = timedelta(minutes=1)

#: How long before a session's own close new entries stop being allowed.
#: 60 minutes, i.e. 16:00 NY inclusive on a regular 17:00 NY close.
ENTRY_BLACKOUT_BEFORE_CLOSE = timedelta(minutes=60)

#: Identifier written into the journal manifest. Any change to the rule --
#: including the blackout length -- must change this string, so a resume
#: against a journal written under a different policy fails loudly instead
#: of silently mixing two execution rule sets.
SESSION_CLOSE_POLICY_ID = "cme-hard-session-close-v1"


def _ny_break_start(day: date) -> datetime:
    """``17:00`` America/New_York on ``day``, as UTC.

    Built as a wall-clock instant in the NY zone, never as UTC arithmetic:
    the offset is whatever ``zoneinfo`` says it is on that date, so a spring-
    forward or fall-back weekend needs no special case. 17:00 is never inside
    the ambiguous or non-existent local hour a DST transition creates, so the
    construction is unambiguous.
    """
    return datetime(day.year, day.month, day.day, DAILY_BREAK_START_HOUR, 0, tzinfo=NY).astimezone(timezone.utc)


def natural_session_close_utc(minute_start_utc: datetime) -> datetime:
    """The next regular ``17:00`` NY break start strictly after this minute.

    Ignores holidays and early closes -- see :func:`session_close_utc`, which
    layers the checksummed special-session schedule on top of this.
    """
    if minute_start_utc.tzinfo is None:
        raise ValueError("minute_start_utc must be timezone-aware")
    local = minute_start_utc.astimezone(NY)
    if local.hour < DAILY_BREAK_START_HOUR:
        # Still before today's break: this session ends at today's 17:00.
        return _ny_break_start(local.date())
    # 17:00-17:59 is the break itself; 18:00 onward belongs to the session
    # that ends at tomorrow's 17:00. Both resolve to the same next boundary.
    if local.hour == DAILY_BREAK_START_HOUR:
        return _ny_break_start(local.date())
    return _ny_break_start(local.date() + timedelta(days=1))


@dataclass(frozen=True)
class SessionClosePolicy:
    """The hard session-close rule, bound to one special-session schedule.

    ``schedule`` is the same checksummed ``SpecialSessionSchedule`` the
    market-data manifest pins and the 1m validation and 5m/15m/1H/4H/1D
    aggregation already use. ``None`` means "regular weekly schedule only" --
    correct for synthetic fixtures, never for production, where the manifest
    always supplies one.
    """

    schedule: SpecialSessionSchedule | None = None

    def __post_init__(self) -> None:
        starts = tuple(item.start_utc for item in self.schedule.closed_intervals) if self.schedule is not None else ()
        object.__setattr__(self, "_interval_starts", starts)

    def session_close_utc(self, minute_start_utc: datetime) -> datetime:
        """The first instant at which the session containing this minute is
        closed -- whichever comes first, the regular 17:00 NY break or a
        holiday/early close from the schedule.

        A minute that is *itself* inside a scheduled closure returns that
        closure's own start, which is at or before the minute. Callers read
        that as "this minute is already past its session's close", which is
        the conservative answer: entries are blocked and an open position is
        force-closed rather than being carried across a closure the calendar
        says exists. A bar should not be there at all in that case, so this
        is a data-quality backstop, not a normal path.
        """
        natural = natural_session_close_utc(minute_start_utc)
        if self.schedule is None:
            return natural
        starts: tuple[datetime, ...] = self._interval_starts  # type: ignore[attr-defined]
        # The interval that could contain this minute is the last one starting
        # at or before it.
        idx = bisect.bisect_right(starts, minute_start_utc) - 1
        if idx >= 0:
            containing = self.schedule.closed_intervals[idx]
            if containing.start_utc <= minute_start_utc < containing.end_utc:
                return containing.start_utc
        # Otherwise: the earliest closure that begins before the regular break
        # cuts this session short.
        nxt = bisect.bisect_right(starts, minute_start_utc)
        if nxt < len(starts) and starts[nxt] < natural:
            return starts[nxt]
        return natural

    def blocks_new_entry(self, minute_start_utc: datetime) -> bool:
        """Whether a *new* position is forbidden on the bar starting here.

        Only new entries. A position opened before the blackout stays open and
        keeps its normal TP/SL until the forced-exit bar.
        """
        close = self.session_close_utc(minute_start_utc)
        return minute_start_utc >= close - ENTRY_BLACKOUT_BEFORE_CLOSE

    def is_forced_exit_bar(self, minute_start_utc: datetime) -> bool:
        """Whether an open position must be closed at this bar's close.

        True for the last minute of the session -- the bar whose own minute
        ends exactly at the session close -- and for any bar at or after that
        close, which only happens when the feed carries a bar the calendar
        says should not exist.
        """
        close = self.session_close_utc(minute_start_utc)
        return minute_start_utc + ONE_MINUTE >= close


class SessionCloseIndex:
    """Per-bar answers to the two policy questions, precomputed once.

    Built once per ``PreparedBacktestContext`` and shared by every strategy
    variant executed against it. The per-bar work is O(1) amortized: bars are
    chronological, so the session close only has to be recomputed when the
    previous one is passed -- once per session, not once per bar.

    ``next_forced_exit_index`` is the reason this is an index rather than a
    predicate: the accelerated execution loop resolves a position's exit with
    an O(log n) range query over ``[lo, hi)`` and needs to know, in O(1),
    whether a forced close cuts that range short.

    **Stored in ``array`` buffers, not tuples.** A production shard carries
    on the order of a million 1m bars, and three Python tuples over that would
    cost ~24 MB of pointers per worker for what is two bits and a small
    integer per bar. ``array('B')``/``array('i')`` make it ~6 bytes per bar
    instead, which matters directly to the worker-count memory gate this same
    task benchmarks.
    """

    __slots__ = ("_forced", "_blocked", "_next_forced", "_size")

    def __init__(self, forced: "array.array", blocked: "array.array") -> None:
        size = len(forced)
        next_forced = array.array("i", bytes(4 * (size + 1)))
        next_forced[size] = size
        for i in range(size - 1, -1, -1):
            next_forced[i] = i if forced[i] else next_forced[i + 1]
        self._forced = forced
        self._blocked = blocked
        self._next_forced = next_forced
        self._size = size

    @classmethod
    def build(cls, bars: Sequence, policy: SessionClosePolicy) -> "SessionCloseIndex":
        forced = array.array("B")
        blocked = array.array("B")
        close: datetime | None = None
        blackout_from: datetime | None = None
        for bar in bars:
            ts = bar.timestamp
            if close is None or ts >= close:
                close = policy.session_close_utc(ts)
                blackout_from = close - ENTRY_BLACKOUT_BEFORE_CLOSE
                if ts >= close:
                    # The minute is at or past its own session's close: a bar
                    # the calendar says should not exist. Treat it as fully
                    # blacked out and force-close on it, and do not cache a
                    # close it has already passed.
                    forced.append(1)
                    blocked.append(1)
                    close = None
                    continue
            forced.append(1 if ts + ONE_MINUTE >= close else 0)
            blocked.append(1 if ts >= blackout_from else 0)
        return cls(forced, blocked)

    def __len__(self) -> int:
        return self._size

    def blocks_new_entry(self, bar_index: int) -> bool:
        return bool(self._blocked[bar_index])

    def is_forced_exit_bar(self, bar_index: int) -> bool:
        return bool(self._forced[bar_index])

    def next_forced_exit_index(self, bar_index: int) -> int | None:
        """The first forced-exit bar at or after ``bar_index``, or ``None``.

        ``bar_index`` may equal the bar count, which answers ``None`` rather
        than raising -- callers pass a cursor that legitimately sits one past
        the end.
        """
        if bar_index >= self._size:
            return None
        found = self._next_forced[bar_index]
        return None if found >= self._size else found
