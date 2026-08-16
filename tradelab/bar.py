"""Normalized 1-minute OHLCV bar model."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Bar1m:
    """A single, closed 1-minute candle.

    ``timestamp`` is the UTC instant the bar *opened*; the bar covers
    ``[timestamp, timestamp + 1 minute)`` and is only valid once that
    interval has fully elapsed (see ``tradelab.aggregate`` for how this
    propagates to higher timeframes via ``known_at``).
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    contract: str

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(f"Bar1m.timestamp must be timezone-aware: {self.timestamp!r}")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        if self.timestamp.second != 0 or self.timestamp.microsecond != 0:
            raise ValueError(
                f"Bar1m.timestamp must fall exactly on a minute boundary (second=0, microsecond=0): {self.timestamp!r}"
            )
        if not self.contract:
            raise ValueError("Bar1m.contract must be a non-empty source contract identifier")
