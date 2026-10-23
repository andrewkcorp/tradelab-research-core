"""Fail-closed discovery/holdout separation.

The bundled policy is an illustrative fixture, not a statement about any
real dataset. Discovery and holdout periods must be predeclared for each actual
study. This policy makes that separation machine-checkable.  Production discovery rejects future
holdout bars; final validation requires an explicit call and at least the
configured amount of genuinely later data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Sequence

from tradelab.bar import Bar1m

SCHEMA_VERSION = "tradelab-research-period-policy-v1"
DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "_resources" / "research_period_policy.json"


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("research period timestamps must be timezone-aware")
    return parsed


@dataclass(frozen=True)
class ResearchPeriodPolicy:
    discovery_start_utc: datetime
    discovery_end_exclusive_utc: datetime
    holdout_start_utc: datetime
    minimum_holdout_calendar_days: int
    rationale: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported research-period schema {self.schema_version!r}")
        if self.discovery_end_exclusive_utc != self.holdout_start_utc:
            raise ValueError("discovery end must equal holdout start; overlap/gap is not allowed")
        if not self.discovery_start_utc < self.discovery_end_exclusive_utc:
            raise ValueError("discovery period must be non-empty")
        if self.minimum_holdout_calendar_days < 1 or not self.rationale.strip():
            raise ValueError("holdout duration and rationale must be explicit")


def load_research_period_policy(path: Path | str = DEFAULT_POLICY_PATH) -> ResearchPeriodPolicy:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return ResearchPeriodPolicy(
            schema_version=raw["schema_version"],
            discovery_start_utc=_timestamp(raw["discovery_start_utc"]),
            discovery_end_exclusive_utc=_timestamp(raw["discovery_end_exclusive_utc"]),
            holdout_start_utc=_timestamp(raw["holdout_start_utc"]),
            minimum_holdout_calendar_days=int(raw["minimum_holdout_calendar_days"]),
            rationale=raw["rationale"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid research-period policy {path}: {error}") from error


def require_discovery_only(bars: Sequence[Bar1m], policy: ResearchPeriodPolicy) -> None:
    """Refuse to let strategy discovery consume any future holdout bar."""
    outside = [bar.timestamp for bar in bars if not policy.discovery_start_utc <= bar.timestamp < policy.discovery_end_exclusive_utc]
    if outside:
        raise ValueError(
            f"discovery input contains {len(outside)} bar(s) outside the locked discovery period; "
            f"first={outside[0].isoformat()}, holdout begins {policy.holdout_start_utc.isoformat()}"
        )


def require_mature_holdout(bars: Sequence[Bar1m], policy: ResearchPeriodPolicy, *, explicitly_unlocked: bool) -> None:
    """Permit one final holdout evaluation only after an explicit unlock."""
    if not explicitly_unlocked:
        raise ValueError("holdout access requires explicitly_unlocked=True")
    if not bars or min(bar.timestamp for bar in bars) < policy.holdout_start_utc:
        raise ValueError("holdout input is empty or contains discovery-period bars")
    observed_span = max(bar.timestamp for bar in bars) - min(bar.timestamp for bar in bars)
    minimum = timedelta(days=policy.minimum_holdout_calendar_days)
    if observed_span < minimum:
        raise ValueError(f"holdout is immature: observed {observed_span.days} days, requires at least {minimum.days}")
