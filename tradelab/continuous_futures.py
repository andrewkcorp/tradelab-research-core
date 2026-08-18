"""Additive back-adjusted continuous futures series for NQ/MNQ.

Quarterly NQ/MNQ contracts each have their own price series; splicing
them together naively at rollover produces an artificial price jump
that has nothing to do with real market movement -- purely an artifact
of switching which contract is being quoted. Left uncorrected, that
jump can manufacture spurious FVGs, liquidity sweeps, and HTF bias
right at every rollover, contaminating years of otherwise-real
confluence with fake signals. This module removes that artifact by
additively shifting every older contract's own OHLC by a caller-
supplied number of points, chaining backward from the newest (anchor)
contract, so the whole spliced series reads as one continuous
instrument with no artificial rollover gap.

**Additive, not percentage, adjustment** (this task's own spec): NQ/MNQ
tick size is a fixed 0.25 points regardless of price level, and every
technical-analysis distance in this package (FVG gaps, liquidity
sweeps, stop/target distances) is measured in points/ticks, not
percentage return -- a percentage (ratio) adjustment would distort
those absolute distances differently at every price level, silently
breaking every downstream detector's notion of "how many points/ticks
apart". A flat point offset preserves every absolute distance in the
older, shifted contract's own history exactly as it was, just moved to
a new price origin.

**The roll manifest is the only source of truth for *where* and *by how
much* to adjust.** This module never inspects contract prices to guess
a roll date or an adjustment size, and never infers a rollover from a
volume dropoff, an open-interest crossover, or any other heuristic --
those decisions belong entirely to whoever builds the ``RollAdjustment``
manifest (typically from vendor-published settlement prices at the
actual roll date), an explicit, auditable input to this module, never
computed by it. All this module does is (1) verify that manifest is
internally consistent and matches the actual contract transitions
present in ``bars``, and (2) apply it. Without this step, several years
of spliced NQ/MNQ 1-minute data would still contain a handful of raw,
unadjusted contract-switch jumps -- each one a plausible-looking but
entirely fake price gap that today's or a future detector could easily
mistake for a real FVG, liquidity sweep, or bias flip.

**Newest contract is the back-adjustment anchor.** Per this task's
convention, the most recent contract observed in ``bars`` keeps its
prices completely unadjusted (cumulative offset ``0.0``); every older
contract is shifted by its own cumulative offset -- the sum of every
``RollAdjustment.adjustment_points`` along the chain from that
contract's own exit roll up to the anchor. ``RollAdjustment.
adjustment_points`` is always this single-hop, additive meaning: "how
many points to add to the *older* (``outgoing_contract``) contract's
own history to splice it onto the *newer* (``incoming_contract``)
contract's own price level" -- never a total/cumulative figure itself,
even though a chain of more than one roll compounds those single-hop
figures into a cumulative offset for anything further back than one
hop.

**Every bar keeps its original identity except its own OHLC and
``contract`` label.** ``volume`` and ``timestamp`` are copied through
completely untouched -- rollover realigns price, never trading
activity or time. Every output bar's own ``contract`` becomes the
single ``canonical_contract`` label this function was called with
(e.g. ``"NQ_CONTINUOUS"``/``"MNQ_CONTINUOUS"``), regardless of which
raw quarterly contract (e.g. ``"NQZ24"``) it actually came from --
that raw label survives only in this module's own audit trail
(``ContinuousSeriesResult.applied_rolls``/``cumulative_offsets``), not
on the bar itself.

**Input bars are never mutated.** ``Bar1m`` is already an immutable,
frozen dataclass; this module additionally never calls
``object.__setattr__`` on any input bar and builds entirely new
``Bar1m`` instances for its output, so the caller's own input list and
every bar object in it remain exactly as they were, safely reusable
for anything else the caller does with them.

**Already-continuous input is a first-class case, not a special
case.** If every bar in ``bars`` already carries the same raw
``contract`` label, there are zero contract transitions to reconcile,
so an empty ``rolls`` is the only valid input (any non-empty ``rolls``
in that case necessarily references at least one contract absent from
``bars``, and is rejected by the same "unknown contract" validation
that would reject it in any other case -- no separate code path is
needed). The result is still a fully-formed, auditable
``ContinuousSeriesResult`` (with an empty ``applied_rolls`` and a
single, zero-offset ``cumulative_offsets`` entry); prices are not
altered.

Deliberately out of scope for this module (later, separate steps, per
this task): ``tradelab.backtest``, any confluence detector, TP/SL
target logic, and the CSV/Parquet vendor loaders are all untouched --
this module only prepares an already-loaded ``list[Bar1m]`` (e.g. the
concatenated output of several ``read_csv_bars``/``read_parquet_bars``
calls, one per contract) into the single, back-adjusted series those
existing modules already expect to consume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from tradelab.bar import Bar1m
from tradelab.validation import validate_bars


@dataclass(frozen=True)
class RollAdjustment:
    """One explicit, audited rollover from ``outgoing_contract`` (older)
    to ``incoming_contract`` (newer). ``adjustment_points`` is always the
    single-hop additive shift: how many points must be added to
    ``outgoing_contract``'s own history to splice it onto
    ``incoming_contract``'s own price level -- never a cumulative figure
    across more than this one hop (see the module docstring). This
    module never computes ``effective_at``/``adjustment_points`` itself;
    both must come from the caller's own audited roll manifest.
    """

    outgoing_contract: str
    incoming_contract: str
    effective_at: datetime
    adjustment_points: float

    def __post_init__(self) -> None:
        if not self.outgoing_contract:
            raise ValueError("RollAdjustment.outgoing_contract must be non-empty")
        if not self.incoming_contract:
            raise ValueError("RollAdjustment.incoming_contract must be non-empty")
        if self.outgoing_contract == self.incoming_contract:
            raise ValueError(
                f"RollAdjustment.outgoing_contract and incoming_contract must differ, both are {self.outgoing_contract!r}"
            )
        if self.effective_at.tzinfo is None:
            raise ValueError(f"RollAdjustment.effective_at must be timezone-aware: {self.effective_at!r}")
        if not isinstance(self.adjustment_points, (int, float)) or isinstance(self.adjustment_points, bool):
            raise ValueError(f"RollAdjustment.adjustment_points must be a real number, got {self.adjustment_points!r}")
        if not math.isfinite(self.adjustment_points):
            raise ValueError(f"RollAdjustment.adjustment_points must be explicit and finite, got {self.adjustment_points!r}")


@dataclass(frozen=True)
class ContractOffset:
    """One raw contract's own cumulative additive offset -- how many
    points were added to every one of its bars to splice it onto
    ``canonical_contract``. Zero for the newest observed contract (the
    back-adjustment anchor, never adjusted); every older contract's own
    value is the sum of every ``RollAdjustment.adjustment_points`` along
    its own chain up to the anchor."""

    contract: str
    cumulative_offset_points: float


@dataclass(frozen=True)
class ContinuousSeriesResult:
    """``normalized_bars`` is the full back-adjusted series, in the same
    chronological order as the input, every bar re-stamped with
    ``canonical_contract``. ``applied_rolls`` is every ``RollAdjustment``
    that was actually validated and used, in chronological order of its
    own contract transition (never merely the order the caller happened
    to pass them in). ``cumulative_offsets`` is the full per-contract
    audit trail (oldest contract first, ending with the zero-offset
    anchor) -- together with ``applied_rolls`` it fully explains, for
    every raw contract, both *how much* it was shifted and *which*
    rolls produced that figure.
    """

    normalized_bars: tuple[Bar1m, ...]
    canonical_contract: str
    applied_rolls: tuple[RollAdjustment, ...]
    cumulative_offsets: tuple[ContractOffset, ...]


def build_back_adjusted_continuous_series(
    bars: list[Bar1m], rolls: list[RollAdjustment], canonical_contract: str
) -> ContinuousSeriesResult:
    """Splice ``bars`` (spanning one or more raw quarterly contracts,
    already in chronological order) into one additive back-adjusted
    continuous series labeled ``canonical_contract``, using ``rolls`` as
    the sole, explicit source of every rollover date and adjustment
    size -- see the module docstring for the full contract. Raises
    ``ValueError`` on any inconsistency between ``rolls`` and the actual
    contract transitions observed in ``bars`` (missing roll, unknown
    contract, conflicting/duplicate roll for the same transition, a
    roll whose ``effective_at`` does not line up with where the data
    actually switches contracts, or a contract that reappears after the
    series had already moved past it). Never mutates ``bars`` or any
    bar in it.
    """
    if not canonical_contract:
        raise ValueError("canonical_contract must be non-empty")

    bars = list(bars)
    rolls = list(rolls)

    validate_bars(bars)  # chronological, no duplicates, valid OHLCV -- reused, never reimplemented here

    if not bars:
        if rolls:
            raise ValueError("rolls must be empty when bars is empty -- there is no contract transition to apply them to")
        return ContinuousSeriesResult(normalized_bars=(), canonical_contract=canonical_contract, applied_rolls=(), cumulative_offsets=())

    contract_chain = _contract_chain(bars)
    transitions = _find_transitions(bars)
    roll_by_pair = _validate_rolls(rolls, contract_chain, transitions)

    cumulative: dict[str, float] = {contract_chain[-1]: 0.0}
    ordered_rolls: list[RollAdjustment] = []
    for outgoing, incoming in reversed(list(zip(contract_chain, contract_chain[1:]))):
        roll = roll_by_pair[(outgoing, incoming)]
        cumulative[outgoing] = roll.adjustment_points + cumulative[incoming]
        ordered_rolls.append(roll)
    ordered_rolls.reverse()  # chronological order: oldest transition first

    normalized_bars = tuple(
        Bar1m(
            timestamp=bar.timestamp,
            open=bar.open + cumulative[bar.contract],
            high=bar.high + cumulative[bar.contract],
            low=bar.low + cumulative[bar.contract],
            close=bar.close + cumulative[bar.contract],
            volume=bar.volume,
            contract=canonical_contract,
        )
        for bar in bars
    )

    cumulative_offsets = tuple(ContractOffset(contract=c, cumulative_offset_points=cumulative[c]) for c in contract_chain)

    return ContinuousSeriesResult(
        normalized_bars=normalized_bars,
        canonical_contract=canonical_contract,
        applied_rolls=tuple(ordered_rolls),
        cumulative_offsets=cumulative_offsets,
    )


def _contract_chain(bars: list[Bar1m]) -> list[str]:
    """The distinct raw contracts in ``bars``, oldest first, each
    required to occupy exactly one contiguous chronological block --
    a contract reappearing after the series has already moved past it
    makes "the" cumulative offset for that label ambiguous, so it is
    rejected rather than silently picking one occurrence."""
    chain: list[str] = []
    seen: set[str] = set()
    for bar in bars:
        if not chain or chain[-1] != bar.contract:
            if bar.contract in seen:
                raise ValueError(
                    f"contract {bar.contract!r} reappears after the series had already moved past it -- "
                    "each contract must occupy exactly one contiguous chronological block in bars"
                )
            chain.append(bar.contract)
            seen.add(bar.contract)
    return chain


def _find_transitions(bars: list[Bar1m]) -> list[tuple[str, str, datetime, datetime]]:
    """Every point in ``bars`` where the contract label actually
    changes, as ``(outgoing, incoming, previous_bar_timestamp,
    transition_bar_timestamp)``, in chronological order."""
    return [
        (previous.contract, bar.contract, previous.timestamp, bar.timestamp)
        for previous, bar in zip(bars, bars[1:])
        if bar.contract != previous.contract
    ]


def _validate_rolls(
    rolls: list[RollAdjustment],
    contract_chain: list[str],
    transitions: list[tuple[str, str, datetime, datetime]],
) -> dict[tuple[str, str], RollAdjustment]:
    contract_set = set(contract_chain)

    roll_by_pair: dict[tuple[str, str], RollAdjustment] = {}
    for roll in rolls:
        pair = (roll.outgoing_contract, roll.incoming_contract)
        if pair in roll_by_pair:
            raise ValueError(
                f"conflicting/duplicate RollAdjustment for the same transition {roll.outgoing_contract!r} -> "
                f"{roll.incoming_contract!r} -- exactly one roll is allowed per transition"
            )
        roll_by_pair[pair] = roll

    for roll in rolls:
        if roll.outgoing_contract not in contract_set:
            raise ValueError(f"RollAdjustment references unknown contract {roll.outgoing_contract!r}, not present in bars")
        if roll.incoming_contract not in contract_set:
            raise ValueError(f"RollAdjustment references unknown contract {roll.incoming_contract!r}, not present in bars")

    transition_pairs = {(outgoing, incoming) for outgoing, incoming, _, _ in transitions}
    for outgoing, incoming, previous_ts, ts in transitions:
        pair = (outgoing, incoming)
        if pair not in roll_by_pair:
            raise ValueError(
                f"bars transitions from contract {outgoing!r} to {incoming!r} at {ts!r} with no matching "
                "RollAdjustment -- every contract change in bars requires an explicit roll"
            )
        roll = roll_by_pair[pair]
        if not (previous_ts < roll.effective_at <= ts):
            raise ValueError(
                f"RollAdjustment {outgoing!r} -> {incoming!r} effective_at {roll.effective_at!r} is not consistent "
                f"with the actual contract transition in bars (must fall strictly after {previous_ts!r} and at or "
                f"before {ts!r})"
            )

    extraneous = roll_by_pair.keys() - transition_pairs
    if extraneous:
        outgoing, incoming = sorted(extraneous)[0]
        raise ValueError(
            f"RollAdjustment {outgoing!r} -> {incoming!r} does not correspond to any actual contract transition "
            "in bars -- rolls must exactly match the sequence of contract changes observed in the data"
        )

    return roll_by_pair
