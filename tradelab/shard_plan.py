"""Deterministic partitioning of a backtest run into independent shards.

A worker today holds the entire entry universe: ~9.9 GB steady on the real
artifact, which is why a second process does not fit on a 16 GB machine (see
``docs/pre-multiprocessing-memory-readiness.md``). The fix is not to make the
universe smaller but to stop every worker from holding all of it. This module
decides, once and reproducibly, which equivalence classes belong to which
shard.

**Sharding is by equivalence class, never within one.** A class's exit
variants share its merged entry stream and its per-class ``EntryTargetCache``,
so splitting a class across shards would recompute that work per shard for no
memory benefit. More importantly, a ``both`` class is one strategy with one
position state and one equity curve: its long and short streams are simulated
together and must never be separated. The class is therefore the atomic unit,
and ``shard_of_class`` is the whole assignment rule.

**Determinism.** Assignment never depends on dict iteration order, hash
randomisation, or the order classes came out of discovery. Classes are sorted
by ``(-entry_count, class_id)`` and dealt to whichever shard currently holds
the fewest entries -- longest-processing-time-first bin packing. Ties break on
``class_id``. The same universe therefore always produces the same plan, on any
machine, in any process.

**The plan is not a sampling.** Every class lands in exactly one shard and no
class is dropped, so the union of all shards is the original run, variant for
variant. ``verify_partitions`` asserts this and is exercised by the tests.

**Fingerprints** cover everything that would make two plans incomparable:
market data, run configuration, code commit, the universe itself, and the exit
variants. A worker that loads a payload built from different inputs is a
correctness bug, not a performance one, so the fingerprint is checked rather
than assumed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

SHARD_PLAN_SCHEMA_VERSION = "1.0"


def _canonical_digest(payload: Any) -> str:
    """A stable digest of ``payload``.

    ``sort_keys`` plus a fixed separator makes this independent of dict
    insertion order, so the same logical inputs always digest identically --
    which is the entire point of a fingerprint.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ShardFingerprint:
    """What a plan was built from.

    ``combined`` is the single value worth comparing; the components exist so
    a mismatch says *which* input changed instead of only that something did.
    """

    data: str
    config: str
    code: str | None
    universe: str
    exit_variants: str
    combined: str = ""

    def __post_init__(self) -> None:
        if not self.combined:
            object.__setattr__(self, "combined", _canonical_digest({
                "data": self.data, "config": self.config, "code": self.code,
                "universe": self.universe, "exit_variants": self.exit_variants,
            }))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ShardFingerprint":
        return ShardFingerprint(
            data=d["data"], config=d["config"], code=d.get("code"),
            universe=d["universe"], exit_variants=d["exit_variants"], combined=d.get("combined", ""),
        )

    @staticmethod
    def build(
        *,
        market_artifact_sha256: str | None,
        news_csv_sha256: str | None,
        parameters: Mapping[str, Any],
        risk_config: Mapping[str, Any],
        equity_config: Mapping[str, Any],
        instruments: Sequence[str],
        git_commit: str | None,
        class_ids: Sequence[str],
        class_entry_counts: Sequence[int],
        exit_variant_ids: Sequence[str],
    ) -> "ShardFingerprint":
        return ShardFingerprint(
            data=_canonical_digest({"market": market_artifact_sha256, "news": news_csv_sha256}),
            config=_canonical_digest({
                "parameters": dict(parameters), "risk": dict(risk_config),
                "equity": dict(equity_config), "instruments": sorted(instruments),
            }),
            code=git_commit,
            # Sorted pairs, so a universe that came out of discovery in a
            # different order but contains the same classes fingerprints the
            # same -- the universe is a set of classes, not a sequence.
            universe=_canonical_digest(sorted(zip(class_ids, class_entry_counts))),
            exit_variants=_canonical_digest(sorted(exit_variant_ids)),
        )


@dataclass(frozen=True)
class ShardAssignment:
    """One shard's own work: which classes, and which exit variants over them."""

    shard_index: int
    class_ids: tuple[str, ...]
    exit_variant_ids: tuple[str, ...]
    entry_count: int

    @property
    def variant_run_count(self) -> int:
        return len(self.class_ids) * len(self.exit_variant_ids)

    def to_dict(self) -> dict:
        return {
            "shard_index": self.shard_index,
            "class_ids": list(self.class_ids),
            "exit_variant_ids": list(self.exit_variant_ids),
            "entry_count": self.entry_count,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ShardAssignment":
        return ShardAssignment(
            shard_index=d["shard_index"], class_ids=tuple(d["class_ids"]),
            exit_variant_ids=tuple(d["exit_variant_ids"]), entry_count=d["entry_count"],
        )


@dataclass(frozen=True)
class BacktestShardPlan:
    """A whole run, partitioned. Serializable, comparable, and verifiable."""

    schema_version: str
    shard_count: int
    fingerprint: ShardFingerprint
    shards: tuple[ShardAssignment, ...]
    instruments: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.shard_count < 1:
            raise ValueError("BacktestShardPlan.shard_count must be >= 1")
        if len(self.shards) != self.shard_count:
            raise ValueError(
                f"BacktestShardPlan has {len(self.shards)} shards but shard_count={self.shard_count}"
            )
        for index, shard in enumerate(self.shards):
            if shard.shard_index != index:
                raise ValueError(f"shard at position {index} reports shard_index={shard.shard_index}")

    def shard(self, shard_index: int) -> ShardAssignment:
        if not 0 <= shard_index < self.shard_count:
            raise ValueError(f"shard_index {shard_index} outside 0..{self.shard_count - 1}")
        return self.shards[shard_index]

    def total_variant_runs(self) -> int:
        return sum(shard.variant_run_count for shard in self.shards) * max(len(self.instruments), 1)

    def iter_strategy_variant_ids(self, shard_index: int) -> Iterable[str]:
        """Every ``strategy_variant_id`` this shard owns.

        Built the same way ``tradelab.backtest_v2.strategy_variant_id`` builds
        it -- deliberately duplicated as a plain format here rather than
        imported, so this module stays free of a dependency on the simulation
        engine. ``PartitionTests`` pins the two together.
        """
        shard = self.shard(shard_index)
        for class_id in shard.class_ids:
            for variant_id in shard.exit_variant_ids:
                for instrument in self.instruments:
                    yield f"{class_id}::{variant_id}::{instrument}"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "shard_count": self.shard_count,
            "fingerprint": self.fingerprint.to_dict(),
            "instruments": list(self.instruments),
            "shards": [shard.to_dict() for shard in self.shards],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "BacktestShardPlan":
        return BacktestShardPlan(
            schema_version=d["schema_version"], shard_count=d["shard_count"],
            fingerprint=ShardFingerprint.from_dict(d["fingerprint"]),
            shards=tuple(ShardAssignment.from_dict(s) for s in d["shards"]),
            instruments=tuple(d.get("instruments", ())),
        )


def build_shard_plan(
    *,
    shard_count: int,
    class_ids: Sequence[str],
    class_entry_counts: Sequence[int],
    exit_variant_ids: Sequence[str],
    instruments: Sequence[str],
    fingerprint: ShardFingerprint,
) -> BacktestShardPlan:
    """Partition ``class_ids`` across ``shard_count`` shards, deterministically.

    Balances on entry count rather than class count: a dense class can carry
    twenty thousand entries while a sparse one carries a hundred, so dealing
    equal *numbers* of classes would produce wildly unequal work. Longest-
    processing-time-first is the standard greedy approximation and is stable
    because both the sort key and the tie-break are total orders.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if len(class_ids) != len(class_entry_counts):
        raise ValueError("class_ids and class_entry_counts must be the same length")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must be unique")

    ordered = sorted(zip(class_ids, class_entry_counts), key=lambda pair: (-pair[1], pair[0]))
    buckets: list[list[str]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for class_id, entry_count in ordered:
        # min() over (load, index) keeps the choice deterministic when several
        # shards are equally loaded -- notably at the very start, when all are 0.
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        buckets[target].append(class_id)
        loads[target] += entry_count

    # Restore the caller's own class order within each shard. Execution order
    # inside a shard is then exactly the global order restricted to its
    # classes, which is what keeps results and the journal identical to an
    # unsharded run. Done with one position lookup rather than a scan per
    # shard, which matters at 27k classes.
    position = {class_id: index for index, class_id in enumerate(class_ids)}
    variant_ids = tuple(exit_variant_ids)
    shards = tuple(
        ShardAssignment(
            shard_index=index,
            class_ids=tuple(sorted(bucket, key=position.__getitem__)),
            exit_variant_ids=variant_ids,
            entry_count=loads[index],
        )
        for index, bucket in enumerate(buckets)
    )
    return BacktestShardPlan(
        schema_version=SHARD_PLAN_SCHEMA_VERSION, shard_count=shard_count,
        fingerprint=fingerprint, shards=shards, instruments=tuple(instruments),
    )


def verify_partitions(plan: BacktestShardPlan, expected_class_ids: Sequence[str]) -> None:
    """Raise unless the plan is an exact partition of ``expected_class_ids``.

    Cheap enough to call before every sharded run, and worth doing: a plan that
    silently drops or duplicates a class produces a run that looks successful
    and is wrong.
    """
    seen: dict[str, int] = {}
    for shard in plan.shards:
        for class_id in shard.class_ids:
            if class_id in seen:
                raise ValueError(
                    f"class {class_id!r} is assigned to both shard {seen[class_id]} and {shard.shard_index}"
                )
            seen[class_id] = shard.shard_index
    expected = set(expected_class_ids)
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing:
        raise ValueError(f"{len(missing)} class(es) are in the universe but no shard, e.g. {sorted(missing)[:3]}")
    if extra:
        raise ValueError(f"{len(extra)} class(es) are in the plan but not the universe, e.g. {sorted(extra)[:3]}")
