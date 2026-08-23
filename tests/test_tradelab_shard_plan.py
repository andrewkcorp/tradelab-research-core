#!/usr/bin/env python3
"""tradelab.shard_plan tests: the partition must be exact and deterministic.

A plan that silently drops, duplicates or reorders a class produces a run that
looks successful and is wrong, so these tests are about the two properties
that make sharding safe at all -- every class in exactly one shard, and the
same plan every time regardless of how the universe was ordered or which
process built it.
"""
from __future__ import annotations

import json
import random
import unittest

from tradelab.shard_plan import (
    SHARD_PLAN_SCHEMA_VERSION,
    BacktestShardPlan,
    ShardFingerprint,
    build_shard_plan,
    verify_partitions,
)


def _fingerprint(**overrides) -> ShardFingerprint:
    kwargs = dict(
        market_artifact_sha256="aaaa", news_csv_sha256="bbbb",
        parameters={"pivot_strength": 2}, risk_config={"target": 500}, equity_config={"start": 50_000},
        instruments=["MNQ"], git_commit="c0ffee",
        class_ids=["a", "b"], class_entry_counts=[1, 2], exit_variant_ids=["v1"],
    )
    kwargs.update(overrides)
    return ShardFingerprint.build(**kwargs)


def _universe(count: int, rng: random.Random) -> tuple[list[str], list[int]]:
    class_ids = [f"both|class-{i:05d}" for i in range(count)]
    entry_counts = [rng.randint(1, 20_000) for _ in class_ids]
    return class_ids, entry_counts


class PartitionTests(unittest.TestCase):
    def test_every_class_lands_in_exactly_one_shard(self):
        rng = random.Random(11)
        class_ids, counts = _universe(500, rng)
        for shard_count in (1, 2, 3, 7, 16):
            plan = build_shard_plan(
                shard_count=shard_count, class_ids=class_ids, class_entry_counts=counts,
                exit_variant_ids=["v1", "v2"], instruments=["MNQ"], fingerprint=_fingerprint(),
            )
            verify_partitions(plan, class_ids)
            assigned = [cid for shard in plan.shards for cid in shard.class_ids]
            self.assertEqual(sorted(assigned), sorted(class_ids))
            self.assertEqual(len(assigned), len(set(assigned)), "a class was assigned twice")

    def test_every_strategy_variant_id_belongs_to_exactly_one_shard(self):
        from tradelab.backtest_v2 import strategy_variant_id

        rng = random.Random(12)
        class_ids, counts = _universe(60, rng)
        plan = build_shard_plan(
            shard_count=4, class_ids=class_ids, class_entry_counts=counts,
            exit_variant_ids=["v1", "v2", "v3"], instruments=["MNQ", "NQ"], fingerprint=_fingerprint(),
        )
        seen: dict[str, int] = {}
        for shard_index in range(plan.shard_count):
            for variant_id in plan.iter_strategy_variant_ids(shard_index):
                self.assertNotIn(variant_id, seen, f"{variant_id} in shards {seen.get(variant_id)} and {shard_index}")
                seen[variant_id] = shard_index
        # The ids the plan hands out must be the ids the engine will actually
        # write, or resume and the leaderboard would not line up.
        expected = {
            strategy_variant_id(class_id, variant_id, instrument)
            for class_id in class_ids for variant_id in ("v1", "v2", "v3") for instrument in ("MNQ", "NQ")
        }
        self.assertEqual(set(seen), expected)
        self.assertEqual(len(seen), plan.total_variant_runs())

    def test_assignment_is_independent_of_universe_order(self):
        rng = random.Random(13)
        class_ids, counts = _universe(300, rng)
        shuffled = list(zip(class_ids, counts))
        random.Random(99).shuffle(shuffled)
        shuffled_ids = [cid for cid, _ in shuffled]
        shuffled_counts = [n for _, n in shuffled]

        a = build_shard_plan(
            shard_count=5, class_ids=class_ids, class_entry_counts=counts,
            exit_variant_ids=["v1"], instruments=["MNQ"], fingerprint=_fingerprint(),
        )
        b = build_shard_plan(
            shard_count=5, class_ids=shuffled_ids, class_entry_counts=shuffled_counts,
            exit_variant_ids=["v1"], instruments=["MNQ"], fingerprint=_fingerprint(),
        )
        # Same *membership* per shard regardless of input order. The within-shard
        # sequence deliberately follows the caller's own order, so compare sets.
        self.assertEqual(
            [set(s.class_ids) for s in a.shards],
            [set(s.class_ids) for s in b.shards],
        )

    def test_within_shard_order_follows_the_universe_order(self):
        class_ids = [f"c{i}" for i in range(20)]
        counts = [100] * 20
        plan = build_shard_plan(
            shard_count=3, class_ids=class_ids, class_entry_counts=counts,
            exit_variant_ids=["v1"], instruments=["MNQ"], fingerprint=_fingerprint(),
        )
        position = {cid: i for i, cid in enumerate(class_ids)}
        for shard in plan.shards:
            offsets = [position[cid] for cid in shard.class_ids]
            self.assertEqual(offsets, sorted(offsets), "shard classes must keep the global order")

    def test_repeated_builds_are_byte_identical(self):
        rng = random.Random(14)
        class_ids, counts = _universe(200, rng)
        plans = [
            build_shard_plan(
                shard_count=6, class_ids=class_ids, class_entry_counts=counts,
                exit_variant_ids=["v1", "v2"], instruments=["MNQ"], fingerprint=_fingerprint(),
            ).to_dict()
            for _ in range(3)
        ]
        self.assertEqual(json.dumps(plans[0], sort_keys=True), json.dumps(plans[1], sort_keys=True))
        self.assertEqual(json.dumps(plans[1], sort_keys=True), json.dumps(plans[2], sort_keys=True))

    def test_balances_on_entries_not_class_count(self):
        # One class carries as many entries as a thousand small ones. Balancing
        # by class count would put the whale and half the minnows together.
        class_ids = ["whale"] + [f"minnow-{i}" for i in range(1000)]
        counts = [50_000] + [50] * 1000
        plan = build_shard_plan(
            shard_count=2, class_ids=class_ids, class_entry_counts=counts,
            exit_variant_ids=["v1"], instruments=["MNQ"], fingerprint=_fingerprint(),
        )
        loads = [shard.entry_count for shard in plan.shards]
        self.assertLess(max(loads) - min(loads), 50_000, f"entry load is badly skewed: {loads}")

    def test_verify_partitions_rejects_a_missing_class(self):
        plan = build_shard_plan(
            shard_count=2, class_ids=["a", "b", "c"], class_entry_counts=[1, 1, 1],
            exit_variant_ids=["v1"], instruments=["MNQ"], fingerprint=_fingerprint(),
        )
        with self.assertRaisesRegex(ValueError, "no shard"):
            verify_partitions(plan, ["a", "b", "c", "d"])

    def test_more_shards_than_classes_is_allowed_and_leaves_empties(self):
        plan = build_shard_plan(
            shard_count=5, class_ids=["a", "b"], class_entry_counts=[1, 1],
            exit_variant_ids=["v1"], instruments=["MNQ"], fingerprint=_fingerprint(),
        )
        verify_partitions(plan, ["a", "b"])
        self.assertEqual(sum(len(s.class_ids) for s in plan.shards), 2)


class FingerprintTests(unittest.TestCase):
    def test_same_inputs_same_fingerprint(self):
        self.assertEqual(_fingerprint().combined, _fingerprint().combined)

    def test_universe_order_does_not_change_the_fingerprint(self):
        a = _fingerprint(class_ids=["a", "b", "c"], class_entry_counts=[1, 2, 3])
        b = _fingerprint(class_ids=["c", "b", "a"], class_entry_counts=[3, 2, 1])
        self.assertEqual(a.universe, b.universe)
        self.assertEqual(a.combined, b.combined)

    def test_each_input_dimension_changes_the_fingerprint(self):
        base = _fingerprint().combined
        for name, override in (
            ("data", {"market_artifact_sha256": "zzzz"}),
            ("config", {"parameters": {"pivot_strength": 3}}),
            ("code", {"git_commit": "deadbee"}),
            ("universe", {"class_ids": ["a", "b"], "class_entry_counts": [1, 3]}),
            ("exit_variants", {"exit_variant_ids": ["v2"]}),
        ):
            with self.subTest(dimension=name):
                self.assertNotEqual(base, _fingerprint(**override).combined)

    def test_round_trips_through_the_plan_json(self):
        plan = build_shard_plan(
            shard_count=3, class_ids=["a", "b", "c"], class_entry_counts=[3, 2, 1],
            exit_variant_ids=["v1"], instruments=["MNQ"], fingerprint=_fingerprint(),
        )
        restored = BacktestShardPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        self.assertEqual(restored, plan)
        self.assertEqual(restored.schema_version, SHARD_PLAN_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
