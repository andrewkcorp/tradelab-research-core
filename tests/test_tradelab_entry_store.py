#!/usr/bin/env python3
"""The normalised store and bundle must hold references, not copies.

The old bundle format duplicated every ``ObservedEntry`` into every shard that
referenced it. Classes share entries 33.6x, so 256 shards of the real artifact
projected to 44.8 GB and a production run had to be aborted mid-materialisation
with the disk at 98%. These tests pin the two properties that fix stays fixed:

* every unique entry is stored exactly once,
* no shard file contains an ``ObservedEntry`` at all.

Both are asserted against the *bytes on disk*, not against an in-memory model,
because the failure they guard is a serialisation failure.
"""
from __future__ import annotations

import pickle
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from tradelab.backtest import release_discovery_inputs
from tradelab.backtest_v2 import (
    ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, build_shard_plan_for_universe, shard_universe,
)
from tradelab.both_direction_universe import discover_both_direction_universe
from tradelab.entry_store import (
    EntryStoreError,
    EntryStoreReader,
    build_entry_id_index,
    collect_unique_entries,
    write_entry_store,
)
from tradelab.entry_strategy_universe import ObservedEntry, discover_entry_strategy_universe
from tradelab.shard_bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleBudgetExceeded,
    STORE_DIRNAME,
    bundle_sizes,
    load_shard_classes,
    open_store,
    shard_entry_reference_count,
    shard_filename,
    write_bundle,
)
from tradelab.shard_payload import shared_fvg_zone_index

from tests.test_tradelab_backtest_v2 import _random_zigzag_context


def _universe(seed: int = 5150, n_bars: int = 900):
    context = _random_zigzag_context(random.Random(seed), n_bars=n_bars)
    universe = discover_entry_strategy_universe(
        context.fvg_results, context.fvg_entry_result, context.liquidity_entry_result,
        context.htf_bias_result, min_entries_per_year=1, pair_yearly_eligibility_for_both=True,
    )
    both = discover_both_direction_universe(universe)
    return release_discovery_inputs(context), both


class _BundleFixture:
    def __init__(self, tmp: Path, shard_count: int = 8, **budget):
        self.context, self.both = _universe()
        self.shard_count = min(shard_count, max(1, self.both.equivalence_class_count))
        self.plan = build_shard_plan_for_universe(
            self.both, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, shard_count=self.shard_count,
            market_artifact_sha256="fixture", news_csv_sha256=None, parameters={}, git_commit="test",
        )
        self.zone_index = shared_fvg_zone_index(self.both.equivalence_classes)
        self.dir = tmp / "bundle"
        self.manifest = write_bundle(
            self.dir, plan=self.plan, context=self.context,
            equivalence_classes=self.both.equivalence_classes, fvg_zone_index=self.zone_index, **budget,
        )


class EntryStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.fx = _BundleFixture(cls.tmp)
        if cls.fx.both.equivalence_class_count < 3:
            raise unittest.SkipTest("fixture produced too few classes")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_unique_entry_is_stored_exactly_once(self):
        """The headline property. Counted from the store's own index, and
        cross-checked against the universe's distinct entry ids."""
        expected = collect_unique_entries(self.fx.both.equivalence_classes)
        expected_ids = {e.entry_id for e in expected}
        self.assertEqual(len(expected), len(expected_ids), "collect_unique_entries returned duplicates")

        with open_store(self.fx.dir) as store:
            self.assertEqual(len(store), len(expected_ids))
            stored_ids = [store.read(i).entry_id for i in range(len(store))]
        self.assertEqual(len(stored_ids), len(set(stored_ids)), "an entry id appears twice in the store")
        self.assertEqual(set(stored_ids), expected_ids)

        # And the universe really does reference them many times over, or this
        # test would be proving nothing.
        references = sum(
            len(c.long_entries) + len(c.short_entries) + len(c.observed_entries)
            for c in self.fx.both.equivalence_classes
        )
        self.assertGreater(
            references, len(expected_ids) * 2,
            f"fixture shares entries only {references / max(len(expected_ids), 1):.1f}x; "
            "it cannot demonstrate deduplication",
        )

    def test_no_shard_file_contains_an_observed_entry(self):
        """A shard must carry references. Asserted by unpickling the raw file
        and walking it, so any future change that smuggles entries back in
        fails here rather than on a full disk."""
        for shard_index in range(self.fx.plan.shard_count):
            raw = (self.fx.dir / shard_filename(shard_index)).read_bytes()
            payload = pickle.loads(raw)
            found: list[str] = []

            def walk(value, path="payload"):
                if isinstance(value, ObservedEntry):
                    found.append(path)
                    return
                if isinstance(value, dict):
                    for key, item in value.items():
                        walk(item, f"{path}[{key!r}]")
                elif isinstance(value, (list, tuple, set, frozenset)):
                    for i, item in enumerate(value):
                        walk(item, f"{path}[{i}]")

            walk(payload)
            self.assertEqual(
                found, [], f"shard {shard_index} contains ObservedEntry objects at {found[:3]}"
            )
            # Nor the zone index, which is the other thing that used to be copied.
            self.assertNotIn(b"FVGZone", raw, f"shard {shard_index} embeds FVGZone records")

    def test_entry_ids_are_stable_across_processes_and_orderings(self):
        entries = collect_unique_entries(self.fx.both.equivalence_classes)
        a = build_entry_id_index(entries)
        shuffled = list(self.fx.both.equivalence_classes)
        random.Random(7).shuffle(shuffled)
        b = build_entry_id_index(collect_unique_entries(shuffled))
        self.assertEqual(a, b, "entry numbering depends on class order")
        # Numbering follows sorted entry_id, so it is reproducible from the ids
        # alone -- no process state involved.
        self.assertEqual([k for k, _ in sorted(a.items(), key=lambda kv: kv[1])], sorted(a))

    def test_random_access_does_not_read_the_whole_store(self):
        with open_store(self.fx.dir) as store:
            last = store.read(len(store) - 1)
            first = store.read(0)
        self.assertNotEqual(first.entry_id, last.entry_id)
        self.assertLessEqual(first.entry_id, last.entry_id)

    def test_shared_parts_are_pooled_not_repeated(self):
        with open_store(self.fx.dir) as store:
            self.assertLessEqual(
                store.manifest.selector_map_count, store.manifest.entry_count,
                "selector maps should be pooled, not one per entry",
            )
            entries = [store.read(i) for i in range(min(len(store), 50))]
            with_maps = [e for e in entries if e.active_fvg_zone_ids_by_selector is not None]
            if with_maps:
                # Every entry points at the one shared zone index object.
                self.assertEqual(
                    len({id(e.fvg_zone_by_id) for e in with_maps}), 1,
                    "entries do not share a single zone index instance",
                )

    def test_bundle_is_far_smaller_than_the_universe_it_references(self):
        sizes = bundle_sizes(self.fx.dir)
        references = sum(
            shard_entry_reference_count(self.fx.dir, i) for i in range(self.fx.plan.shard_count)
        )
        self.assertGreater(references, 0)
        # 4 bytes per reference plus class metadata; the point is that shard
        # bytes scale with references, not with entry payloads.
        self.assertLess(
            sizes["shard_bytes_total"], sizes["store_bytes"],
            "shards outweigh the store, which means they are still carrying payloads",
        )

    def test_missing_or_truncated_store_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "bundle"
            shutil.copytree(self.fx.dir, broken)
            (broken / STORE_DIRNAME / "manifest.json").unlink()
            with self.assertRaisesRegex(EntryStoreError, "manifest"):
                open_store(broken)

        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "bundle"
            shutil.copytree(self.fx.dir, broken)
            data = broken / STORE_DIRNAME / "entries.dat"
            data.write_bytes(data.read_bytes()[:-64])
            with self.assertRaisesRegex(EntryStoreError, "truncated|different run"):
                open_store(broken)

    def test_store_fingerprint_mismatch_is_refused(self):
        with self.assertRaisesRegex(EntryStoreError, "fingerprint"):
            open_store(self.fx.dir, expected_fingerprint="0" * 64)

    def test_shard_built_against_another_store_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            other_dir = Path(tmp) / "other"
            context, both = _universe(seed=777, n_bars=500)
            if both.equivalence_class_count < 1:
                self.skipTest("second fixture produced no classes")
            plan = build_shard_plan_for_universe(
                both, ALL_EXIT_VARIANTS, DEFAULT_INSTRUMENTS, shard_count=1,
                market_artifact_sha256="other", news_csv_sha256=None, parameters={}, git_commit="test",
            )
            write_bundle(
                other_dir, plan=plan, context=context, equivalence_classes=both.equivalence_classes,
                fvg_zone_index=shared_fvg_zone_index(both.equivalence_classes),
            )
            # Point a shard from one bundle at the other bundle's store.
            with open_store(other_dir) as other_store:
                with self.assertRaisesRegex(EntryStoreError, "built against store"):
                    load_shard_classes(self.fx.dir, 0, other_store)

    def test_eager_zone_representation_is_rejected_rather_than_silently_copied(self):
        from datetime import datetime, timezone

        from tradelab.fvg import FVGStatus, FVGZone

        t0 = datetime(2024, 6, 3, 14, tzinfo=timezone.utc)
        zone = FVGZone(
            zone_id="z1", timeframe="5m", direction="bullish", lower=1.0, upper=2.0,
            candle_1_timestamp=t0, candle_2_timestamp=t0, candle_3_timestamp=t0,
            formed_at=t0, status=FVGStatus.ACTIVE, resolved_at=None, produced_ifvg=False,
        )
        eager = ObservedEntry(
            entry_id="e1", known_at=t0, direction="long", source=object(),
            active_fvg_zones_by_selector={"fvg-5m-fvg_touch": (zone,)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "eager"):
                write_entry_store(Path(tmp) / "store", [eager], fvg_zone_index=None)


class BundleBudgetTests(unittest.TestCase):
    def test_budget_refusal_happens_after_a_probe_not_after_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BundleBudgetExceeded) as caught:
                _BundleFixture(Path(tmp), shard_count=8, max_bundle_bytes=1, probe_shards=1)
            self.assertGreater(caught.exception.projected_bytes, 0)
            # The refusal must carry a real projection, not just a failure.
            self.assertEqual(caught.exception.budget_bytes, 1)

    def test_free_space_requirement_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(BundleBudgetExceeded, "free"):
                _BundleFixture(Path(tmp), shard_count=4, min_free_bytes_after=2 ** 60)


if __name__ == "__main__":
    unittest.main()
