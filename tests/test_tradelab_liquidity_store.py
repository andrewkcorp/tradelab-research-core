#!/usr/bin/env python3
"""Mapped liquidity levels must answer identically, and stay lazy.

Two independent obligations:

* **identical answers** -- `MappedLiquidityTargetIndex.nearest()` must return
  the same levels, in the same order, as `LiquidityTargetIndex.nearest()`, for
  every side, count and eligibility boundary. Rank 1/2/3 and the equal-price
  tie groups are the whole point; an "almost right" ordering silently changes
  which target a trade uses.
* **laziness** -- the reader must not rebuild the store as Python objects. A
  correct-but-eager reader would pass every equivalence test while saving
  nothing, so that property is asserted directly rather than assumed.

The tie-break deserves special attention: the object index orders equal
candidates by `level_id` as a *string*, while the mapped index orders them by a
numeric ordinal. Those agree only because ordinals are assigned in sorted
`level_id` order, and there is a test for exactly that.
"""
from __future__ import annotations

import json
import random
import struct
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradelab.liquidity import (
    HIGH,
    LOW,
    LiquidityDetectionResult,
    LiquidityLevel,
    LiquidityLevelRevision,
    LiquidityStatus,
    LiquidityType,
)
from tradelab.liquidity_store import (
    LIQUIDITY_STORE_FORMAT_VERSION,
    LiquidityStoreError,
    open_liquidity_store,
    write_liquidity_store,
)
from tradelab.liquidity_target_index import LiquidityTargetIndex, MappedLiquidityTargetIndex

_T0 = datetime(2024, 6, 3, 13, 30, tzinfo=timezone.utc)


def _fixture(seed: int = 4242, levels_per_tf: int = 60):
    """Levels dense enough to produce equal-price groups, revisions, taken and
    absorbed states -- the cases the tie-break and eligibility rules exist for."""
    rng = random.Random(seed)
    swing: dict[str, LiquidityDetectionResult] = {}
    session: dict[str, LiquidityDetectionResult] = {}

    for family, table, types in (
        ("sw", swing, (LiquidityType.SWING_HIGH, LiquidityType.SWING_LOW,
                       LiquidityType.EQUAL_HIGHS, LiquidityType.EQUAL_LOWS)),
        ("se", session, (LiquidityType.SESSION_ASIA_HIGH, LiquidityType.SESSION_ASIA_LOW)),
    ):
        for timeframe in ("1m", "5m"):
            levels: list[LiquidityLevel] = []
            revisions: list[LiquidityLevelRevision] = []
            for i in range(levels_per_tf):
                liquidity_type = types[i % len(types)]
                side = HIGH if "high" in liquidity_type.value else LOW
                # A deliberately small price alphabet, so many levels collide
                # on price and the equal-price tie groups get exercised.
                price = 17000.0 + (i % 7) * 0.25
                formed_at = _T0 + timedelta(minutes=i * 3)
                taken = rng.random() < 0.3
                absorbed = (not taken) and rng.random() < 0.15
                status = (LiquidityStatus.TAKEN if taken
                          else LiquidityStatus.ABSORBED_IN_EQUAL_CLUSTER if absorbed
                          else LiquidityStatus.ACTIVE)
                level = LiquidityLevel(
                    level_id=f"{family}-{timeframe}-{i:04d}",
                    timeframe=timeframe, liquidity_type=liquidity_type, side=side,
                    lower=price, upper=price + (0.5 if i % 3 == 0 else 0.0),
                    member_count=1 + (i % 3),
                    member_timestamps=tuple(
                        formed_at - timedelta(minutes=m) for m in range(1 + (i % 3))
                    ),
                    formed_at=formed_at, status=status,
                    taken_at=formed_at + timedelta(minutes=20) if taken else None,
                    absorbed_by_level_id=f"{family}-{timeframe}-0000" if absorbed else None,
                )
                levels.append(level)
                # An earlier, ACTIVE version for some levels, so effective
                # windows open and close rather than being open-ended.
                if i % 4 == 0:
                    earlier = LiquidityLevel(
                        **{**asdict(level), "status": LiquidityStatus.ACTIVE, "taken_at": None,
                           "absorbed_by_level_id": None,
                           "liquidity_type": level.liquidity_type,
                           "member_timestamps": level.member_timestamps}
                    )
                    revisions.append(LiquidityLevelRevision(effective_at=formed_at, level=earlier))
                    revisions.append(LiquidityLevelRevision(
                        effective_at=formed_at + timedelta(minutes=10), level=level
                    ))
                else:
                    revisions.append(LiquidityLevelRevision(effective_at=formed_at, level=level))
            revisions.sort(key=lambda r: r.effective_at)
            table[timeframe] = LiquidityDetectionResult(
                levels=levels, takes=[], reclaims=[], level_revisions=tuple(revisions)
            )
    return swing, session


class LiquidityStoreEquivalenceTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.swing, cls.session = _fixture()
        cls.dir = cls.tmp / "liquidity"
        cls.manifest = write_liquidity_store(
            cls.dir, cls.swing, cls.session, fingerprint="fp-liq"
        )
        cls.reference = LiquidityTargetIndex.build(cls.swing, cls.session)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_nearest_is_identical_across_sides_counts_and_boundaries(self):
        with open_liquidity_store(self.dir) as store:
            mapped = MappedLiquidityTargetIndex(store)

            instants = []
            for results in (self.swing, self.session):
                for result in results.values():
                    for level in result.levels:
                        instants.append(level.formed_at)
                        # The exact boundary: eligibility uses <= on formed_at
                        # and <= on taken_at, so one minute either side of each
                        # must be probed or the rule is untested.
                        instants.append(level.formed_at - timedelta(minutes=1))
                        if level.taken_at is not None:
                            instants.append(level.taken_at)
                            instants.append(level.taken_at - timedelta(minutes=1))
            prices = [17000.0 + step * 0.125 for step in range(-4, 20)]

            compared = 0
            nonempty = 0
            for side in (HIGH, LOW):
                for count in (1, 2, 3):
                    for price in prices:
                        for known_at in instants:
                            expected = self.reference.nearest(
                                side=side, entry_price=price, entry_known_at=known_at, count=count
                            )
                            got = mapped.nearest(
                                side=side, entry_price=price, entry_known_at=known_at, count=count
                            )
                            compared += 1
                            nonempty += 1 if expected else 0
                            self.assertEqual(
                                [asdict(level) for level in expected],
                                [asdict(level) for level in got],
                                f"side={side} price={price} known_at={known_at} count={count}",
                            )
            # The comparison is only meaningful if it actually returned levels.
            self.assertGreater(compared, 1000)
            self.assertGreater(
                nonempty, compared // 10,
                f"only {nonempty}/{compared} probes returned any level; the fixture is too sparse",
            )

    def test_equal_price_groups_really_occur_in_the_fixture(self):
        """Guards the guard: without price collisions the tie-break is untested."""
        with open_liquidity_store(self.dir) as store:
            prices = list(store.high_prices)
            duplicates = len(prices) - len(set(prices))
            self.assertGreater(
                duplicates, 20,
                f"only {duplicates} equal-price collisions on the high side; "
                "the equal-price tie group path is barely exercised",
            )

    def test_numeric_ordinals_order_exactly_as_the_level_id_strings_do(self):
        """The mapped index breaks ties on a numeric ordinal where the object
        index breaks them on the `level_id` string. That substitution is only
        valid because ordinals follow sorted string order."""
        with open_liquidity_store(self.dir) as store:
            ids = [store.level_id(ordinal) for ordinal in range(store.level_count)]
        self.assertEqual(ids, sorted(ids), "ordinals do not follow sorted level_id order")
        self.assertEqual(len(ids), len(set(ids)), "an ordinal maps to a duplicate level_id")

    def test_every_level_version_round_trips_field_for_field(self):
        with open_liquidity_store(self.dir) as store:
            self.assertEqual(store.version_count, self.manifest["version_count"])
            by_id: dict[tuple, LiquidityLevel] = {}
            for results in (self.swing, self.session):
                for result in results.values():
                    for revision in result.level_revisions:
                        by_id[(revision.level.level_id, revision.effective_at)] = revision.level
            seen = 0
            for version in range(store.version_count):
                level = store.level_at(version)
                formed_at, taken_at, effective_from, _until = store.times(version)
                original = by_id.get((level.level_id, __import__(
                    "tradelab.liquidity_store", fromlist=["_from_micros"]
                )._from_micros(effective_from)))
                if original is None:
                    continue
                seen += 1
                self.assertEqual(asdict(level), asdict(original), f"version {version} differs")
                # Prices bit-exact, not merely equal-comparing.
                self.assertEqual(struct.pack("=d", level.lower), struct.pack("=d", original.lower))
                self.assertEqual(struct.pack("=d", level.upper), struct.pack("=d", original.upper))
            self.assertGreater(seen, 0, "no versions were matched back to the fixture")


class LiquidityStoreLazinessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.swing, self.session = _fixture(seed=99)
        self.dir = self.tmp / "liquidity"
        write_liquidity_store(self.dir, self.swing, self.session, fingerprint="fp-lazy")

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_index_does_not_materialise_the_store(self):
        """The whole reason this exists. After many queries the reader must
        hold at most its bounded cache, not one object per version."""
        with open_liquidity_store(self.dir, cache_size=16) as store:
            mapped = MappedLiquidityTargetIndex(store)
            self.assertEqual(len(store._cache), 0, "opening the store materialised levels")
            for step in range(400):
                mapped.nearest(
                    side=HIGH if step % 2 else LOW,
                    entry_price=17000.0 + (step % 17) * 0.25,
                    entry_known_at=_T0 + timedelta(minutes=step),
                    count=3,
                )
            self.assertLessEqual(
                len(store._cache), 16,
                f"cache grew to {len(store._cache)} for a 16-entry bound",
            )
            self.assertGreater(store.version_count, 100, "fixture too small to prove anything")

    def test_cache_size_does_not_change_results(self):
        """A cache that could change an answer would be a correctness bug, not
        an optimisation."""
        probes = [
            (side, 17000.0 + step * 0.25, _T0 + timedelta(minutes=step * 7), count)
            for side in (HIGH, LOW) for step in range(12) for count in (1, 3)
        ]
        answers = []
        for cache_size in (1, 8, 4096):
            with open_liquidity_store(self.dir, cache_size=cache_size) as store:
                mapped = MappedLiquidityTargetIndex(store)
                answers.append([
                    [asdict(level) for level in mapped.nearest(
                        side=side, entry_price=price, entry_known_at=known_at, count=count
                    )]
                    for side, price, known_at, count in probes
                ])
        self.assertEqual(answers[0], answers[1])
        self.assertEqual(answers[0], answers[2])

    def test_returned_levels_are_immutable(self):
        with open_liquidity_store(self.dir) as store:
            level = store.level_at(0)
            with self.assertRaises(Exception):
                level.lower = 1.0  # frozen dataclass


class LiquidityStoreValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        swing, session = _fixture(seed=7, levels_per_tf=12)
        self.dir = self.tmp / "liquidity"
        write_liquidity_store(self.dir, swing, session, fingerprint="fp-v")

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_manifest_is_refused(self):
        (self.dir / "manifest.json").unlink()
        with self.assertRaisesRegex(LiquidityStoreError, "manifest"):
            open_liquidity_store(self.dir)

    def test_truncated_column_is_refused(self):
        path = self.dir / "versions.dat"
        path.write_bytes(path.read_bytes()[:-76])
        with self.assertRaisesRegex(LiquidityStoreError, "bytes"):
            open_liquidity_store(self.dir)

    def test_missing_column_is_refused(self):
        (self.dir / "high_prices.f8").unlink()
        with self.assertRaisesRegex(LiquidityStoreError, "missing"):
            open_liquidity_store(self.dir)

    def test_fingerprint_mismatch_is_refused(self):
        with self.assertRaisesRegex(LiquidityStoreError, "fingerprint"):
            open_liquidity_store(self.dir, expected_fingerprint="other")
        with open_liquidity_store(self.dir, expected_fingerprint="fp-v") as store:
            self.assertGreater(store.version_count, 0)

    def test_format_version_mismatch_is_refused(self):
        path = self.dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["format_version"] = "0.1"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(LiquidityStoreError, "format version"):
            open_liquidity_store(self.dir)

    def test_foreign_endianness_is_refused(self):
        path = self.dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["endianness"] = "big" if manifest["endianness"] == "little" else "little"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(LiquidityStoreError, "endian"):
            open_liquidity_store(self.dir)

    def test_corrupted_bytes_are_caught_by_the_checksum(self):
        path = self.dir / "versions.dat"
        blob = bytearray(path.read_bytes())
        blob[100] ^= 0xFF
        path.write_bytes(bytes(blob))
        # Length still matches, so only the checksum can see this.
        with self.assertRaisesRegex(LiquidityStoreError, "checksum"):
            open_liquidity_store(self.dir, verify_checksums=True)

    def test_no_temp_files_survive_a_successful_write(self):
        leftovers = [p.name for p in self.dir.iterdir() if ".tmp." in p.name]
        self.assertEqual(leftovers, [], f"temp files left behind: {leftovers}")

    def test_manifest_records_counts_dtypes_and_checksums(self):
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.assertEqual(manifest["format_version"], LIQUIDITY_STORE_FORMAT_VERSION)
        for key in ("version_count", "level_count", "high_count", "low_count",
                    "record_bytes", "record_format", "endianness", "bytes", "sha256"):
            self.assertIn(key, manifest)
        self.assertEqual(set(manifest["sha256"]), set(manifest["bytes"]))


if __name__ == "__main__":
    unittest.main()
