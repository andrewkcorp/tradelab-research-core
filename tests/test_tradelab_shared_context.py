#!/usr/bin/env python3
"""A memory-mapped context must reproduce the bars exactly, or refuse to open.

Two properties, and they pull in opposite directions:

* **exactness** -- a bar read back through the mapping is indistinguishable
  from the original, field by field, including float bit patterns and the UTC
  instant. Anything less silently changes what the backtest simulates.
* **fail-closed** -- a store that is missing, truncated, half-written, built on
  a different-endian machine, or built from different inputs must raise before
  anything is mapped. A run that read the wrong context would produce
  plausible, wrong results, which is far worse than failing to start.
"""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradelab.bar import Bar1m
from tradelab.shared_context import (
    SHARED_CONTEXT_FORMAT_VERSION,
    SharedContextError,
    open_shared_context,
    shared_context_bytes,
    write_shared_context,
)


def _bars(count: int = 500, contract: str = "MNQZ24") -> list[Bar1m]:
    """Prices chosen to be awkward: values that are not exact tick multiples and
    do not survive a naive round-trip through a scaled integer."""
    start = datetime(2024, 6, 3, 13, 30, tzinfo=timezone.utc)
    out = []
    for i in range(count):
        base = 17234.25 + i * 0.37109375 + (0.0000001 if i % 7 == 0 else 0.0)
        out.append(Bar1m(
            timestamp=start + timedelta(minutes=i),
            open=base, high=base + 3.7, low=base - 2.9, close=base + 0.125,
            volume=float(i) + 0.5,
            contract=contract if i % 3 else "MNQH25",
        ))
    return out


class SharedContextRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.bars = _bars()
        cls.dir = cls.tmp / "simctx"
        cls.manifest = write_shared_context(cls.dir, cls.bars, fingerprint="fp-abc")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_bar_round_trips_exactly(self):
        with open_shared_context(self.dir) as reader:
            self.assertEqual(len(reader.bars), len(self.bars))
            for index, original in enumerate(self.bars):
                got = reader.bars[index]
                self.assertEqual(got, original, f"bar {index} differs")
                # Equality on a float dataclass is value equality; assert the
                # exact bit patterns too, so a lossy conversion cannot hide
                # behind a comparison that happens to round the same way.
                for field in ("open", "high", "low", "close", "volume"):
                    self.assertEqual(
                        struct.pack("=d", getattr(got, field)),
                        struct.pack("=d", getattr(original, field)),
                        f"bar {index}.{field} is not bit-identical",
                    )
                self.assertEqual(got.timestamp, original.timestamp)
                self.assertEqual(got.timestamp.utcoffset(), timedelta(0))
                self.assertEqual(got.contract, original.contract)

    def test_the_fixture_would_catch_a_tick_grid_rounding_bug(self):
        """The round-trip test only proves something if the prices are hostile
        to a scaled-integer representation."""
        tick = 0.25
        off_grid = [b for b in self.bars if abs(b.open / tick - round(b.open / tick)) > 1e-9]
        self.assertGreater(len(off_grid), len(self.bars) // 2,
                           "fixture prices sit on the tick grid; it cannot detect rounding")

    def test_negative_indexing_and_slicing_match_the_original_sequence(self):
        with open_shared_context(self.dir) as reader:
            self.assertEqual(reader.bars[-1], self.bars[-1])
            self.assertEqual(reader.bars[-len(self.bars)], self.bars[0])
            self.assertEqual(reader.bars[10:15], self.bars[10:15])
            with self.assertRaises(IndexError):
                reader.bars[len(self.bars)]

    def test_columnar_accessors_agree_with_the_materialised_bars(self):
        """Index building reads columns directly; it must see the same numbers
        the bar objects would report, or the index and the bars disagree."""
        with open_shared_context(self.dir) as reader:
            highs = list(reader.bars.highs())
            lows = list(reader.bars.lows())
            self.assertEqual(highs, [b.high for b in self.bars])
            self.assertEqual(lows, [b.low for b in self.bars])
            self.assertEqual(list(reader.bars.timestamps()), [b.timestamp for b in self.bars])
            for index in (0, 1, len(self.bars) // 2, len(self.bars) - 1):
                self.assertEqual(reader.bars.high_at(index), self.bars[index].high)
                self.assertEqual(reader.bars.low_at(index), self.bars[index].low)
                self.assertEqual(reader.bars.timestamp_at(index), self.bars[index].timestamp)

    def test_materialised_bars_are_not_retained(self):
        """The whole point is that bars are temporary. Two reads of the same
        index must produce equal but distinct objects, so nothing is being
        cached into a resident million."""
        with open_shared_context(self.dir) as reader:
            first, second = reader.bars[7], reader.bars[7]
            self.assertEqual(first, second)
            self.assertIsNot(first, second)

    def test_store_is_far_smaller_than_the_objects_it_replaces(self):
        sizes = shared_context_bytes(self.dir)
        # 8 (timestamp) + 40 (ohlcv) + 2 (contract) bytes per bar, exactly.
        self.assertEqual(sizes["total"], len(self.bars) * 50)

    def test_manifest_records_what_a_reader_must_verify(self):
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.assertEqual(manifest["format_version"], SHARED_CONTEXT_FORMAT_VERSION)
        self.assertEqual(manifest["fingerprint"], "fp-abc")
        self.assertIn(manifest["endianness"], ("little", "big"))
        self.assertEqual(manifest["bar_count"], len(self.bars))
        for column in ("bars.timestamp", "bars.ohlcv", "bars.contract"):
            spec = manifest["columns"][column]
            self.assertIn("dtype", spec)
            self.assertIn("itemsize", spec)
            self.assertIn("items_per_bar", spec)

    def test_contract_strings_are_pooled_not_repeated_per_bar(self):
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.assertEqual(sorted(manifest["contracts"]), ["MNQH25", "MNQZ24"])
        # Two distinct contracts, 500 bars, two bytes each -- the strings live
        # in the manifest, not in the column.
        self.assertEqual((self.dir / "bars.contract").stat().st_size, len(self.bars) * 2)


class BarRangeIndexOnSharedBarsTests(unittest.TestCase):
    """The index is the hot path. Built over mapped columns it must be the
    same index, query for query, as one built over the original objects."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.bars = _bars(700)
        cls.dir = cls.tmp / "simctx"
        write_shared_context(cls.dir, cls.bars, fingerprint="fp-idx")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_index_over_mapped_columns_answers_identically(self):
        from tradelab.bar_range_index import BarRangeIndex

        reference = BarRangeIndex.build(self.bars)
        with open_shared_context(self.dir) as reader:
            mapped = BarRangeIndex.build(reader.bars)

            self.assertEqual(mapped.bar_count, reference.bar_count)
            # The segment trees are the index. Identical trees mean identical
            # answers for every possible query, not just the sampled ones.
            self.assertEqual(mapped._highs_max, reference._highs_max)
            self.assertEqual(mapped._lows_min, reference._lows_min)

            # And the public queries, exercised against real thresholds drawn
            # from the data so hits and misses both occur.
            probes = [self.bars[i].high for i in (0, 3, 51, 340, 699)]
            probes += [self.bars[i].low for i in (7, 120, 500)]
            probes += [min(b.low for b in self.bars) - 1.0, max(b.high for b in self.bars) + 1.0]
            for lo, hi in ((0, len(self.bars)), (10, 400), (399, 400), (500, 700)):
                self.assertEqual(mapped.range_max_high(lo, hi), reference.range_max_high(lo, hi))
                self.assertEqual(mapped.range_min_low(lo, hi), reference.range_min_low(lo, hi))
                for threshold in probes:
                    self.assertEqual(
                        mapped.first_high_at_least(lo, hi, threshold),
                        reference.first_high_at_least(lo, hi, threshold),
                        f"first_high_at_least({lo},{hi},{threshold})",
                    )
                    self.assertEqual(
                        mapped.first_low_at_most(lo, hi, threshold),
                        reference.first_low_at_most(lo, hi, threshold),
                        f"first_low_at_most({lo},{hi},{threshold})",
                    )

    def test_timestamp_lookup_agrees_including_between_bars_and_past_the_end(self):
        from tradelab.bar_range_index import BarRangeIndex

        reference = BarRangeIndex.build(self.bars)
        with open_shared_context(self.dir) as reader:
            mapped = BarRangeIndex.build(reader.bars)
            probes = [self.bars[0].timestamp, self.bars[1].timestamp, self.bars[-1].timestamp,
                      self.bars[0].timestamp - timedelta(minutes=5),
                      self.bars[0].timestamp + timedelta(seconds=30),
                      self.bars[-1].timestamp + timedelta(minutes=1)]
            for probe in probes:
                self.assertEqual(
                    mapped.find_bar_at_or_after(probe),
                    reference.find_bar_at_or_after(probe),
                    f"find_bar_at_or_after({probe})",
                )

    def test_the_index_does_not_retain_a_datetime_per_bar(self):
        """The reason for the columnar path. A tuple of a million datetimes is
        exactly the per-worker cost this change exists to remove."""
        from tradelab.bar_range_index import BarRangeIndex

        with open_shared_context(self.dir) as reader:
            mapped = BarRangeIndex.build(reader.bars)
            self.assertNotIsInstance(mapped.timestamps, tuple)
            # Still an ordered sequence that bisect can search.
            self.assertEqual(len(mapped.timestamps), len(self.bars))
            self.assertEqual(mapped.timestamps[0], self.bars[0].timestamp)
            self.assertEqual(mapped.timestamps[-1], self.bars[-1].timestamp)
            # Decoded on access, never cached.
            self.assertIsNot(mapped.timestamps[5], mapped.timestamps[5])

        plain = BarRangeIndex.build(self.bars)
        self.assertIsInstance(plain.timestamps, tuple)


class SharedContextValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.dir = self.tmp / "simctx"
        self.bars = _bars(64)
        write_shared_context(self.dir, self.bars, fingerprint="fp-abc")

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_manifest_is_refused(self):
        (self.dir / "manifest.json").unlink()
        with self.assertRaisesRegex(SharedContextError, "manifest"):
            open_shared_context(self.dir)

    def test_an_unfinished_write_is_detectable_because_the_manifest_comes_last(self):
        """Columns without a manifest is exactly the state an interrupted write
        leaves behind, and it must not look like an empty-but-valid store."""
        (self.dir / "manifest.json").unlink()
        self.assertTrue((self.dir / "bars.ohlcv").is_file())
        with self.assertRaises(SharedContextError):
            open_shared_context(self.dir)

    def test_truncated_column_is_refused(self):
        path = self.dir / "bars.ohlcv"
        path.write_bytes(path.read_bytes()[:-8])
        with self.assertRaisesRegex(SharedContextError, "truncated|bytes, expected"):
            open_shared_context(self.dir)

    def test_missing_column_is_refused(self):
        (self.dir / "bars.timestamp").unlink()
        with self.assertRaisesRegex(SharedContextError, "missing"):
            open_shared_context(self.dir)

    def test_fingerprint_mismatch_is_refused(self):
        with self.assertRaisesRegex(SharedContextError, "fingerprint"):
            open_shared_context(self.dir, expected_fingerprint="fp-other")
        # And the matching fingerprint still opens.
        with open_shared_context(self.dir, expected_fingerprint="fp-abc") as reader:
            self.assertEqual(len(reader.bars), len(self.bars))

    def test_format_version_mismatch_is_refused_rather_than_guessed(self):
        path = self.dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["format_version"] = "0.9"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SharedContextError, "format version"):
            open_shared_context(self.dir)

    def test_foreign_endianness_is_refused(self):
        path = self.dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["endianness"] = "big" if manifest["endianness"] == "little" else "little"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SharedContextError, "endian"):
            open_shared_context(self.dir)

    def test_bar_count_disagreeing_with_column_length_is_refused(self):
        """A manifest claiming more bars than the columns hold must fail, not
        read past the end or silently serve fewer."""
        path = self.dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["bar_count"] = len(self.bars) + 10
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SharedContextError, "bytes, expected"):
            open_shared_context(self.dir)

    def test_no_temp_files_survive_a_successful_write(self):
        leftovers = [p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [], f"temp files left behind: {leftovers}")

    def test_empty_series_round_trips(self):
        empty = self.tmp / "empty"
        write_shared_context(empty, [], fingerprint="fp-empty")
        with open_shared_context(empty) as reader:
            self.assertEqual(len(reader.bars), 0)
            with self.assertRaises(IndexError):
                reader.bars[0]


if __name__ == "__main__":
    unittest.main()
