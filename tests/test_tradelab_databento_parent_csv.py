#!/usr/bin/env python3
"""Databento parent-symbology CSV parser tests: the exact real column
layout parses correctly, parallel contracts sharing the same timestamp
are preserved (never deduplicated), rtype/columns/timestamp-format
violations are rejected with clear errors, and the file must be
non-decreasing in ts_event overall (ties across symbols are fine, a
genuine regression is not)."""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tradelab.databento_parent_csv import DatabentoParentRecord, read_databento_parent_csv

UTC = timezone.utc

HEADER = "ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume,symbol\n"
REAL_ROW = "2023-08-18T00:00:00.000000000Z,33,1,34070,14938.000000000,14938.000000000,14930.000000000,14930.000000000,9,MNQZ3\n"


def write(tmp: Path, name: str, content: str) -> Path:
    path = tmp / name
    path.write_text(content, encoding="utf-8")
    return path


class RealColumnLayoutTests(unittest.TestCase):
    def test_parses_the_real_export_row_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "raw.csv", HEADER + REAL_ROW)
            [record] = read_databento_parent_csv(path)
            self.assertEqual(record.ts_event, datetime(2023, 8, 18, 0, 0, tzinfo=UTC))
            self.assertEqual(record.rtype, 33)
            self.assertEqual(record.publisher_id, 1)
            self.assertEqual(record.instrument_id, 34070)
            self.assertEqual(record.open, 14938.0)
            self.assertEqual(record.high, 14938.0)
            self.assertEqual(record.low, 14930.0)
            self.assertEqual(record.close, 14930.0)
            self.assertEqual(record.volume, 9.0)
            self.assertEqual(record.symbol, "MNQZ3")

    def test_column_order_case_insensitive_matching(self):
        # header uses different case and order -- csv.DictReader + our own
        # lowercase lookup should still resolve every column correctly.
        header = "SYMBOL,Volume,Close,Low,High,Open,Instrument_Id,Publisher_Id,RType,TS_EVENT\n"
        row = "MNQZ3,9,14930.000000000,14930.000000000,14938.000000000,14938.000000000,34070,1,33,2023-08-18T00:00:00.000000000Z\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "raw.csv", header + row)
            [record] = read_databento_parent_csv(path)
            self.assertEqual(record.symbol, "MNQZ3")
            self.assertEqual(record.close, 14930.0)


class ParallelContractsTests(unittest.TestCase):
    def test_identical_timestamps_across_symbols_are_preserved_not_deduplicated(self):
        rows = (
            "2023-08-18T00:00:00.000000000Z,33,1,1,100,100,100,100.0,10,MNQZ3\n"
            "2023-08-18T00:00:00.000000000Z,33,1,2,105,105,105,105.0,5,MNQU3\n"
            "2023-08-18T00:00:00.000000000Z,33,1,3,110,110,110,110.0,2,MNQH4\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "raw.csv", HEADER + rows)
            records = read_databento_parent_csv(path)
            self.assertEqual(len(records), 3)
            self.assertEqual({r.symbol for r in records}, {"MNQZ3", "MNQU3", "MNQH4"})
            self.assertTrue(all(r.ts_event == datetime(2023, 8, 18, 0, 0, tzinfo=UTC) for r in records))


class ValidationTests(unittest.TestCase):
    def test_missing_required_column_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "raw.csv", "ts_event,rtype,open,high,low,close,volume,symbol\n")
            with self.assertRaises(ValueError):
                read_databento_parent_csv(path)

    def test_wrong_rtype_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_row = REAL_ROW.replace(",33,", ",34,")
            path = write(Path(tmp), "raw.csv", HEADER + bad_row)
            with self.assertRaises(ValueError) as ctx:
                read_databento_parent_csv(path)
            self.assertIn("rtype", str(ctx.exception))

    def test_missing_utc_z_suffix_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_row = REAL_ROW.replace("Z,33", ",33")
            path = write(Path(tmp), "raw.csv", HEADER + bad_row)
            with self.assertRaises(ValueError):
                read_databento_parent_csv(path)

    def test_missing_nanosecond_fraction_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_row = "2023-08-18T00:00:00Z,33,1,34070,14938,14938,14930,14930,9,MNQZ3\n"
            path = write(Path(tmp), "raw.csv", HEADER + bad_row)
            with self.assertRaises(ValueError):
                read_databento_parent_csv(path)

    def test_non_minute_aligned_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_row = REAL_ROW.replace("00:00:00.000000000Z", "00:00:30.000000000Z")
            path = write(Path(tmp), "raw.csv", HEADER + bad_row)
            with self.assertRaises(ValueError):
                read_databento_parent_csv(path)

    def test_impossible_ohlc_range_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_row = "2023-08-18T00:00:00.000000000Z,33,1,34070,14938,14920,14930,14930,9,MNQZ3\n"  # low > open
            path = write(Path(tmp), "raw.csv", HEADER + bad_row)
            with self.assertRaises(ValueError):
                read_databento_parent_csv(path)

    def test_negative_volume_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_row = REAL_ROW.replace(",9,MNQZ3", ",-1,MNQZ3")
            path = write(Path(tmp), "raw.csv", HEADER + bad_row)
            with self.assertRaises(ValueError):
                read_databento_parent_csv(path)

    def test_non_numeric_price_rejected_with_row_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_row = REAL_ROW.replace("14938.000000000,14938", "not-a-number,14938", 1)
            path = write(Path(tmp), "raw.csv", HEADER + bad_row)
            with self.assertRaises(ValueError) as ctx:
                read_databento_parent_csv(path)
            self.assertIn("row 2", str(ctx.exception).lower())

    def test_empty_file_has_no_header_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "raw.csv", "")
            with self.assertRaises(ValueError):
                read_databento_parent_csv(path)

    def test_decreasing_ts_event_rejected(self):
        rows = (
            "2023-08-18T00:01:00.000000000Z,33,1,1,100,100,100,100.0,10,MNQZ3\n"
            "2023-08-18T00:00:00.000000000Z,33,1,2,105,105,105,105.0,5,MNQU3\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "raw.csv", HEADER + rows)
            with self.assertRaises(ValueError):
                read_databento_parent_csv(path)

    def test_equal_consecutive_ts_event_is_allowed(self):
        rows = (
            "2023-08-18T00:00:00.000000000Z,33,1,1,100,100,100,100.0,10,MNQZ3\n"
            "2023-08-18T00:00:00.000000000Z,33,1,2,105,105,105,105.0,5,MNQU3\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "raw.csv", HEADER + rows)
            records = read_databento_parent_csv(path)  # no raise
            self.assertEqual(len(records), 2)


class DatabentoParentRecordValidationTests(unittest.TestCase):
    def test_naive_ts_event_rejected(self):
        with self.assertRaises(ValueError):
            DatabentoParentRecord(
                ts_event=datetime(2023, 8, 18, 0, 0), rtype=33, publisher_id=1, instrument_id=1,
                open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0, symbol="MNQZ3",
            )

    def test_wrong_rtype_rejected(self):
        with self.assertRaises(ValueError):
            DatabentoParentRecord(
                ts_event=datetime(2023, 8, 18, 0, 0, tzinfo=UTC), rtype=1, publisher_id=1, instrument_id=1,
                open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0, symbol="MNQZ3",
            )

    def test_empty_symbol_rejected(self):
        with self.assertRaises(ValueError):
            DatabentoParentRecord(
                ts_event=datetime(2023, 8, 18, 0, 0, tzinfo=UTC), rtype=33, publisher_id=1, instrument_id=1,
                open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0, symbol="",
            )


if __name__ == "__main__":
    unittest.main()
