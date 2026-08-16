#!/usr/bin/env python3
"""Loader tests: local CSV always, local Parquet only when pyarrow is installed."""
import builtins
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tradelab.loaders import read_csv_bars, read_parquet_bars

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


CSV_HEADER = "timestamp,open,high,low,close,volume,contract\n"
CSV_ROW = "2026-01-05T14:30:00Z,100,100.5,99.5,100.2,10,NQZ25\n"


class CsvLoaderTests(unittest.TestCase):
    def test_reads_well_formed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            path.write_text(CSV_HEADER + CSV_ROW, encoding="utf-8")
            [bar] = read_csv_bars(path)
            self.assertEqual(bar.timestamp, datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc))
            self.assertEqual(bar.open, 100.0)
            self.assertEqual(bar.volume, 10.0)
            self.assertEqual(bar.contract, "NQZ25")

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            path.write_text("timestamp,open,high,low,close\n2026-01-05T14:30:00Z,1,1,1,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_csv_bars(path)

    def test_missing_contract_column_requires_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            path.write_text("timestamp,open,high,low,close,volume\n2026-01-05T14:30:00Z,1,1.1,0.9,1,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_csv_bars(path)
            [bar] = read_csv_bars(path, default_contract="NQZ25")
            self.assertEqual(bar.contract, "NQZ25")

    def test_naive_timestamp_without_utc_offset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            path.write_text(CSV_HEADER + "2026-01-05T14:30:00,100,100.5,99.5,100.2,10,NQZ25\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_csv_bars(path)

    def test_invalid_numeric_field_raises_with_row_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            path.write_text(CSV_HEADER + "2026-01-05T14:30:00Z,not-a-number,100.5,99.5,100.2,10,NQZ25\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_csv_bars(path)
            self.assertIn("row 2", str(ctx.exception).lower())


class ParquetImportGuardTests(unittest.TestCase):
    def test_missing_pyarrow_raises_clear_import_error(self):
        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("simulated missing pyarrow")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import
        try:
            with self.assertRaises(ImportError) as ctx:
                read_parquet_bars("does-not-matter.parquet")
            self.assertIn("pyarrow", str(ctx.exception))
            self.assertIn("pip install", str(ctx.exception))
        finally:
            builtins.__import__ = real_import


@unittest.skipUnless(HAS_PYARROW, "pyarrow is an optional dependency, not installed in this environment")
class ParquetLoaderTests(unittest.TestCase):
    def test_reads_well_formed_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "timestamp": pa.array([datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)], type=pa.timestamp("us", tz="UTC")),
                "open": [100.0],
                "high": [100.5],
                "low": [99.5],
                "close": [100.2],
                "volume": [10.0],
                "contract": ["NQZ25"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.parquet"
            pq.write_table(table, str(path))
            [bar] = read_parquet_bars(path)
            self.assertEqual(bar.timestamp, datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc))
            self.assertEqual(bar.close, 100.2)
            self.assertEqual(bar.contract, "NQZ25")

    def test_missing_required_column_raises(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.parquet"
            pq.write_table(table, str(path))
            with self.assertRaises(ValueError):
                read_parquet_bars(path)


if __name__ == "__main__":
    unittest.main()
