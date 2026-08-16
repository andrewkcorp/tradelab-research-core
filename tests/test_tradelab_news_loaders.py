#!/usr/bin/env python3
"""NewsEvent CSV/Parquet loader tests. No network access: local temp files only."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tradelab.news_loaders import read_news_events_csv, read_news_events_parquet

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


HEADER = "event_id,scheduled_at,country,currency,event_type,source_agency,source_url,source_reference,time_is_exact,schema_version\n"
ROW_1 = "CPI-2026-01,2026-01-13T13:30:00Z,United States,USD,CPI,BLS,https://www.bls.gov/cpi/,CPI-2026-01-REL,true,1.0\n"
ROW_2 = "FOMC-2026-01,2026-01-28T19:00:00Z,United States,USD,FOMC_RATE_DECISION,Federal Reserve,https://www.federalreserve.gov/,FOMC-2026-01-STMT,true,1.0\n"


class CsvLoaderTests(unittest.TestCase):
    def test_reads_well_formed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.csv"
            path.write_text(HEADER + ROW_1 + ROW_2, encoding="utf-8")
            events = read_news_events_csv(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].event_id, "CPI-2026-01")
            self.assertEqual(events[0].scheduled_at, datetime(2026, 1, 13, 13, 30, tzinfo=timezone.utc))
            self.assertEqual(events[1].event_type, "FOMC_RATE_DECISION")

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.csv"
            path.write_text("event_id,scheduled_at\nCPI-2026-01,2026-01-13T13:30:00Z\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_news_events_csv(path)

    def test_naive_scheduled_at_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.csv"
            path.write_text(HEADER + ROW_1.replace("2026-01-13T13:30:00Z", "2026-01-13T13:30:00"), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_news_events_csv(path)

    def test_duplicate_event_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.csv"
            duplicate_row = ROW_1.replace("2026-01-13T13:30:00Z", "2026-02-13T13:30:00Z")
            path.write_text(HEADER + ROW_1 + duplicate_row, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_news_events_csv(path)

    def test_unknown_event_type_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.csv"
            path.write_text(HEADER + ROW_1.replace(",CPI,", ",NOT_A_REAL_EVENT,"), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_news_events_csv(path)

    def test_non_us_country_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.csv"
            path.write_text(HEADER + ROW_1.replace("United States", "Canada"), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_news_events_csv(path)

    def test_time_is_exact_false_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.csv"
            path.write_text(HEADER + ROW_1.replace(",true,", ",false,"), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_news_events_csv(path)


class ParquetImportGuardTests(unittest.TestCase):
    def test_missing_pyarrow_raises_clear_import_error(self):
        import builtins

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("simulated missing pyarrow")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import
        try:
            with self.assertRaises(ImportError) as ctx:
                read_news_events_parquet("does-not-matter.parquet")
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
                "event_id": ["CPI-2026-01"],
                "scheduled_at": pa.array([datetime(2026, 1, 13, 13, 30, tzinfo=timezone.utc)], type=pa.timestamp("us", tz="UTC")),
                "country": ["United States"],
                "currency": ["USD"],
                "event_type": ["CPI"],
                "source_agency": ["BLS"],
                "source_url": ["https://www.bls.gov/cpi/"],
                "source_reference": ["CPI-2026-01-REL"],
                "time_is_exact": [True],
                "schema_version": ["1.0"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.parquet"
            pq.write_table(table, str(path))
            events = read_news_events_parquet(path)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_id, "CPI-2026-01")
            self.assertEqual(events[0].scheduled_at, datetime(2026, 1, 13, 13, 30, tzinfo=timezone.utc))

    def test_duplicate_event_id_raises(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "event_id": ["CPI-2026-01", "CPI-2026-01"],
                "scheduled_at": pa.array(
                    [datetime(2026, 1, 13, 13, 30, tzinfo=timezone.utc), datetime(2026, 2, 13, 13, 30, tzinfo=timezone.utc)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "country": ["United States", "United States"],
                "currency": ["USD", "USD"],
                "event_type": ["CPI", "PPI"],
                "source_agency": ["BLS", "BLS"],
                "source_url": ["https://www.bls.gov/cpi/", "https://www.bls.gov/ppi/"],
                "source_reference": ["ref1", "ref2"],
                "time_is_exact": [True, True],
                "schema_version": ["1.0", "1.0"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.parquet"
            pq.write_table(table, str(path))
            with self.assertRaises(ValueError):
                read_news_events_parquet(path)

    def test_missing_required_column_raises(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"event_id": ["x"], "country": ["United States"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.parquet"
            pq.write_table(table, str(path))
            with self.assertRaises(ValueError):
                read_news_events_parquet(path)


if __name__ == "__main__":
    unittest.main()
