#!/usr/bin/env python3
"""tradelab.diagnose_context_preparation tests: the deterministic,
contiguous prefix selector always returns exactly the first N bars
(never a random sample, never re-sorted), rejects a non-positive or
too-large N rather than silently truncating, ``diagnose_one_prefix``
runs prepare_backtest_context and returns a real context without
writing anything, and ``main()`` never writes any file, handles
multiple comma-separated prefixes against one shared artifact load, and
propagates a too-large prefix as a clean exit-code-1 error. No real
market data anywhere -- every fixture is a tiny synthetic CSV/JSON
written to a temporary directory."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradelab.backtest import PreparedBacktestContext
from tradelab.bar import Bar1m
from tradelab.diagnose_context_preparation import diagnose_one_prefix, main, select_deterministic_prefix

UTC = timezone.utc

NEWS_HEADER = "event_id,scheduled_at,country,currency,event_type,source_agency,source_url,source_reference,time_is_exact,schema_version\n"
NEWS_ROW = "CPI-2024-01,2024-01-02T14:30:00Z,United States,USD,CPI,BLS,https://bls.gov/cpi,ref-1,true,1.0\n"


def _make_bars(count: int, base: datetime = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)) -> list[Bar1m]:
    return [
        Bar1m(
            timestamp=base + timedelta(minutes=i),
            open=100.0 + i * 0.1,
            high=100.5 + i * 0.1,
            low=99.5 + i * 0.1,
            close=100.2 + i * 0.1,
            volume=10.0,
            contract="MNQ_CONTINUOUS",
        )
        for i in range(count)
    ]


def _bars_to_csv(bars: list[Bar1m]) -> str:
    lines = ["timestamp,open,high,low,close,volume,contract\n"]
    for b in bars:
        lines.append(f"{b.timestamp.isoformat().replace('+00:00', 'Z')},{b.open},{b.high},{b.low},{b.close},{b.volume},{b.contract}\n")
    return "".join(lines)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(tmp: Path, bar_count: int) -> tuple[Path, Path]:
    """Writes a small, internally-consistent continuous CSV (``bar_count``
    1m bars) + manifest + valid news CSV to ``tmp``; returns
    ``(manifest_path, news_path)``."""
    bars = _make_bars(bar_count)
    csv_path = tmp / "mnq_continuous_1m.csv"
    csv_path.write_text(_bars_to_csv(bars), encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "source_name": "test-source",
        "instrument_root": "MNQ",
        "bar_interval": "1m",
        "timestamp_timezone": "UTC",
        "series_kind": "continuous",
        "data_path": csv_path.name,
        "coverage_start": bars[0].timestamp.isoformat().replace("+00:00", "Z"),
        "coverage_end": (bars[-1].timestamp + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "sha256": sha256_of(csv_path),
        "roll_manifest_path": None,
        "normalization_method": "volume_front_month_v1+additive_back_adjustment",
    }
    manifest_path = tmp / "continuous_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    news_path = tmp / "news.csv"
    news_path.write_text(NEWS_HEADER + NEWS_ROW, encoding="utf-8")
    return manifest_path, news_path


class SelectDeterministicPrefixTests(unittest.TestCase):
    def test_returns_exactly_the_first_n_bars_in_existing_order(self):
        bars = tuple(_make_bars(10))
        prefix = select_deterministic_prefix(bars, 4)
        self.assertEqual(prefix, bars[:4])
        self.assertEqual(len(prefix), 4)

    def test_prefix_is_contiguous_and_never_reordered(self):
        bars = tuple(_make_bars(10))
        prefix = select_deterministic_prefix(bars, 5)
        self.assertEqual([b.timestamp for b in prefix], [b.timestamp for b in bars[:5]])

    def test_repeated_calls_are_byte_for_byte_identical(self):
        bars = tuple(_make_bars(20))
        first = select_deterministic_prefix(bars, 7)
        second = select_deterministic_prefix(bars, 7)
        self.assertEqual(first, second)

    def test_prefix_equal_to_full_length_returns_everything(self):
        bars = tuple(_make_bars(6))
        self.assertEqual(select_deterministic_prefix(bars, 6), bars)

    def test_zero_or_negative_prefix_bars_raises(self):
        bars = tuple(_make_bars(6))
        with self.assertRaises(ValueError):
            select_deterministic_prefix(bars, 0)
        with self.assertRaises(ValueError):
            select_deterministic_prefix(bars, -1)

    def test_prefix_bars_exceeding_available_count_raises_rather_than_truncating(self):
        bars = tuple(_make_bars(6))
        with self.assertRaises(ValueError):
            select_deterministic_prefix(bars, 7)


class DiagnoseOnePrefixTests(unittest.TestCase):
    def test_returns_a_real_prepared_context_sized_to_the_prefix(self):
        bars = tuple(_make_bars(30))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            context = diagnose_one_prefix(bars, (), 10)

        self.assertIsInstance(context, PreparedBacktestContext)
        self.assertEqual(len(context.bars_1m), 10)
        self.assertEqual(context.bars_1m, bars[:10])

    def test_prints_phase_timings_and_summary_counts(self):
        bars = tuple(_make_bars(20))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            diagnose_one_prefix(bars, (), 10)
        output = buffer.getvalue()

        self.assertIn("prefix_bars=10", output)
        self.assertIn("1m_validation_loading", output)
        self.assertIn("timeframe_aggregation", output)
        self.assertIn("fvg_detection:1m", output)
        self.assertIn("liquidity_detection:swing_equal:1m", output)
        self.assertIn("liquidity_detection:session:1m", output)
        self.assertIn("data_wick_detection", output)
        self.assertIn("htf_bias_construction", output)
        self.assertIn("entry_signal_timeline_construction", output)
        self.assertIn("1m bars (prefix):            10", output)


class MainCLITests(unittest.TestCase):
    def test_never_writes_any_file(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_fixture(tmp, 20)
            before = sorted(p.name for p in tmp.iterdir())

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["--market-artifact", str(manifest_path), "--news-csv", str(news_path), "--prefix-bars", "10"])

            after = sorted(p.name for p in tmp.iterdir())
            self.assertEqual(exit_code, 0)
            self.assertEqual(before, after, "diagnose_context_preparation must never write any artifact")

    def test_multiple_comma_separated_prefixes_are_each_diagnosed_once(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_fixture(tmp, 30)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["--market-artifact", str(manifest_path), "--news-csv", str(news_path), "--prefix-bars", "5,10,20"])
            output = buffer.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertEqual(output.count("--- prefix_bars=5 "), 1)
            self.assertEqual(output.count("--- prefix_bars=10 "), 1)
            self.assertEqual(output.count("--- prefix_bars=20 "), 1)
            self.assertIn("artifact load:", output)
            # Exactly one artifact load line -- the (potentially slow) full
            # series load must happen once, shared across every prefix.
            self.assertEqual(output.count("artifact load:"), 1)

    def test_prefix_bars_exceeding_available_count_is_a_clean_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_fixture(tmp, 10)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["--market-artifact", str(manifest_path), "--news-csv", str(news_path), "--prefix-bars", "999"])

            self.assertEqual(exit_code, 1)

    def test_non_integer_prefix_bars_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            manifest_path, news_path = write_fixture(tmp, 10)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(["--market-artifact", str(manifest_path), "--news-csv", str(news_path), "--prefix-bars", "not-a-number"])

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
