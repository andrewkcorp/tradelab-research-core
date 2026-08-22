#!/usr/bin/env python3
"""Market data artifact tests: a valid single continuous artifact loads
cleanly with no critical findings; a checksum mismatch is a critical
finding that blocks the pipeline; a raw_quarterly manifest without a
roll_manifest_path is rejected at construction; a duplicate timestamp
is caught (via tradelab.validation.validate_bars, reused not
reimplemented) as a critical finding; the audit report is fully
deterministic for identical input; and the raw_quarterly pipeline path
calls tradelab.continuous_futures.build_back_adjusted_continuous_series
correctly, producing the same result as calling it directly. No real
market data is used anywhere -- every fixture is a tiny synthetic CSV
written to a temporary directory."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradelab.accounting import Instrument
from tradelab.bar import Bar1m
from tradelab.continuous_futures import RollAdjustment, build_back_adjusted_continuous_series
from tradelab.market_data_artifact import (
    AuditSeverity,
    MarketDataArtifact,
    SeriesKind,
    audit_market_data_artifact,
    load_market_data_artifact,
    load_market_data_series,
    read_roll_manifest,
)

UTC = timezone.utc
T0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)

CONTINUOUS_CSV = (
    "timestamp,open,high,low,close,volume,contract\n"
    "2024-01-02T14:30:00Z,100,101,99,100.5,10,NQ_CONTINUOUS\n"
    "2024-01-02T14:31:00Z,100.5,101.5,99.5,101,12,NQ_CONTINUOUS\n"
)

RAW_QUARTERLY_CSV = (
    "timestamp,open,high,low,close,volume,contract\n"
    "2024-01-02T14:30:00Z,100,101,99,100.5,10,NQZ24\n"
    "2024-01-02T14:31:00Z,100.5,101.5,99.5,101,12,NQZ24\n"
    "2024-01-02T14:32:00Z,150,151,149,150.5,11,NQH25\n"
)

DUPLICATE_TIMESTAMP_CSV = (
    "timestamp,open,high,low,close,volume,contract\n"
    "2024-01-02T14:30:00Z,100,101,99,100.5,10,NQ_CONTINUOUS\n"
    "2024-01-02T14:30:00Z,100,101,99,100.5,10,NQ_CONTINUOUS\n"
)

OFF_TICK_CSV = (
    "timestamp,open,high,low,close,volume,contract\n"
    "2024-01-02T14:30:00Z,100,101,99,100.2,10,NQ_CONTINUOUS\n"
)

ROLL_MANIFEST = [
    {"outgoing_contract": "NQZ24", "incoming_contract": "NQH25", "effective_at": "2024-01-02T14:32:00Z", "adjustment_points": 50.0}
]

DATABENTO_HEADER = "ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume,symbol\n"
# Session boundaries (EST, UTC-5): 2024-01-15 -> [2024-01-15T23:00Z, ...),
# 2024-01-16 -> [2024-01-16T23:00Z, ...), 2024-01-17 -> [2024-01-17T23:00Z, 2024-01-18T22:00Z).
# S0 (warm-up): MNQZ3 wins volume (30 vs 3) -> S1 selects MNQZ3.
# S1: MNQU3 wins volume (80 vs 20) -> S2 selects MNQU3 (transition, adjustment = 106.5-101.5 = 5.0).
# The trailing ESZ3 row (a non-outright symbol, never a selection candidate)
# is dated well past S2's own nominal end so S2 itself counts as fully
# covered by the file's own data range and isn't excluded as a trailing
# partial session -- see tradelab.volume_roll_continuous's coverage policy.
DATABENTO_PARENT_CSV = (
    DATABENTO_HEADER
    + "2024-01-15T23:00:00.000000000Z,33,1,1,100,100,100,100.0,20,MNQZ3\n"
    + "2024-01-15T23:00:00.000000000Z,33,1,2,105,105,105,105.0,2,MNQU3\n"
    + "2024-01-15T23:01:00.000000000Z,33,1,1,100.5,100.5,100.5,100.5,10,MNQZ3\n"
    + "2024-01-15T23:01:00.000000000Z,33,1,2,105.5,105.5,105.5,105.5,1,MNQU3\n"
    + "2024-01-16T23:00:00.000000000Z,33,1,1,101,101,101,101.0,10,MNQZ3\n"
    + "2024-01-16T23:00:00.000000000Z,33,1,2,106,106,106,106.0,40,MNQU3\n"
    + "2024-01-16T23:01:00.000000000Z,33,1,1,101.5,101.5,101.5,101.5,10,MNQZ3\n"
    + "2024-01-16T23:01:00.000000000Z,33,1,2,106.5,106.5,106.5,106.5,40,MNQU3\n"
    + "2024-01-17T23:00:00.000000000Z,33,1,2,107,107,107,107.0,50,MNQU3\n"
    + "2024-01-19T23:00:00.000000000Z,33,1,3,1,1,1,1.0,1,ESZ3\n"
)


def databento_artifact(sha256: str, *, data_path: str = "mnq_parent.csv") -> MarketDataArtifact:
    return MarketDataArtifact(
        schema_version="1.0",
        source_name="databento-export",
        instrument_root=Instrument.MNQ,
        bar_interval="1m",
        timestamp_timezone="UTC",
        series_kind=SeriesKind.DATABENTO_PARENT_CSV,
        data_path=data_path,
        coverage_start=datetime(2024, 1, 16, 23, 0, tzinfo=UTC),
        coverage_end=datetime(2024, 1, 17, 23, 1, tzinfo=UTC),
        sha256=sha256,
        normalization_method="volume_front_month_v1",
    )


def write(tmp: Path, name: str, content: str) -> Path:
    path = tmp / name
    path.write_text(content, encoding="utf-8")
    return path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def continuous_artifact(sha256: str, *, coverage_end: datetime = T0 + timedelta(minutes=2), data_path: str = "bars.csv") -> MarketDataArtifact:
    return MarketDataArtifact(
        schema_version="1.0",
        source_name="local-export",
        instrument_root=Instrument.NQ,
        bar_interval="1m",
        timestamp_timezone="UTC",
        series_kind=SeriesKind.CONTINUOUS,
        data_path=data_path,
        coverage_start=T0,
        coverage_end=coverage_end,
        sha256=sha256,
        normalization_method="additive_back_adjustment",
    )


def raw_quarterly_artifact(sha256: str, *, roll_manifest_path: str | None = "rolls.json", data_path: str = "bars.csv") -> MarketDataArtifact:
    return MarketDataArtifact(
        schema_version="1.0",
        source_name="local-export",
        instrument_root=Instrument.NQ,
        bar_interval="1m",
        timestamp_timezone="UTC",
        series_kind=SeriesKind.RAW_QUARTERLY,
        data_path=data_path,
        coverage_start=T0,
        coverage_end=T0.replace(minute=33),
        sha256=sha256,
        roll_manifest_path=roll_manifest_path,
    )


class ValidContinuousArtifactTests(unittest.TestCase):
    def test_valid_single_continuous_artifact_loads_with_no_critical_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", CONTINUOUS_CSV)
            artifact = continuous_artifact(sha256_of(csv_path))

            result = load_market_data_series(artifact, base_dir=tmp)

            self.assertEqual(len(result.bars), 2)
            self.assertTrue(all(isinstance(b, Bar1m) for b in result.bars))
            self.assertTrue(all(b.contract == "NQ_CONTINUOUS" for b in result.bars))
            self.assertFalse(result.audit_report.has_critical_findings)
            self.assertTrue(result.audit_report.sha256_match)
            self.assertIsNone(result.continuous_series_result)

    def test_manifest_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", CONTINUOUS_CSV)
            artifact = continuous_artifact(sha256_of(csv_path))
            manifest_path = write(tmp, "manifest.json", json.dumps(artifact.to_dict()))

            loaded = load_market_data_artifact(manifest_path)
            self.assertEqual(loaded, artifact)


class BadChecksumTests(unittest.TestCase):
    def test_checksum_mismatch_is_critical_and_blocks_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write(tmp, "bars.csv", CONTINUOUS_CSV)
            artifact = continuous_artifact("0" * 64)  # deliberately wrong

            report = audit_market_data_artifact(artifact, [], base_dir=tmp)
            self.assertTrue(report.has_critical_findings)
            self.assertFalse(report.sha256_match)
            self.assertIn("checksum_mismatch", [f.category for f in report.findings])
            self.assertTrue(any(f.severity == AuditSeverity.CRITICAL and f.category == "checksum_mismatch" for f in report.findings))

            with self.assertRaises(ValueError):
                load_market_data_series(artifact, base_dir=tmp)


class RawQuarterlyMissingRollManifestTests(unittest.TestCase):
    def test_raw_quarterly_without_roll_manifest_path_rejected(self):
        with self.assertRaises(ValueError):
            raw_quarterly_artifact("0" * 64, roll_manifest_path=None)

    def test_raw_quarterly_with_empty_roll_manifest_path_rejected(self):
        with self.assertRaises(ValueError):
            raw_quarterly_artifact("0" * 64, roll_manifest_path="")

    def test_continuous_requires_normalization_method(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0.replace(minute=31), sha256="0" * 64,
            )

    def test_raw_quarterly_must_not_have_normalization_method(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.RAW_QUARTERLY, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0.replace(minute=33), sha256="0" * 64,
                roll_manifest_path="rolls.json", normalization_method="additive_back_adjustment",
            )


class DuplicateTimestampTests(unittest.TestCase):
    def test_duplicate_timestamp_is_a_critical_finding(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", DUPLICATE_TIMESTAMP_CSV)
            artifact = continuous_artifact(sha256_of(csv_path), coverage_end=T0.replace(minute=31))

            with self.assertRaises(ValueError):
                load_market_data_series(artifact, base_dir=tmp)

            from tradelab.loaders import read_csv_bars

            bars = read_csv_bars(csv_path)
            report = audit_market_data_artifact(artifact, bars, base_dir=tmp)
            self.assertTrue(report.has_critical_findings)
            self.assertIn("bar_series_invalid", [f.category for f in report.findings])


class PriceGridArtifactTests(unittest.TestCase):
    def test_off_tick_vendor_price_is_critical_and_blocks_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", OFF_TICK_CSV)
            artifact = continuous_artifact(sha256_of(csv_path), coverage_end=T0.replace(minute=31))

            with self.assertRaises(ValueError):
                load_market_data_series(artifact, base_dir=tmp)

            from tradelab.loaders import read_csv_bars

            report = audit_market_data_artifact(artifact, read_csv_bars(csv_path), base_dir=tmp)
            self.assertTrue(report.has_critical_findings)
            self.assertIn("price_grid_invalid", [finding.category for finding in report.findings])


class ReportDeterminismTests(unittest.TestCase):
    def test_identical_input_produces_identical_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", CONTINUOUS_CSV)
            artifact = continuous_artifact(sha256_of(csv_path))

            from tradelab.loaders import read_csv_bars

            bars = read_csv_bars(csv_path)
            report_a = audit_market_data_artifact(artifact, bars, base_dir=tmp)
            report_b = audit_market_data_artifact(artifact, bars, base_dir=tmp)

            self.assertEqual(json.dumps(report_a.to_dict(), sort_keys=True), json.dumps(report_b.to_dict(), sort_keys=True))
            self.assertEqual(report_a, report_b)

    def test_report_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", CONTINUOUS_CSV)
            artifact = continuous_artifact(sha256_of(csv_path))
            result = load_market_data_series(artifact, base_dir=tmp)
            serialized = json.dumps(result.audit_report.to_dict())
            self.assertIsInstance(serialized, str)

    def test_audit_never_mutates_input_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", CONTINUOUS_CSV)
            artifact = continuous_artifact(sha256_of(csv_path))
            from tradelab.loaders import read_csv_bars

            bars = read_csv_bars(csv_path)
            snapshot = [(b.open, b.high, b.low, b.close, b.volume, b.contract) for b in bars]
            audit_market_data_artifact(artifact, bars, base_dir=tmp)
            after = [(b.open, b.high, b.low, b.close, b.volume, b.contract) for b in bars]
            self.assertEqual(snapshot, after)


class RawQuarterlyPipelineUsesContinuousFuturesCorrectlyTests(unittest.TestCase):
    def test_pipeline_matches_calling_build_back_adjusted_continuous_series_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", RAW_QUARTERLY_CSV)
            write(tmp, "rolls.json", json.dumps(ROLL_MANIFEST))
            artifact = raw_quarterly_artifact(sha256_of(csv_path))

            result = load_market_data_series(artifact, base_dir=tmp)
            self.assertIsNotNone(result.continuous_series_result)
            self.assertTrue(all(b.contract == "NQ_CONTINUOUS" for b in result.bars))

            from tradelab.loaders import read_csv_bars

            raw_bars = read_csv_bars(csv_path)
            rolls = read_roll_manifest(tmp / "rolls.json")
            expected = build_back_adjusted_continuous_series(raw_bars, rolls, "NQ_CONTINUOUS")

            self.assertEqual(result.bars, expected.normalized_bars)
            self.assertEqual(result.continuous_series_result.cumulative_offsets, expected.cumulative_offsets)
            self.assertEqual(result.continuous_series_result.applied_rolls, expected.applied_rolls)

    def test_custom_canonical_contract_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", RAW_QUARTERLY_CSV)
            write(tmp, "rolls.json", json.dumps(ROLL_MANIFEST))
            artifact = raw_quarterly_artifact(sha256_of(csv_path))

            result = load_market_data_series(artifact, base_dir=tmp, canonical_contract="NQ_MY_CONTINUOUS")
            self.assertTrue(all(b.contract == "NQ_MY_CONTINUOUS" for b in result.bars))


class RollManifestLoaderTests(unittest.TestCase):
    def test_reads_valid_roll_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            roll_path = write(tmp, "rolls.json", json.dumps(ROLL_MANIFEST))
            rolls = read_roll_manifest(roll_path)
            self.assertEqual(len(rolls), 1)
            self.assertIsInstance(rolls[0], RollAdjustment)
            self.assertEqual(rolls[0].outgoing_contract, "NQZ24")
            self.assertEqual(rolls[0].adjustment_points, 50.0)

    def test_non_array_roll_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            roll_path = write(tmp, "rolls.json", json.dumps({"not": "a list"}))
            with self.assertRaises(ValueError):
                read_roll_manifest(roll_path)

    def test_malformed_roll_entry_rejected_with_index_in_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            roll_path = write(tmp, "rolls.json", json.dumps([{"outgoing_contract": "A"}]))
            with self.assertRaises(ValueError) as ctx:
                read_roll_manifest(roll_path)
            self.assertIn("entry 0", str(ctx.exception))


class ArtifactFieldValidationTests(unittest.TestCase):
    def test_roll_audit_path_and_checksum_must_be_supplied_together(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0.replace(minute=31), sha256="0" * 64,
                normalization_method="m", roll_audit_path="roll.json",
            )

    def test_declared_roll_audit_checksum_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", CONTINUOUS_CSV)
            write(tmp, "roll.json", "{}\n")
            artifact = MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0 + timedelta(minutes=2), sha256=sha256_of(csv_path),
                normalization_method="volume_front_month_v1", roll_audit_path="roll.json",
                roll_audit_sha256="0" * 64,
            )
            with self.assertRaises(ValueError):
                load_market_data_series(artifact, base_dir=tmp)
            from tradelab.loaders import read_csv_bars
            report = audit_market_data_artifact(artifact, read_csv_bars(csv_path), base_dir=tmp)
            self.assertIn("roll_audit_checksum_mismatch", [finding.category for finding in report.findings])

    def test_wrong_schema_version_rejected(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="9.9", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0.replace(minute=31), sha256="0" * 64, normalization_method="m",
            )

    def test_wrong_bar_interval_rejected(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="5m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0.replace(minute=31), sha256="0" * 64, normalization_method="m",
            )

    def test_wrong_timezone_rejected(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="America/New_York", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0.replace(minute=31), sha256="0" * 64, normalization_method="m",
            )

    def test_coverage_start_not_before_end_rejected(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0, sha256="0" * 64, normalization_method="m",
            )

    def test_malformed_sha256_rejected(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0.replace(minute=31), sha256="not-hex", normalization_method="m",
            )

    def test_uppercase_sha256_rejected(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=T0, coverage_end=T0.replace(minute=31), sha256="A" * 64, normalization_method="m",
            )

    def test_naive_coverage_start_rejected(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="bars.csv",
                coverage_start=datetime(2024, 1, 1), coverage_end=T0.replace(minute=31), sha256="0" * 64, normalization_method="m",
            )

    def test_empty_data_path_rejected(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.NQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.CONTINUOUS, data_path="",
                coverage_start=T0, coverage_end=T0.replace(minute=31), sha256="0" * 64, normalization_method="m",
            )


class AuditConsistencyChecksTests(unittest.TestCase):
    def test_one_minute_bar_end_matches_half_open_declared_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", CONTINUOUS_CSV)
            artifact = continuous_artifact(sha256_of(csv_path), coverage_end=T0 + timedelta(minutes=2))
            from tradelab.loaders import read_csv_bars

            report = audit_market_data_artifact(artifact, read_csv_bars(csv_path), base_dir=tmp)
            self.assertNotIn("declared_coverage_not_fully_observed", [f.category for f in report.findings])

    def test_continuous_series_with_multiple_contracts_is_critical(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", RAW_QUARTERLY_CSV)  # two contracts, mislabeled as continuous
            artifact = continuous_artifact(sha256_of(csv_path), coverage_end=T0.replace(minute=33))
            from tradelab.loaders import read_csv_bars

            bars = read_csv_bars(csv_path)
            report = audit_market_data_artifact(artifact, bars, base_dir=tmp)
            self.assertTrue(report.has_critical_findings)
            self.assertIn("continuous_series_has_multiple_contracts", [f.category for f in report.findings])

    def test_contract_label_instrument_mismatch_is_a_warning_not_critical(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = (
                "timestamp,open,high,low,close,volume,contract\n"
                "2024-01-02T14:30:00Z,100,101,99,100.5,10,ESZ24\n"
            )
            csv_path = write(tmp, "bars.csv", csv)
            write(tmp, "rolls.json", json.dumps([]))
            artifact = raw_quarterly_artifact(sha256_of(csv_path), roll_manifest_path="rolls.json")
            from tradelab.loaders import read_csv_bars

            bars = read_csv_bars(csv_path)
            report = audit_market_data_artifact(artifact, bars, base_dir=tmp)
            mismatch_findings = [f for f in report.findings if f.category == "contract_label_instrument_mismatch"]
            self.assertEqual(len(mismatch_findings), 1)
            self.assertEqual(mismatch_findings[0].severity, AuditSeverity.WARNING)
            self.assertFalse(report.has_critical_findings)

    def test_gaps_are_reported_as_warnings_and_never_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = (
                "timestamp,open,high,low,close,volume,contract\n"
                "2024-01-02T14:30:00Z,100,101,99,100.5,10,NQ_CONTINUOUS\n"
                "2024-01-02T14:35:00Z,100,101,99,100.5,10,NQ_CONTINUOUS\n"
            )
            csv_path = write(tmp, "bars.csv", csv)
            artifact = continuous_artifact(sha256_of(csv_path), coverage_end=T0.replace(minute=36))
            from tradelab.loaders import read_csv_bars

            bars = read_csv_bars(csv_path)
            report = audit_market_data_artifact(artifact, bars, base_dir=tmp)
            self.assertGreater(report.gap_count, 0)
            self.assertGreater(report.total_missing_minutes, 0)
            self.assertIn("missing_minutes_gaps", [f.category for f in report.findings])
            self.assertTrue(all(f.severity == AuditSeverity.WARNING for f in report.findings if f.category == "missing_minutes_gaps"))

    def test_coverage_exceeding_declared_window_is_critical(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", CONTINUOUS_CSV)
            # declared window ends at 14:30:30, before the data's actual last bar at 14:31
            artifact = continuous_artifact(sha256_of(csv_path), coverage_end=T0 + timedelta(seconds=30))
            from tradelab.loaders import read_csv_bars

            bars = read_csv_bars(csv_path)
            report = audit_market_data_artifact(artifact, bars, base_dir=tmp)
            self.assertTrue(report.has_critical_findings)
            self.assertIn("coverage_exceeds_declared_window", [f.category for f in report.findings])


class UnsupportedFileExtensionTests(unittest.TestCase):
    def test_unsupported_extension_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bad_path = write(tmp, "bars.txt", CONTINUOUS_CSV)
            artifact = continuous_artifact(sha256_of(bad_path), data_path="bars.txt")
            with self.assertRaises(ValueError):
                load_market_data_series(artifact, base_dir=tmp)


class DatabentoParentCsvManifestValidationTests(unittest.TestCase):
    def test_roll_manifest_path_must_be_none_for_databento_parent_csv(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.MNQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.DATABENTO_PARENT_CSV, data_path="mnq.csv",
                coverage_start=T0, coverage_end=T0 + timedelta(minutes=1), sha256="0" * 64,
                normalization_method="volume_front_month_v1", roll_manifest_path="rolls.json",
            )

    def test_normalization_method_required_for_databento_parent_csv(self):
        with self.assertRaises(ValueError):
            MarketDataArtifact(
                schema_version="1.0", source_name="x", instrument_root=Instrument.MNQ, bar_interval="1m",
                timestamp_timezone="UTC", series_kind=SeriesKind.DATABENTO_PARENT_CSV, data_path="mnq.csv",
                coverage_start=T0, coverage_end=T0 + timedelta(minutes=1), sha256="0" * 64,
            )


class DatabentoParentCsvPipelineTests(unittest.TestCase):
    def test_pipeline_selects_front_month_and_back_adjusts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "mnq_parent.csv", DATABENTO_PARENT_CSV)
            artifact = databento_artifact(sha256_of(csv_path))

            result = load_market_data_series(artifact, base_dir=tmp)

            self.assertFalse(result.audit_report.has_critical_findings)
            self.assertTrue(all(b.contract == "MNQ_CONTINUOUS" for b in result.bars))
            self.assertIsNotNone(result.continuous_series_result)
            self.assertEqual(len(result.continuous_series_result.applied_rolls), 1)
            roll = result.continuous_series_result.applied_rolls[0]
            self.assertEqual(roll.outgoing_contract, "MNQZ3")
            self.assertEqual(roll.incoming_contract, "MNQU3")
            self.assertEqual(roll.adjustment_points, 5.0)
            # warm-up session (2024-01-15) never enters the final bars
            self.assertTrue(all(b.timestamp >= datetime(2024, 1, 16, 23, 0, tzinfo=UTC) for b in result.bars))

    def test_audit_runs_on_post_selector_bars_not_the_raw_parallel_file(self):
        # The raw file has many same-timestamp rows across MNQZ3/MNQU3 --
        # if the audit were run against it directly (instead of the
        # post-selector, single-contract-per-session bars), the generic
        # validate_bars duplicate-timestamp check would immediately fail.
        # A clean pass here proves the audit saw the already-collapsed series.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "mnq_parent.csv", DATABENTO_PARENT_CSV)
            artifact = databento_artifact(sha256_of(csv_path))
            result = load_market_data_series(artifact, base_dir=tmp)
            self.assertNotIn("bar_series_invalid", [f.category for f in result.audit_report.findings])
            # far fewer audited bars than raw rows, since only the selected
            # contract's own bars per session ever reach the audit.
            self.assertLess(result.audit_report.bar_count, DATABENTO_PARENT_CSV.count("\n") - 1)

    def test_bad_checksum_blocks_databento_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write(tmp, "mnq_parent.csv", DATABENTO_PARENT_CSV)
            artifact = databento_artifact("0" * 64)  # deliberately wrong
            with self.assertRaises(ValueError):
                load_market_data_series(artifact, base_dir=tmp)

    def test_raw_quarterly_and_continuous_behavior_unchanged(self):
        # Same fixtures/assertions as the pre-existing raw_quarterly/continuous
        # tests -- guards against the new branch altering their behavior.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write(tmp, "bars.csv", RAW_QUARTERLY_CSV)
            write(tmp, "rolls.json", json.dumps(ROLL_MANIFEST))
            artifact = raw_quarterly_artifact(sha256_of(csv_path))
            result = load_market_data_series(artifact, base_dir=tmp)
            self.assertTrue(all(b.contract == "NQ_CONTINUOUS" for b in result.bars))

            csv_path2 = write(tmp, "cont.csv", CONTINUOUS_CSV)
            cont_artifact = continuous_artifact(sha256_of(csv_path2), data_path="cont.csv")
            cont_result = load_market_data_series(cont_artifact, base_dir=tmp)
            self.assertIsNone(cont_result.continuous_series_result)


if __name__ == "__main__":
    unittest.main()
