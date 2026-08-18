"""Local, vendor-independent market data artifact: a versioned JSON
manifest describing exactly one on-disk NQ/MNQ 1-minute bar file, plus
an audit that verifies the file actually matches what the manifest
claims before any research code is allowed to consume it.

**This module never fetches data.** It has no HTTP client, no vendor
SDK, no API keys, and no network calls of any kind -- every path a
manifest references (the bar data file, and for a raw-quarterly
series, the rollover manifest) must already exist locally, produced by
whatever process the caller used to obtain and export their own CME
NQ/MNQ 1-minute history (at least 3 years, per the eventual research
plan this module exists to support). This module's only job is to
describe that local file precisely enough, and check it thoroughly
enough, that a research pipeline can trust it.

**Two accepted series kinds**, matching ``tradelab.continuous_futures``
exactly:

- ``raw_quarterly`` -- the file holds one or more raw quarterly
  contracts back to back (e.g. ``NQZ24`` then ``NQH25``), and the
  manifest *must* also point at an explicit rollover manifest
  (``roll_manifest_path``, loaded into ``tradelab.continuous_futures.
  RollAdjustment`` objects by ``read_roll_manifest`` below) that this
  module's own pipeline function feeds straight into ``tradelab.
  continuous_futures.build_back_adjusted_continuous_series`` --
  unchanged, with no new adjustment semantics of its own.
- ``continuous`` -- the file is already one back-adjusted series (e.g.
  already produced by a prior run of that same module) and is loaded
  as-is, never re-normalized. Its manifest still records
  ``normalization_method`` and, when known, the ``roll_manifest_path``
  that originally produced it -- provenance only, never re-applied.

**Rollover dates and adjustment sizes are never inferred, here or
anywhere downstream.** Exactly as ``tradelab.continuous_futures``
already requires, every ``RollAdjustment`` this module loads comes
verbatim from an explicit rollover manifest file; this module adds no
heuristic of its own (no volume-dropoff detection, no open-interest
crossover, nothing) for guessing a roll date or a price adjustment.

**Timestamps are UTC throughout**, matching ``tradelab.bar.Bar1m``'s own
unconditional normalization -- the manifest's own ``timestamp_timezone``
field must literally be ``"UTC"``, and every bar this module loads (via
the existing ``tradelab.loaders`` CSV/Parquet readers, never
reimplemented here) is already UTC by the time this module sees it.

**No synthetic bars, ever.** A gap between two observed 1-minute bars
is reported -- as a warning/metric on the audit report, using ``tradelab
.validation``'s own existing missing-minutes registry -- never filled
in, interpolated, or otherwise fabricated. This module also never
assumes its own CME holiday/early-close calendar (mirroring ``tradelab.
validation``'s own documented limitation): a real holiday and a genuine
data problem both surface as the same kind of reported gap, left for a
human (or a calendar this package does not yet have) to tell apart.

**The audit never mutates its input** and never raises on a data-
quality problem -- ``audit_market_data_artifact`` always returns a
complete, deterministic, JSON-serializable ``MarketDataArtifactAuditReport``
(``.to_dict()``), whether or not it found anything wrong; only the
pipeline function, ``load_market_data_series``, decides to raise
(after inspecting the report) when a *critical* finding means the data
cannot safely be used.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from tradelab.accounting import Instrument
from tradelab.bar import Bar1m
from tradelab.continuous_futures import ContinuousSeriesResult, RollAdjustment, build_back_adjusted_continuous_series
from tradelab.loaders import _parse_timestamp, read_csv_bars, read_parquet_bars
from tradelab.validation import MissingMinutesGap, validate_bars

SCHEMA_VERSION = "1.0"
SUPPORTED_BAR_INTERVAL = "1m"
REQUIRED_TIMESTAMP_TIMEZONE = "UTC"

_SHA256_HEX_LENGTH = 64
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


class SeriesKind(str, Enum):
    RAW_QUARTERLY = "raw_quarterly"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class MarketDataArtifact:
    """A versioned, validated description of one local NQ/MNQ 1-minute
    bar file. Construction alone validates every field; it never reads
    ``data_path`` or any other file off disk -- that is exactly what
    ``audit_market_data_artifact``/``load_market_data_series`` do.

    ``roll_manifest_path`` is mandatory (non-empty) for ``series_kind=
    raw_quarterly`` (the pipeline will load and apply it) and must be
    left unset for ``series_kind=continuous`` normalization purposes --
    it may still be supplied there purely as audit provenance (the
    manifest that originally produced this already-continuous file),
    never re-applied. ``normalization_method`` is the mirror image:
    mandatory (non-empty) for ``continuous`` (describing how it was
    built, e.g. ``"additive_back_adjustment"``), and must be ``None``
    for ``raw_quarterly`` (the data has not been normalized yet -- that
    field would be misleading before normalization has even happened).
    """

    schema_version: str
    source_name: str
    instrument_root: Instrument
    bar_interval: str
    timestamp_timezone: str
    series_kind: SeriesKind
    data_path: str
    coverage_start: datetime
    coverage_end: datetime
    sha256: str
    roll_manifest_path: str | None = None
    normalization_method: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"MarketDataArtifact.schema_version must be exactly {SCHEMA_VERSION!r}, got {self.schema_version!r}")
        if not self.source_name:
            raise ValueError("MarketDataArtifact.source_name must be non-empty")
        if not isinstance(self.instrument_root, Instrument):
            raise ValueError(f"MarketDataArtifact.instrument_root must be a tradelab.accounting.Instrument, got {self.instrument_root!r}")
        if self.bar_interval != SUPPORTED_BAR_INTERVAL:
            raise ValueError(f"MarketDataArtifact.bar_interval must be exactly {SUPPORTED_BAR_INTERVAL!r}, got {self.bar_interval!r}")
        if self.timestamp_timezone != REQUIRED_TIMESTAMP_TIMEZONE:
            raise ValueError(
                f"MarketDataArtifact.timestamp_timezone must be exactly {REQUIRED_TIMESTAMP_TIMEZONE!r}, got {self.timestamp_timezone!r}"
            )
        if not isinstance(self.series_kind, SeriesKind):
            raise ValueError(f"MarketDataArtifact.series_kind must be a SeriesKind, got {self.series_kind!r}")
        if not self.data_path:
            raise ValueError("MarketDataArtifact.data_path must be non-empty")
        if self.coverage_start.tzinfo is None:
            raise ValueError(f"MarketDataArtifact.coverage_start must be timezone-aware: {self.coverage_start!r}")
        if self.coverage_end.tzinfo is None:
            raise ValueError(f"MarketDataArtifact.coverage_end must be timezone-aware: {self.coverage_end!r}")
        if not self.coverage_start < self.coverage_end:
            raise ValueError(f"MarketDataArtifact.coverage_start ({self.coverage_start!r}) must be < coverage_end ({self.coverage_end!r})")
        if not _looks_like_sha256(self.sha256):
            raise ValueError(f"MarketDataArtifact.sha256 must be a 64-character lowercase hex digest, got {self.sha256!r}")

        if self.series_kind == SeriesKind.RAW_QUARTERLY:
            if not self.roll_manifest_path:
                raise ValueError("MarketDataArtifact.roll_manifest_path is required for series_kind=raw_quarterly")
            if self.normalization_method is not None:
                raise ValueError("MarketDataArtifact.normalization_method must be None for series_kind=raw_quarterly (not yet normalized)")
        else:  # CONTINUOUS
            if not self.normalization_method:
                raise ValueError("MarketDataArtifact.normalization_method is required for series_kind=continuous")

    def to_dict(self) -> dict:
        """A plain, JSON-compatible dict -- the exact shape ``load_market_data_artifact`` reads back."""
        return {
            "schema_version": self.schema_version,
            "source_name": self.source_name,
            "instrument_root": self.instrument_root.value,
            "bar_interval": self.bar_interval,
            "timestamp_timezone": self.timestamp_timezone,
            "series_kind": self.series_kind.value,
            "data_path": self.data_path,
            "coverage_start": self.coverage_start.isoformat(),
            "coverage_end": self.coverage_end.isoformat(),
            "sha256": self.sha256,
            "roll_manifest_path": self.roll_manifest_path,
            "normalization_method": self.normalization_method,
        }


def _looks_like_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_HEX_LENGTH and set(value) <= _SHA256_HEX_DIGITS


def load_market_data_artifact(path: Path | str) -> MarketDataArtifact:
    """Read and validate a JSON market data artifact manifest from ``path``."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"market data artifact manifest {path} must be a JSON object")
    try:
        return MarketDataArtifact(
            schema_version=raw["schema_version"],
            source_name=raw["source_name"],
            instrument_root=Instrument(raw["instrument_root"]),
            bar_interval=raw["bar_interval"],
            timestamp_timezone=raw["timestamp_timezone"],
            series_kind=SeriesKind(raw["series_kind"]),
            data_path=raw["data_path"],
            coverage_start=_parse_timestamp(raw["coverage_start"]),
            coverage_end=_parse_timestamp(raw["coverage_end"]),
            sha256=raw["sha256"],
            roll_manifest_path=raw.get("roll_manifest_path"),
            normalization_method=raw.get("normalization_method"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid market data artifact manifest {path}: {error}") from error


def read_roll_manifest(path: Path | str) -> list[RollAdjustment]:
    """Read a JSON rollover manifest -- a plain array of objects, each
    with ``outgoing_contract``/``incoming_contract``/``effective_at``/
    ``adjustment_points`` -- into ``tradelab.continuous_futures.
    RollAdjustment`` objects. Every field must be explicit in the file;
    this function derives nothing and reuses ``RollAdjustment``'s own
    constructor/validation verbatim, so adjustment semantics can never
    drift between this loader and ``tradelab.continuous_futures``
    itself."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"roll manifest {path} must be a JSON array of roll objects")

    rolls: list[RollAdjustment] = []
    for index, entry in enumerate(raw):
        try:
            rolls.append(
                RollAdjustment(
                    outgoing_contract=entry["outgoing_contract"],
                    incoming_contract=entry["incoming_contract"],
                    effective_at=_parse_timestamp(entry["effective_at"]),
                    adjustment_points=float(entry["adjustment_points"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid roll manifest entry {index} in {path}: {error}") from error
    return rolls


class AuditSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AuditFinding:
    severity: AuditSeverity
    category: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity.value, "category": self.category, "message": self.message}


@dataclass(frozen=True)
class MarketDataArtifactAuditReport:
    """A complete, deterministic, JSON-serializable audit result for one
    ``MarketDataArtifact`` against one already-loaded ``bars`` series.
    ``findings`` is always in the same fixed check order for the same
    input, so two audits of identical input always produce an identical
    report (down to key order in ``to_dict()``)."""

    source_name: str
    instrument_root: Instrument
    series_kind: SeriesKind
    data_path: str
    sha256_expected: str
    sha256_actual: str
    sha256_match: bool
    bar_count: int
    declared_coverage_start: datetime
    declared_coverage_end: datetime
    observed_coverage_start: datetime | None
    observed_coverage_end: datetime | None
    gap_count: int
    total_missing_minutes: int
    gaps: tuple[MissingMinutesGap, ...]
    findings: tuple[AuditFinding, ...]

    @property
    def has_critical_findings(self) -> bool:
        return any(finding.severity == AuditSeverity.CRITICAL for finding in self.findings)

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "instrument_root": self.instrument_root.value,
            "series_kind": self.series_kind.value,
            "data_path": self.data_path,
            "sha256_expected": self.sha256_expected,
            "sha256_actual": self.sha256_actual,
            "sha256_match": self.sha256_match,
            "bar_count": self.bar_count,
            "declared_coverage_start": self.declared_coverage_start.isoformat(),
            "declared_coverage_end": self.declared_coverage_end.isoformat(),
            "observed_coverage_start": self.observed_coverage_start.isoformat() if self.observed_coverage_start else None,
            "observed_coverage_end": self.observed_coverage_end.isoformat() if self.observed_coverage_end else None,
            "gap_count": self.gap_count,
            "total_missing_minutes": self.total_missing_minutes,
            "gaps": [
                {"first_missing": gap.first_missing.isoformat(), "last_missing": gap.last_missing.isoformat(), "missing_count": gap.missing_count}
                for gap in self.gaps
            ],
            "findings": [finding.to_dict() for finding in self.findings],
            "has_critical_findings": self.has_critical_findings,
        }


def audit_market_data_artifact(
    artifact: MarketDataArtifact, bars: list[Bar1m], *, base_dir: Path | str | None = None
) -> MarketDataArtifactAuditReport:
    """Audit ``bars`` (already loaded, e.g. via ``tradelab.loaders``)
    against ``artifact``'s own claims: checksum, chronology/duplicates/
    OHLCV (via ``tradelab.validation.validate_bars``, never
    reimplemented here), coverage range, and -- when determinable --
    instrument/contract-label consistency. Missing-minute gaps are
    always reported as warnings/metrics, never assumed to be a CME
    holiday or early close and never used to reject the data. Never
    mutates ``bars`` or ``artifact``, and never raises for a data-
    quality problem -- every problem becomes an ``AuditFinding`` in the
    returned report instead; only ``load_market_data_series`` decides
    whether a critical finding should stop the pipeline.
    """
    bars = list(bars)
    findings: list[AuditFinding] = []

    data_path = _resolve_path(artifact.data_path, base_dir)
    sha256_actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
    sha256_match = sha256_actual == artifact.sha256
    if not sha256_match:
        findings.append(
            AuditFinding(
                severity=AuditSeverity.CRITICAL,
                category="checksum_mismatch",
                message=f"sha256 of {data_path} is {sha256_actual!r}, manifest declares {artifact.sha256!r}",
            )
        )

    try:
        gaps = validate_bars(bars).gaps
    except ValueError as error:
        findings.append(AuditFinding(severity=AuditSeverity.CRITICAL, category="bar_series_invalid", message=str(error)))
        gaps = ()

    observed_start = bars[0].timestamp if bars else None
    observed_end = bars[-1].timestamp if bars else None
    if observed_start is not None and observed_end is not None:
        if observed_start < artifact.coverage_start or observed_end > artifact.coverage_end:
            findings.append(
                AuditFinding(
                    severity=AuditSeverity.CRITICAL,
                    category="coverage_exceeds_declared_window",
                    message=(
                        f"observed data [{observed_start.isoformat()}, {observed_end.isoformat()}] extends beyond "
                        f"declared coverage [{artifact.coverage_start.isoformat()}, {artifact.coverage_end.isoformat()}]"
                    ),
                )
            )
        elif observed_start > artifact.coverage_start or observed_end < artifact.coverage_end:
            findings.append(
                AuditFinding(
                    severity=AuditSeverity.WARNING,
                    category="declared_coverage_not_fully_observed",
                    message=(
                        f"declared coverage [{artifact.coverage_start.isoformat()}, {artifact.coverage_end.isoformat()}] is not "
                        f"fully covered by observed data [{observed_start.isoformat()}, {observed_end.isoformat()}]"
                    ),
                )
            )

    if artifact.series_kind == SeriesKind.CONTINUOUS:
        distinct_contracts = sorted({bar.contract for bar in bars})
        if len(distinct_contracts) > 1:
            findings.append(
                AuditFinding(
                    severity=AuditSeverity.CRITICAL,
                    category="continuous_series_has_multiple_contracts",
                    message=f"series_kind=continuous but bars carry {len(distinct_contracts)} distinct contract labels: {distinct_contracts}",
                )
            )
    else:  # RAW_QUARTERLY -- a soft, non-rollover-related labeling sanity check only
        mismatched = sorted({bar.contract for bar in bars if not bar.contract.startswith(artifact.instrument_root.value)})
        if mismatched:
            findings.append(
                AuditFinding(
                    severity=AuditSeverity.WARNING,
                    category="contract_label_instrument_mismatch",
                    message=(
                        f"{len(mismatched)} contract label(s) do not start with instrument_root "
                        f"{artifact.instrument_root.value!r}: {mismatched}"
                    ),
                )
            )

    total_missing_minutes = sum(gap.missing_count for gap in gaps)
    if gaps:
        findings.append(
            AuditFinding(
                severity=AuditSeverity.WARNING,
                category="missing_minutes_gaps",
                message=(
                    f"{len(gaps)} gap(s) totalling {total_missing_minutes} missing expected-open minute(s) -- "
                    "never assumed to be a CME holiday/early close, always reported"
                ),
            )
        )

    return MarketDataArtifactAuditReport(
        source_name=artifact.source_name,
        instrument_root=artifact.instrument_root,
        series_kind=artifact.series_kind,
        data_path=artifact.data_path,
        sha256_expected=artifact.sha256,
        sha256_actual=sha256_actual,
        sha256_match=sha256_match,
        bar_count=len(bars),
        declared_coverage_start=artifact.coverage_start,
        declared_coverage_end=artifact.coverage_end,
        observed_coverage_start=observed_start,
        observed_coverage_end=observed_end,
        gap_count=len(gaps),
        total_missing_minutes=total_missing_minutes,
        gaps=tuple(gaps),
        findings=tuple(findings),
    )


@dataclass(frozen=True)
class MarketDataSeriesResult:
    """The final output of ``load_market_data_series``: ``bars`` is
    ready to hand to ``tradelab.validation``/``tradelab.aggregate``/
    ``tradelab.backtest`` as-is. ``continuous_series_result`` is the
    full ``tradelab.continuous_futures`` audit trail, present only when
    normalization actually ran (``series_kind=raw_quarterly``) -- ``None``
    for an already-continuous artifact, which this module passes
    through unchanged."""

    bars: tuple[Bar1m, ...]
    audit_report: MarketDataArtifactAuditReport
    continuous_series_result: ContinuousSeriesResult | None


def load_market_data_series(
    artifact: MarketDataArtifact, *, base_dir: Path | str | None = None, canonical_contract: str | None = None
) -> MarketDataSeriesResult:
    """Load ``artifact``'s data file, audit it, and -- only if the audit
    found no critical finding -- return the ready-to-use bar series.
    For ``series_kind=raw_quarterly`` this also loads ``artifact.
    roll_manifest_path`` and calls ``tradelab.continuous_futures.
    build_back_adjusted_continuous_series`` unchanged (``canonical_
    contract`` defaults to ``f"{instrument_root}_CONTINUOUS"`` when not
    given); for ``series_kind=continuous`` the loaded bars are returned
    exactly as read, never re-normalized. Raises ``ValueError`` -- never
    silently -- when the audit reports any critical finding; call
    ``audit_market_data_artifact`` directly for the full report in that
    case.
    """
    data_path = _resolve_path(artifact.data_path, base_dir)
    bars = _read_bars(data_path)

    audit_report = audit_market_data_artifact(artifact, bars, base_dir=base_dir)
    if audit_report.has_critical_findings:
        critical_messages = [finding.message for finding in audit_report.findings if finding.severity == AuditSeverity.CRITICAL]
        raise ValueError(
            f"market data artifact {artifact.source_name!r} ({artifact.data_path}) failed audit with "
            f"{len(critical_messages)} critical finding(s): {'; '.join(critical_messages)}"
        )

    if artifact.series_kind == SeriesKind.RAW_QUARTERLY:
        roll_manifest_path = _resolve_path(artifact.roll_manifest_path, base_dir)
        rolls = read_roll_manifest(roll_manifest_path)
        label = canonical_contract or f"{artifact.instrument_root.value}_CONTINUOUS"
        continuous_result = build_back_adjusted_continuous_series(bars, rolls, label)
        return MarketDataSeriesResult(bars=continuous_result.normalized_bars, audit_report=audit_report, continuous_series_result=continuous_result)

    return MarketDataSeriesResult(bars=tuple(bars), audit_report=audit_report, continuous_series_result=None)


def _resolve_path(raw_path: str, base_dir: Path | str | None) -> Path:
    path = Path(raw_path)
    if base_dir is not None and not path.is_absolute():
        return Path(base_dir) / path
    return path


def _read_bars(path: Path) -> list[Bar1m]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_bars(path)
    if suffix == ".parquet":
        return read_parquet_bars(path)
    raise ValueError(f"unsupported market data file extension {suffix!r} for {path}; expected .csv or .parquet")
