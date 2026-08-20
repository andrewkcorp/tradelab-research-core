"""Durable run-artifact protocol and reporter API for a future long
entry-discovery/backtest runner -- a dashboard-facing observability
layer only. This module never enumerates strategies, never simulates a
trade, and never computes TP/SL/P&L/drawdown itself; it only records,
atomically and durably, whatever a caller (a future runner) reports
about its own progress, so a local dashboard (``tradelab.research_
dashboard``) can render it live, and so a crash leaves a coherent,
inspectable trail.

**Not the same thing as ``tradelab.research_journal``.** That module is
a real, SQLite-backed, crash-*resumable* checkpoint of the actual
``iter_research_evaluations`` stream -- it can skip already-completed
work on restart. This module's own ``CheckpointRecord`` is deliberately
**not** that: it is a lightweight, JSON, dashboard-facing snapshot of
*where a run currently is* (``resumable=False`` unless a caller
explicitly opts in with a real resume mechanism and an honest
``resumable_note`` describing it -- see ``RunReporter.checkpoint_
written``). A future long-running engine that wants real crash-resume
should compose this module (for live status/telemetry) with
``tradelab.research_journal`` (for actual resumability), not expect
this module's checkpoints alone to resume anything.

**Run directory layout** (``runs/<run_id>/``)::

    run_manifest.json       -- written once, at create_run()
    live_status.json        -- overwritten atomically on every reporter call
    progress_events.jsonl   -- append-only, one JSON object per line
    leaderboard.json         -- overwritten atomically on update_leaderboard()
    results/                 -- reserved for future per-strategy result files
    checkpoints/checkpoint_NNNN.json -- one file per checkpoint_written() call

**Atomicity.** Every JSON *snapshot* (``run_manifest.json``, ``live_
status.json``, ``leaderboard.json``, each checkpoint file) is written
to a sibling temp file, ``fsync``'d, then moved into place with
``os.replace`` (POSIX-atomic) -- a reader opening the real path at any
instant always sees either the previous complete file or the new
complete one, never a partial write. This is exactly why ``live_
status.json`` is safe to read while a run is still active (spec,
this task) -- no lock, no coordination needed on the reader's side.
``progress_events.jsonl`` is append-only: each call opens in append
mode, writes one complete JSON line, flushes, and ``fsync``'s -- a
reader tailing the file never sees a torn line either.

**The ETA rule -- only when statistically defensible, else ``None``.**
``eta_seconds`` is computed only when *all* of: the current phase is
not terminal; ``items_total`` is known; at least ``_MIN_ETA_SAMPLES``
(3) ``progress()`` calls have been recorded in the *current* phase;
elapsed time and completed-item delta across the recorded window are
both strictly positive (a non-zero, forward-moving throughput). When
an ETA is reported, ``eta_basis`` names exactly the method and sample
size used, so a dashboard (or a human) can judge its own confidence --
never a silent, unexplained number. Any other situation reports
``eta_seconds=None`` -- this module never fabricates an ETA from a
single sample or a stalled phase.

**Checkpoint honesty.** ``checkpoint_written(resumable=True, ...)``
*requires* an explicit, non-empty ``resumable_note`` describing how
resume actually works -- there is no way to call it with ``resumable=
True`` and no explanation. The default, ``resumable=False``, always
carries an honest, non-empty note (auto-filled if the caller doesn't
supply one) stating plainly that no engine currently reads this
checkpoint back to resume a run -- never a silently-empty or misleading
field.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace as _dataclasses_replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

RUN_ARTIFACT_SCHEMA_VERSION = "1.0"
_UTC = timezone.utc
_MIN_ETA_SAMPLES = 3
_MAX_PROGRESS_SAMPLES = 50  # bounded ring buffer -- never grows with run length
_HEARTBEAT_MIN_INTERVAL_SECONDS = 1.0  # task 044: at most one live-status write per second from heartbeat()


class RunPhase(str, Enum):
    PREPARING_DATA = "preparing_data"
    DISCOVERING_ENTRIES = "discovering_entries"
    DEDUPLICATING_ENTRIES = "deduplicating_entries"
    EXPANDING_EXIT_VARIANTS = "expanding_exit_variants"
    SIMULATING_BACKTESTS = "simulating_backtests"
    RANKING_RESULTS = "ranking_results"
    COMPLETED = "completed"
    FAILED = "failed"


_TERMINAL_PHASES = frozenset({RunPhase.COMPLETED, RunPhase.FAILED})
_WORK_PHASES = frozenset(RunPhase) - _TERMINAL_PHASES


def _utc_now_iso() -> str:
    return datetime.now(_UTC).isoformat()


def _sha256_of_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_provenance(repo_root: Path) -> tuple[str | None, bool | None]:
    """Best-effort ``(commit, dirty)`` of ``repo_root`` -- ``(None,
    None)`` if this is not a git checkout or ``git`` is unavailable,
    mirroring ``tradelab.profile_research_context._git_provenance``'s
    own never-fabricate contract."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True, timeout=10
        ).stdout
        return commit, status.strip() != ""
    except Exception:
        return None, None


def _sample_resource_usage() -> tuple[int | None, float | None]:
    """Best-effort ``(rss_bytes, cpu_seconds)`` for this process --
    ``(None, None)`` if unavailable (``resource`` is POSIX-only, never
    fabricated elsewhere). RSS is normalized to bytes on every platform
    (``ru_maxrss`` is KiB on Linux, already bytes on macOS) -- the same
    normalization ``tradelab.backtest._peak_rss_bytes`` already applies,
    reused here via that exact helper rather than re-derived."""
    try:
        from tradelab.backtest import _peak_rss_bytes

        rss_bytes = _peak_rss_bytes()
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = usage.ru_utime + usage.ru_stime
        return rss_bytes, cpu_seconds
    except Exception:
        return None, None


class ProgressThrottle:
    """Wraps a callback so it fires at most once per ``min_interval_
    seconds`` (a monotonic-clock check), *and* so that clock check itself
    is only evaluated once every ``check_every`` calls to :meth:`maybe` --
    never on every single call. This is what makes it safe to call
    ``maybe(...)`` from inside a tight, per-bar simulation loop that may
    run millions of times: the overwhelming majority of calls do nothing
    more than increment one integer counter and compare it, never touch
    the clock or invoke the wrapped callback at all (task 044 spec: "the
    callback must be throttled by monotonic time to no more than once
    per second; never once per bar").

    The very first call always fires immediately (regardless of
    ``check_every``), so a live dashboard sees an instant first update
    rather than waiting for the first stride to complete. ``force()``
    bypasses both the stride and the time check entirely -- useful for a
    caller that wants to guarantee one final, definite update (e.g. "this
    bounded unit of work just finished") without waiting on the throttle.
    """

    def __init__(self, callback: Callable[..., None] | None, *, min_interval_seconds: float = _HEARTBEAT_MIN_INTERVAL_SECONDS, check_every: int = 2000) -> None:
        _require(check_every >= 1, "ProgressThrottle(check_every=...) must be >= 1")
        _require(min_interval_seconds >= 0, "ProgressThrottle(min_interval_seconds=...) must be >= 0")
        self._callback = callback
        self._min_interval = min_interval_seconds
        self._check_every = check_every
        self._calls_since_check = 0
        self._last_fired_monotonic: float | None = None

    def maybe(self, *args: Any, **kwargs: Any) -> bool:
        if self._callback is None:
            return False
        first_call = self._last_fired_monotonic is None
        self._calls_since_check += 1
        if not first_call and self._calls_since_check < self._check_every:
            return False
        self._calls_since_check = 0
        now = time.monotonic()
        if not first_call and (now - self._last_fired_monotonic) < self._min_interval:
            return False
        self._last_fired_monotonic = now
        self._callback(*args, **kwargs)
        return True

    def force(self, *args: Any, **kwargs: Any) -> bool:
        if self._callback is None:
            return False
        self._last_fired_monotonic = time.monotonic()
        self._calls_since_check = 0
        self._callback(*args, **kwargs)
        return True


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically: temp file in the same
    directory, ``fsync``'d, then ``os.replace`` -- see module docstring.
    On any failure the temp file is removed and the exception
    propagates; ``path`` itself is left untouched (either its previous
    content, or never created at all).

    The temp filename includes both the PID *and* the calling thread's
    own identity (task 046) -- two threads in the same process racing to
    write the *same* path (e.g. ``RunReporter``'s own liveness thread
    and its main thread both flushing ``live_status.json``) each get
    their own temp file, so neither's ``os.replace`` can ever race
    against a temp file the other has already consumed. This is a
    file-level safety net; ``RunReporter`` itself additionally
    serializes *which* thread's write actually happens via its own lock
    (see ``RunReporter._flush_status``) -- this just guarantees that
    even an unsynchronized concurrent caller can never crash with a
    ``FileNotFoundError`` racing on a shared temp path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # best-effort: not every platform/filesystem supports fsync on a directory fd
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path | str, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a single line to ``path`` -- flushed
    and ``fsync``'d before returning, so a concurrent reader tailing the
    file never observes a torn/partial line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ResourceUsage:
    rss_bytes: int | None
    cpu_seconds: float | None

    def __post_init__(self) -> None:
        if self.rss_bytes is not None:
            _require(self.rss_bytes >= 0, "ResourceUsage.rss_bytes must be >= 0")
        if self.cpu_seconds is not None:
            _require(self.cpu_seconds >= 0, "ResourceUsage.cpu_seconds must be >= 0")

    def to_dict(self) -> dict:
        return {"rss_bytes": self.rss_bytes, "cpu_seconds": self.cpu_seconds}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ResourceUsage":
        return ResourceUsage(rss_bytes=d.get("rss_bytes"), cpu_seconds=d.get("cpu_seconds"))


@dataclass(frozen=True)
class ActiveWork:
    """A live, best-effort heartbeat snapshot of *what a runner is doing
    right now*, inside a single phase, between the coarser ``progress()``
    milestones -- task 044. Written by :meth:`RunReporter.heartbeat`,
    throttled to at most once per second (see ``_HEARTBEAT_MIN_INTERVAL_
    SECONDS``) and never durably logged as its own event (see that
    method's own docstring for why): this is pure, ephemeral, in-flight
    status, not a milestone worth a permanent record.

    ``unit_index``/``unit_total`` are deliberately generic -- the same
    two fields serve both "current sample number of N" (preflight
    throughput sampling) and "current strategy-variant number of N"
    (production simulation); ``operation`` names which one a given
    snapshot means. ``bars_scanned``/``bars_total`` are populated only
    while genuinely inside a single variant's own bar-by-bar simulation
    loop -- ``None`` otherwise."""

    operation: str
    variant_id: str | None
    unit_index: int | None
    unit_total: int | None
    bars_scanned: int | None
    bars_total: int | None
    updated_at: str

    def __post_init__(self) -> None:
        _require(bool(self.operation), "ActiveWork.operation must be non-empty")
        for name, value in (
            ("unit_index", self.unit_index), ("unit_total", self.unit_total),
            ("bars_scanned", self.bars_scanned), ("bars_total", self.bars_total),
        ):
            if value is not None:
                _require(value >= 0, f"ActiveWork.{name} must be >= 0")

    def to_dict(self) -> dict:
        return {
            "operation": self.operation, "variant_id": self.variant_id, "unit_index": self.unit_index,
            "unit_total": self.unit_total, "bars_scanned": self.bars_scanned, "bars_total": self.bars_total,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ActiveWork":
        return ActiveWork(
            operation=d["operation"], variant_id=d.get("variant_id"), unit_index=d.get("unit_index"),
            unit_total=d.get("unit_total"), bars_scanned=d.get("bars_scanned"), bars_total=d.get("bars_total"),
            updated_at=d["updated_at"],
        )


@dataclass(frozen=True)
class TopStrategy:
    """One leaderboard row as embedded in ``live_status.json`` -- a
    trimmed view; the full ``LeaderboardEntry`` (with equity curve,
    per-trade P&L, monthly breakdown) lives in ``leaderboard.json``."""

    rank: int
    strategy_id: str
    pnl: float
    roi: float
    win_rate: float
    trade_count: int
    max_drawdown: float
    drawdowns_over_2000_count: int

    def __post_init__(self) -> None:
        _require(self.rank >= 1, "TopStrategy.rank must be >= 1")
        _require(bool(self.strategy_id), "TopStrategy.strategy_id must be non-empty")
        _require(self.trade_count >= 0, "TopStrategy.trade_count must be >= 0")
        _require(self.max_drawdown >= 0, "TopStrategy.max_drawdown must be >= 0")
        _require(self.drawdowns_over_2000_count >= 0, "TopStrategy.drawdowns_over_2000_count must be >= 0")
        _require(0.0 <= self.win_rate <= 1.0, "TopStrategy.win_rate must be within [0.0, 1.0]")

    def to_dict(self) -> dict:
        return {
            "rank": self.rank, "strategy_id": self.strategy_id, "pnl": self.pnl, "roi": self.roi,
            "win_rate": self.win_rate, "trade_count": self.trade_count, "max_drawdown": self.max_drawdown,
            "drawdowns_over_2000_count": self.drawdowns_over_2000_count,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TopStrategy":
        return TopStrategy(
            rank=d["rank"], strategy_id=d["strategy_id"], pnl=d["pnl"], roi=d["roi"], win_rate=d["win_rate"],
            trade_count=d["trade_count"], max_drawdown=d["max_drawdown"], drawdowns_over_2000_count=d["drawdowns_over_2000_count"],
        )


@dataclass(frozen=True)
class LeaderboardEntry:
    """One full leaderboard row -- everything ``TopStrategy`` has, plus
    chart-source detail: an equity curve, per-trade P&L (for the P&L
    distribution chart), and a trades-by-month breakdown."""

    rank: int
    strategy_id: str
    pnl: float
    roi: float
    win_rate: float
    trade_count: int
    max_drawdown: float
    drawdowns_over_2000_count: int
    equity_curve: tuple[Mapping[str, Any], ...] = ()  # [{"ts": iso str, "equity": float}, ...]
    trade_pnls: tuple[float, ...] = ()
    trades_by_month: Mapping[str, int] = None  # {"2024-01": count, ...}

    def __post_init__(self) -> None:
        _require(self.rank >= 1, "LeaderboardEntry.rank must be >= 1")
        _require(bool(self.strategy_id), "LeaderboardEntry.strategy_id must be non-empty")
        _require(self.trade_count >= 0, "LeaderboardEntry.trade_count must be >= 0")
        _require(self.max_drawdown >= 0, "LeaderboardEntry.max_drawdown must be >= 0")
        _require(self.drawdowns_over_2000_count >= 0, "LeaderboardEntry.drawdowns_over_2000_count must be >= 0")
        _require(0.0 <= self.win_rate <= 1.0, "LeaderboardEntry.win_rate must be within [0.0, 1.0]")
        if self.trades_by_month is None:
            object.__setattr__(self, "trades_by_month", {})

    def to_dict(self) -> dict:
        return {
            "rank": self.rank, "strategy_id": self.strategy_id, "pnl": self.pnl, "roi": self.roi,
            "win_rate": self.win_rate, "trade_count": self.trade_count, "max_drawdown": self.max_drawdown,
            "drawdowns_over_2000_count": self.drawdowns_over_2000_count,
            "equity_curve": list(self.equity_curve), "trade_pnls": list(self.trade_pnls),
            "trades_by_month": dict(self.trades_by_month),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LeaderboardEntry":
        return LeaderboardEntry(
            rank=d["rank"], strategy_id=d["strategy_id"], pnl=d["pnl"], roi=d["roi"], win_rate=d["win_rate"],
            trade_count=d["trade_count"], max_drawdown=d["max_drawdown"], drawdowns_over_2000_count=d["drawdowns_over_2000_count"],
            equity_curve=tuple(d.get("equity_curve", ())), trade_pnls=tuple(d.get("trade_pnls", ())),
            trades_by_month=dict(d.get("trades_by_month", {})),
        )

    def as_top_strategy(self) -> TopStrategy:
        return TopStrategy(
            rank=self.rank, strategy_id=self.strategy_id, pnl=self.pnl, roi=self.roi, win_rate=self.win_rate,
            trade_count=self.trade_count, max_drawdown=self.max_drawdown, drawdowns_over_2000_count=self.drawdowns_over_2000_count,
        )


@dataclass(frozen=True)
class Leaderboard:
    schema_version: str
    run_id: str
    updated_at: str
    entries: tuple[LeaderboardEntry, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "run_id": self.run_id, "updated_at": self.updated_at,
            "entries": [e.to_dict() for e in self.entries],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Leaderboard":
        return Leaderboard(
            schema_version=d["schema_version"], run_id=d["run_id"], updated_at=d["updated_at"],
            entries=tuple(LeaderboardEntry.from_dict(e) for e in d.get("entries", ())),
        )


@dataclass(frozen=True)
class CheckpointRecord:
    schema_version: str
    run_id: str
    checkpoint_id: str
    created_at: str
    phase: str
    cursor: Mapping[str, Any]
    resumable: bool
    resumable_note: str

    def __post_init__(self) -> None:
        RunPhase(self.phase)  # raises ValueError if not a legal phase
        _require(bool(self.checkpoint_id), "CheckpointRecord.checkpoint_id must be non-empty")
        _require(bool(self.resumable_note) and self.resumable_note.strip() != "", "CheckpointRecord.resumable_note must be a non-empty, honest description of resumability status")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "run_id": self.run_id, "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at, "phase": self.phase, "cursor": dict(self.cursor),
            "resumable": self.resumable, "resumable_note": self.resumable_note,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "CheckpointRecord":
        return CheckpointRecord(
            schema_version=d["schema_version"], run_id=d["run_id"], checkpoint_id=d["checkpoint_id"],
            created_at=d["created_at"], phase=d["phase"], cursor=dict(d.get("cursor", {})),
            resumable=d["resumable"], resumable_note=d["resumable_note"],
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    created_at: str
    git_commit: str | None
    git_commit_dirty: bool | None
    market_artifact_path: str | None
    market_artifact_sha256: str | None
    news_csv_path: str | None
    news_csv_sha256: str | None
    parameters: Mapping[str, Any]
    python_version: str
    command: str | None
    is_demo: bool

    def __post_init__(self) -> None:
        _require(self.schema_version == RUN_ARTIFACT_SCHEMA_VERSION, f"RunManifest.schema_version must be {RUN_ARTIFACT_SCHEMA_VERSION!r}")
        _require(bool(self.run_id), "RunManifest.run_id must be non-empty")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "run_id": self.run_id, "created_at": self.created_at,
            "git_commit": self.git_commit, "git_commit_dirty": self.git_commit_dirty,
            "market_artifact_path": self.market_artifact_path, "market_artifact_sha256": self.market_artifact_sha256,
            "news_csv_path": self.news_csv_path, "news_csv_sha256": self.news_csv_sha256,
            "parameters": dict(self.parameters), "python_version": self.python_version,
            "command": self.command, "is_demo": self.is_demo,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "RunManifest":
        return RunManifest(
            schema_version=d["schema_version"], run_id=d["run_id"], created_at=d["created_at"],
            git_commit=d.get("git_commit"), git_commit_dirty=d.get("git_commit_dirty"),
            market_artifact_path=d.get("market_artifact_path"), market_artifact_sha256=d.get("market_artifact_sha256"),
            news_csv_path=d.get("news_csv_path"), news_csv_sha256=d.get("news_csv_sha256"),
            parameters=dict(d.get("parameters", {})), python_version=d["python_version"],
            command=d.get("command"), is_demo=d.get("is_demo", False),
        )


@dataclass(frozen=True)
class LiveStatus:
    schema_version: str
    run_id: str
    updated_at: str
    phase: str
    is_demo: bool
    items_completed: int | None
    items_total: int | None
    elapsed_seconds: float
    throughput_per_second: float | None
    eta_seconds: float | None
    eta_basis: Mapping[str, Any] | None
    latest_checkpoint: Mapping[str, Any] | None
    resource_usage: Mapping[str, Any] | None
    discovered_specs: int | None
    rejected_specs: int | None
    eligible_specs: int | None
    unique_entry_classes: int | None
    exit_variants_generated: int | None
    backtests_completed: int | None
    top_strategies: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    active_work: Mapping[str, Any] | None = None  # task 044 -- see ActiveWork; None on older run folders, always backward compatible

    def __post_init__(self) -> None:
        RunPhase(self.phase)  # raises ValueError if not a legal phase
        _require(self.elapsed_seconds >= 0, "LiveStatus.elapsed_seconds must be >= 0")
        if self.items_completed is not None:
            _require(self.items_completed >= 0, "LiveStatus.items_completed must be >= 0")
        if self.items_total is not None:
            _require(self.items_total >= 0, "LiveStatus.items_total must be >= 0")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "run_id": self.run_id, "updated_at": self.updated_at,
            "phase": self.phase, "is_demo": self.is_demo,
            "items_completed": self.items_completed, "items_total": self.items_total,
            "elapsed_seconds": self.elapsed_seconds, "throughput_per_second": self.throughput_per_second,
            "eta_seconds": self.eta_seconds, "eta_basis": (dict(self.eta_basis) if self.eta_basis is not None else None),
            "latest_checkpoint": (dict(self.latest_checkpoint) if self.latest_checkpoint is not None else None),
            "resource_usage": (dict(self.resource_usage) if self.resource_usage is not None else None),
            "discovered_specs": self.discovered_specs, "rejected_specs": self.rejected_specs,
            "eligible_specs": self.eligible_specs, "unique_entry_classes": self.unique_entry_classes,
            "exit_variants_generated": self.exit_variants_generated, "backtests_completed": self.backtests_completed,
            "top_strategies": [dict(s) for s in self.top_strategies],
            "warnings": list(self.warnings), "errors": list(self.errors),
            "active_work": (dict(self.active_work) if self.active_work is not None else None),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LiveStatus":
        return LiveStatus(
            schema_version=d["schema_version"], run_id=d["run_id"], updated_at=d["updated_at"], phase=d["phase"],
            is_demo=d.get("is_demo", False), items_completed=d.get("items_completed"), items_total=d.get("items_total"),
            elapsed_seconds=d["elapsed_seconds"], throughput_per_second=d.get("throughput_per_second"),
            eta_seconds=d.get("eta_seconds"), eta_basis=d.get("eta_basis"),
            latest_checkpoint=d.get("latest_checkpoint"), resource_usage=d.get("resource_usage"),
            discovered_specs=d.get("discovered_specs"), rejected_specs=d.get("rejected_specs"),
            eligible_specs=d.get("eligible_specs"), unique_entry_classes=d.get("unique_entry_classes"),
            exit_variants_generated=d.get("exit_variants_generated"), backtests_completed=d.get("backtests_completed"),
            top_strategies=tuple(d.get("top_strategies", ())), warnings=tuple(d.get("warnings", ())),
            errors=tuple(d.get("errors", ())), active_work=d.get("active_work"),  # absent on older run folders -- None, never a KeyError
        )


def _default_run_id(*, is_demo: bool) -> str:
    prefix = "demo" if is_demo else "run"
    timestamp = datetime.now(_UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


class RunReporter:
    """The mutable, in-process side of one run -- every method both
    updates in-memory state and durably flushes it (``live_status.json``
    atomically rewritten, an event appended to ``progress_events.jsonl``)
    before returning, so a crash immediately after any call still leaves
    the run directory in a coherent, fully-flushed state. Never
    constructed directly by a caller -- use :func:`create_run`."""

    def __init__(self, run_dir: Path, manifest: RunManifest) -> None:
        self._run_dir = run_dir
        self._manifest = manifest
        self._start_time = time.perf_counter()
        self._phase: RunPhase | None = None
        self._phase_start_time: float | None = None
        self._items_completed: int | None = None
        self._items_total: int | None = None
        self._progress_samples: list[tuple[float, int]] = []
        self._checkpoint_seq = 0
        self._latest_checkpoint: CheckpointRecord | None = None
        self._resource_usage: ResourceUsage | None = None
        self._discovered_specs: int | None = None
        self._rejected_specs: int | None = None
        self._eligible_specs: int | None = None
        self._unique_entry_classes: int | None = None
        self._exit_variants_generated: int | None = None
        self._backtests_completed: int | None = None
        self._top_strategies: tuple[Mapping[str, Any], ...] = ()
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._active_work: ActiveWork | None = None
        self._last_heartbeat_monotonic: float | None = None
        # task 046: an RLock (not a plain Lock) -- _flush_status() itself
        # acquires this too (see its own docstring), and heartbeat()/
        # _tick_liveness() already hold it when they call _flush_status(),
        # so a plain Lock would deadlock on that reentrant acquisition.
        self._heartbeat_lock = threading.RLock()
        self._liveness_thread: threading.Thread | None = None
        self._liveness_stop_event: threading.Event | None = None

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def run_id(self) -> str:
        return self._manifest.run_id

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    def phase_started(self, phase: RunPhase | str, *, items_total: int | None = None) -> None:
        phase = RunPhase(phase)
        _require(phase in _WORK_PHASES, f"phase_started() must be one of {sorted(p.value for p in _WORK_PHASES)}, got {phase.value!r} -- use completed()/failed() for terminal phases")
        if items_total is not None:
            _require(items_total >= 0, "phase_started(items_total=...) must be >= 0")
        self._phase = phase
        self._phase_start_time = time.perf_counter()
        self._items_total = items_total
        self._items_completed = 0 if items_total is not None else None
        self._progress_samples = []
        with self._heartbeat_lock:  # task 046 -- the liveness thread must never read a torn active_work mid-transition
            self._active_work = None  # stale in-flight status from the previous phase must not linger
            self._last_heartbeat_monotonic = None
        self._append_event("phase_started", {"items_total": items_total})
        self._flush_status()

    def progress(
        self,
        *,
        items_completed: int | None = None,
        items_total: int | None = None,
        discovered_specs: int | None = None,
        rejected_specs: int | None = None,
        eligible_specs: int | None = None,
        unique_entry_classes: int | None = None,
        exit_variants_generated: int | None = None,
        backtests_completed: int | None = None,
        warning: str | None = None,
    ) -> None:
        if self._phase is None:
            raise RuntimeError("RunReporter.progress() called before phase_started()")
        if items_completed is not None:
            _require(items_completed >= 0, "progress(items_completed=...) must be >= 0")
            self._items_completed = items_completed
        if items_total is not None:
            _require(items_total >= 0, "progress(items_total=...) must be >= 0")
            self._items_total = items_total
        for name, value in (
            ("discovered_specs", discovered_specs), ("rejected_specs", rejected_specs), ("eligible_specs", eligible_specs),
            ("unique_entry_classes", unique_entry_classes), ("exit_variants_generated", exit_variants_generated),
            ("backtests_completed", backtests_completed),
        ):
            if value is not None:
                _require(value >= 0, f"progress({name}=...) must be >= 0")
                setattr(self, f"_{name}", value)

        if self._items_completed is not None:
            elapsed_in_phase = time.perf_counter() - self._phase_start_time
            self._progress_samples.append((elapsed_in_phase, self._items_completed))
            if len(self._progress_samples) > _MAX_PROGRESS_SAMPLES:
                self._progress_samples = self._progress_samples[-_MAX_PROGRESS_SAMPLES:]

        self._append_event(
            "progress",
            {"items_completed": self._items_completed, "items_total": self._items_total, "warning": warning},
        )
        if warning:
            self._warnings.append(warning)
        self._flush_status()

    def record_resource_usage(self, *, rss_bytes: int | None = None, cpu_seconds: float | None = None) -> None:
        if rss_bytes is None and cpu_seconds is None:
            rss_bytes, cpu_seconds = _sample_resource_usage()
        self._resource_usage = ResourceUsage(rss_bytes=rss_bytes, cpu_seconds=cpu_seconds)
        self._append_event("resource_usage", self._resource_usage.to_dict())
        self._flush_status()

    def heartbeat(
        self,
        *,
        operation: str,
        variant_id: str | None = None,
        unit_index: int | None = None,
        unit_total: int | None = None,
        bars_scanned: int | None = None,
        bars_total: int | None = None,
    ) -> bool:
        """A live, in-flight status update for the dashboard -- task 044.

        Unlike every other ``RunReporter`` method, this one deliberately
        does **not** append to ``progress_events.jsonl``: it exists
        precisely so a caller inside a long, tight loop (potentially
        millions of iterations -- one strategy variant's own bar-by-bar
        simulation, or the full production variant sweep) can report
        fine-grained, second-by-second position without growing the
        durable event log without bound. ``live_status.json`` (via
        ``_flush_status``) is the only artifact this writes to.

        **Hard cap: at most one actual write per second -- no exceptions,
        no bypass.** This is a strict contract, not a best-effort default:
        every call updates the in-memory ``active_work`` snapshot
        immediately (so no milestone is ever silently lost), but the
        write to ``live_status.json`` itself only happens on the first
        call since the current phase started, or once at least ``_
        HEARTBEAT_MIN_INTERVAL_SECONDS`` (monotonic-clock based) has
        elapsed since this method's own last write -- whichever a given
        call satisfies. There used to be a ``force=True`` escape hatch
        for "this bounded unit of work just finished" callers (e.g. one
        completed preflight throughput sample); it was removed because it
        let several such calls, each individually reasonable, land inside
        the same second and blow the one-write-per-second budget in
        aggregate. **Coalescing, not forcing, is the fix**: a caller that
        reports several milestones faster than once a second no longer
        gets every single one written -- it gets the *latest* one written
        as soon as the next write is actually due, because the in-memory
        snapshot this method updates unconditionally is exactly what
        every other flushing method (``progress()``, ``checkpoint_
        written()``, ``phase_started()``, ...) also serializes into
        ``live_status.json`` on its own next call. A caller that wants a
        milestone to become visible sooner than the throttle would
        otherwise allow should call one of those methods (each already
        milestone-triggered by design, not subject to this cap) rather
        than reach for a bypass here.

        Callers inside a genuinely per-bar loop should still wrap their
        own callback in a :class:`ProgressThrottle` first, purely to
        avoid paying this method's own function-call/monotonic-check cost
        on every single bar -- see that class's own docstring; this
        method's cap is the final, authoritative guarantee regardless of
        how disciplined (or not) the caller's own wrapping is.

        Returns ``True`` if this call actually wrote a new status,
        ``False`` if only the in-memory snapshot was updated (purely
        informational, useful for tests).

        As a side effect, only when it actually writes, this also
        refreshes ``resource_usage`` with a fresh, un-logged CPU/RSS
        sample -- so CPU/RAM stay visibly live during long work without
        needing their own separate once-a-second event either.

        **Thread-safe (task 046)**: the write-throttle decision is made
        under the same lock ``start_liveness_heartbeat()``'s background
        thread uses for its own tick, so the two together still never
        write more than once per second combined -- see that method's
        own docstring.
        """
        _require(bool(operation), "heartbeat(operation=...) must be non-empty")
        active_work = ActiveWork(
            operation=operation, variant_id=variant_id, unit_index=unit_index, unit_total=unit_total,
            bars_scanned=bars_scanned, bars_total=bars_total, updated_at=_utc_now_iso(),
        )
        with self._heartbeat_lock:
            self._active_work = active_work
            return self._maybe_write_heartbeat_locked()

    def _maybe_write_heartbeat_locked(self) -> bool:
        """The shared, once-per-second-throttled write decision behind
        both ``heartbeat()`` (new operation/progress data) and
        ``_tick_liveness()`` (no new data -- a bare liveness refresh) --
        task 046. Caller must already hold ``self._heartbeat_lock``."""
        now = time.monotonic()
        if self._last_heartbeat_monotonic is not None and (now - self._last_heartbeat_monotonic) < _HEARTBEAT_MIN_INTERVAL_SECONDS:
            return False
        self._last_heartbeat_monotonic = now
        rss_bytes, cpu_seconds = _sample_resource_usage()
        if rss_bytes is not None or cpu_seconds is not None:
            self._resource_usage = ResourceUsage(rss_bytes=rss_bytes, cpu_seconds=cpu_seconds)
        self._flush_status()
        return True

    def _tick_liveness(self) -> bool:
        """task 046: called only by the background liveness thread,
        roughly once per ``interval_seconds`` (see ``start_liveness_
        heartbeat``) -- refreshes ``live_status.json`` (``updated_at``,
        ``elapsed_seconds``, a fresh CPU/RAM sample) using whatever
        ``active_work`` is already recorded, **without inventing any new
        operation or progress data**. This is what guarantees ``live_
        status`` keeps refreshing at least once a second even while the
        main thread is deep inside a single long, callback-free
        operation (e.g. building a ``tradelab.bar_range_index.
        BarRangeIndex``, or a detector phase with no intra-phase
        progress hook) -- "still working on X, N seconds elapsed" is
        genuinely true and worth showing, never a fabricated bar count
        or fake sub-progress. Shares ``heartbeat()``'s own once-per-
        second write cap via the same lock -- see ``_maybe_write_
        heartbeat_locked``."""
        with self._heartbeat_lock:
            if self._active_work is not None:
                # bump updated_at on the *existing* snapshot so "updated
                # Ns ago" reflects this genuine tick, not a stale
                # timestamp from whenever the operation last changed.
                self._active_work = _dataclasses_replace(self._active_work, updated_at=_utc_now_iso())
            return self._maybe_write_heartbeat_locked()

    def start_liveness_heartbeat(self, *, interval_seconds: float = 1.0) -> None:
        """task 046: starts a daemon background thread that calls
        ``_tick_liveness()`` roughly every ``interval_seconds``, for as
        long as this reporter's run is active. Guarantees ``live_status.
        json`` keeps refreshing at least once a second even during a
        long, single operation with no completed high-level milestone
        and no finer progress callback available -- never fabricates
        new operation/progress data, only confirms the process is still
        alive and current on whatever it last genuinely reported.

        Idempotent: calling this twice on an already-started reporter is
        a no-op. Always pair with ``stop_liveness_heartbeat()`` (ideally
        in a ``try``/``finally``) once the run reaches a terminal state
        -- the thread itself is a daemon (never blocks process exit on
        its own), but leaving it running past a run's own lifetime
        wastes a thread and keeps sampling resource usage for no
        reason."""
        if self._liveness_thread is not None:
            return
        stop_event = threading.Event()

        def _loop() -> None:
            while not stop_event.wait(interval_seconds):
                try:
                    self._tick_liveness()
                except Exception:  # noqa: BLE001 -- a liveness thread must never crash the run it's monitoring
                    pass

        thread = threading.Thread(target=_loop, name=f"RunReporter-liveness-{self.run_id}", daemon=True)
        self._liveness_stop_event = stop_event
        self._liveness_thread = thread
        thread.start()

    def stop_liveness_heartbeat(self, *, timeout: float = 5.0) -> None:
        """Stops the background liveness thread started by ``start_
        liveness_heartbeat()``, if any, and waits (bounded by
        ``timeout``) for it to actually exit. A no-op if no thread is
        running."""
        if self._liveness_thread is None:
            return
        self._liveness_stop_event.set()
        self._liveness_thread.join(timeout=timeout)
        self._liveness_thread = None
        self._liveness_stop_event = None

    def update_leaderboard(self, entries: Sequence[LeaderboardEntry]) -> None:
        for entry in entries:
            _require(isinstance(entry, LeaderboardEntry), "update_leaderboard(entries=...) must contain only LeaderboardEntry instances")
        leaderboard = Leaderboard(schema_version=RUN_ARTIFACT_SCHEMA_VERSION, run_id=self.run_id, updated_at=_utc_now_iso(), entries=tuple(entries))
        atomic_write_json(self._run_dir / "leaderboard.json", leaderboard.to_dict())
        ranked = sorted(entries, key=lambda e: e.rank)[:10]
        self._top_strategies = tuple(e.as_top_strategy().to_dict() for e in ranked)
        self._append_event("leaderboard_updated", {"entry_count": len(entries)})
        self._flush_status()

    def checkpoint_written(self, *, cursor: Mapping[str, Any], resumable: bool = False, resumable_note: str | None = None) -> CheckpointRecord:
        if resumable:
            _require(bool(resumable_note) and resumable_note.strip() != "", "checkpoint_written(resumable=True) requires an explicit, non-empty resumable_note describing how resume actually works")
        else:
            resumable_note = resumable_note or (
                "Not yet resumable: this checkpoint records progress position for the live dashboard only. "
                "No engine currently reads it back to resume a run -- see tradelab.research_journal for real, "
                "SQLite-backed resumability of the backtest evaluation stream."
            )
        self._checkpoint_seq += 1
        checkpoint_id = f"checkpoint_{self._checkpoint_seq:04d}"
        record = CheckpointRecord(
            schema_version=RUN_ARTIFACT_SCHEMA_VERSION, run_id=self.run_id, checkpoint_id=checkpoint_id, created_at=_utc_now_iso(),
            phase=(self._phase.value if self._phase is not None else RunPhase.PREPARING_DATA.value),
            cursor=dict(cursor), resumable=resumable, resumable_note=resumable_note,
        )
        atomic_write_json(self._run_dir / "checkpoints" / f"{checkpoint_id}.json", record.to_dict())
        self._latest_checkpoint = record
        self._append_event("checkpoint_written", {"checkpoint_id": checkpoint_id, "resumable": resumable})
        self._flush_status()
        return record

    def completed(self, *, summary: Mapping[str, Any] | None = None) -> None:
        self._phase = RunPhase.COMPLETED
        self._append_event("completed", {"summary": dict(summary) if summary else {}})
        self._flush_status()

    def failed(self, *, error: str, traceback_text: str | None = None) -> None:
        _require(bool(error), "failed(error=...) must be non-empty")
        self._phase = RunPhase.FAILED
        self._errors.append(error)
        self._append_event("failed", {"error": error, "traceback": traceback_text})
        self._flush_status()

    def _elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start_time

    def _compute_throughput(self) -> float | None:
        if len(self._progress_samples) < 2:
            return None
        t0, c0 = self._progress_samples[0]
        t1, c1 = self._progress_samples[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        return (c1 - c0) / dt

    def _compute_eta(self) -> tuple[float | None, dict[str, Any] | None]:
        if self._phase is None or self._phase in _TERMINAL_PHASES:
            return None, None
        if self._items_total is None or self._items_completed is None:
            return None, None
        if len(self._progress_samples) < _MIN_ETA_SAMPLES:
            return None, None
        t0, c0 = self._progress_samples[0]
        t1, c1 = self._progress_samples[-1]
        dt = t1 - t0
        dc = c1 - c0
        if dt <= 0 or dc <= 0:
            return None, None
        throughput = dc / dt
        remaining = max(self._items_total - self._items_completed, 0)
        eta = remaining / throughput
        basis = {"method": "linear_recent_window", "sample_size": len(self._progress_samples), "throughput_per_second": throughput}
        return eta, basis

    def _append_event(self, event_type: str, detail: Mapping[str, Any]) -> None:
        append_jsonl(
            self._run_dir / "progress_events.jsonl",
            {"ts": _utc_now_iso(), "event_type": event_type, "phase": (self._phase.value if self._phase is not None else None), "detail": dict(detail)},
        )

    def _flush_status(self) -> None:
        """Writes ``live_status.json`` -- task 046: guarded by the same
        ``self._heartbeat_lock`` ``heartbeat()``/``_tick_liveness()`` use
        for their own throttle decision (an ``RLock``, so their own
        already-held lock reenters here without deadlocking). Every
        *other* caller (``progress()``, ``phase_started()``, ``checkpoint_
        written()``, ``update_leaderboard()``, ``completed()``,
        ``failed()``, ...) runs only on the main thread in normal use,
        but the background liveness thread can call ``_tick_liveness()``
        at literally any moment -- without this lock, two threads
        writing ``live_status.json`` at the same instant could race on
        ``atomic_write_json``'s own temp-file rename (``os.replace``)
        and raise ``FileNotFoundError``. Serializing every write here
        closes that race at its root, rather than only papering over the
        file-level symptom (``atomic_write_json`` also gives each thread
        its own temp filename as a second, independent safety net)."""
        with self._heartbeat_lock:
            eta_seconds, eta_basis = self._compute_eta()
            status = LiveStatus(
                schema_version=RUN_ARTIFACT_SCHEMA_VERSION, run_id=self.run_id, updated_at=_utc_now_iso(),
                phase=(self._phase.value if self._phase is not None else RunPhase.PREPARING_DATA.value),
                is_demo=self._manifest.is_demo, items_completed=self._items_completed, items_total=self._items_total,
                elapsed_seconds=self._elapsed_seconds(), throughput_per_second=self._compute_throughput(),
                eta_seconds=eta_seconds, eta_basis=eta_basis,
                latest_checkpoint=(self._latest_checkpoint.to_dict() if self._latest_checkpoint is not None else None),
                resource_usage=(self._resource_usage.to_dict() if self._resource_usage is not None else None),
                discovered_specs=self._discovered_specs, rejected_specs=self._rejected_specs, eligible_specs=self._eligible_specs,
                unique_entry_classes=self._unique_entry_classes, exit_variants_generated=self._exit_variants_generated,
                backtests_completed=self._backtests_completed, top_strategies=self._top_strategies,
                warnings=tuple(self._warnings), errors=tuple(self._errors),
                active_work=(self._active_work.to_dict() if self._active_work is not None else None),
            )
            atomic_write_json(self._run_dir / "live_status.json", status.to_dict())


def create_run(
    *,
    runs_dir: Path | str,
    market_artifact_path: Path | str | None,
    news_csv_path: Path | str | None,
    parameters: Mapping[str, Any],
    run_id: str | None = None,
    command: str | None = None,
    is_demo: bool = False,
) -> RunReporter:
    """Create a new run directory under ``runs_dir`` (default ``run_id``:
    a UTC timestamp + short random suffix, prefixed ``demo-`` when
    ``is_demo``), write its ``run_manifest.json`` (commit, artifact/news
    hashes -- ``None`` if the given paths don't actually exist, never
    fabricated -- parameters, start time, schema version), create the
    ``results/``/``checkpoints/`` subdirectories, write an initial
    ``live_status.json`` (phase ``preparing_data``, nothing else known
    yet), and return the :class:`RunReporter` to drive it."""
    runs_dir = Path(runs_dir)
    run_id = run_id or _default_run_id(is_demo=is_demo)
    run_dir = runs_dir / run_id
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    market_artifact_path = Path(market_artifact_path) if market_artifact_path is not None else None
    news_csv_path = Path(news_csv_path) if news_csv_path is not None else None
    repo_root = Path(__file__).resolve().parents[1]
    git_commit, git_commit_dirty = _git_provenance(repo_root)

    manifest = RunManifest(
        schema_version=RUN_ARTIFACT_SCHEMA_VERSION, run_id=run_id, created_at=_utc_now_iso(),
        git_commit=git_commit, git_commit_dirty=git_commit_dirty,
        market_artifact_path=(str(market_artifact_path) if market_artifact_path is not None else None),
        market_artifact_sha256=(_sha256_of_file(market_artifact_path) if market_artifact_path is not None else None),
        news_csv_path=(str(news_csv_path) if news_csv_path is not None else None),
        news_csv_sha256=(_sha256_of_file(news_csv_path) if news_csv_path is not None else None),
        parameters=dict(parameters), python_version=sys.version, command=command, is_demo=is_demo,
    )
    atomic_write_json(run_dir / "run_manifest.json", manifest.to_dict())

    reporter = RunReporter(run_dir, manifest)
    reporter._flush_status()
    return reporter


def read_run_manifest(run_dir: Path | str) -> RunManifest:
    with open(Path(run_dir) / "run_manifest.json", encoding="utf-8") as f:
        return RunManifest.from_dict(json.load(f))


def read_live_status(run_dir: Path | str) -> LiveStatus:
    with open(Path(run_dir) / "live_status.json", encoding="utf-8") as f:
        return LiveStatus.from_dict(json.load(f))


def read_leaderboard(run_dir: Path | str) -> Leaderboard | None:
    path = Path(run_dir) / "leaderboard.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return Leaderboard.from_dict(json.load(f))


def read_progress_events(run_dir: Path | str, *, limit: int | None = None) -> list[dict]:
    """Every event in ``progress_events.jsonl``, oldest first -- or, if
    ``limit`` is given, only the last ``limit`` lines (a cheap tail,
    never re-parsing the whole file's semantics beyond line-splitting).
    A line that fails to parse as JSON is skipped (never raises) -- an
    event log being tailed live can, in principle, be read mid-append by
    a caller not using this module's own append_jsonl; skipping a torn
    trailing line is safer than crashing the reader."""
    path = Path(run_dir) / "progress_events.jsonl"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if limit is not None:
        lines = lines[-limit:]
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def list_run_ids(runs_dir: Path | str) -> list[str]:
    """Every run_id under ``runs_dir`` with a readable ``run_manifest.
    json``, sorted by that manifest's own ``created_at`` descending
    (newest first) -- directories without a manifest (e.g. still being
    created) are silently skipped."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    candidates = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            manifest = read_run_manifest(child)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
        candidates.append((manifest.created_at, child.name))
    candidates.sort(reverse=True)
    return [run_id for _created_at, run_id in candidates]
