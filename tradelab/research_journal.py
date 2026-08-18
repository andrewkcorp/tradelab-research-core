"""Persistent, resumable journal for long ``tradelab.research_runner``
sweeps, built entirely on the standard library's ``sqlite3`` -- no new
dependency, no new strategy/confluence/TP/SL/bias/risk-management/
backtest semantics of any kind. This module only checkpoints and
resumes the exact same ``ResearchEvaluation`` stream ``tradelab.
research_runner.iter_research_evaluations`` already produces.

**Lifecycle: create -> run -> interrupt -> resume -> export.**

1. **Create** a ``ResearchJournal(db_path, ...)`` at a caller-chosen
   local file path. Nothing about this path, or anything written to
   it, is ever committed to this repository -- see ``.gitignore``.
   Construction writes (or, on resume, verifies) a single ``run_
   manifest`` row identifying exactly what this run *is*: a
   deterministic ``run_id`` (a sha256 over every manifest field, so
   identical configuration always reproduces the identical id), the
   journal's own schema version, a caller-supplied ``data_artifact_id``
   (typically a market data artifact's own sha256 -- this module has no
   way to derive that from a bare ``PreparedBacktestContext``, so it is
   never guessed), a caller-supplied ``policy_version`` (e.g. the
   continuous-series normalization method that produced the underlying
   bars) and ``generator_version`` (whatever produced the ``StrategySpec``
   stream -- this module doesn't generate strategies itself and has no
   way to know what did), an automatically-derived ``selector_catalog_
   fingerprint`` (a sha256 over ``tradelab.selector_catalog.ALL_
   CANONICAL_ENTRY_SELECTORS`` as it actually is right now -- never
   hand-versioned, since the real catalog is already right there to
   hash), and the full ``ResearchConfig`` (every window, the risk
   config, the equity config).
2. **Run** ``journal.run(strategy_specs, context)`` -- a generator
   wrapping ``iter_research_evaluations`` unchanged. Every ``(strategy_
   id, instrument)`` result is committed to SQLite in its own
   transaction *before* being yielded back to the caller, so a crash
   can only ever lose work that was never durably checkpointed in the
   first place -- it can never fabricate a checkpoint for work that
   didn't finish. Strategies whose every configured instrument is
   already committed are skipped *before* their (expensive) backtest
   ever runs; a strategy only partially committed (e.g. NQ done, MNQ
   missing after an earlier interruption) is still re-evaluated in
   full, but only its still-missing ``(strategy_id, instrument)`` pairs
   are (re-)committed -- already-committed ones are silently skipped,
   never duplicated (the ``evaluations`` table's own primary key is
   ``(strategy_id, instrument)``).
3. **Interrupt** -- kill the process, let it crash, whatever. Nothing
   special has to happen: every already-completed transaction is
   already durable, and the one (if any) that was in flight was never
   committed, so it simply doesn't exist yet.
4. **Resume** -- construct a new ``ResearchJournal`` at the same
   ``db_path`` with the exact same manifest-defining arguments, and call
   ``run(...)`` again with the same (or a superset of the same)
   ``strategy_specs``. Already-committed pairs are skipped as described
   above. If *any* manifest field differs from what's already stored --
   a different data artifact, a different research config, even a
   different window -- construction itself raises ``ValueError``
   immediately, naming exactly which field(s) mismatched, and refuses
   to touch the existing checkpoint data at all.
5. **Export** ``journal.export_qualifying(path, fmt="csv"|"jsonl")`` --
   a deterministic snapshot of every currently-qualified evaluation,
   sorted by ``realized_net_pnl_usd`` descending, then ``max_drawdown_
   usd`` ascending, then ``trade_count`` descending, then ``strategy_id``
   ascending. This is a **sortable export, not a final ranking** -- this
   module declares no "winner" and adds no selection criterion beyond
   what ``tradelab.research_runner`` already applies.

**This remains research-only.** Nothing in this module places, sizes,
or manages a live order, and nothing here changes what "qualifies",
what a trade is, or how P&L/drawdown/frequency are computed -- all of
that is entirely ``tradelab.research_runner``'s (and, beneath it,
``tradelab.backtest``/``tradelab.accounting``/``tradelab.equity``'s)
existing, unmodified responsibility. This module only remembers, across
process restarts, which ``(strategy, instrument)`` pairs of that
existing computation have already been durably completed.

**Frequency windows are always explicit, never guessed.** Exactly like
``tradelab.research_runner.ResearchWindow`` itself, this module never
invents or infers a window boundary -- the ``ResearchConfig`` bound to
a journal at construction is whatever the caller explicitly built (e.g.
two full, explicit calendar-year windows, ``2024-01-01T00:00:00Z``
through ``2026-01-01T00:00:00Z``, split at the year boundary) and is
recorded verbatim in the manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

from tradelab.accounting import Instrument
from tradelab.backtest import PreparedBacktestContext
from tradelab.research_runner import ResearchConfig, ResearchEvaluation, iter_research_evaluations
from tradelab.selector_catalog import ALL_CANONICAL_ENTRY_SELECTORS
from tradelab.strategy_spec import StrategySpec

JOURNAL_SCHEMA_VERSION = "1.0"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS run_manifest (
    run_id TEXT PRIMARY KEY,
    journal_schema_version TEXT NOT NULL,
    data_artifact_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    generator_version TEXT NOT NULL,
    selector_catalog_fingerprint TEXT NOT NULL,
    research_config_json TEXT NOT NULL,
    instruments_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    strategy_id TEXT NOT NULL,
    instrument TEXT NOT NULL,
    status TEXT NOT NULL,
    strategy_spec_json TEXT NOT NULL,
    rejection_reason TEXT,
    window_trade_counts_json TEXT NOT NULL,
    realized_net_pnl_usd REAL,
    roi REAL,
    max_drawdown_usd REAL,
    drawdown_episode_count INTEGER,
    trade_count INTEGER,
    entry_costs_usd_total REAL,
    committed_at TEXT NOT NULL,
    PRIMARY KEY (strategy_id, instrument)
);
"""


class EvaluationStatus(str, Enum):
    QUALIFIED = "qualified"
    REJECTED = "rejected"
    PREFILTERED = "prefiltered"


@dataclass(frozen=True)
class RunManifest:
    """Everything that identifies *this* run, exactly as stored in
    ``run_manifest`` -- see the module docstring for what each field
    means and why it is never guessed."""

    run_id: str
    journal_schema_version: str
    data_artifact_id: str
    policy_version: str
    generator_version: str
    selector_catalog_fingerprint: str
    research_config: dict
    instruments: tuple[str, ...]


@dataclass(frozen=True)
class JournalEntry:
    """One freshly-committed checkpoint, yielded by ``ResearchJournal.
    run`` the instant its transaction durably commits -- never before."""

    strategy_id: str
    instrument: Instrument
    status: EvaluationStatus
    committed_at: str


@dataclass(frozen=True)
class LastCheckpoint:
    strategy_id: str
    instrument: str
    committed_at: str


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    completed_count: int
    qualified_count: int
    rejected_count: int
    prefiltered_count: int
    last_checkpoint: LastCheckpoint | None


class ResearchJournal:
    """A resumable, crash-safe SQLite checkpoint log over ``tradelab.
    research_runner.iter_research_evaluations``. See the module
    docstring for the full create/run/interrupt/resume/export lifecycle.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        data_artifact_id: str,
        policy_version: str,
        generator_version: str,
        research_config: ResearchConfig,
        instruments: Iterable[Instrument],
    ) -> None:
        if not data_artifact_id:
            raise ValueError("data_artifact_id must be non-empty")
        if not policy_version:
            raise ValueError("policy_version must be non-empty")
        if not generator_version:
            raise ValueError("generator_version must be non-empty")
        if not isinstance(research_config, ResearchConfig):
            raise ValueError(f"research_config must be a ResearchConfig, got {research_config!r}")
        instruments = tuple(instruments)
        if not instruments:
            raise ValueError("instruments must not be empty")

        self._db_path = Path(db_path)
        self._research_config = research_config
        self._instruments = instruments

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

        self._manifest = self._open_or_verify_manifest(
            data_artifact_id=data_artifact_id,
            policy_version=policy_version,
            generator_version=generator_version,
            research_config=research_config,
            instruments=instruments,
        )

    @property
    def run_id(self) -> str:
        return self._manifest.run_id

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    def run(self, strategy_specs: Iterable[StrategySpec], context: PreparedBacktestContext) -> Iterator[JournalEntry]:
        """Lazily stream already-committed ``(strategy_id, instrument)``
        pairs out of the underlying ``iter_research_evaluations`` sweep,
        committing and yielding exactly one ``JournalEntry`` per newly
        completed pair. Never materializes ``strategy_specs`` -- a
        strategy is skipped *before* its backtest runs only once every
        one of ``self._instruments`` is already committed for it;
        otherwise it is re-evaluated in full, but any of its pairs
        already committed are silently skipped rather than recommitted.
        """
        committed_pairs = self._load_committed_pairs()
        fully_done = self._fully_done_strategy_ids(committed_pairs)

        def _remaining() -> Iterator[StrategySpec]:
            for spec in strategy_specs:
                if spec.strategy_id in fully_done:
                    continue
                yield spec

        last_pair: tuple[str, str] | None = None
        try:
            for evaluation in iter_research_evaluations(_remaining(), context, self._research_config, self._instruments):
                pair = (evaluation.strategy_id, evaluation.instrument.value)
                last_pair = pair
                if pair in committed_pairs:
                    continue
                committed_at = self._commit_evaluation(evaluation)
                committed_pairs.add(pair)
                yield JournalEntry(
                    strategy_id=evaluation.strategy_id,
                    instrument=evaluation.instrument,
                    status=_status_for(evaluation),
                    committed_at=committed_at,
                )
        except Exception as error:
            raise RuntimeError(
                f"research journal run (run_id={self.run_id!r}) failed while processing {last_pair!r}; "
                f"every checkpoint committed before this failure remains intact in {self._db_path}"
            ) from error

    def status(self) -> RunStatus:
        counts = dict(self._conn.execute("SELECT status, COUNT(*) FROM evaluations GROUP BY status").fetchall())
        last_row = self._conn.execute(
            "SELECT strategy_id, instrument, committed_at FROM evaluations ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        last_checkpoint = LastCheckpoint(strategy_id=last_row[0], instrument=last_row[1], committed_at=last_row[2]) if last_row else None
        return RunStatus(
            run_id=self.run_id,
            completed_count=sum(counts.values()),
            qualified_count=counts.get(EvaluationStatus.QUALIFIED.value, 0),
            rejected_count=counts.get(EvaluationStatus.REJECTED.value, 0),
            prefiltered_count=counts.get(EvaluationStatus.PREFILTERED.value, 0),
            last_checkpoint=last_checkpoint,
        )

    def export_qualifying(self, path: Path | str, fmt: str = "csv") -> int:
        """Deterministically export every currently-``QUALIFIED`` row to
        ``path`` (``fmt="csv"`` or ``"jsonl"``), ordered by ``realized_
        net_pnl_usd`` descending, then ``max_drawdown_usd`` ascending,
        then ``trade_count`` descending, then ``strategy_id`` ascending
        -- a sortable export, never a declared "winner" (see module
        docstring). Written atomically (temp file + rename) so a failed
        export never leaves a partial file at ``path``. Returns the
        number of rows exported.
        """
        if fmt not in ("csv", "jsonl"):
            raise ValueError(f"fmt must be 'csv' or 'jsonl', got {fmt!r}")

        rows = self._conn.execute(
            """
            SELECT strategy_id, instrument, strategy_spec_json, realized_net_pnl_usd, roi,
                   max_drawdown_usd, drawdown_episode_count, trade_count, entry_costs_usd_total
            FROM evaluations
            WHERE status = ?
            ORDER BY realized_net_pnl_usd DESC, max_drawdown_usd ASC, trade_count DESC, strategy_id ASC
            """,
            (EvaluationStatus.QUALIFIED.value,),
        ).fetchall()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        try:
            if fmt == "csv":
                _write_csv(tmp_path, rows)
            else:
                _write_jsonl(tmp_path, rows)
            os.replace(str(tmp_path), str(path))
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        return len(rows)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResearchJournal":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _open_or_verify_manifest(
        self,
        *,
        data_artifact_id: str,
        policy_version: str,
        generator_version: str,
        research_config: ResearchConfig,
        instruments: tuple[Instrument, ...],
    ) -> RunManifest:
        selector_fingerprint = _selector_catalog_fingerprint()
        research_config_dict = _serialize_research_config(research_config)
        instruments_sorted = tuple(sorted(instrument.value for instrument in instruments))
        fields = {
            "journal_schema_version": JOURNAL_SCHEMA_VERSION,
            "data_artifact_id": data_artifact_id,
            "policy_version": policy_version,
            "generator_version": generator_version,
            "selector_catalog_fingerprint": selector_fingerprint,
            "research_config": research_config_dict,
            "instruments": list(instruments_sorted),
        }
        computed_run_id = _compute_run_id(fields)

        row = self._conn.execute(
            "SELECT run_id, journal_schema_version, data_artifact_id, policy_version, generator_version, "
            "selector_catalog_fingerprint, research_config_json, instruments_json FROM run_manifest"
        ).fetchone()

        if row is None:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO run_manifest (run_id, journal_schema_version, data_artifact_id, policy_version, "
                    "generator_version, selector_catalog_fingerprint, research_config_json, instruments_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        computed_run_id,
                        JOURNAL_SCHEMA_VERSION,
                        data_artifact_id,
                        policy_version,
                        generator_version,
                        selector_fingerprint,
                        json.dumps(research_config_dict, sort_keys=True),
                        json.dumps(list(instruments_sorted), sort_keys=True),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            return RunManifest(
                run_id=computed_run_id,
                journal_schema_version=JOURNAL_SCHEMA_VERSION,
                data_artifact_id=data_artifact_id,
                policy_version=policy_version,
                generator_version=generator_version,
                selector_catalog_fingerprint=selector_fingerprint,
                research_config=research_config_dict,
                instruments=instruments_sorted,
            )

        (
            stored_run_id,
            stored_schema,
            stored_artifact,
            stored_policy,
            stored_generator,
            stored_selector_fp,
            stored_rc_json,
            stored_instruments_json,
        ) = row
        stored_fields = {
            "journal_schema_version": stored_schema,
            "data_artifact_id": stored_artifact,
            "policy_version": stored_policy,
            "generator_version": stored_generator,
            "selector_catalog_fingerprint": stored_selector_fp,
            "research_config": json.loads(stored_rc_json),
            "instruments": json.loads(stored_instruments_json),
        }
        mismatched = sorted(key for key in fields if fields[key] != stored_fields[key])
        if mismatched:
            raise ValueError(
                f"cannot resume research journal at {self._db_path}: manifest mismatch in field(s) {mismatched} "
                f"against the existing run (run_id={stored_run_id!r}) -- refusing to resume with a different configuration"
            )
        return RunManifest(
            run_id=stored_run_id,
            journal_schema_version=stored_schema,
            data_artifact_id=stored_artifact,
            policy_version=stored_policy,
            generator_version=stored_generator,
            selector_catalog_fingerprint=stored_selector_fp,
            research_config=stored_fields["research_config"],
            instruments=tuple(stored_fields["instruments"]),
        )

    def _load_committed_pairs(self) -> set[tuple[str, str]]:
        rows = self._conn.execute("SELECT strategy_id, instrument FROM evaluations").fetchall()
        return {(strategy_id, instrument) for strategy_id, instrument in rows}

    def _fully_done_strategy_ids(self, committed_pairs: set[tuple[str, str]]) -> set[str]:
        by_strategy: dict[str, set[str]] = {}
        for strategy_id, instrument_value in committed_pairs:
            by_strategy.setdefault(strategy_id, set()).add(instrument_value)
        required = {instrument.value for instrument in self._instruments}
        return {strategy_id for strategy_id, seen in by_strategy.items() if required <= seen}

    def _commit_evaluation(self, evaluation: ResearchEvaluation) -> str:
        status = _status_for(evaluation)
        strategy_spec_json = json.dumps(evaluation.strategy.to_dict(), sort_keys=True)
        window_trade_counts_json = json.dumps(
            [
                {"window_id": wtc.window_id, "trade_count": wtc.trade_count, "meets_minimum": wtc.meets_minimum}
                for wtc in evaluation.window_trade_counts
            ],
            sort_keys=True,
        )

        realized_net_pnl_usd = roi = max_drawdown_usd = None
        drawdown_episode_count = trade_count = entry_costs_usd_total = None
        if status == EvaluationStatus.QUALIFIED:
            equity = evaluation.equity_result
            account = evaluation.account_result
            realized_net_pnl_usd = equity.net_cash_pnl_usd
            roi = equity.net_cash_pnl_usd / self._research_config.equity_config.initial_balance_usd
            max_drawdown_usd = equity.max_drawdown_usd
            drawdown_episode_count = equity.drawdown_over_threshold_episode_count
            trade_count = account.trade_count
            entry_costs_usd_total = account.entry_costs_charged_usd_total

        committed_at = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT INTO evaluations (strategy_id, instrument, status, strategy_spec_json, rejection_reason, "
                "window_trade_counts_json, realized_net_pnl_usd, roi, max_drawdown_usd, drawdown_episode_count, "
                "trade_count, entry_costs_usd_total, committed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evaluation.strategy_id,
                    evaluation.instrument.value,
                    status.value,
                    strategy_spec_json,
                    evaluation.rejection_reason.value if evaluation.rejection_reason is not None else None,
                    window_trade_counts_json,
                    realized_net_pnl_usd,
                    roi,
                    max_drawdown_usd,
                    drawdown_episode_count,
                    trade_count,
                    entry_costs_usd_total,
                    committed_at,
                ),
            )
        return committed_at


def _status_for(evaluation: ResearchEvaluation) -> EvaluationStatus:
    if evaluation.qualifies_frequency:
        return EvaluationStatus.QUALIFIED
    if evaluation.prefiltered:
        return EvaluationStatus.PREFILTERED
    return EvaluationStatus.REJECTED


def _selector_catalog_fingerprint() -> str:
    return hashlib.sha256("|".join(ALL_CANONICAL_ENTRY_SELECTORS).encode("utf-8")).hexdigest()


def _serialize_research_config(research_config: ResearchConfig) -> dict:
    windows = sorted(research_config.windows, key=lambda w: w.window_id)
    return {
        "min_trades_per_window": research_config.min_trades_per_window,
        "windows": [
            {"window_id": w.window_id, "start_utc": w.start_utc.isoformat(), "end_utc": w.end_utc.isoformat()} for w in windows
        ],
        "risk_config": {
            "target_risk_usd": research_config.risk_config.target_risk_usd,
            "max_stop_risk_usd": research_config.risk_config.max_stop_risk_usd,
            "entry_cost_usd": research_config.risk_config.entry_cost_usd,
        },
        "equity_config": {
            "initial_balance_usd": research_config.equity_config.initial_balance_usd,
            "drawdown_threshold_usd": research_config.equity_config.drawdown_threshold_usd,
        },
    }


def _compute_run_id(manifest_fields: dict) -> str:
    canonical = json.dumps(manifest_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_EXPORT_COLUMNS = (
    "strategy_id",
    "instrument",
    "realized_net_pnl_usd",
    "roi",
    "max_drawdown_usd",
    "drawdown_episode_count",
    "trade_count",
    "entry_costs_usd_total",
    "strategy_spec",
)


def _write_csv(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_EXPORT_COLUMNS)
        for strategy_id, instrument, strategy_spec_json, pnl, roi, dd, dd_episodes, trades, costs in rows:
            writer.writerow([strategy_id, instrument, pnl, roi, dd, dd_episodes, trades, costs, strategy_spec_json])


def _write_jsonl(path: Path, rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for strategy_id, instrument, strategy_spec_json, pnl, roi, dd, dd_episodes, trades, costs in rows:
            record = {
                "strategy_id": strategy_id,
                "instrument": instrument,
                "realized_net_pnl_usd": pnl,
                "roi": roi,
                "max_drawdown_usd": dd,
                "drawdown_episode_count": dd_episodes,
                "trade_count": trades,
                "entry_costs_usd_total": costs,
                "strategy_spec": json.loads(strategy_spec_json),
            }
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
