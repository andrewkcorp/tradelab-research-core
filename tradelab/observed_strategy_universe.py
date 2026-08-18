"""Data-driven, exhaustive-on-real-events strategy space generator.

Builds every ``StrategySpec`` (primary entry x required confluence
subset x bias policy x TP choice x SL choice) that could ever actually
fire against a given, already-prepared ``PreparedBacktestContext`` --
never an abstract combinatorial sweep over the full 438-selector
canonical catalog.

**"Every possible combination on real events" is not ``2**438``.** The
canonical selector catalog (``tradelab.selector_catalog.ALL_CANONICAL_
ENTRY_SELECTORS``) names 438 selector *families* this package's
detectors can ever produce -- it says nothing about which of them were
ever simultaneously *active* in a specific dataset. A required-
confluence subset is only ever a real strategy candidate if every one
of its selectors was genuinely active, in the same direction, at some
real primary signal's own ``known_at`` -- this module discovers that
observed co-activity directly from ``context.timeline`` (via the exact
same ``tradelab.entry_signal_timeline.query_active_signals`` mechanism
a real backtest already uses to evaluate confluence) rather than
enumerating subsets of the full catalog and hoping most of them turn out
to be reachable. In real market data, the number of selectors *actually*
co-active at any one instant is small and sparse compared to 438, so the
real, reachable combination space -- while it can still be large -- is
many orders of magnitude smaller than the abstract power set. **There is
still no artificial cap on how many required confluences a generated
strategy may carry** (this module never sets ``max_required_confluences``
in the sense ``tradelab.strategy_spec.iter_strategy_specs`` uses that
parameter) -- the only bound is what the data itself ever actually
produced.

**Two phases, deliberately split, mirroring ``tradelab.backtest``'s own
"preparation vs. execution" split:**

1. ``discover_observed_universe(context)`` -- a single, eager pass over
   every real ``EntrySignal`` in ``context.timeline``, producing an
   ``ObservedUniverse``: the deduplicated set of every really-observed
   ``(direction, primary_entry_selector, required_confluence_selectors)``
   combination (an ``ObservedConfluenceCombo``, still no TP/SL/bias
   attached), plus the empirical FVG-boundary-SL variant bounds (see
   below) and a few audit counts. This *is* eagerly materialized -- but
   it is the bounded, "real observed combos" layer the module docstring
   above distinguishes from the combinatorial explosion; it is never as
   large as the full expanded strategy space, since it carries no TP/SL/
   bias multiplication yet.
2. ``iter_observed_strategy_specs(universe)`` -- a lazy generator over
   that already-discovered, bounded set of combos, expanding each one
   across all three ``BiasPolicy`` values, all four TP choices
   (liquidity rank 1/2/3, nearest data-wick), and every SL choice
   (liquidity rank 1/2/3, plus one ``FVG_BOUNDARY`` variant per
   qualifying own fvg-5m/15m selector up to its own empirical bound) --
   this is the layer that can be astronomically larger than step 1's own
   combo count, and it is never materialized: calling this function runs
   none of its body, and each ``StrategySpec`` is only constructed the
   instant a caller asks for the next one.

``StrategyUniverseGenerator`` (the stateful, SQLite-backed class) wraps
both phases with exact-once, resumable checkpointing of *emitted*
``strategy_id``s -- see its own docstring for the full contract. Its
``.run()`` yields plain ``StrategySpec`` objects, so it can be passed
directly as the ``strategy_specs`` argument to ``tradelab.research_
journal.ResearchJournal.run(strategy_specs, context)`` unchanged.

**Empirical FVG-boundary-SL variant bounds** (this task's own
requirement): ``SLChoice(source_type=FVG_BOUNDARY, fvg_variant_number=N)``
has no *artificial* global cap on ``N`` -- but generating every ``N``
from 1 to infinity would be nonsensical, since ``tradelab.sl_targets.
detect_sl_target_candidates`` can never return more qualifying boundaries
than there are currently-active ``EntrySignal``s of the linked fvg-5m/15m
selector at any one real entry instant. For each ``(fvg_link_selector,
direction)`` pair this module tracks, across every real primary-signal
occurrence, the maximum number of *simultaneously active*, same-
direction signals of that selector ever observed (via the same ``query_
active_signals`` call already used for confluence discovery -- no extra
detector run) -- a provably *safe upper bound*: ``len(sl_result.
fvg_targets)`` can never exceed the count of currently-active candidates
that feed it, so a variant number above this bound could never produce a
qualifying boundary anywhere in this context, and every variant number up
to and including the bound is never wrongly excluded either. This bound
may occasionally be a little generous (a handful of the active
candidates at a given instant might not individually qualify as an SL
boundary, e.g. wrong side relative to entry) -- that only ever costs a
combination that ends up ``MISSING_TARGET`` at backtest time, never a
missed real variant and never a crash.

**Never filters by historical results.** Exactly like ``tradelab.
strategy_spec.iter_strategy_specs``, nothing here inspects P&L, win
rate, or trade count -- only whether a combination is structurally
*reachable* given what the timeline actually contains. That filtering
(minimum trade frequency, ranking) is entirely ``tradelab.research_
runner``/``tradelab.research_journal``'s job, downstream and unchanged.

**Reuses, never duplicates, existing validation and ID construction.**
Every generated ``StrategySpec`` is built and validated by the exact
same ``StrategySpec.__post_init__`` and ``tradelab.strategy_spec.
_build_strategy_id`` that ``iter_strategy_specs`` itself uses internally
(cross-module private-helper reuse, the same established convention
this package already uses elsewhere, e.g. ``tradelab.fvg_touch``
importing ``tradelab.liquidity._validate_chronology``) -- this module
adds no second selector-validation or strategy-id scheme.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterator, Mapping

from tradelab.backtest import PreparedBacktestContext
from tradelab.entry_signal_timeline import query_active_signals
from tradelab.selector_catalog import ALL_CANONICAL_ENTRY_SELECTORS, selector_fixed_direction, selector_timeframe
from tradelab.sl_targets import FVG_SL_TIMEFRAMES, MAX_LIQUIDITY_SL_TARGETS, SLSourceType
from tradelab.strategy_spec import LONG, SHORT, BiasPolicy, SLChoice, StrategySpec, TPChoice, _build_strategy_id
from tradelab.tp_targets import MAX_LIQUIDITY_TARGETS, TPSourceType

UNIVERSE_SCHEMA_VERSION = "1.0"
GENERATOR_POLICY_VERSION = "observed_strategy_universe_v1"

ALL_BIAS_POLICIES: tuple[BiasPolicy, ...] = tuple(BiasPolicy)
ALL_TP_CHOICES: tuple[TPChoice, ...] = tuple(
    TPChoice(source_type=TPSourceType.LIQUIDITY, rank=rank) for rank in range(1, MAX_LIQUIDITY_TARGETS + 1)
) + (TPChoice(source_type=TPSourceType.DATA_WICK, rank=None),)

_DIRECTIONS: tuple[str, ...] = (LONG, SHORT)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS universe_manifest (
    run_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    context_artifact_id TEXT NOT NULL,
    selector_catalog_fingerprint TEXT NOT NULL,
    generator_policy_version TEXT NOT NULL,
    primary_pair_count INTEGER NOT NULL,
    primary_occurrence_count INTEGER NOT NULL,
    combo_count INTEGER NOT NULL,
    fvg_boundary_bounds_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emitted_strategies (
    strategy_id TEXT PRIMARY KEY,
    committed_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ObservedConfluenceCombo:
    """One really-observed ``(direction, primary_entry_selector,
    required_confluence_selectors)`` triple -- every selector in
    ``required_confluence_selectors`` was genuinely active, in this same
    ``direction``, at the ``known_at`` of at least one real primary
    signal occurrence of ``primary_entry_selector``. No TP/SL/bias
    attached yet -- that expansion is ``iter_observed_strategy_specs``'s
    job."""

    direction: str
    primary_entry_selector: str
    required_confluence_selectors: tuple[str, ...]


@dataclass(frozen=True)
class ObservedUniverse:
    """The result of one ``discover_observed_universe`` pass: the
    deduplicated, deterministically-ordered set of every observed
    confluence combo, the empirical FVG-boundary-SL bounds keyed by
    ``(fvg_link_selector, direction)``, and a few audit counts. Small
    and bounded by what the data actually contains -- never the full
    TP/SL/bias-expanded strategy space (see module docstring)."""

    combos: tuple[ObservedConfluenceCombo, ...]
    fvg_boundary_bounds: Mapping[tuple[str, str], int]
    primary_pair_count: int
    primary_occurrence_count: int
    distinct_pool_count: int


def discover_observed_universe(context: PreparedBacktestContext) -> ObservedUniverse:
    """Walk every real ``EntrySignal`` in ``context.timeline`` exactly
    once and return the bounded, deduplicated ``ObservedUniverse`` it
    actually supports -- see the module docstring for why this eager
    pass is safe (it is the small, real-data-bounded layer, not the
    combinatorial TP/SL/bias expansion) and for the empirical FVG-
    boundary-SL bound derivation.
    """
    if not isinstance(context, PreparedBacktestContext):
        raise ValueError(f"context must be a PreparedBacktestContext, got {context!r}")

    timeline = context.timeline
    all_selectors = tuple(sorted(timeline.signals_by_selector.keys()))
    all_selectors_set = frozenset(all_selectors)

    seen_combo_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    combos: list[ObservedConfluenceCombo] = []
    fvg_bounds: dict[tuple[str, str], int] = {}
    primary_pair_count = 0
    primary_occurrence_count = 0
    distinct_pool_count = 0

    for primary in all_selectors:
        fixed_direction = selector_fixed_direction(primary)
        directions = (fixed_direction,) if fixed_direction is not None else _DIRECTIONS
        for direction in directions:
            occurrences = [signal for signal in timeline.signals_by_selector[primary] if signal.direction == direction]
            if not occurrences:
                continue
            primary_pair_count += 1

            seen_pools: set[frozenset[str]] = set()
            for signal in occurrences:
                primary_occurrence_count += 1
                query_result = query_active_signals(timeline, all_selectors_set, direction, signal.known_at)

                active_by_selector: dict[str, list] = {}
                for active_signal in query_result.active_signals:
                    active_by_selector.setdefault(active_signal.selector, []).append(active_signal)

                # Empirical FVG-boundary-SL bound bookkeeping: every fvg-5m/15m
                # selector actually active at this real instant -- whether it
                # is this occurrence's own primary or a confluence -- has its
                # own active count folded into the running max for (selector,
                # direction). Reuses the exact same query result already
                # computed above; no extra detector call.
                for selector, active_list in active_by_selector.items():
                    if selector.startswith("fvg-") and selector_timeframe(selector) in FVG_SL_TIMEFRAMES:
                        key = (selector, direction)
                        fvg_bounds[key] = max(fvg_bounds.get(key, 0), len(active_list))

                pool = frozenset(active_by_selector.keys()) - {primary}
                if pool in seen_pools:
                    continue
                seen_pools.add(pool)
                distinct_pool_count += 1

                pool_sorted = tuple(sorted(pool))
                for size in range(0, len(pool_sorted) + 1):
                    for confluence in combinations(pool_sorted, size):
                        combo_key = (direction, primary, confluence)
                        if combo_key in seen_combo_keys:
                            continue
                        seen_combo_keys.add(combo_key)
                        combos.append(ObservedConfluenceCombo(direction, primary, confluence))

    combos.sort(key=lambda combo: (combo.direction, combo.primary_entry_selector, combo.required_confluence_selectors))
    return ObservedUniverse(
        combos=tuple(combos),
        fvg_boundary_bounds=dict(fvg_bounds),
        primary_pair_count=primary_pair_count,
        primary_occurrence_count=primary_occurrence_count,
        distinct_pool_count=distinct_pool_count,
    )


def _sl_choices_for_combo(combo: ObservedConfluenceCombo, fvg_boundary_bounds: Mapping[tuple[str, str], int]) -> Iterator[SLChoice]:
    for rank in range(1, MAX_LIQUIDITY_SL_TARGETS + 1):
        yield SLChoice(source_type=SLSourceType.LIQUIDITY, rank=rank)

    candidate_selectors = (combo.primary_entry_selector,) + combo.required_confluence_selectors
    seen: set[str] = set()
    for selector in candidate_selectors:
        if selector in seen:
            continue
        seen.add(selector)
        if not selector.startswith("fvg-") or selector_timeframe(selector) not in FVG_SL_TIMEFRAMES:
            continue
        bound = fvg_boundary_bounds.get((selector, combo.direction), 0)
        for variant in range(1, bound + 1):
            yield SLChoice(source_type=SLSourceType.FVG_BOUNDARY, fvg_link_selector=selector, fvg_variant_number=variant)


def iter_observed_strategy_specs(universe: ObservedUniverse) -> Iterator[StrategySpec]:
    """Lazily yield every valid ``StrategySpec`` obtainable by expanding
    ``universe.combos`` across all bias policies, TP choices, and SL
    choices (liquidity ranks + empirically-bounded FVG boundary
    variants) -- a generator function: calling it runs none of its body,
    and materializes nothing beyond ``universe`` itself (already small
    and bounded -- see ``discover_observed_universe``). Structurally
    invalid combinations (same validation ``StrategySpec.__post_init__``
    already enforces) are silently skipped, exactly like ``tradelab.
    strategy_spec.iter_strategy_specs`` does for its own combinations --
    this is never a historical-results filter.
    """
    for combo in universe.combos:
        for bias_policy in ALL_BIAS_POLICIES:
            for tp_choice in ALL_TP_CHOICES:
                for sl_choice in _sl_choices_for_combo(combo, universe.fvg_boundary_bounds):
                    try:
                        strategy_id = _build_strategy_id(
                            combo.direction, combo.primary_entry_selector, combo.required_confluence_selectors, bias_policy, tp_choice, sl_choice
                        )
                        spec = StrategySpec(
                            strategy_id=strategy_id,
                            direction=combo.direction,
                            primary_entry_selector=combo.primary_entry_selector,
                            required_confluence_selectors=combo.required_confluence_selectors,
                            bias_policy=bias_policy,
                            tp_choice=tp_choice,
                            sl_choice=sl_choice,
                        )
                    except ValueError:
                        continue
                    yield spec


@dataclass(frozen=True)
class UniverseManifest:
    """Everything that identifies *this* universe-generation run -- see
    the module docstring for what each field means. ``fvg_boundary_
    bounds`` is stored as ``{"selector|direction": bound}`` (plain,
    JSON-stable string keys) rather than a dict keyed by tuples, which
    JSON cannot represent natively."""

    run_id: str
    schema_version: str
    context_artifact_id: str
    selector_catalog_fingerprint: str
    generator_policy_version: str
    primary_pair_count: int
    primary_occurrence_count: int
    combo_count: int
    fvg_boundary_bounds: dict[str, int]


@dataclass(frozen=True)
class UniverseProgress:
    run_id: str
    emitted_count: int
    last_committed_at: str | None


class StrategyUniverseGenerator:
    """A resumable, crash-safe SQLite checkpoint layer over ``iter_
    observed_strategy_specs``. Construction runs ``discover_observed_
    universe(context)`` once (see that function's own docstring for why
    this bounded, eager pass is safe) and writes (fresh run) or verifies
    (resume) a single ``universe_manifest`` row identifying this run --
    the discovered universe's own counts and FVG bounds are folded into
    the manifest's own identity fields, so resuming against a
    meaningfully different context is blocked exactly like a mismatched
    ``context_artifact_id`` would be, even if a caller forgot to bump
    that id themselves.

    ``.run()`` streams ``StrategySpec`` objects, committing each newly-
    seen ``strategy_id`` to the ``emitted_strategies`` table in its own
    transaction *before* yielding it -- so a crash can only ever lose
    work that was never durably checkpointed, and a resumed run never
    re-emits an already-registered ``strategy_id``, exactly mirroring
    ``tradelab.research_journal.ResearchJournal``'s own checkpoint
    contract. The result is a plain iterator of ``StrategySpec``, so it
    can be passed directly as ``ResearchJournal.run(strategy_specs,
    context)``'s own ``strategy_specs`` argument, unchanged.

    Never persists ``context`` itself or any market data -- only counts,
    bounds, and the emitted ``strategy_id`` strings.
    """

    def __init__(self, db_path: Path | str, *, context_artifact_id: str, context: PreparedBacktestContext) -> None:
        if not context_artifact_id:
            raise ValueError("context_artifact_id must be non-empty")
        if not isinstance(context, PreparedBacktestContext):
            raise ValueError(f"context must be a PreparedBacktestContext, got {context!r}")

        self._db_path = Path(db_path)
        self._context_artifact_id = context_artifact_id
        self._universe = discover_observed_universe(context)

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

        self._manifest = self._open_or_verify_manifest()

    @property
    def run_id(self) -> str:
        return self._manifest.run_id

    @property
    def manifest(self) -> UniverseManifest:
        return self._manifest

    @property
    def universe(self) -> ObservedUniverse:
        return self._universe

    def run(self) -> Iterator[StrategySpec]:
        committed = self._load_committed_ids()
        for spec in iter_observed_strategy_specs(self._universe):
            if spec.strategy_id in committed:
                continue
            self._commit(spec.strategy_id)
            committed.add(spec.strategy_id)
            yield spec

    def progress(self) -> UniverseProgress:
        row = self._conn.execute("SELECT COUNT(*), MAX(committed_at) FROM emitted_strategies").fetchone()
        count, last_committed_at = row
        return UniverseProgress(run_id=self.run_id, emitted_count=count or 0, last_committed_at=last_committed_at)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StrategyUniverseGenerator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _load_committed_ids(self) -> set[str]:
        return {row[0] for row in self._conn.execute("SELECT strategy_id FROM emitted_strategies").fetchall()}

    def _commit(self, strategy_id: str) -> str:
        committed_at = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute("INSERT INTO emitted_strategies (strategy_id, committed_at) VALUES (?, ?)", (strategy_id, committed_at))
        return committed_at

    def _open_or_verify_manifest(self) -> UniverseManifest:
        selector_fingerprint = _selector_catalog_fingerprint()
        fvg_bounds_dict = {f"{selector}|{direction}": bound for (selector, direction), bound in sorted(self._universe.fvg_boundary_bounds.items())}
        fields = {
            "schema_version": UNIVERSE_SCHEMA_VERSION,
            "context_artifact_id": self._context_artifact_id,
            "selector_catalog_fingerprint": selector_fingerprint,
            "generator_policy_version": GENERATOR_POLICY_VERSION,
            "primary_pair_count": self._universe.primary_pair_count,
            "primary_occurrence_count": self._universe.primary_occurrence_count,
            "combo_count": len(self._universe.combos),
            "fvg_boundary_bounds": fvg_bounds_dict,
        }
        computed_run_id = _compute_run_id(fields)

        row = self._conn.execute(
            "SELECT run_id, schema_version, context_artifact_id, selector_catalog_fingerprint, generator_policy_version, "
            "primary_pair_count, primary_occurrence_count, combo_count, fvg_boundary_bounds_json FROM universe_manifest"
        ).fetchone()

        if row is None:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO universe_manifest (run_id, schema_version, context_artifact_id, selector_catalog_fingerprint, "
                    "generator_policy_version, primary_pair_count, primary_occurrence_count, combo_count, fvg_boundary_bounds_json, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        computed_run_id,
                        UNIVERSE_SCHEMA_VERSION,
                        self._context_artifact_id,
                        selector_fingerprint,
                        GENERATOR_POLICY_VERSION,
                        fields["primary_pair_count"],
                        fields["primary_occurrence_count"],
                        fields["combo_count"],
                        json.dumps(fvg_bounds_dict, sort_keys=True),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            return UniverseManifest(run_id=computed_run_id, **fields)

        (
            stored_run_id,
            stored_schema,
            stored_artifact,
            stored_fingerprint,
            stored_policy,
            stored_pp,
            stored_po,
            stored_cc,
            stored_bounds_json,
        ) = row
        stored_fields = {
            "schema_version": stored_schema,
            "context_artifact_id": stored_artifact,
            "selector_catalog_fingerprint": stored_fingerprint,
            "generator_policy_version": stored_policy,
            "primary_pair_count": stored_pp,
            "primary_occurrence_count": stored_po,
            "combo_count": stored_cc,
            "fvg_boundary_bounds": json.loads(stored_bounds_json),
        }
        mismatched = sorted(key for key in fields if fields[key] != stored_fields[key])
        if mismatched:
            raise ValueError(
                f"cannot resume observed strategy universe at {self._db_path}: manifest mismatch in field(s) {mismatched} "
                f"against the existing run (run_id={stored_run_id!r}) -- refusing to resume with a different context/universe"
            )
        return UniverseManifest(run_id=stored_run_id, **stored_fields)


def _selector_catalog_fingerprint() -> str:
    return hashlib.sha256("|".join(ALL_CANONICAL_ENTRY_SELECTORS).encode("utf-8")).hexdigest()


def _compute_run_id(manifest_fields: dict) -> str:
    canonical = json.dumps(manifest_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
