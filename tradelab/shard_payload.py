"""On-disk, read-only payloads so a worker loads only its own shard.

The parent runs data preparation and entry discovery **once** -- they are
global and cannot be sharded -- then writes what a worker needs:

    <bundle>/plan.json            the partition and its fingerprints
    <bundle>/shared.payload       simulation-only context: bars, news,
                                  liquidity detector results, data wicks
    <bundle>/shard_0000.payload   only the equivalence classes of shard 0
    <bundle>/shard_0001.payload   ...

A worker loads ``shared`` plus exactly one ``shard`` file. It never sees the
other shards' classes, which is the entire point: the universe, not the market
data, is what makes a worker 9.9 GB.

**Why pickle prunes correctly on its own.** A shard file is a pickle of that
shard's classes. Pickle serialises the transitive closure of what it is given,
so an entry's source event, its linked FVG zone-id tuples and its specs travel
with it, while another shard's entries simply are not reachable and are never
written.

**The global FVG zone index is stored once, not per shard.** Every
``ObservedEntry`` points at the same ``fvg_zone_by_id`` dict, so a naive pickle
copies that whole index into every shard file -- measured at 3.13 GB of bundle
for 64 shards. Rebuilding the entries to strip it is not an option (that is the
exact 78-million-object mistake ``docs/dense-class-memory-diagnostic.md``
removed), so the index is instead written once into ``shared.payload`` and
referenced from shard files through pickle's own ``persistent_id`` hook. The
entries are untouched; only the bytes on disk change, and every worker ends up
sharing one in-memory index per process rather than one per shard.

**Nothing here changes results.** A loaded context and a loaded class carry
the same values as the originals; ``ShardPayloadEquivalenceTests`` proves a
sharded run reproduces an unsharded one exactly.
"""
from __future__ import annotations

import json
import pickle
import types
from dataclasses import replace as _dataclass_replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from tradelab.shard_plan import BacktestShardPlan

PLAN_FILENAME = "plan.json"
SHARED_FILENAME = "shared.payload"

# Pickle persistent-id token standing in for the one global FVG zone index.
# Shard files reference it; ``shared.payload`` carries the real thing.
_ZONE_INDEX_TOKEN = "tradelab:fvg_zone_index"


class _SharedRefPickler(pickle.Pickler):
    """Writes a reference, not a copy, wherever ``shared`` itself appears."""

    def __init__(self, file, shared) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._shared = shared  # held so the identity check stays valid

    def persistent_id(self, obj):
        if self._shared is not None and obj is self._shared:
            return _ZONE_INDEX_TOKEN
        return None


class _SharedRefUnpickler(pickle.Unpickler):
    """Resolves that reference back to the already-loaded shared object."""

    def __init__(self, file, shared) -> None:
        super().__init__(file)
        self._shared = shared

    def persistent_load(self, pid):
        if pid == _ZONE_INDEX_TOKEN:
            if self._shared is None:
                raise pickle.UnpicklingError(
                    "shard payload references the shared FVG zone index, but none was supplied -- "
                    "load shared.payload first (see load_shard)"
                )
            return self._shared
        raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")


def shared_fvg_zone_index(equivalence_classes) -> dict | None:
    """The single ``fvg_zone_by_id`` dict every entry points at, if there is one.

    Real discovery builds exactly one and shares it; small fixtures and
    hand-built entries may not, in which case there is nothing to hoist and
    ``None`` disables the optimisation rather than risking a wrong rebind.
    """
    found = None
    for equivalence_class in equivalence_classes:
        for entry in equivalence_class.long_entries + equivalence_class.short_entries:
            index = entry.fvg_zone_by_id
            if index is None:
                continue
            if found is None:
                found = index
            elif index is not found:
                return None  # more than one index in play; do not hoist
    return found

# Context fields exposed as ``types.MappingProxyType``. A mappingproxy cannot
# be pickled, so these are unwrapped on write and re-wrapped on read, leaving a
# loaded context indistinguishable from a freshly prepared one.
_MAPPING_PROXY_FIELDS = (
    "fvg_results",
    "fvg_touch_results",
    "swing_equal_liquidity_results",
    "session_liquidity_results",
)


def shard_filename(shard_index: int) -> str:
    return f"shard_{shard_index:04d}.payload"


def _unwrap_proxies(context):
    replacements = {}
    for name in _MAPPING_PROXY_FIELDS:
        value = getattr(context, name, None)
        if isinstance(value, types.MappingProxyType):
            replacements[name] = dict(value)
    return _dataclass_replace(context, **replacements) if replacements else context


def _rewrap_proxies(context):
    replacements = {}
    for name in _MAPPING_PROXY_FIELDS:
        value = getattr(context, name, None)
        if isinstance(value, dict):
            replacements[name] = types.MappingProxyType(value)
    return _dataclass_replace(context, **replacements) if replacements else context


def write_shard_bundle(
    bundle_dir: Path | str,
    *,
    plan: BacktestShardPlan,
    context,
    equivalence_classes: Sequence,
) -> Path:
    """Write the plan, the shared context and one file per shard.

    ``context`` should already have had its discovery-only inputs released
    (``tradelab.backtest.release_discovery_inputs``) -- writing them would put
    megabytes of detector wrappers into every worker for nothing.
    """
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    (bundle / PLAN_FILENAME).write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    # The zone index is hoisted here so shard files can reference it instead of
    # each carrying a copy. Written as part of the same object so the two can
    # never drift apart or be loaded from mismatched bundles.
    zone_index = shared_fvg_zone_index(equivalence_classes)
    with open(bundle / SHARED_FILENAME, "wb") as handle:
        pickle.dump(
            {"context": _unwrap_proxies(context), "fvg_zone_index": zone_index},
            handle, protocol=pickle.HIGHEST_PROTOCOL,
        )

    by_class_id = {equivalence_class.class_id: equivalence_class for equivalence_class in equivalence_classes}
    for shard in plan.shards:
        missing = [class_id for class_id in shard.class_ids if class_id not in by_class_id]
        if missing:
            raise ValueError(
                f"shard {shard.shard_index} names {len(missing)} class(es) absent from the universe, "
                f"e.g. {missing[:3]}"
            )
        # Written in the shard's own (globally-ordered) sequence, so a worker
        # iterates exactly the order an unsharded run would for these classes.
        selected = tuple(by_class_id[class_id] for class_id in shard.class_ids)
        with open(bundle / shard_filename(shard.shard_index), "wb") as handle:
            _SharedRefPickler(handle, zone_index).dump(selected)
    return bundle


def read_shard_plan(bundle_dir: Path | str) -> BacktestShardPlan:
    return BacktestShardPlan.from_dict(json.loads((Path(bundle_dir) / PLAN_FILENAME).read_text(encoding="utf-8")))


def load_shared_payload(bundle_dir: Path | str) -> tuple:
    """``(context, fvg_zone_index)`` -- the read-only data every worker needs."""
    with open(Path(bundle_dir) / SHARED_FILENAME, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "context" in payload:
        return _rewrap_proxies(payload["context"]), payload.get("fvg_zone_index")
    # A bundle written before the zone index was hoisted: the shard files still
    # carry their own copies, so there is nothing to supply.
    return _rewrap_proxies(payload), None


def load_shared_context(bundle_dir: Path | str):
    return load_shared_payload(bundle_dir)[0]


def load_shard_classes(bundle_dir: Path | str, shard_index: int, *, fvg_zone_index=None) -> tuple:
    path = Path(bundle_dir) / shard_filename(shard_index)
    if not path.is_file():
        raise FileNotFoundError(f"shard {shard_index} payload missing at {path}")
    with open(path, "rb") as handle:
        return tuple(_SharedRefUnpickler(handle, fvg_zone_index).load())


def load_shard(bundle_dir: Path | str, shard_index: int, *, expected_fingerprint: str | None = None):
    """Load one shard: ``(plan, context, classes)``.

    ``expected_fingerprint`` is checked when supplied. Running a worker against
    a payload built from different data or configuration is a correctness
    failure that would otherwise surface as quietly wrong results, so it is
    refused loudly.
    """
    bundle = Path(bundle_dir)
    plan = read_shard_plan(bundle)
    if expected_fingerprint is not None and plan.fingerprint.combined != expected_fingerprint:
        raise ValueError(
            f"shard bundle fingerprint {plan.fingerprint.combined} does not match the expected "
            f"{expected_fingerprint}; the payload was built from different data, config or code"
        )
    context, zone_index = load_shared_payload(bundle)
    return plan, context, load_shard_classes(bundle, shard_index, fvg_zone_index=zone_index)


def bundle_sizes(bundle_dir: Path | str) -> dict[str, Any]:
    """On-disk sizes, for reporting what sharding actually costs in storage."""
    bundle = Path(bundle_dir)
    shards = sorted(bundle.glob("shard_*.payload"))
    shared_bytes = (bundle / SHARED_FILENAME).stat().st_size if (bundle / SHARED_FILENAME).is_file() else 0
    shard_bytes = [path.stat().st_size for path in shards]
    return {
        "shared_bytes": shared_bytes,
        "shard_bytes": shard_bytes,
        "shard_count": len(shards),
        "largest_shard_bytes": max(shard_bytes) if shard_bytes else 0,
        "total_bytes": shared_bytes + sum(shard_bytes),
    }
