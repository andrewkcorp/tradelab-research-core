"""Normalised shard bundles: references, never copies.

A bundle is a shared ``EntryStoreReader``-backed store plus one small file per
shard. A shard file carries **only**:

* class metadata (id, spec objects, entry counts, link selectors),
* ``array('I')`` arrays of ``uint32`` entry ids for the long, short and merged
  streams,
* the fingerprints that tie it to a specific store and plan.

It carries no ``ObservedEntry`` and no FVG zone index. Those live once in the
store; ``load_shard_classes`` materialises only the entries the shard actually
names, and only for as long as the caller holds them.

**Why this replaces the old format.** The previous bundle pickled full entry
objects per shard. Classes share entries 33.6x, so 256 shards of the real
artifact projected to 44.8 GB -- the run was stopped mid-materialisation with
the volume at 98%. References cost 4 bytes each, so the same 9,065,223
references cost ~36 MB in total instead.

**Disk budget.** ``estimate_bundle_bytes`` projects the total from the first
few shards, so an over-budget configuration is refused after writing megabytes
rather than tens of gigabytes.

Every file is written tmp -> fsync -> rename, and each shard names the store
fingerprint it was built against, so a mismatched or half-written bundle fails
loudly before any simulation.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
from array import array
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tradelab.entry_store import (
    EntryStoreError,
    EntryStoreReader,
    build_entry_id_index,
    collect_unique_entries,
    write_entry_store,
)
from tradelab.shard_plan import BacktestShardPlan

BUNDLE_FORMAT_VERSION = "2.0"
PLAN_FILENAME = "plan.json"
CONTEXT_FILENAME = "context.payload"
BUNDLE_MANIFEST_FILENAME = "bundle.json"
STORE_DIRNAME = "store"

_MAPPING_PROXY_FIELDS = (
    "fvg_results", "fvg_touch_results", "swing_equal_liquidity_results", "session_liquidity_results",
)


def shard_filename(shard_index: int) -> str:
    return f"shard_{shard_index:05d}.bin"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _unwrap_proxies(context):
    import types
    from dataclasses import replace

    replacements = {
        name: dict(getattr(context, name))
        for name in _MAPPING_PROXY_FIELDS
        if isinstance(getattr(context, name, None), types.MappingProxyType)
    }
    return replace(context, **replacements) if replacements else context


def _rewrap_proxies(context):
    import types
    from dataclasses import replace

    replacements = {
        name: types.MappingProxyType(getattr(context, name))
        for name in _MAPPING_PROXY_FIELDS
        if isinstance(getattr(context, name, None), dict)
    }
    return replace(context, **replacements) if replacements else context


def _class_record(equivalence_class, entry_ids: Mapping[str, int]) -> dict:
    """One class, with its entry streams reduced to uint32 id arrays."""

    def ids(stream) -> array:
        out = array("I")
        for entry in stream or ():
            out.append(entry_ids[entry.entry_id])
        return out

    return {
        "class_id": equivalence_class.class_id,
        "member_specs": equivalence_class.member_specs,
        "representative_spec": equivalence_class.representative_spec,
        "entry_count_2024": equivalence_class.entry_count_2024,
        "entry_count_2025": equivalence_class.entry_count_2025,
        "fvg_link_selectors": equivalence_class.fvg_link_selectors,
        "long_ids": ids(equivalence_class.long_entries),
        "short_ids": ids(equivalence_class.short_entries),
        "observed_ids": ids(equivalence_class.observed_entries),
    }


def estimate_bundle_bytes(
    written_shard_bytes: Sequence[int], shard_count: int, store_bytes: int, context_bytes: int
) -> int:
    """Project the finished bundle from the shards written so far."""
    if not written_shard_bytes:
        return store_bytes + context_bytes
    mean = sum(written_shard_bytes) / len(written_shard_bytes)
    return int(store_bytes + context_bytes + mean * shard_count)


class BundleBudgetExceeded(RuntimeError):
    """Raised when the projected bundle would breach the disk budget.

    Carries the projection so the caller can report a real number rather than
    "it was too big".
    """

    def __init__(self, message: str, *, projected_bytes: int, budget_bytes: int, free_bytes: int) -> None:
        super().__init__(message)
        self.projected_bytes = projected_bytes
        self.budget_bytes = budget_bytes
        self.free_bytes = free_bytes


def write_bundle(
    bundle_dir: Path | str,
    *,
    plan: BacktestShardPlan,
    context,
    equivalence_classes: Sequence,
    fvg_zone_index: Mapping | None,
    max_bundle_bytes: int | None = None,
    min_free_bytes_after: int | None = None,
    probe_shards: int = 8,
) -> dict[str, Any]:
    """Write the store once, then one small reference file per shard.

    ``max_bundle_bytes`` and ``min_free_bytes_after`` are enforced twice: once
    up front against the store's own measured size, and again after
    ``probe_shards`` shards, when the per-shard cost is known. Both raise
    ``BundleBudgetExceeded`` and leave the partial bundle for inspection --
    the caller decides whether to delete it.
    """
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    entries = collect_unique_entries(equivalence_classes)
    entry_ids = build_entry_id_index(entries)
    store_manifest = write_entry_store(bundle / STORE_DIRNAME, entries, fvg_zone_index=fvg_zone_index)
    # Released before any shard is written: the caller's own universe is still
    # alive, but this local materialisation must not double it.
    del entries

    context_blob = pickle.dumps(_unwrap_proxies(context), protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(bundle / CONTEXT_FILENAME, context_blob)
    _atomic_write_bytes(
        bundle / PLAN_FILENAME,
        json.dumps(plan.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
    )

    store_bytes = store_manifest.data_bytes + store_manifest.index_bytes + store_manifest.pools_bytes

    def check_budget(projected: int) -> None:
        free = shutil.disk_usage(bundle).free
        if max_bundle_bytes is not None and projected > max_bundle_bytes:
            raise BundleBudgetExceeded(
                f"projected bundle {projected / 2**30:.2f} GB exceeds the "
                f"{max_bundle_bytes / 2**30:.2f} GB budget",
                projected_bytes=projected, budget_bytes=max_bundle_bytes, free_bytes=free,
            )
        if min_free_bytes_after is not None:
            already = _bundle_bytes_on_disk(bundle)
            remaining = max(0, projected - already)
            if free - remaining < min_free_bytes_after:
                raise BundleBudgetExceeded(
                    f"finishing the bundle would leave {(free - remaining) / 2**30:.2f} GB free, "
                    f"below the required {min_free_bytes_after / 2**30:.2f} GB",
                    projected_bytes=projected, budget_bytes=max_bundle_bytes or 0, free_bytes=free,
                )

    check_budget(store_bytes + len(context_blob))

    by_class_id = {c.class_id: c for c in equivalence_classes}
    written: list[int] = []
    for position, shard in enumerate(plan.shards):
        missing = [cid for cid in shard.class_ids if cid not in by_class_id]
        if missing:
            raise ValueError(
                f"shard {shard.shard_index} names {len(missing)} class(es) absent from the universe, "
                f"e.g. {missing[:3]}"
            )
        payload = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "shard_index": shard.shard_index,
            "store_fingerprint": store_manifest.fingerprint,
            "plan_fingerprint": plan.fingerprint.combined,
            "classes": [_class_record(by_class_id[cid], entry_ids) for cid in shard.class_ids],
        }
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write_bytes(bundle / shard_filename(shard.shard_index), blob)
        written.append(len(blob))
        if position + 1 == probe_shards:
            check_budget(estimate_bundle_bytes(written, plan.shard_count, store_bytes, len(context_blob)))

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "shard_count": plan.shard_count,
        "store_fingerprint": store_manifest.fingerprint,
        "plan_fingerprint": plan.fingerprint.combined,
        "store": store_manifest.to_dict(),
        "context_bytes": len(context_blob),
        "shard_bytes_total": sum(written),
        "largest_shard_bytes": max(written) if written else 0,
        "entry_references": None,
    }
    _atomic_write_bytes(
        bundle / BUNDLE_MANIFEST_FILENAME, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    )
    return manifest


def _bundle_bytes_on_disk(bundle: Path) -> int:
    total = 0
    for path in bundle.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def bundle_sizes(bundle_dir: Path | str) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    shards = sorted(bundle.glob("shard_*.bin"))
    shard_bytes = [p.stat().st_size for p in shards]
    store = bundle / STORE_DIRNAME
    store_bytes = sum(p.stat().st_size for p in store.glob("*")) if store.is_dir() else 0
    context_bytes = (bundle / CONTEXT_FILENAME).stat().st_size if (bundle / CONTEXT_FILENAME).is_file() else 0
    return {
        "store_bytes": store_bytes,
        "context_bytes": context_bytes,
        "shard_count": len(shards),
        "shard_bytes_total": sum(shard_bytes),
        "largest_shard_bytes": max(shard_bytes) if shard_bytes else 0,
        "total_bytes": store_bytes + context_bytes + sum(shard_bytes),
    }


def read_plan(bundle_dir: Path | str) -> BacktestShardPlan:
    return BacktestShardPlan.from_dict(
        json.loads((Path(bundle_dir) / PLAN_FILENAME).read_text(encoding="utf-8"))
    )


def load_context(bundle_dir: Path | str):
    with open(Path(bundle_dir) / CONTEXT_FILENAME, "rb") as handle:
        return _rewrap_proxies(pickle.load(handle))


def open_store(bundle_dir: Path | str, *, expected_fingerprint: str | None = None) -> EntryStoreReader:
    return EntryStoreReader(Path(bundle_dir) / STORE_DIRNAME, expected_fingerprint=expected_fingerprint)


def load_shard_classes(bundle_dir: Path | str, shard_index: int, store: EntryStoreReader) -> tuple:
    """Materialise one shard's classes, and only the entries they name.

    Entries are materialised once per shard and shared between the streams that
    reference them, exactly as discovery shared them -- so the object graph a
    worker sees matches an unsharded run's. Nothing store-wide is cached: when
    the caller drops the returned classes, the entries go with them.
    """
    from tradelab.both_direction_universe import BothDirectionEquivalenceClass

    path = Path(bundle_dir) / shard_filename(shard_index)
    if not path.is_file():
        raise EntryStoreError(f"shard {shard_index} missing at {path}")
    payload = pickle.loads(path.read_bytes())
    if payload.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise EntryStoreError(
            f"shard {shard_index} has bundle format {payload.get('format_version')!r}, expected "
            f"{BUNDLE_FORMAT_VERSION!r}; a run must never mix bundle formats"
        )
    if payload.get("store_fingerprint") != store.manifest.fingerprint:
        raise EntryStoreError(
            f"shard {shard_index} was built against store {payload.get('store_fingerprint', '')[:16]}... "
            f"but the opened store is {store.manifest.fingerprint[:16]}..."
        )

    needed: set[int] = set()
    for record in payload["classes"]:
        needed.update(record["long_ids"])
        needed.update(record["short_ids"])
        needed.update(record["observed_ids"])
    materialised = store.read_many(sorted(needed))

    classes = []
    for record in payload["classes"]:
        classes.append(
            BothDirectionEquivalenceClass(
                class_id=record["class_id"],
                observed_entries=tuple(materialised[i] for i in record["observed_ids"]),
                long_entries=tuple(materialised[i] for i in record["long_ids"]),
                short_entries=tuple(materialised[i] for i in record["short_ids"]),
                member_specs=record["member_specs"],
                representative_spec=record["representative_spec"],
                entry_count_2024=record["entry_count_2024"],
                entry_count_2025=record["entry_count_2025"],
                fvg_link_selectors=record["fvg_link_selectors"],
            )
        )
    return tuple(classes)


def shard_entry_reference_count(bundle_dir: Path | str, shard_index: int) -> int:
    """How many entry references one shard holds -- for duplication reporting."""
    payload = pickle.loads((Path(bundle_dir) / shard_filename(shard_index)).read_bytes())
    return sum(
        len(r["long_ids"]) + len(r["short_ids"]) + len(r["observed_ids"]) for r in payload["classes"]
    )
