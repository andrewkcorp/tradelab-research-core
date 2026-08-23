"""A normalised, memory-mapped store holding every unique ``ObservedEntry`` once.

The previous bundle format was denormalised: every shard file carried full
copies of the ``ObservedEntry`` objects its classes referenced. Because classes
share entries **33.6x** (96,408 classes reference 270,091 distinct entries
through 9,065,223 references), 256 shards of the real artifact projected to
**44.8 GB** on disk -- the same overlap that makes per-worker memory expensive,
resurfacing as duplication.

This store fixes that at the root. Each unique entry is written exactly once;
shards carry only ``uint32`` references into it.

**Layout** (a directory, written atomically):

    manifest.json      counts, fingerprint, format version
    entries.idx        N x (uint64 offset, uint32 length) -- mmap'd, random access
    entries.dat        one pickled record per entry, in id order
    pools.pkl          the shared, deduplicated parts (see below)

**What is normalised.** Three things are shared across entries and are stored
once rather than per entry:

* the global FVG zone index (``fvg_zone_by_id``) -- one dict for the whole run;
* the per-entry active zone-id maps -- measured at only **29,814 distinct maps
  for 270,091 entries**, so storing them per entry would be almost pure waste;
* the zone-id tuples inside those maps.

An entry record therefore holds only what is genuinely its own: its id string,
``known_at``, direction, its source event, and an index into the shared map
pool. Pools are small enough to load once per process; entry records are read
on demand.

**Stable numeric ids.** Entries are sorted by their own ``entry_id`` string and
numbered from zero. That is independent of dict iteration order, of hash
randomisation and of process restarts, so the same universe always produces the
same ids -- which is what lets a shard reference entries by number at all.

**Random access, not bulk load.** ``EntryStoreReader`` mmaps the index and
seeks into the data file. Materialising one shard touches only that shard's
entries; nothing ever builds a ``dict[entry_id, ObservedEntry]`` spanning the
whole store.

**Read-only during simulation** and safe for both workers to open at once: no
process writes to it after ``write_entry_store`` returns.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

ENTRY_STORE_FORMAT_VERSION = "1.0"

MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "entries.idx"
DATA_FILENAME = "entries.dat"
POOLS_FILENAME = "pools.pkl"

# (offset, length) per entry. Fixed width so the index is a flat, mmap'able
# array addressable by entry id without any parsing.
_INDEX_STRUCT = struct.Struct("<QI")
INDEX_RECORD_SIZE = _INDEX_STRUCT.size  # 12 bytes

MAX_ENTRIES = 2 ** 32 - 1
"""Entry ids are uint32 in shard bundles; refuse a universe that would overflow
rather than silently truncating one."""


def _atomic_write(path: Path, write_body) -> None:
    """tmp -> fsync -> rename, so a crash never leaves a half-written file
    that a later run would happily read."""
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            write_body(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class EntryStoreManifest:
    format_version: str
    entry_count: int
    fingerprint: str
    data_bytes: int
    index_bytes: int
    pools_bytes: int
    selector_map_count: int
    zone_tuple_count: int
    zone_index_size: int

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version, "entry_count": self.entry_count,
            "fingerprint": self.fingerprint, "data_bytes": self.data_bytes,
            "index_bytes": self.index_bytes, "pools_bytes": self.pools_bytes,
            "selector_map_count": self.selector_map_count, "zone_tuple_count": self.zone_tuple_count,
            "zone_index_size": self.zone_index_size,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "EntryStoreManifest":
        return EntryStoreManifest(
            format_version=d["format_version"], entry_count=d["entry_count"], fingerprint=d["fingerprint"],
            data_bytes=d["data_bytes"], index_bytes=d["index_bytes"], pools_bytes=d["pools_bytes"],
            selector_map_count=d.get("selector_map_count", 0), zone_tuple_count=d.get("zone_tuple_count", 0),
            zone_index_size=d.get("zone_index_size", 0),
        )


def collect_unique_entries(equivalence_classes: Iterable) -> list:
    """Every distinct ``ObservedEntry``, ordered by ``entry_id``.

    Deduplicates on ``entry_id`` rather than object identity: two references to
    the same logical entry must land on one record even if discovery happened
    to produce distinct objects. Sorting gives the stable numbering the whole
    format depends on.
    """
    by_id: dict[str, Any] = {}
    for equivalence_class in equivalence_classes:
        for stream in (
            getattr(equivalence_class, "long_entries", ()) or (),
            getattr(equivalence_class, "short_entries", ()) or (),
            getattr(equivalence_class, "observed_entries", ()) or (),
        ):
            for entry in stream:
                if entry.entry_id not in by_id:
                    by_id[entry.entry_id] = entry
    return [by_id[key] for key in sorted(by_id)]


def build_entry_id_index(entries: Sequence) -> dict[str, int]:
    """``entry_id`` -> numeric id, matching ``collect_unique_entries`` order."""
    return {entry.entry_id: index for index, entry in enumerate(entries)}


def write_entry_store(
    store_dir: Path | str, entries: Sequence, *, fvg_zone_index: Mapping | None
) -> EntryStoreManifest:
    """Write ``entries`` exactly once each, with their shared parts normalised."""
    if len(entries) > MAX_ENTRIES:
        raise ValueError(f"{len(entries)} entries exceeds the uint32 id space ({MAX_ENTRIES})")
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)

    # --- normalise the shared parts -------------------------------------
    # Zone-id tuples first: many selector maps point at the same tuples, and
    # many entries share the same map. Both are pooled by value.
    zone_tuple_pool: list[tuple[str, ...]] = []
    zone_tuple_ids: dict[tuple[str, ...], int] = {}
    selector_map_pool: list[tuple[tuple[str, int], ...]] = []
    selector_map_ids: dict[tuple[tuple[str, int], ...], int] = {}

    def pool_selector_map(mapping: Mapping[str, tuple[str, ...]] | None) -> int:
        if mapping is None:
            return -1
        normalised = []
        for selector in mapping:  # preserve the map's own key order
            zones = tuple(mapping[selector])
            tuple_id = zone_tuple_ids.get(zones)
            if tuple_id is None:
                tuple_id = len(zone_tuple_pool)
                zone_tuple_pool.append(zones)
                zone_tuple_ids[zones] = tuple_id
            normalised.append((selector, tuple_id))
        key = tuple(normalised)
        map_id = selector_map_ids.get(key)
        if map_id is None:
            map_id = len(selector_map_pool)
            selector_map_pool.append(key)
            selector_map_ids[key] = map_id
        return map_id

    # Streamed, not accumulated: holding every pickled blob in a list before
    # writing would add roughly a gigabyte to the parent's peak on the real
    # artifact, at exactly the moment the universe is still alive. Blobs go
    # straight to the data file and only their offsets are kept.
    offsets: list[tuple[int, int]] = []
    digest = hashlib.sha256()
    data_tmp = store / (DATA_FILENAME + f".tmp.{os.getpid()}")
    total_bytes = 0
    data_handle = open(data_tmp, "wb")
    for entry in entries:
        if entry.active_fvg_zones_by_selector is not None:
            # The eager representation carries FVGZone objects per entry, which
            # is exactly what this store exists to avoid. Small fixtures may use
            # it; production discovery never does.
            raise ValueError(
                f"entry {entry.entry_id!r} uses the eager active_fvg_zones_by_selector representation, "
                "which cannot be normalised; rebuild the universe with the shared zone-id form"
            )
        record = (
            entry.entry_id,
            entry.known_at,
            entry.direction,
            entry.source,
            pool_selector_map(entry.active_fvg_zone_ids_by_selector),
            tuple(entry.strategy_fvg_link_selectors or ()),
        )
        blob = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        offsets.append((total_bytes, len(blob)))
        total_bytes += len(blob)
        digest.update(blob)
        data_handle.write(blob)
    data_handle.flush()
    os.fsync(data_handle.fileno())
    data_handle.close()

    pools_blob = pickle.dumps(
        {
            "zone_tuple_pool": zone_tuple_pool,
            "selector_map_pool": selector_map_pool,
            "fvg_zone_index": fvg_zone_index,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    digest.update(pools_blob)
    fingerprint = digest.hexdigest()

    def write_index(handle) -> None:
        for offset, length in offsets:
            handle.write(_INDEX_STRUCT.pack(offset, length))

    # The data file was streamed to a temp path above; publish it the same
    # atomic way as everything else.
    os.replace(data_tmp, store / DATA_FILENAME)
    _atomic_write(store / INDEX_FILENAME, write_index)
    _atomic_write(store / POOLS_FILENAME, lambda handle: handle.write(pools_blob))

    manifest = EntryStoreManifest(
        format_version=ENTRY_STORE_FORMAT_VERSION, entry_count=len(entries), fingerprint=fingerprint,
        data_bytes=total_bytes,
        index_bytes=len(offsets) * INDEX_RECORD_SIZE, pools_bytes=len(pools_blob),
        selector_map_count=len(selector_map_pool), zone_tuple_count=len(zone_tuple_pool),
        zone_index_size=len(fvg_zone_index) if fvg_zone_index else 0,
    )
    # Written last: its presence is what marks the store complete, so a crash
    # mid-write leaves a store that is detectably unfinished rather than
    # plausibly whole.
    _atomic_write(
        store / MANIFEST_FILENAME,
        lambda handle: handle.write(json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")),
    )
    return manifest


class EntryStoreError(RuntimeError):
    """Raised for a missing, incomplete, corrupt or mismatched store.

    Always fatal before simulation: running against a store that does not match
    the plan would produce quietly wrong results, not a crash.
    """


class EntryStoreReader:
    """Random-access, memory-mapped reader. Open once per process."""

    def __init__(self, store_dir: Path | str, *, expected_fingerprint: str | None = None) -> None:
        self._dir = Path(store_dir)
        manifest_path = self._dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise EntryStoreError(
                f"entry store at {self._dir} has no {MANIFEST_FILENAME}; it is missing or was never "
                "finished (the manifest is written last, precisely so this is detectable)"
            )
        self.manifest = EntryStoreManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        if self.manifest.format_version != ENTRY_STORE_FORMAT_VERSION:
            raise EntryStoreError(
                f"entry store format {self.manifest.format_version} != expected {ENTRY_STORE_FORMAT_VERSION}"
            )
        if expected_fingerprint is not None and self.manifest.fingerprint != expected_fingerprint:
            raise EntryStoreError(
                f"entry store fingerprint {self.manifest.fingerprint[:16]}... does not match the "
                f"{expected_fingerprint[:16]}... the plan was built against; the store and the shards "
                "come from different universes"
            )

        index_path, data_path, pools_path = (
            self._dir / INDEX_FILENAME, self._dir / DATA_FILENAME, self._dir / POOLS_FILENAME
        )
        for path in (index_path, data_path, pools_path):
            if not path.is_file():
                raise EntryStoreError(f"entry store at {self._dir} is missing {path.name}")
        expected_index = self.manifest.entry_count * INDEX_RECORD_SIZE
        if index_path.stat().st_size != expected_index:
            raise EntryStoreError(
                f"{INDEX_FILENAME} is {index_path.stat().st_size} bytes, expected {expected_index} for "
                f"{self.manifest.entry_count} entries -- the store is truncated"
            )
        if data_path.stat().st_size != self.manifest.data_bytes:
            raise EntryStoreError(
                f"{DATA_FILENAME} is {data_path.stat().st_size} bytes, expected {self.manifest.data_bytes} "
                "-- the store is truncated or was written by a different run"
            )

        self._data_file = open(data_path, "rb")
        self._data = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._index_file = open(index_path, "rb")
        self._index = mmap.mmap(self._index_file.fileno(), 0, access=mmap.ACCESS_READ)

        pools = pickle.loads(pools_path.read_bytes())
        self._zone_tuple_pool = pools["zone_tuple_pool"]
        self._selector_map_pool = pools["selector_map_pool"]
        self.fvg_zone_index = pools["fvg_zone_index"]
        # Selector maps are rebuilt lazily and cached: there are far fewer of
        # them than entries (29,814 vs 270,091 on the audit window), so this
        # cache is bounded by the pool, not by how many entries are read.
        self._selector_map_cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return self.manifest.entry_count

    def _selector_map(self, map_id: int):
        if map_id < 0:
            return None
        cached = self._selector_map_cache.get(map_id)
        if cached is None:
            cached = {
                selector: self._zone_tuple_pool[tuple_id]
                for selector, tuple_id in self._selector_map_pool[map_id]
            }
            self._selector_map_cache[map_id] = cached
        return cached

    def read(self, entry_index: int):
        """Materialise one ``ObservedEntry`` by numeric id."""
        from tradelab.entry_strategy_universe import ObservedEntry

        if not 0 <= entry_index < self.manifest.entry_count:
            raise EntryStoreError(f"entry id {entry_index} outside 0..{self.manifest.entry_count - 1}")
        start = entry_index * INDEX_RECORD_SIZE
        offset, length = _INDEX_STRUCT.unpack(self._index[start : start + INDEX_RECORD_SIZE])
        entry_id, known_at, direction, source, map_id, link_selectors = pickle.loads(
            self._data[offset : offset + length]
        )
        return ObservedEntry(
            entry_id=entry_id, known_at=known_at, direction=direction, source=source,
            active_fvg_zone_ids_by_selector=self._selector_map(map_id),
            fvg_zone_by_id=self.fvg_zone_index,
            strategy_fvg_link_selectors=link_selectors,
        )

    def read_many(self, entry_indices: Iterable[int]) -> dict[int, Any]:
        """Materialise a set of entries once each.

        Deliberately keyed by numeric id and scoped to the caller: a shard's
        materialisation lives exactly as long as the caller holds this dict,
        and no store-wide cache of entries is ever built.
        """
        out: dict[int, Any] = {}
        for index in entry_indices:
            if index not in out:
                out[index] = self.read(index)
        return out

    def iter_all(self) -> Iterator:
        for index in range(self.manifest.entry_count):
            yield self.read(index)

    def close(self) -> None:
        for closeable in (self._index, self._data, self._index_file, self._data_file):
            try:
                closeable.close()
            except Exception:
                pass
        self._selector_map_cache.clear()

    def __enter__(self) -> "EntryStoreReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
