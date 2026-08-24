"""Liquidity levels as memory-mapped columns, shared by every worker.

Mapping the bars removed 998 MB per worker and left ~2.66 GB, which is not
bars: it is the liquidity detector results. On the real MNQ artifact those are
359,931 + 32,184 `LiquidityLevel` objects across 785,102 causally-distinct
versions, rebuilt independently in every `spawn` worker.

**What the simulation actually reads.** An AST walk over `tp_targets.py`,
`sl_targets.py` and `liquidity_target_index.py`, attributing every attribute
read to its enclosing function, gives exactly nine level fields --
`level_id`, `timeframe`, `liquidity_type`, `side`, `lower`, `upper`,
`formed_at`, `status`, `taken_at` -- plus `effective_from`/`effective_until`
from the revision. `member_count`, `member_timestamps` and
`absorbed_by_level_id` are read only by `causality_audit.py`, which is not on
the production path. They are stored anyway, so a mapped context is a faithful
substitute rather than a lossy one; `member_timestamps` is variable-length and
therefore lives in a blob read only on demand.

**Why the hot path never builds an object.** `LiquidityTargetIndex._eligible`
tests only `effective_from`, `effective_until`, `status`, `formed_at` and
`taken_at` -- all numeric. `nearest()` then ranks by `(formed_at, level_id)`
and finally by `(|price - entry_price|, formed_at, level_id)`, keeping at most
three. So filtering and ranking run entirely on mapped columns, and a
`LiquidityLevel` is materialised only for the survivors: at most three per
call, never a resident 392,000.

**Tie-breaking is preserved exactly.** Ranking compares `level_id` as a
*string*. Numeric ids here are assigned in **sorted `level_id` order**, so
comparing ids is provably the same ordering as comparing the strings -- the
same device `build_entry_id_index` uses for entries. Assigning ids in
encounter order would silently reorder ties, which is why it is not done.

**Prices** are stored as `float64`, the type they already are, so comparisons
and the tick grid are bit-identical. Converting to scaled integers would
assume every price is an exact multiple of the tick and round any that is not.

**Layout.** Fixed-width records, so a version is addressed arithmetically with
no per-record index; the variable-length parts (level id strings, member
timestamps) use the index-plus-blob shape `EntryStoreReader` already uses.

    liquidity/
      manifest.json     schema version, fingerprint, dtypes, endianness,
                        counts, and a checksum per file
      versions.dat      fixed-width, one record per level version
      level_ids.idx     (uint64 offset, uint32 length) per distinct level
      level_ids.dat     the level id strings
      members.dat       int64 microsecond member timestamps
      high_order.u4     version indices sorted by (price, effective_from, id)
      high_prices.f8    the matching prices, for bisect
      low_order.u4      the same for the low side
      low_prices.f8

Every file is written temp -> fsync -> rename, and the manifest is renamed
last, so an interrupted write is detectable by the manifest's absence rather
than being half-readable.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

LIQUIDITY_STORE_FORMAT_VERSION = "1.0"
LIQUIDITY_DIRNAME = "liquidity"

_MANIFEST = "manifest.json"
_VERSIONS = "versions.dat"
_LEVEL_IDS_IDX = "level_ids.idx"
_LEVEL_IDS_DAT = "level_ids.dat"
_MEMBERS = "members.dat"
_HIGH_ORDER = "high_order.u4"
_HIGH_PRICES = "high_prices.f8"
_LOW_ORDER = "low_order.u4"
_LOW_PRICES = "low_prices.f8"

# level_ord, timeframe, type, side, status, lower, upper, formed_at, taken_at,
# effective_from, effective_until, member_offset, member_ts_count, member_count,
# absorbed_ord
_RECORD = struct.Struct("=IBBBBddqqqqQIII")
RECORD_BYTES = _RECORD.size

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_NULL_TIME = -(2 ** 63)
_NULL_ORD = 2 ** 32 - 1

_ID_ENTRY = struct.Struct("=QI")


class LiquidityStoreError(RuntimeError):
    """The store is missing, truncated, corrupt, or built from other inputs.

    Always raised before anything is mapped or simulated: a run that silently
    read liquidity levels from the wrong data would produce plausible, wrong
    results, which is far worse than refusing to start."""


def _endianness() -> str:
    return sys.byteorder


def _to_micros(value: datetime | None) -> int:
    if value is None:
        return _NULL_TIME
    delta = value.astimezone(timezone.utc) - _EPOCH
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _from_micros(value: int) -> datetime | None:
    if value == _NULL_TIME:
        return None
    return _EPOCH + timedelta(microseconds=value)


def _atomic_write(path: Path, chunks: Iterable[bytes]) -> tuple[int, str]:
    """temp -> fsync -> rename. Returns ``(bytes, sha256)``."""
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    digest = hashlib.sha256()
    written = 0
    try:
        with open(tmp, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return written, digest.hexdigest()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _expand_versions(swing_equal_results: Mapping, session_results: Mapping) -> list:
    """Every causally-distinct level version, exactly as ``LiquidityTargetIndex.build``
    derives them -- including its validation, which must fail at write time
    rather than being discovered per worker at simulation time.

    Returns ``(price, side, effective_from, effective_until, level)`` tuples.
    """
    from tradelab.liquidity import HIGH, LiquidityStatus

    out: list[tuple[float, str, datetime, datetime | None, Any]] = []
    seen: set[str] = set()
    for results in (swing_equal_results, session_results):
        for timeframe, result in (results or {}).items():
            for level in result.levels:
                if level.timeframe != timeframe:
                    raise LiquidityStoreError(
                        f"LiquidityLevel {level.level_id!r} has timeframe {level.timeframe!r}, "
                        f"expected {timeframe!r}"
                    )
                if level.level_id in seen:
                    raise LiquidityStoreError(f"Duplicate liquidity level_id {level.level_id!r}")
                seen.add(level.level_id)
                if level.status == LiquidityStatus.TAKEN and (
                    level.taken_at is None or level.taken_at <= level.formed_at
                ):
                    raise LiquidityStoreError(
                        f"LiquidityLevel {level.level_id!r} has invalid taken_at"
                    )

            revisions_by_id: dict[str, list] = {}
            for revision in result.level_revisions or ():
                revisions_by_id.setdefault(revision.level.level_id, []).append(
                    (revision.effective_at, revision.level)
                )
            for level in result.levels:
                revisions = revisions_by_id.get(level.level_id, [(level.formed_at, level)])
                revisions.sort(key=lambda item: item[0])
                for position, (effective_from, version) in enumerate(revisions):
                    price = version.upper if version.side == HIGH else version.lower
                    if price <= 0:
                        raise LiquidityStoreError(
                            f"LiquidityLevel {version.level_id!r} has non-positive target price {price!r}"
                        )
                    effective_until = (
                        revisions[position + 1][0] if position + 1 < len(revisions) else None
                    )
                    out.append((price, version.side, effective_from, effective_until, version))
    return out


def write_liquidity_store(
    directory: Path | str,
    swing_equal_results: Mapping,
    session_results: Mapping,
    *,
    fingerprint: str,
) -> dict:
    """Write the store. Returns the manifest.

    ``fingerprint`` identifies the inputs these levels were derived from. A
    reader that expects a different one refuses to open the store.
    """
    from tradelab.liquidity import HIGH, LOW

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    versions = _expand_versions(swing_equal_results, session_results)

    # Numeric level ids follow sorted level_id order, so comparing ids is the
    # same ordering as comparing the strings the engine's tie-break uses.
    level_ids = sorted({version[4].level_id for version in versions})
    ordinal = {level_id: position for position, level_id in enumerate(level_ids)}

    timeframes: list[str] = []
    timeframe_ids: dict[str, int] = {}
    types: list[str] = []
    type_ids: dict[str, int] = {}
    statuses: list[str] = []
    status_ids: dict[str, int] = {}

    def intern(value: str, table: list, ids: dict) -> int:
        got = ids.get(value)
        if got is None:
            if len(table) >= 255:
                raise LiquidityStoreError(f"more than 255 distinct values for {value!r}'s column")
            got = len(table)
            ids[value] = got
            table.append(value)
        return got

    members: list[bytes] = []
    member_offset = 0
    records: list[bytes] = []
    for price, side, effective_from, effective_until, level in versions:
        stamps = tuple(level.member_timestamps or ())
        blob = struct.pack(f"={len(stamps)}q", *(_to_micros(s) for s in stamps))
        members.append(blob)
        records.append(_RECORD.pack(
            ordinal[level.level_id],
            intern(level.timeframe, timeframes, timeframe_ids),
            intern(str(level.liquidity_type.value), types, type_ids),
            0 if level.side == HIGH else 1,
            intern(str(level.status.value), statuses, status_ids),
            level.lower, level.upper,
            _to_micros(level.formed_at), _to_micros(level.taken_at),
            _to_micros(effective_from), _to_micros(effective_until),
            member_offset, len(stamps), int(level.member_count),
            _NULL_ORD if level.absorbed_by_level_id is None
            else ordinal.get(level.absorbed_by_level_id, _NULL_ORD),
        ))
        member_offset += len(blob)

    # The two per-side orderings, byte-identical in intent to
    # ``sorted(..., key=(price, effective_from, level_id))`` in the index.
    def order_for(wanted: str) -> list[int]:
        positions = [i for i, v in enumerate(versions) if v[1] == wanted]
        positions.sort(key=lambda i: (versions[i][0], versions[i][2], versions[i][4].level_id))
        return positions

    high_order = order_for(HIGH)
    low_order = order_for(LOW)

    checksums: dict[str, str] = {}
    sizes: dict[str, int] = {}

    def emit(name: str, chunks: Iterable[bytes]) -> None:
        written, digest = _atomic_write(directory / name, chunks)
        sizes[name] = written
        checksums[name] = digest

    emit(_VERSIONS, records)
    emit(_MEMBERS, members)

    id_blobs = [level_id.encode("utf-8") for level_id in level_ids]
    offsets: list[bytes] = []
    running = 0
    for blob in id_blobs:
        offsets.append(_ID_ENTRY.pack(running, len(blob)))
        running += len(blob)
    emit(_LEVEL_IDS_IDX, offsets)
    emit(_LEVEL_IDS_DAT, id_blobs)

    emit(_HIGH_ORDER, [struct.pack(f"={len(high_order)}I", *high_order)] if high_order else [])
    emit(_HIGH_PRICES, [struct.pack(f"={len(high_order)}d", *(versions[i][0] for i in high_order))] if high_order else [])
    emit(_LOW_ORDER, [struct.pack(f"={len(low_order)}I", *low_order)] if low_order else [])
    emit(_LOW_PRICES, [struct.pack(f"={len(low_order)}d", *(versions[i][0] for i in low_order))] if low_order else [])

    manifest = {
        "format_version": LIQUIDITY_STORE_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "endianness": _endianness(),
        "version_count": len(versions),
        "level_count": len(level_ids),
        "high_count": len(high_order),
        "low_count": len(low_order),
        "record_bytes": RECORD_BYTES,
        "record_format": _RECORD.format if isinstance(_RECORD.format, str) else _RECORD.format.decode(),
        "timeframes": timeframes,
        "liquidity_types": types,
        "statuses": statuses,
        "bytes": sizes,
        "sha256": checksums,
    }
    _atomic_write(directory / _MANIFEST, [json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")])
    return manifest


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

# Fixed offsets inside a record. "=" means no padding, so these are exact and
# let the hot path read one field without unpacking the whole record.
_OFF_ORD = 0
_OFF_TIMEFRAME = 4
_OFF_TYPE = 5
_OFF_SIDE = 6
_OFF_STATUS = 7
_OFF_LOWER = 8
_OFF_TIMES = 24  # formed_at, taken_at, effective_from, effective_until

_ORD_ONLY = struct.Struct("=I")
_PRICES_ONLY = struct.Struct("=dd")
_TIMES_ONLY = struct.Struct("=qqqq")
_TAIL = struct.Struct("=QIII")

_STATUS_ABSORBED = "absorbed_in_equal_cluster"
_STATUS_TAKEN = "taken"


class LiquidityStoreReader:
    """Read-only, multi-process-mappable view over a written liquidity store.

    Every `spawn` worker maps the same files; the pages are clean and
    file-backed, so they are shared by the kernel and counted as no process's
    private memory.

    The eligibility and ranking accessors deliberately return primitives, not
    objects: that is what keeps the hot path from allocating. ``level_at``
    materialises a real ``LiquidityLevel``, and is called only for the handful
    of levels a caller actually receives.
    """

    def __init__(self, directory: Path | str, *, expected_fingerprint: str | None = None,
                 cache_size: int = 256, verify_checksums: bool = False) -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / _MANIFEST
        if not manifest_path.is_file():
            raise LiquidityStoreError(
                f"no liquidity manifest at {manifest_path}; the store is missing or was never "
                "finished (the manifest is written last)"
            )
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LiquidityStoreError(f"unreadable liquidity manifest at {manifest_path}: {exc}") from exc

        version = self.manifest.get("format_version")
        if version != LIQUIDITY_STORE_FORMAT_VERSION:
            raise LiquidityStoreError(
                f"liquidity store has format version {version!r}, this build reads "
                f"{LIQUIDITY_STORE_FORMAT_VERSION!r}; rebuild it rather than mixing formats"
            )
        if self.manifest.get("endianness") != _endianness():
            raise LiquidityStoreError(
                f"liquidity store was written {self.manifest.get('endianness')!r}-endian, this "
                f"machine is {_endianness()!r}-endian"
            )
        if self.manifest.get("record_bytes") != RECORD_BYTES:
            raise LiquidityStoreError(
                f"liquidity store record is {self.manifest.get('record_bytes')} bytes, this build "
                f"expects {RECORD_BYTES}"
            )
        if expected_fingerprint is not None and self.manifest.get("fingerprint") != expected_fingerprint:
            raise LiquidityStoreError(
                "liquidity store fingerprint mismatch: it was built from different inputs "
                f"({self.manifest.get('fingerprint')!r} != {expected_fingerprint!r})"
            )

        self.version_count = int(self.manifest.get("version_count", 0))
        self.level_count = int(self.manifest.get("level_count", 0))
        self._timeframes = tuple(self.manifest.get("timeframes") or ())
        self._types = tuple(self.manifest.get("liquidity_types") or ())
        self._statuses = tuple(self.manifest.get("statuses") or ())

        expected_bytes = {
            _VERSIONS: self.version_count * RECORD_BYTES,
            _LEVEL_IDS_IDX: self.level_count * _ID_ENTRY.size,
            _HIGH_ORDER: int(self.manifest.get("high_count", 0)) * 4,
            _HIGH_PRICES: int(self.manifest.get("high_count", 0)) * 8,
            _LOW_ORDER: int(self.manifest.get("low_count", 0)) * 4,
            _LOW_PRICES: int(self.manifest.get("low_count", 0)) * 8,
        }
        self._handles: list = []
        self._maps: list = []
        self._views: list = []
        try:
            raw: dict[str, Any] = {}
            for name in (_VERSIONS, _LEVEL_IDS_IDX, _LEVEL_IDS_DAT, _MEMBERS,
                         _HIGH_ORDER, _HIGH_PRICES, _LOW_ORDER, _LOW_PRICES):
                path = self.directory / name
                if not path.is_file():
                    raise LiquidityStoreError(f"liquidity store column {name!r} is missing from {self.directory}")
                size = path.stat().st_size
                declared = (self.manifest.get("bytes") or {}).get(name)
                if declared is not None and size != declared:
                    raise LiquidityStoreError(
                        f"liquidity store column {name!r} is {size} bytes, the manifest records "
                        f"{declared}; it is truncated or belongs to a different run"
                    )
                if name in expected_bytes and size != expected_bytes[name]:
                    raise LiquidityStoreError(
                        f"liquidity store column {name!r} is {size} bytes, expected "
                        f"{expected_bytes[name]} for the counts the manifest declares"
                    )
                if verify_checksums:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest != (self.manifest.get("sha256") or {}).get(name):
                        raise LiquidityStoreError(f"liquidity store column {name!r} fails its checksum")
                if size == 0:
                    raw[name] = memoryview(b"")
                    continue
                handle = open(path, "rb")
                self._handles.append(handle)
                mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                self._maps.append(mapping)
                view = memoryview(mapping)
                self._views.append(view)
                raw[name] = view
        except BaseException:
            self.close()
            raise

        self._versions = raw[_VERSIONS]
        self._ids_idx = raw[_LEVEL_IDS_IDX]
        self._ids_dat = raw[_LEVEL_IDS_DAT]
        self._members = raw[_MEMBERS]
        self.high_order = raw[_HIGH_ORDER].cast("I") if len(raw[_HIGH_ORDER]) else ()
        self.high_prices = raw[_HIGH_PRICES].cast("d") if len(raw[_HIGH_PRICES]) else ()
        self.low_order = raw[_LOW_ORDER].cast("I") if len(raw[_LOW_ORDER]) else ()
        self.low_prices = raw[_LOW_PRICES].cast("d") if len(raw[_LOW_PRICES]) else ()
        for extra in (self.high_order, self.high_prices, self.low_order, self.low_prices):
            if isinstance(extra, memoryview):
                self._views.append(extra)

        self._cache_size = max(1, cache_size)
        self._cache: dict[int, Any] = {}

    # -- primitives, no object built ---------------------------------------

    def times(self, version: int) -> tuple[int, int, int, int]:
        """``(formed_at, taken_at, effective_from, effective_until)`` in
        microseconds, with ``_NULL_TIME`` for absent values."""
        return _TIMES_ONLY.unpack_from(self._versions, version * RECORD_BYTES + _OFF_TIMES)

    def status_id(self, version: int) -> int:
        return self._versions[version * RECORD_BYTES + _OFF_STATUS]

    def level_ord(self, version: int) -> int:
        return _ORD_ONLY.unpack_from(self._versions, version * RECORD_BYTES + _OFF_ORD)[0]

    def side_is_high(self, version: int) -> bool:
        return self._versions[version * RECORD_BYTES + _OFF_SIDE] == 0

    def prices(self, version: int) -> tuple[float, float]:
        return _PRICES_ONLY.unpack_from(self._versions, version * RECORD_BYTES + _OFF_LOWER)

    def status_name(self, status_id: int) -> str:
        return self._statuses[status_id]

    @property
    def absorbed_status_id(self) -> int:
        return self._statuses.index(_STATUS_ABSORBED) if _STATUS_ABSORBED in self._statuses else -1

    @property
    def taken_status_id(self) -> int:
        return self._statuses.index(_STATUS_TAKEN) if _STATUS_TAKEN in self._statuses else -1

    def level_id(self, ordinal: int) -> str:
        offset, length = _ID_ENTRY.unpack_from(self._ids_idx, ordinal * _ID_ENTRY.size)
        return bytes(self._ids_dat[offset:offset + length]).decode("utf-8")

    # -- materialisation, bounded ------------------------------------------

    def level_at(self, version: int):
        """The ``LiquidityLevel`` for one version.

        Cached in a small bounded dict. **The cache changes nothing about the
        result**: it is keyed by version index and every value is an immutable
        frozen dataclass derived only from the mapped bytes, so a hit and a
        miss are indistinguishable to the caller. Its size therefore does not
        affect output, and it cannot grow with the number of shards.
        """
        cached = self._cache.get(version)
        if cached is not None:
            return cached
        from tradelab.liquidity import HIGH, LOW, LiquidityLevel, LiquidityStatus, LiquidityType

        base = version * RECORD_BYTES
        ordinal = _ORD_ONLY.unpack_from(self._versions, base + _OFF_ORD)[0]
        timeframe = self._timeframes[self._versions[base + _OFF_TIMEFRAME]]
        liquidity_type = LiquidityType(self._types[self._versions[base + _OFF_TYPE]])
        side = HIGH if self._versions[base + _OFF_SIDE] == 0 else LOW
        status = LiquidityStatus(self._statuses[self._versions[base + _OFF_STATUS]])
        lower, upper = _PRICES_ONLY.unpack_from(self._versions, base + _OFF_LOWER)
        formed_at, taken_at, _from, _until = _TIMES_ONLY.unpack_from(self._versions, base + _OFF_TIMES)
        member_offset, member_ts_count, member_count, absorbed_ord = _TAIL.unpack_from(
            self._versions, base + 56
        )
        stamps = ()
        if member_ts_count:
            raw = bytes(self._members[member_offset:member_offset + member_ts_count * 8])
            stamps = tuple(_from_micros(v) for v in struct.unpack(f"={member_ts_count}q", raw))
        level = LiquidityLevel(
            level_id=self.level_id(ordinal),
            timeframe=timeframe,
            liquidity_type=liquidity_type,
            side=side,
            lower=lower,
            upper=upper,
            member_count=member_count,
            member_timestamps=stamps,
            formed_at=_from_micros(formed_at),
            status=status,
            taken_at=_from_micros(taken_at),
            absorbed_by_level_id=None if absorbed_ord == _NULL_ORD else self.level_id(absorbed_ord),
        )
        if len(self._cache) >= self._cache_size:
            # Plain bounded eviction: order is irrelevant because every entry is
            # equivalent to recomputing it from the mapping.
            self._cache.clear()
        self._cache[version] = level
        return level

    def close(self) -> None:
        for view in getattr(self, "_views", []):
            try:
                view.release()
            except (BufferError, ValueError):
                pass
        for mapping in getattr(self, "_maps", []):
            mapping.close()
        for handle in getattr(self, "_handles", []):
            handle.close()
        self._views, self._maps, self._handles = [], [], []
        self._cache = {}

    def __enter__(self) -> "LiquidityStoreReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_liquidity_store(directory: Path | str, **kwargs) -> LiquidityStoreReader:
    return LiquidityStoreReader(directory, **kwargs)


def liquidity_store_bytes(directory: Path | str) -> dict[str, int]:
    directory = Path(directory)
    sizes = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file()}
    sizes["total"] = sum(sizes.values())
    return sizes
