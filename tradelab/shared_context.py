"""STAGE B: the simulation context as memory-mapped columns, shared by every worker.

`spawn` gives each worker its own copy of the prepared context -- measured at
**~4.0 GB per worker**, essentially all of it private. `fork` does not fix that:
copy-on-write defers the copy rather than avoiding it, because CPython writes an
object's refcount when it touches it, and simulating touches the bars. Measured
with two independent tools, a forked child's own dirty+swapped memory equals its
whole footprint, and forking additionally forces the parent to keep the context
resident purely to share it. See the STAGE A section of
``docs/multiprocessing-readiness-and-design.md``.

So the context is stored as **columns in files**, mapped read-only into every
process. File-backed clean pages are shared by the kernel for free, are never
counted as any process's private memory, and can be evicted and re-read instead
of compressed or swapped.

**Why an on-demand accessor is cheap here.** The obvious objection is that
rebuilding a ``Bar1m`` per access would cost more in allocation than it saves in
residency. It does not, because the hot loop never touches bar objects:

* the exit search runs inside ``BarRangeIndex.first_high_at_least`` /
  ``first_low_at_most``, which query two flat ``array("d")`` segment trees,
* ``bars[fill_index]`` is read once per observed entry (``backtest_v2.py:1183``),
* ``bars[exit_index]`` once per trade (``backtest_v2.py:1203,1268``).

A handful of temporary objects per entry, never a resident million.

**What is stored.** An AST walk over the simulation path (every ``context.<field>``
read, attributed to its enclosing function) shows it reads exactly five fields,
and two of those -- ``swing_equal_liquidity_results`` and
``session_liquidity_results`` -- are never dereferenced when a
``LiquidityTargetIndex`` is supplied, which production always does: both target
detectors return from the index first (``tp_targets.py:251``, ``sl_targets.py:239``).
This module therefore stores the bars; the liquidity and news indexes are built
from their own sources and are small by comparison.

**Format.** Fixed-width little-or-native-endian columns, one file each, plus a
manifest written *last* so a half-written store is detectable by its absence:

    simctx/
      manifest.json      schema version, fingerprints, dtypes, shapes, endianness
      bars.timestamp     int64 microseconds since the Unix epoch, UTC
      bars.ohlcv         float64, five values per bar, C-order
      bars.contract      uint16 index into the manifest's contract pool

Prices are stored as ``float64`` rather than an integer tick grid. The artifact's
prices already *are* float64, so this round-trips bit-exactly; converting to tick
counts would have to assume every price is an exact multiple of the tick and
would silently round any that is not.

Writes are atomic: each column goes to a temp file, is fsynced and renamed, and
the manifest is renamed last. A reader validates schema version, fingerprint,
endianness, dtype and every file's exact length before mapping anything.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from tradelab.bar import Bar1m

SHARED_CONTEXT_FORMAT_VERSION = "1.0"
SIMCTX_DIRNAME = "simctx"

_MANIFEST = "manifest.json"
_TIMESTAMP_FILE = "bars.timestamp"
_OHLCV_FILE = "bars.ohlcv"
_CONTRACT_FILE = "bars.contract"

_TIMESTAMP_DTYPE = "int64"
_OHLCV_DTYPE = "float64"
_CONTRACT_DTYPE = "uint16"

_TIMESTAMP_ITEMSIZE = 8
_OHLCV_ITEMSIZE = 8
_OHLCV_FIELDS = 5
_CONTRACT_ITEMSIZE = 2

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MAX_CONTRACTS = 2 ** 16 - 1


class SharedContextError(RuntimeError):
    """A shared context is missing, truncated, or was built for other inputs.

    Always raised before anything is mapped or simulated. A run that silently
    read a context built from different market data would produce plausible,
    wrong results -- far worse than failing to start."""


def _endianness() -> str:
    return sys.byteorder


def _atomic_write(path: Path, chunks) -> int:
    """temp -> fsync -> rename, returning the byte count written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with open(tmp, "wb") as handle:
        for chunk in chunks:
            handle.write(chunk)
            written += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return written


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_shared_context(
    directory: Path | str,
    bars: Sequence[Bar1m],
    *,
    fingerprint: str,
    chunk_bars: int = 65536,
) -> dict:
    """Write ``bars`` as memory-mappable columns. Returns the manifest.

    ``fingerprint`` identifies the inputs this context was built from (market
    artifact, news, config, code). A reader that expects a different one refuses
    to open the store rather than simulating against the wrong data.

    Bars are streamed in chunks rather than buffered whole: the writer runs in
    the same process that just built the universe, and materialising three full
    column buffers there would add hundreds of megabytes to the peak this whole
    change exists to reduce.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = len(bars)

    contracts: list[str] = []
    contract_ids: dict[str, int] = {}

    def timestamp_chunks() -> Iterator[bytes]:
        packer = struct.Struct(f"={chunk_bars}q")
        buffer: list[int] = []
        for bar in bars:
            delta = bar.timestamp.astimezone(timezone.utc) - _EPOCH
            buffer.append(delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds)
            if len(buffer) == chunk_bars:
                yield packer.pack(*buffer)
                buffer.clear()
        if buffer:
            yield struct.pack(f"={len(buffer)}q", *buffer)

    def ohlcv_chunks() -> Iterator[bytes]:
        per_chunk = chunk_bars * _OHLCV_FIELDS
        packer = struct.Struct(f"={per_chunk}d")
        buffer: list[float] = []
        for bar in bars:
            buffer.extend((bar.open, bar.high, bar.low, bar.close, bar.volume))
            if len(buffer) == per_chunk:
                yield packer.pack(*buffer)
                buffer.clear()
        if buffer:
            yield struct.pack(f"={len(buffer)}d", *buffer)

    def contract_chunks() -> Iterator[bytes]:
        packer = struct.Struct(f"={chunk_bars}H")
        buffer: list[int] = []
        for bar in bars:
            identifier = contract_ids.get(bar.contract)
            if identifier is None:
                if len(contracts) >= _MAX_CONTRACTS:
                    raise SharedContextError(
                        f"more than {_MAX_CONTRACTS} distinct contracts; the uint16 "
                        "contract column cannot represent them"
                    )
                identifier = len(contracts)
                contract_ids[bar.contract] = identifier
                contracts.append(bar.contract)
            buffer.append(identifier)
            if len(buffer) == chunk_bars:
                yield packer.pack(*buffer)
                buffer.clear()
        if buffer:
            yield struct.pack(f"={len(buffer)}H", *buffer)

    timestamp_bytes = _atomic_write(directory / _TIMESTAMP_FILE, timestamp_chunks())
    ohlcv_bytes = _atomic_write(directory / _OHLCV_FILE, ohlcv_chunks())
    contract_bytes = _atomic_write(directory / _CONTRACT_FILE, contract_chunks())

    manifest = {
        "format_version": SHARED_CONTEXT_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "endianness": _endianness(),
        "bar_count": count,
        "contracts": contracts,
        "columns": {
            _TIMESTAMP_FILE: {
                "dtype": _TIMESTAMP_DTYPE, "itemsize": _TIMESTAMP_ITEMSIZE,
                "items_per_bar": 1, "bytes": timestamp_bytes,
                "unit": "microseconds since 1970-01-01T00:00:00Z",
            },
            _OHLCV_FILE: {
                "dtype": _OHLCV_DTYPE, "itemsize": _OHLCV_ITEMSIZE,
                "items_per_bar": _OHLCV_FIELDS, "bytes": ohlcv_bytes,
                "fields": ["open", "high", "low", "close", "volume"],
            },
            _CONTRACT_FILE: {
                "dtype": _CONTRACT_DTYPE, "itemsize": _CONTRACT_ITEMSIZE,
                "items_per_bar": 1, "bytes": contract_bytes,
            },
        },
    }
    # Manifest last, and only after every column is durable: its absence is what
    # makes an interrupted write detectable instead of half-readable.
    _atomic_write(directory / _MANIFEST, [json.dumps(manifest, indent=2).encode("utf-8")])
    return manifest


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class SharedBars(Sequence):
    """A read-only ``Sequence[Bar1m]`` backed by mapped columns.

    ``__getitem__`` builds a ``Bar1m`` on demand and keeps no reference to it,
    so nothing accumulates. The columnar accessors -- ``highs``, ``lows``,
    ``timestamps_us`` -- exist so index building can read the numbers directly
    without ever creating a bar object.
    """

    __slots__ = ("_timestamps", "_ohlcv", "_contract_ids", "_contracts", "_count")

    def __init__(self, timestamps, ohlcv, contract_ids, contracts: Sequence[str], count: int) -> None:
        self._timestamps = timestamps
        self._ohlcv = ohlcv
        self._contract_ids = contract_ids
        self._contracts = tuple(contracts)
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(f"bar index out of range: {index}")
        base = index * _OHLCV_FIELDS
        return Bar1m(
            timestamp=_EPOCH + timedelta(microseconds=self._timestamps[index]),
            open=self._ohlcv[base],
            high=self._ohlcv[base + 1],
            low=self._ohlcv[base + 2],
            close=self._ohlcv[base + 3],
            volume=self._ohlcv[base + 4],
            contract=self._contracts[self._contract_ids[index]],
        )

    # -- columnar access, no bar objects created ---------------------------

    def timestamp_at(self, index: int) -> datetime:
        return _EPOCH + timedelta(microseconds=self._timestamps[index])

    def high_at(self, index: int) -> float:
        return self._ohlcv[index * _OHLCV_FIELDS + 1]

    def low_at(self, index: int) -> float:
        return self._ohlcv[index * _OHLCV_FIELDS + 2]

    def highs(self) -> Iterator[float]:
        ohlcv = self._ohlcv
        for index in range(self._count):
            yield ohlcv[index * _OHLCV_FIELDS + 1]

    def lows(self) -> Iterator[float]:
        ohlcv = self._ohlcv
        for index in range(self._count):
            yield ohlcv[index * _OHLCV_FIELDS + 2]

    def timestamps(self) -> Iterator[datetime]:
        for index in range(self._count):
            yield _EPOCH + timedelta(microseconds=self._timestamps[index])

    def as_columns(self):
        """``(timestamps, highs, lows)`` as sized, lazy, read-only sequences.

        ``BarRangeIndex.build`` uses this instead of iterating bar objects. The
        timestamp sequence materialises a ``datetime`` only for the element
        asked for, which matters because the index keeps it for the lifetime of
        the run and binary-searches it: a ``bisect`` over 1,057,182 bars touches
        about twenty entries, so twenty temporary datetimes replace a million
        resident ones.
        """
        return (
            _LazyTimestamps(self._timestamps, self._count),
            _StridedFloats(self._ohlcv, self._count, offset=1),
            _StridedFloats(self._ohlcv, self._count, offset=2),
        )


class _LazyTimestamps(Sequence):
    """Ordered UTC datetimes decoded on access from an int64 microsecond column."""

    __slots__ = ("_view", "_count")

    def __init__(self, view, count: int) -> None:
        self._view = view
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        return _EPOCH + timedelta(microseconds=self._view[index])


class _StridedFloats(Sequence):
    """One field of the interleaved OHLCV column, read without copying."""

    __slots__ = ("_view", "_count", "_offset")

    def __init__(self, view, count: int, *, offset: int) -> None:
        self._view = view
        self._count = count
        self._offset = offset

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        return self._view[index * _OHLCV_FIELDS + self._offset]


@dataclass
class SharedContextReader:
    """Maps a written shared context read-only. Close it to unmap."""

    directory: Path
    manifest: dict
    bars: SharedBars
    _maps: list
    _handles: list
    _views: list

    def close(self) -> None:
        # Order matters: an mmap refuses to close while a memoryview still
        # exports it, so every view is released first.
        self.bars = SharedBars((), (), (), (), 0)
        for view in self._views:
            view.release()
        for mapping in self._maps:
            mapping.close()
        for handle in self._handles:
            handle.close()
        self._views = []
        self._maps = []
        self._handles = []

    def __enter__(self) -> "SharedContextReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SharedContextError(message)


def open_shared_context(
    directory: Path | str, *, expected_fingerprint: str | None = None
) -> SharedContextReader:
    """Validate and map a shared context.

    Everything that could make the data wrong is checked *before* any mapping:
    the manifest must exist (its absence is how an interrupted write shows up),
    the schema version and endianness must match this machine, and every column
    file must be exactly the length its dtype and bar count imply. A truncated
    column is refused rather than read as a shorter series.
    """
    directory = Path(directory)
    manifest_path = directory / _MANIFEST
    if not manifest_path.is_file():
        raise SharedContextError(
            f"no shared-context manifest at {manifest_path}; the store is missing or "
            "was never finished (the manifest is written last)"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SharedContextError(f"unreadable shared-context manifest at {manifest_path}: {exc}") from exc

    version = manifest.get("format_version")
    _require(
        version == SHARED_CONTEXT_FORMAT_VERSION,
        f"shared context has format version {version!r}, this build reads "
        f"{SHARED_CONTEXT_FORMAT_VERSION!r}; rebuild it rather than mixing formats",
    )
    _require(
        manifest.get("endianness") == _endianness(),
        f"shared context was written {manifest.get('endianness')!r}-endian, this machine is "
        f"{_endianness()!r}-endian",
    )
    if expected_fingerprint is not None:
        _require(
            manifest.get("fingerprint") == expected_fingerprint,
            "shared context fingerprint mismatch: it was built from different inputs "
            f"({manifest.get('fingerprint')!r} != {expected_fingerprint!r})",
        )

    count = int(manifest.get("bar_count", 0))
    columns = manifest.get("columns") or {}
    maps: list = []
    handles: list = []
    views: dict = {}
    try:
        for name, spec in (
            (_TIMESTAMP_FILE, columns.get(_TIMESTAMP_FILE)),
            (_OHLCV_FILE, columns.get(_OHLCV_FILE)),
            (_CONTRACT_FILE, columns.get(_CONTRACT_FILE)),
        ):
            _require(spec is not None, f"shared context manifest does not describe column {name!r}")
            path = directory / name
            _require(path.is_file(), f"shared context column {name!r} is missing from {directory}")
            expected = count * int(spec["itemsize"]) * int(spec["items_per_bar"])
            actual = path.stat().st_size
            _require(
                actual == expected,
                f"shared context column {name!r} is {actual} bytes, expected {expected} for "
                f"{count} bars; it is truncated or belongs to a different run",
            )
            if count == 0:
                views[name] = ()
                continue
            handle = open(path, "rb")
            handles.append(handle)
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            maps.append(mapping)
            views[name] = memoryview(mapping).cast(
                {"int64": "q", "float64": "d", "uint16": "H"}[spec["dtype"]]
            )
    except BaseException:
        # Same ordering the successful path uses: an mmap refuses to close
        # while a memoryview still exports it, so a validation failure part-way
        # through would otherwise raise BufferError and mask the real reason.
        for view in views.values():
            if isinstance(view, memoryview):
                view.release()
        for mapping in maps:
            mapping.close()
        for handle in handles:
            handle.close()
        raise

    reader = SharedContextReader(
        directory=directory,
        manifest=manifest,
        bars=SharedBars(
            views[_TIMESTAMP_FILE], views[_OHLCV_FILE], views[_CONTRACT_FILE],
            manifest.get("contracts") or (), count,
        ),
        _maps=maps,
        _handles=handles,
        _views=[view for view in views.values() if isinstance(view, memoryview)],
    )
    return reader


def shared_context_bytes(directory: Path | str) -> dict[str, Any]:
    """Size of each column on disk, for reporting."""
    directory = Path(directory)
    sizes = {
        name: (directory / name).stat().st_size
        for name in (_TIMESTAMP_FILE, _OHLCV_FILE, _CONTRACT_FILE)
        if (directory / name).is_file()
    }
    sizes["total"] = sum(sizes.values())
    return sizes
