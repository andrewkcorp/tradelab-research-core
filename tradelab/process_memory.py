"""Trustworthy per-process memory metrics, especially on macOS.

``resource.getrusage().ru_maxrss`` is a peak **resident** figure. Once the
kernel starts compressing or swapping a process out, resident size stops
tracking how much memory the process actually holds, and ``ru_maxrss`` --
being a high-water mark on top of that -- stops moving entirely. During the
investigation in ``docs/dense-class-memory-diagnostic.md`` it reported a flat
7.81 GB for a process whose real footprint was 34.8 GB, and reported the *same*
7.81 GB for a later, genuinely healthy 9.9 GB run. It cannot distinguish those
two situations, so it must not be used to decide whether a run is memory-safe.

The number that can is **phys_footprint**: the kernel's own accounting of what
a process is charged for, including compressed pages, and exactly what
Activity Monitor's "Memory" column and ``vmmap``'s "Physical footprint" line
report. This module reads it directly from ``libproc``'s ``proc_pid_rusage``
and, where available, ``task_info(TASK_VM_INFO)`` for the compressed/resident
split -- no subprocess, cheap enough to sample once a second.

Every field is ``None`` when it cannot be obtained. Nothing here is ever
estimated or derived from a different measurement: an absent number is
reported as absent.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import platform
import resource
from dataclasses import asdict, dataclass

_IS_MACOS = platform.system() == "Darwin"

# rusage_info_v4 is a 16-byte UUID followed by 35 uint64 fields. Only the
# offsets actually used are named; see <libproc.h>/<sys/resource.h>.
_RUSAGE_INFO_V4 = 4
_RI_FIELD_COUNT_V4 = 35
_RI_RESIDENT_SIZE = 6
_RI_PHYS_FOOTPRINT = 7
_RI_LIFETIME_MAX_PHYS_FOOTPRINT = 28


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16), ("fields", ctypes.c_uint64 * _RI_FIELD_COUNT_V4)]


# task_vm_info, truncated at the fields this module reads. TASK_VM_INFO's
# count is expressed in natural_t (4-byte) units, so the count below is the
# full struct's size in those units -- passing a short count makes the kernel
# fill only the prefix, which is exactly what is wanted here.
_TASK_VM_INFO = 22
_VM_INFO_PREFIX_FIELDS = 20  # up to and including phys_footprint


class _TaskVMInfoPrefix(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_peak", ctypes.c_uint64),
        ("device", ctypes.c_uint64),
        ("device_peak", ctypes.c_uint64),
        ("internal", ctypes.c_uint64),
        ("internal_peak", ctypes.c_uint64),
        ("external", ctypes.c_uint64),
        ("external_peak", ctypes.c_uint64),
        ("reusable", ctypes.c_uint64),
        ("reusable_peak", ctypes.c_uint64),
        ("purgeable_volatile_pmap", ctypes.c_uint64),
        ("purgeable_volatile_resident", ctypes.c_uint64),
        ("purgeable_volatile_virtual", ctypes.c_uint64),
        ("compressed", ctypes.c_uint64),
        ("compressed_peak", ctypes.c_uint64),
        ("compressed_lifetime", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class ProcessMemory:
    """One process-memory sample. All sizes are bytes.

    ``phys_footprint_bytes`` is the authoritative "how much memory is this
    process using" figure on macOS.

    ``rss_bytes`` is reported alongside it, but **neither bounds the other**
    and they must not be compared as if one were a subset of the other: RSS
    counts shared clean pages (the mapped interpreter and its libraries) that
    the process is not charged for, while the footprint counts compressed
    private pages that are no longer resident. A footprint far *above* RSS
    means the difference has been compressed or swapped out; an RSS above
    footprint is ordinary for a small process dominated by shared text.

    ``peak_rss_bytes`` is the legacy ``ru_maxrss`` value, retained only so
    older runs and other platforms keep a number; it is never the basis of a
    memory-safety judgement.
    """

    phys_footprint_bytes: int | None = None
    phys_footprint_peak_bytes: int | None = None
    rss_bytes: int | None = None
    compressed_bytes: int | None = None
    internal_bytes: int | None = None
    peak_rss_bytes: int | None = None
    source: str = "unavailable"

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _legacy_peak_rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    # ru_maxrss is kibibytes on Linux, already bytes on macOS.
    return int(value) if _IS_MACOS else int(value) * 1024


def _macos_rusage() -> tuple[int | None, int | None, int | None]:
    """``(phys_footprint, lifetime_peak_footprint, resident)`` via libproc."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        proc_pid_rusage = libc.proc_pid_rusage
        proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RUsageInfoV4)]
        proc_pid_rusage.restype = ctypes.c_int
        info = _RUsageInfoV4()
        import os

        if proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
            return None, None, None
        return (
            int(info.fields[_RI_PHYS_FOOTPRINT]),
            int(info.fields[_RI_LIFETIME_MAX_PHYS_FOOTPRINT]) or None,
            int(info.fields[_RI_RESIDENT_SIZE]),
        )
    except Exception:
        return None, None, None


def _macos_task_vm_info() -> tuple[int | None, int | None, int | None]:
    """``(compressed, phys_footprint, internal)`` via task_info(TASK_VM_INFO).

    ``internal`` is private anonymous memory -- the pages this process alone is
    charged for. It is the figure that actually answers "did copy-on-write
    hold?": a forked child that still shares its parent's pages has a small
    ``internal``, and one that has privatised them has a large one. Summing
    ``phys_footprint`` across a parent and its forked children double-counts
    whatever they still share, so it cannot answer that question.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        mach_task_self = libc.mach_task_self
        mach_task_self.restype = ctypes.c_uint
        task_info = libc.task_info
        task_info.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        task_info.restype = ctypes.c_int
        info = _TaskVMInfoPrefix()
        count = ctypes.c_uint(ctypes.sizeof(_TaskVMInfoPrefix) // 4)
        if task_info(mach_task_self(), _TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count)) != 0:
            return None, None, None
        return int(info.compressed), int(info.phys_footprint), int(info.internal)
    except Exception:
        return None, None, None


def sample_process_memory() -> ProcessMemory:
    """One cheap, no-subprocess sample of this process's own memory."""
    peak_rss = _legacy_peak_rss_bytes()
    if not _IS_MACOS:
        # Linux: /proc/self/statm's resident pages are a true current RSS,
        # and there is no footprint/compressed equivalent to fabricate.
        rss = None
        try:
            with open("/proc/self/statm", "r", encoding="ascii") as handle:
                rss = int(handle.read().split()[1]) * resource.getpagesize()
        except Exception:
            rss = None
        return ProcessMemory(rss_bytes=rss, peak_rss_bytes=peak_rss, source="linux" if rss is not None else "unavailable")

    footprint, footprint_peak, resident = _macos_rusage()
    compressed, task_footprint, internal = _macos_task_vm_info()
    if footprint is None:
        footprint = task_footprint
    if footprint is None and compressed is None:
        return ProcessMemory(peak_rss_bytes=peak_rss, source="unavailable")
    return ProcessMemory(
        phys_footprint_bytes=footprint,
        phys_footprint_peak_bytes=footprint_peak,
        rss_bytes=resident,
        compressed_bytes=compressed,
        internal_bytes=internal,
        peak_rss_bytes=peak_rss,
        source="macos",
    )


@dataclass(frozen=True)
class SystemMemoryPressure:
    """System-wide swap, deliberately separate from per-process memory.

    A process is not responsible for the machine's total swap usage, and a
    run that is itself healthy can sit on a machine whose swap is full for
    unrelated reasons. Conflating the two is what made the earlier diagnosis
    ambiguous, so they are reported as distinct facts.
    """

    swap_used_bytes: int | None = None
    swap_total_bytes: int | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


class _XswUsage(ctypes.Structure):
    """``struct xsw_usage`` from <sys/sysctl.h>, behind ``vm.swapusage``."""

    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_int32),
    ]


def sample_system_memory_pressure() -> SystemMemoryPressure:
    """System-wide swap, read via ``sysctlbyname`` -- no subprocess.

    This is sampled from the once-per-second telemetry heartbeat, so spawning
    a ``sysctl`` process here would cost a fork per second for the entire
    length of a multi-day run. ctypes reads the same kernel value directly.
    """
    if not _IS_MACOS:
        return SystemMemoryPressure()
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        sysctlbyname = libc.sysctlbyname
        sysctlbyname.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t
        ]
        sysctlbyname.restype = ctypes.c_int
        usage = _XswUsage()
        size = ctypes.c_size_t(ctypes.sizeof(_XswUsage))
        if sysctlbyname(b"vm.swapusage", ctypes.byref(usage), ctypes.byref(size), None, 0) != 0:
            return SystemMemoryPressure()
        return SystemMemoryPressure(swap_used_bytes=int(usage.xsu_used), swap_total_bytes=int(usage.xsu_total))
    except Exception:
        return SystemMemoryPressure()


@dataclass(frozen=True)
class SystemMemory:
    """Machine-wide memory, for judging what a *set* of processes really costs.

    Summing ``phys_footprint`` across a parent and its forked children
    double-counts every page they still share, so it systematically
    over-reports a fork/COW pool. The honest question is what the machine has
    left, which is what ``available_bytes`` answers: free plus inactive plus
    speculative plus purgeable, the pages the kernel can hand out without
    swapping anything.
    """

    total_bytes: int | None = None
    available_bytes: int | None = None
    free_bytes: int | None = None
    compressed_bytes: int | None = None
    swap_used_bytes: int | None = None
    swap_total_bytes: int | None = None

    @property
    def used_bytes(self) -> int | None:
        if self.total_bytes is None or self.available_bytes is None:
            return None
        return self.total_bytes - self.available_bytes

    def to_dict(self) -> dict:
        out = {key: value for key, value in asdict(self).items() if value is not None}
        if self.used_bytes is not None:
            out["used_bytes"] = self.used_bytes
        return out


class _VMStatistics64(ctypes.Structure):
    """``vm_statistics64_data_t`` prefix -- the page counters this module reads."""

    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_HOST_VM_INFO64 = 4
_HOST_VM_INFO64_COUNT = ctypes.sizeof(_VMStatistics64) // 4


def sample_system_memory() -> SystemMemory:
    """Machine-wide memory via ``host_statistics64`` -- no subprocess."""
    swap = sample_system_memory_pressure()
    if not _IS_MACOS:
        return SystemMemory(swap_used_bytes=swap.swap_used_bytes, swap_total_bytes=swap.swap_total_bytes)
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        total = ctypes.c_uint64()
        size = ctypes.c_size_t(ctypes.sizeof(total))
        libc.sysctlbyname(b"hw.memsize", ctypes.byref(total), ctypes.byref(size), None, 0)
        page_size = ctypes.c_uint64()
        psize = ctypes.c_size_t(ctypes.sizeof(page_size))
        if libc.sysctlbyname(b"vm.pagesize", ctypes.byref(page_size), ctypes.byref(psize), None, 0) != 0:
            page_size.value = 16384

        mach_host_self = libc.mach_host_self
        mach_host_self.restype = ctypes.c_uint
        host_statistics64 = libc.host_statistics64
        host_statistics64.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        host_statistics64.restype = ctypes.c_int
        stats = _VMStatistics64()
        count = ctypes.c_uint(_HOST_VM_INFO64_COUNT)
        if host_statistics64(mach_host_self(), _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)) != 0:
            return SystemMemory(
                total_bytes=int(total.value) or None,
                swap_used_bytes=swap.swap_used_bytes, swap_total_bytes=swap.swap_total_bytes,
            )
        page = int(page_size.value)
        # What the kernel can hand out without swapping: free, plus the
        # inactive/speculative/purgeable pages it can reclaim cheaply.
        available = (
            int(stats.free_count) + int(stats.inactive_count)
            + int(stats.speculative_count) + int(stats.purgeable_count)
        ) * page
        return SystemMemory(
            total_bytes=int(total.value) or None,
            available_bytes=available,
            free_bytes=int(stats.free_count) * page,
            compressed_bytes=int(stats.compressor_page_count) * page,
            swap_used_bytes=swap.swap_used_bytes,
            swap_total_bytes=swap.swap_total_bytes,
        )
    except Exception:
        return SystemMemory(swap_used_bytes=swap.swap_used_bytes, swap_total_bytes=swap.swap_total_bytes)
