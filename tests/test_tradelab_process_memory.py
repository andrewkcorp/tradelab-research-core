#!/usr/bin/env python3
"""tradelab.process_memory tests.

The point of this module is that ``ru_maxrss`` cannot be trusted to say how
much memory a process is using, so these tests are mostly about the *new*
number being real: internally consistent, responsive to an actual allocation,
and -- where ``vmmap`` is available -- in agreement with what the operating
system's own tool reports for the same process.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import unittest

from tradelab.process_memory import (
    ProcessMemory,
    SystemMemoryPressure,
    sample_process_memory,
    sample_system_memory_pressure,
)

_IS_MACOS = platform.system() == "Darwin"


class ProcessMemorySampleTests(unittest.TestCase):
    def test_sample_is_internally_consistent(self):
        sample = sample_process_memory()
        self.assertIsInstance(sample, ProcessMemory)
        for name in ("phys_footprint_bytes", "rss_bytes", "compressed_bytes", "peak_rss_bytes"):
            value = getattr(sample, name)
            if value is not None:
                self.assertGreaterEqual(value, 0, name)
        if sample.phys_footprint_bytes is not None and sample.phys_footprint_peak_bytes is not None:
            self.assertGreaterEqual(
                sample.phys_footprint_peak_bytes, sample.phys_footprint_bytes,
                "a lifetime peak below the current value is impossible",
            )

    @unittest.skipUnless(_IS_MACOS, "phys_footprint is a macOS metric")
    def test_macos_sample_reports_the_expected_fields(self):
        sample = sample_process_memory()
        self.assertEqual(sample.source, "macos")
        self.assertIsNotNone(sample.phys_footprint_bytes)
        self.assertIsNotNone(sample.rss_bytes)
        self.assertGreater(sample.phys_footprint_bytes, 1024 * 1024, "a live CPython is never under 1 MB")

    def test_footprint_follows_a_real_allocation(self):
        before = sample_process_memory().phys_footprint_bytes
        if before is None:
            self.skipTest("phys_footprint unavailable on this platform")
        # bytearray touches every page, so this cannot be satisfied by lazy
        # virtual reservation -- the footprint has to move.
        blob = bytearray(320 * 1024 * 1024)
        during = sample_process_memory().phys_footprint_bytes
        self.assertGreater(during - before, 200 * 1024 * 1024, "footprint did not track a real 320 MB allocation")
        del blob

    def test_to_dict_omits_unavailable_fields_rather_than_reporting_zero(self):
        empty = ProcessMemory()
        self.assertEqual(empty.to_dict(), {"source": "unavailable"})
        self.assertNotIn("phys_footprint_bytes", empty.to_dict())

    @unittest.skipUnless(_IS_MACOS and shutil.which("vmmap"), "needs macOS vmmap")
    def test_agrees_with_vmmap_for_this_process(self):
        """The whole module is worthless if this number disagrees with the
        one a human would read off ``vmmap``/Activity Monitor."""
        blob = bytearray(256 * 1024 * 1024)
        sample = sample_process_memory()
        out = subprocess.run(
            ["vmmap", "--summary", str(os.getpid())], capture_output=True, text=True, timeout=120
        ).stdout
        match = re.search(r"Physical footprint:\s+([0-9.]+)([KMG])", out)
        if match is None:
            self.skipTest("vmmap did not report a physical footprint")
        scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[match.group(2)]
        vmmap_bytes = float(match.group(1)) * scale
        # vmmap prints 3 significant figures and samples microseconds later,
        # so require agreement in magnitude, not to the byte.
        self.assertLess(
            abs(vmmap_bytes - sample.phys_footprint_bytes) / vmmap_bytes, 0.05,
            f"ctypes footprint {sample.phys_footprint_bytes} disagrees with vmmap {vmmap_bytes}",
        )
        del blob


class SystemMemoryPressureTests(unittest.TestCase):
    def test_swap_is_reported_as_a_system_wide_fact(self):
        pressure = sample_system_memory_pressure()
        self.assertIsInstance(pressure, SystemMemoryPressure)
        if pressure.swap_used_bytes is not None:
            self.assertGreaterEqual(pressure.swap_used_bytes, 0)
            self.assertGreaterEqual(pressure.swap_total_bytes, pressure.swap_used_bytes)

    def test_sampling_is_cheap_enough_for_a_one_second_heartbeat(self):
        """This is sampled once a second for the length of a multi-day run,
        so it must not fork a subprocess."""
        import time

        start = time.perf_counter()
        for _ in range(200):
            sample_process_memory()
            sample_system_memory_pressure()
        per_sample = (time.perf_counter() - start) / 200
        self.assertLess(per_sample, 0.01, f"{per_sample * 1e3:.1f} ms per sample is too slow for 1 Hz telemetry")


if __name__ == "__main__":
    unittest.main()
