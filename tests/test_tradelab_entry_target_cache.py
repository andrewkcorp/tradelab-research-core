#!/usr/bin/env python3
"""tradelab.entry_target_cache tests (task 047): memoization keying --
same observed entry never recomputes detect_tp_target_candidates/
detect_sl_target_candidates twice, a different entry (or, for SL, a
different confluence for the SAME entry) always does, and __len__
reflects the true number of memoized entries. The exactness of the
memoized VALUES themselves (never just the call count) is proven
end-to-end, against real market data, by tests.test_tradelab_backtest_v2
.EntryTargetCacheEquivalenceTests -- this module only isolates the
cache's own bookkeeping."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from tradelab.entry_target_cache import EntryTargetCache
from tradelab.sl_targets import SelectedFVGConfluence
from tradelab.tp_targets import MarketEntryContext

UTC = timezone.utc


def _market_entry(entry_id: str) -> MarketEntryContext:
    return MarketEntryContext(entry_id=entry_id, direction="long", entry_price=100.0, entry_known_at=datetime(2024, 1, 1, tzinfo=UTC))


class EntryTargetCacheTests(unittest.TestCase):
    def test_tp_candidates_memoized_per_entry_id(self):
        cache = EntryTargetCache()
        entry_a = _market_entry("A")
        entry_a_again = _market_entry("A")  # a distinct object, same entry_id
        entry_b = _market_entry("B")

        with patch("tradelab.entry_target_cache.detect_tp_target_candidates") as mock_detect:
            mock_detect.return_value = "RESULT"
            r1 = cache.tp_candidates(entry_a, {}, {}, None)
            r2 = cache.tp_candidates(entry_a_again, {}, {}, None)
            r3 = cache.tp_candidates(entry_b, {}, {}, None)

        self.assertEqual(mock_detect.call_count, 2)  # once for "A", once for "B" -- never for entry_a_again
        self.assertEqual(r1, "RESULT")
        self.assertEqual(r2, "RESULT")
        self.assertEqual(r3, "RESULT")
        self.assertEqual(len(cache), 2)

    def test_sl_candidates_memoized_per_entry_id_and_confluence_key(self):
        cache = EntryTargetCache()
        entry_a = _market_entry("A")
        empty_confluence = SelectedFVGConfluence(entry_id="A", fvg_candidates=())

        with patch("tradelab.entry_target_cache.detect_sl_target_candidates") as mock_detect:
            mock_detect.return_value = "SL_RESULT"
            cache.sl_candidates(entry_a, None, empty_confluence, {}, {})
            cache.sl_candidates(entry_a, None, empty_confluence, {}, {})  # same (entry_id, confluence_key) -- memoized
            cache.sl_candidates(entry_a, "5m", empty_confluence, {}, {})  # different confluence_key -- new computation

        self.assertEqual(mock_detect.call_count, 2)
        self.assertEqual(len(cache), 2)

    def test_tp_and_sl_caches_are_independent(self):
        cache = EntryTargetCache()
        entry_a = _market_entry("A")
        empty_confluence = SelectedFVGConfluence(entry_id="A", fvg_candidates=())

        with patch("tradelab.entry_target_cache.detect_tp_target_candidates", return_value="TP"), \
             patch("tradelab.entry_target_cache.detect_sl_target_candidates", return_value="SL"):
            cache.tp_candidates(entry_a, {}, {}, None)
            cache.sl_candidates(entry_a, None, empty_confluence, {}, {})

        self.assertEqual(len(cache), 2)  # one TP entry + one SL entry under the identical entry_id -- distinct namespaces

    def test_bounded_cache_evicts_least_recent_item_and_recomputes_exactly(self):
        cache = EntryTargetCache(max_entries=2)
        entry_a, entry_b, entry_c = _market_entry("A"), _market_entry("B"), _market_entry("C")
        with patch("tradelab.entry_target_cache.detect_tp_target_candidates") as mock_detect:
            mock_detect.side_effect = lambda entry, *_args, **_kwargs: f"RESULT-{entry.entry_id}"
            self.assertEqual(cache.tp_candidates(entry_a, {}, {}, None), "RESULT-A")
            self.assertEqual(cache.tp_candidates(entry_b, {}, {}, None), "RESULT-B")
            self.assertEqual(cache.tp_candidates(entry_a, {}, {}, None), "RESULT-A")  # A is now most recent
            self.assertEqual(cache.tp_candidates(entry_c, {}, {}, None), "RESULT-C")  # evicts B
            self.assertEqual(cache.tp_candidates(entry_b, {}, {}, None), "RESULT-B")  # recomputes B

        self.assertEqual(mock_detect.call_count, 4)
        self.assertEqual(len(cache), 2)


if __name__ == "__main__":
    unittest.main()
