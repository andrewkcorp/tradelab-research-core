#!/usr/bin/env python3
"""tradelab.research_dashboard tests: the dashboard HTTP server parses
and re-serves run artifacts correctly (list of runs, status, manifest,
leaderboard, tailed events), behaves reasonably before any run exists,
and is provably read-only -- it implements no write-capable HTTP verb,
and a POST/PUT/DELETE against it is rejected (never reaches any file
write path) rather than silently accepted."""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest

from tradelab.research_dashboard import DASHBOARD_HTML, _DashboardRequestHandler, serve
from tradelab.research_run import LeaderboardEntry, RunPhase, create_run


class _ServerTestCase(unittest.TestCase):
    """Starts a real dashboard server on an ephemeral local port for
    each test, in a background thread, torn down in tearDown."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.runs_dir = self._tmp.name
        self.server = serve(self.runs_dir, port=0, host="127.0.0.1")
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self._tmp.cleanup()

    def _get(self, path: str):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            return resp.status, body
        finally:
            conn.close()

    def _request(self, method: str, path: str):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path)
            resp = conn.getresponse()
            resp.read()
            return resp.status
        finally:
            conn.close()


class ReadOnlyBehaviorTests(_ServerTestCase):
    def test_handler_implements_no_write_verb(self):
        for verb in ("do_POST", "do_PUT", "do_DELETE", "do_PATCH"):
            self.assertFalse(hasattr(_DashboardRequestHandler, verb), f"_DashboardRequestHandler must not implement {verb}")

    def test_post_is_rejected(self):
        status = self._request("POST", "/api/runs")
        self.assertEqual(status, 501)

    def test_put_is_rejected(self):
        status = self._request("PUT", "/api/runs")
        self.assertEqual(status, 501)

    def test_delete_is_rejected(self):
        status = self._request("DELETE", "/api/runs")
        self.assertEqual(status, 501)

    def test_get_never_writes_anything_to_the_runs_dir(self):
        import os

        create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="r1")

        def _list_files():
            found = []
            for root, _dirs, files in os.walk(self.runs_dir):
                found.extend(os.path.join(root, f) for f in files)
            return sorted(found)

        before_files = _list_files()
        self._get("/api/runs")
        self._get("/api/runs/r1/status")
        self._get("/api/runs/r1/leaderboard")
        self._get("/api/runs/r1/events")
        self.assertEqual(_list_files(), before_files)  # no new/removed files from any GET


class IndexPageTests(_ServerTestCase):
    def test_index_serves_the_dashboard_html(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"TradeLab Research Dashboard", body)
        self.assertEqual(body.decode("utf-8"), DASHBOARD_HTML)

    def test_index_is_self_contained_no_external_script_src(self):
        self.assertNotIn("<script src=", DASHBOARD_HTML)
        self.assertNotIn("cdn.", DASHBOARD_HTML.lower())


class EmptyRunsDirTests(_ServerTestCase):
    def test_list_runs_is_empty_before_any_run_exists(self):
        status, body = self._get("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"runs": []})

    def test_unknown_run_id_returns_404(self):
        status, body = self._get("/api/runs/does-not-exist/status")
        self.assertEqual(status, 404)


class ArtifactParsingTests(_ServerTestCase):
    def test_list_runs_reflects_created_runs(self):
        create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-a")
        create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-b", is_demo=True)
        status, body = self._get("/api/runs")
        self.assertEqual(status, 200)
        runs = json.loads(body)["runs"]
        run_ids = {r["run_id"] for r in runs}
        self.assertEqual(run_ids, {"run-a", "run-b"})
        demo_flags = {r["run_id"]: r["is_demo"] for r in runs}
        self.assertFalse(demo_flags["run-a"])
        self.assertTrue(demo_flags["run-b"])

    def test_status_endpoint_reflects_live_reporter_state(self):
        reporter = create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-c")
        reporter.phase_started(RunPhase.DISCOVERING_ENTRIES, items_total=10)
        reporter.progress(items_completed=4, discovered_specs=50)
        status, body = self._get("/api/runs/run-c/status")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["phase"], "discovering_entries")
        self.assertEqual(data["items_completed"], 4)
        self.assertEqual(data["discovered_specs"], 50)

    def test_manifest_endpoint(self):
        create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={"a": 1}, run_id="run-d")
        status, body = self._get("/api/runs/run-d/manifest")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["parameters"], {"a": 1})

    def test_leaderboard_endpoint_is_null_before_any_update(self):
        create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-e")
        status, body = self._get("/api/runs/run-e/leaderboard")
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(body))

    def test_leaderboard_endpoint_after_update(self):
        reporter = create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-f")
        reporter.update_leaderboard([
            LeaderboardEntry(rank=1, strategy_id="s1", pnl=10.0, roi=0.1, win_rate=0.5, trade_count=5, max_drawdown=2.0, drawdowns_over_2000_count=0)
        ])
        status, body = self._get("/api/runs/run-f/leaderboard")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(len(data["entries"]), 1)
        self.assertEqual(data["entries"][0]["strategy_id"], "s1")

    def test_events_endpoint_respects_limit(self):
        reporter = create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-g")
        reporter.phase_started(RunPhase.PREPARING_DATA, items_total=5)
        for i in range(1, 6):
            reporter.progress(items_completed=i)
        status, body = self._get("/api/runs/run-g/events?limit=2")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(len(data["events"]), 2)

    def test_events_endpoint_empty_before_any_event(self):
        create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-h")
        status, body = self._get("/api/runs/run-h/events")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIsInstance(data["events"], list)  # create_run() itself never appends an event, only phase_started()/progress()/etc. do


class LiveActivityBackwardCompatibilityTests(_ServerTestCase):
    """task 044: the /status endpoint must serve active_work correctly
    for a new-schema run, and must not error on an older run folder's
    live_status.json that never had the field at all -- the server-side
    read path (tradelab.research_run.read_live_status) must upgrade it
    to active_work: null transparently."""

    def test_status_endpoint_reflects_a_live_heartbeat(self):
        reporter = create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-i")
        reporter.phase_started(RunPhase.SIMULATING_BACKTESTS, items_total=200)
        reporter.heartbeat(
            operation="simulating_backtests", variant_id="CLASS_A::v1::MNQ", unit_index=42, unit_total=200,
            bars_scanned=734211, bars_total=1057182,
        )
        status, body = self._get("/api/runs/run-i/status")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["active_work"]["operation"], "simulating_backtests")
        self.assertEqual(data["active_work"]["variant_id"], "CLASS_A::v1::MNQ")
        self.assertEqual(data["active_work"]["unit_index"], 42)
        self.assertEqual(data["active_work"]["bars_scanned"], 734211)

    def test_status_endpoint_serves_an_old_schema_run_folder_without_error(self):
        import os

        reporter = create_run(runs_dir=self.runs_dir, market_artifact_path=None, news_csv_path=None, parameters={}, run_id="run-old")
        reporter.phase_started(RunPhase.PREPARING_DATA, items_total=1)
        reporter.progress(items_completed=1)

        # simulate a genuinely pre-task-044 run folder: rewrite live_status.json without the active_work key at all.
        status_path = os.path.join(self.runs_dir, "run-old", "live_status.json")
        with open(status_path, encoding="utf-8") as f:
            raw = json.load(f)
        del raw["active_work"]
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        status, body = self._get("/api/runs/run-old/status")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIsNone(data["active_work"])
        self.assertEqual(data["phase"], "preparing_data")  # every other field is still served correctly


if __name__ == "__main__":
    unittest.main()
