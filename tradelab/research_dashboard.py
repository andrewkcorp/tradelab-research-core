"""Local, read-only, self-contained live dashboard for ``tradelab.
research_run`` artifacts.

**Read-only by construction.** The HTTP server this module runs
implements only ``do_GET`` -- there is no handler for POST/PUT/DELETE/
PATCH at all, so ``http.server.BaseHTTPRequestHandler``'s own default
behavior (501 Unsupported Method) answers any other verb. Every
endpoint only ever reads files already on disk via ``tradelab.
research_run``'s own reader functions (``read_live_status``, ``read_
run_manifest``, ``read_leaderboard``, ``read_progress_events``, ``list_
run_ids``) -- this module never imports, calls, or influences anything
that writes to a run directory, and can never alter, pause, or steer
the process actually producing those artifacts.

**No external dependencies, no CDN.** The dashboard page is one
self-contained HTML document (inline CSS + vanilla JS, no ``<script
src="https://...">``) served directly from a Python string constant in
this module -- nothing is downloaded at request time, so the dashboard
works fully offline. Charts are hand-drawn on ``<canvas>`` with plain
JS (line/bar/histogram/scatter helpers) -- no charting library.

**Polling, not push.** The page polls ``/api/runs/<run_id>/status``
(and the other JSON endpoints) on a short interval while the run's own
``phase`` is not ``completed``/``failed``, and stops polling once it
is (a manual "Refresh" button remains available). This keeps the
server itself trivial (plain request/response, no websockets) while
still being fluid enough for a human watching a live run.

Run it directly: ``python3 -m tradelab.research_dashboard --runs-dir
runs`` (see ``docs/research-dashboard-v1.md``).
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tradelab.research_run import list_run_ids, read_leaderboard, read_live_status, read_progress_events, read_run_manifest

DEFAULT_PORT = 8765


def _json_bytes(payload) -> bytes:
    return json.dumps(payload).encode("utf-8")


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    """Implements only GET -- see module docstring's read-only
    guarantee. ``runs_dir`` is bound per-server via a small factory
    (``_make_handler_class``) rather than a constructor argument,
    because ``http.server`` instantiates a fresh handler per request."""

    runs_dir: Path = Path("runs")  # overridden by _make_handler_class
    server_version = "TradeLabResearchDashboard/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A002 -- matches BaseHTTPRequestHandler's own signature
        pass  # quiet by default; nothing here is a security-relevant access log

    def _send_json(self, payload, status: int = 200) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's own naming
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._send_html(DASHBOARD_HTML)
                return
            if path == "/api/runs":
                self._handle_list_runs()
                return
            if path.startswith("/api/runs/"):
                self._handle_run_endpoint(path, query)
                return
            self._send_json({"error": f"not found: {path}"}, status=404)
        except FileNotFoundError:
            self._send_json({"error": "run artifact not found -- the run may not have written it yet"}, status=404)
        except Exception as exc:  # pragma: no cover -- defensive: never crash the read-only server on a bad artifact
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _handle_list_runs(self) -> None:
        runs = []
        for run_id in list_run_ids(self.runs_dir):
            try:
                manifest = read_run_manifest(self.runs_dir / run_id)
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
            runs.append({"run_id": run_id, "created_at": manifest.created_at, "is_demo": manifest.is_demo})
        self._send_json({"runs": runs})

    def _handle_run_endpoint(self, path: str, query: dict) -> None:
        parts = path.split("/")
        # ["", "api", "runs", "<run_id>", "<endpoint>"]
        if len(parts) < 5:
            self._send_json({"error": "malformed run endpoint path"}, status=400)
            return
        run_id, endpoint = parts[3], parts[4]
        run_dir = self.runs_dir / run_id
        if not run_dir.is_dir():
            self._send_json({"error": f"unknown run_id {run_id!r}"}, status=404)
            return

        if endpoint == "status":
            status = read_live_status(run_dir)
            self._send_json(status.to_dict())
        elif endpoint == "manifest":
            manifest = read_run_manifest(run_dir)
            self._send_json(manifest.to_dict())
        elif endpoint == "leaderboard":
            leaderboard = read_leaderboard(run_dir)
            self._send_json(leaderboard.to_dict() if leaderboard is not None else None)
        elif endpoint == "events":
            limit = int(query.get("limit", ["50"])[0])
            self._send_json({"events": read_progress_events(run_dir, limit=limit)})
        else:
            self._send_json({"error": f"unknown endpoint {endpoint!r}"}, status=404)


def _make_handler_class(runs_dir: Path) -> type:
    return type("_BoundDashboardRequestHandler", (_DashboardRequestHandler,), {"runs_dir": runs_dir})


def serve(runs_dir: Path | str, *, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Build (but do not run the forever-loop of) a
    ``ThreadingHTTPServer`` bound to ``(host, port)``, serving the
    dashboard for runs under ``runs_dir``. Returned so a caller (e.g. a
    test) can call ``.handle_request()`` once, or a CLI can call
    ``.serve_forever()``; either way the server itself does nothing
    beyond respond to HTTP GETs -- see module docstring."""
    handler_class = _make_handler_class(Path(runs_dir))
    return ThreadingHTTPServer((host, port), handler_class)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local, read-only live dashboard for tradelab.research_run artifacts.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run_id subdirectories (default: ./runs)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard in the default browser once the server is up")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    server = serve(runs_dir, port=args.port, host=args.host)
    url = f"http://{args.host}:{args.port}/"
    print(f"TradeLab research dashboard: {url}")
    print(f"Serving runs from: {runs_dir.resolve()}")
    print("Read-only: this server never writes to the runs directory. Press Ctrl+C to stop.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TradeLab Research Dashboard</title>
<style>
  :root {
    --bg: #0b0e14; --bg-panel: #12161f; --bg-card: #161b26; --border: #232938;
    --text: #e6e9f0; --text-dim: #8b93a7; --text-faint: #5b6377;
    --accent: #4f8cff; --good: #34d399; --bad: #f87171; --warn: #fbbf24; --demo: #a855f7;
    --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  header { display: flex; align-items: center; gap: 16px; padding: 14px 20px; background: var(--bg-panel); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: 0.2px; }
  header h1 span { color: var(--accent); }
  select, button { background: var(--bg-card); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  select:hover, button:hover { border-color: var(--accent); }
  .badge { padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }
  .badge-phase { background: rgba(79,140,255,0.15); color: var(--accent); }
  .badge-completed { background: rgba(52,211,153,0.15); color: var(--good); }
  .badge-failed { background: rgba(248,113,113,0.15); color: var(--bad); }
  .badge-demo { background: rgba(168,85,247,0.18); color: var(--demo); }
  .spacer { flex: 1; }
  main { padding: 20px; max-width: 1400px; margin: 0 auto; }
  .demo-banner { background: linear-gradient(90deg, rgba(168,85,247,0.18), rgba(168,85,247,0.05)); border: 1px solid rgba(168,85,247,0.4); color: var(--demo); padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 600; font-size: 13px; display: none; }
  .empty-state { color: var(--text-faint); font-size: 13px; text-align: center; padding: 30px 10px; }
  section { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }
  section h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text-dim); margin: 0 0 12px 0; }
  .progress-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
  .progress-bar-track { flex: 1; min-width: 200px; height: 10px; background: var(--bg-card); border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
  .progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), #7db2ff); transition: width 0.3s ease; }
  .stat { font-family: var(--mono); font-size: 13px; color: var(--text-dim); }
  .stat b { color: var(--text); font-weight: 600; }
  .timeline { display: flex; gap: 6px; flex-wrap: wrap; }
  .timeline-step { flex: 1; min-width: 110px; padding: 8px 10px; border-radius: 6px; background: var(--bg-card); border: 1px solid var(--border); font-size: 12px; text-align: center; color: var(--text-faint); }
  .timeline-step.done { color: var(--good); border-color: rgba(52,211,153,0.35); }
  .timeline-step.current { color: var(--accent); border-color: var(--accent); background: rgba(79,140,255,0.08); font-weight: 600; }
  .timeline-step.failed-step { color: var(--bad); border-color: rgba(248,113,113,0.4); }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--text-faint); margin-bottom: 6px; }
  .card .value { font-family: var(--mono); font-size: 22px; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { color: var(--text-faint); font-weight: 600; text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.3px; }
  td.num, th.num { text-align: right; font-family: var(--mono); }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .pnl-pos { color: var(--good); } .pnl-neg { color: var(--bad); }
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .chart-box { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .chart-box h3 { font-size: 12px; color: var(--text-dim); margin: 0 0 8px 0; font-weight: 600; }
  canvas { width: 100%; height: 200px; display: block; }
  .warnings-list, .events-list { max-height: 220px; overflow-y: auto; font-family: var(--mono); font-size: 12px; }
  .warnings-list div, .events-list div { padding: 5px 0; border-bottom: 1px solid var(--border); color: var(--text-dim); }
  .events-list .ev-type { color: var(--accent); font-weight: 600; }
  .warnings-list div { color: var(--warn); }
  .errors-list div { color: var(--bad); }
  footer { text-align: center; padding: 20px; color: var(--text-faint); font-size: 11.5px; }
  a { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>TradeLab <span>Research Dashboard</span></h1>
  <select id="run-select"><option>Loading runs...</option></select>
  <span id="phase-badge" class="badge badge-phase">-</span>
  <span id="demo-badge" class="badge badge-demo" style="display:none">DEMO</span>
  <div class="spacer"></div>
  <button id="refresh-btn">Refresh</button>
</header>
<main>
  <div id="demo-banner" class="demo-banner">DEMO RUN &mdash; all numbers on this page are synthetic, not real MNQ research results.</div>

  <section id="section-progress">
    <h2>Run status</h2>
    <div id="progress-empty" class="empty-state">No runs found yet. Start a demo with <code>python3 -m tradelab.research_demo</code> or point --runs-dir at a real run.</div>
    <div id="progress-content" style="display:none">
      <div class="progress-row">
        <span class="stat">Elapsed: <b id="stat-elapsed">-</b></span>
        <span class="stat">Items: <b id="stat-items">-</b></span>
        <span class="stat">Throughput: <b id="stat-throughput">-</b></span>
        <span class="stat">ETA: <b id="stat-eta">-</b></span>
        <span class="stat">RAM: <b id="stat-ram">-</b></span>
        <span class="stat">CPU: <b id="stat-cpu">-</b></span>
      </div>
      <div class="progress-row">
        <div class="progress-bar-track"><div id="progress-bar-fill" class="progress-bar-fill" style="width:0%"></div></div>
      </div>
      <div class="timeline" id="phase-timeline"></div>
    </div>
  </section>

  <section id="section-active-work">
    <h2>Live activity</h2>
    <div id="active-work-empty" class="empty-state">No live activity yet.</div>
    <div id="active-work-content" style="display:none">
      <div class="progress-row">
        <span class="stat">Operation: <b id="stat-operation">-</b></span>
        <span class="stat">Strategy variant: <b id="stat-variant-id">-</b></span>
        <span class="stat">Updated: <b id="stat-active-work-updated">-</b></span>
      </div>
      <div class="progress-row" id="active-work-unit-row">
        <span class="stat" style="min-width:170px">Variant/sample: <b id="stat-unit-progress">-</b></span>
        <div class="progress-bar-track"><div id="unit-progress-bar-fill" class="progress-bar-fill" style="width:0%"></div></div>
      </div>
      <div class="progress-row" id="active-work-bars-row">
        <span class="stat" style="min-width:170px">Bars scanned: <b id="stat-bar-progress">-</b></span>
        <div class="progress-bar-track"><div id="bar-progress-bar-fill" class="progress-bar-fill" style="width:0%"></div></div>
      </div>
    </div>
  </section>

  <section id="section-cards">
    <h2>Discovery / backtest counts</h2>
    <div class="cards" id="cards-grid"></div>
  </section>

  <section id="section-leaders">
    <h2>Leaderboard</h2>
    <div id="leaders-empty" class="empty-state">No leaderboard data yet.</div>
    <div style="overflow-x:auto">
      <table id="leaders-table" style="display:none">
        <thead><tr>
          <th class="num">Rank</th><th>Strategy</th><th class="num">P&amp;L</th><th class="num">ROI</th>
          <th class="num">Win rate</th><th class="num">Trades</th><th class="num">Max DD</th><th class="num">DD&gt;$2k</th>
        </tr></thead>
        <tbody id="leaders-tbody"></tbody>
      </table>
    </div>
  </section>

  <section id="section-charts">
    <h2>Charts</h2>
    <div class="charts-grid">
      <div class="chart-box"><h3>Equity curve (current leader)</h3><canvas id="chart-equity"></canvas></div>
      <div class="chart-box"><h3>P&amp;L distribution</h3><canvas id="chart-pnl-dist"></canvas></div>
      <div class="chart-box"><h3>P&amp;L vs max drawdown</h3><canvas id="chart-pnl-dd"></canvas></div>
      <div class="chart-box"><h3>Trade count distribution</h3><canvas id="chart-trade-count"></canvas></div>
      <div class="chart-box"><h3>Win rate distribution</h3><canvas id="chart-win-rate"></canvas></div>
      <div class="chart-box"><h3>Throughput over time</h3><canvas id="chart-throughput"></canvas></div>
      <div class="chart-box" style="grid-column: 1 / -1"><h3>Results by year/month</h3><canvas id="chart-by-month"></canvas></div>
    </div>
  </section>

  <section id="section-warnings">
    <h2>Warnings &amp; errors</h2>
    <div id="warnings-empty" class="empty-state">No warnings or errors.</div>
    <div class="warnings-list" id="warnings-list"></div>
    <div class="errors-list" id="errors-list"></div>
  </section>

  <section id="section-events">
    <h2>Latest events</h2>
    <div id="events-empty" class="empty-state">No events yet.</div>
    <div class="events-list" id="events-list"></div>
  </section>
</main>
<footer>Read-only dashboard &middot; polls run artifacts only &middot; never alters a running process</footer>

<script>
const PHASES = ["preparing_data","discovering_entries","deduplicating_entries","expanding_exit_variants","simulating_backtests","ranking_results"];
const PHASE_LABELS = {preparing_data:"Preparing data", discovering_entries:"Discovering entries", deduplicating_entries:"Deduplicating", expanding_exit_variants:"Exit variants", simulating_backtests:"Backtests", ranking_results:"Ranking", completed:"Completed", failed:"Failed"};

let currentRunId = null;
let pollTimer = null;
const POLL_MS = 1000; // task 044: live status can update at most once/second server-side -- poll to match
const throughputHistory = []; // [{t, throughput}]

function fmtNum(n) { if (n === null || n === undefined) return "-"; return Number(n).toLocaleString(undefined, {maximumFractionDigits: 2}); }
function fmtSeconds(s) { if (s === null || s === undefined) return "-"; s = Math.round(s); const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60; return (h>0?h+"h ":"") + (m>0||h>0?m+"m ":"") + sec+"s"; }
function fmtMoney(n) { if (n === null || n === undefined) return "-"; const sign = n < 0 ? "-" : ""; return sign + "$" + Math.abs(n).toLocaleString(undefined, {maximumFractionDigits: 2}); }
function fmtPct(n) { if (n === null || n === undefined) return "-"; return (n*100).toFixed(1) + "%"; }
function fmtBytes(n) { if (n === null || n === undefined) return "-"; const units = ["B","KB","MB","GB","TB"]; let i = 0; let v = n; while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; } return v.toFixed(v >= 10 || i === 0 ? 0 : 1) + " " + units[i]; }
function fmtAgo(iso) { if (!iso) return "-"; const s = (Date.now() - new Date(iso).getTime()) / 1000; if (s < 0) return "just now"; if (s < 2) return "just now"; return fmtSeconds(s) + " ago"; }

async function fetchJson(url) {
  const res = await fetch(url, {cache: "no-store"});
  if (!res.ok) return null;
  return res.json();
}

async function loadRuns() {
  const data = await fetchJson("/api/runs");
  const select = document.getElementById("run-select");
  if (!data || !data.runs || data.runs.length === 0) {
    select.innerHTML = "<option>No runs found</option>";
    showEmptyProgress();
    return;
  }
  const previouslySelected = currentRunId;
  select.innerHTML = data.runs.map(r => `<option value="${r.run_id}">${r.run_id}${r.is_demo ? " (demo)" : ""}</option>`).join("");
  if (previouslySelected && data.runs.some(r => r.run_id === previouslySelected)) {
    select.value = previouslySelected;
  } else {
    currentRunId = data.runs[0].run_id;
    select.value = currentRunId;
  }
}

function showEmptyProgress() {
  document.getElementById("progress-empty").style.display = "block";
  document.getElementById("progress-content").style.display = "none";
}

function renderTimeline(phase) {
  const el = document.getElementById("phase-timeline");
  const failed = phase === "failed";
  const currentIndex = PHASES.indexOf(phase);
  el.innerHTML = PHASES.map((p, i) => {
    let cls = "timeline-step";
    if (failed) cls += "";
    else if (phase === "completed" || i < currentIndex) cls += " done";
    else if (i === currentIndex) cls += " current";
    return `<div class="${cls}">${PHASE_LABELS[p]}</div>`;
  }).join("") + (failed ? `<div class="timeline-step failed-step">Failed</div>` : (phase === "completed" ? `<div class="timeline-step done">Completed</div>` : ""));
}

function renderCards(status) {
  const cards = [
    ["Discovered specs", status.discovered_specs],
    ["Rejected specs", status.rejected_specs],
    ["Eligible specs", status.eligible_specs],
    ["Unique entry classes", status.unique_entry_classes],
    ["Exit variants generated", status.exit_variants_generated],
    ["Backtests completed", status.backtests_completed],
  ];
  document.getElementById("cards-grid").innerHTML = cards.map(([label, value]) => `
    <div class="card"><div class="label">${label}</div><div class="value">${value === null || value === undefined ? "&mdash;" : fmtNum(value)}</div></div>
  `).join("");
}

// task 044: live_status.active_work is optional -- absent entirely on
// older run folders written before this field existed, and null on any
// run (old or new) that isn't currently between heartbeats. Both cases
// render the same "no live activity yet" empty state, never an error.
function renderActiveWork(status) {
  const empty = document.getElementById("active-work-empty");
  const content = document.getElementById("active-work-content");
  const work = status && status.active_work;
  if (!work) {
    empty.style.display = "block";
    content.style.display = "none";
    return;
  }
  empty.style.display = "none";
  content.style.display = "block";

  document.getElementById("stat-operation").textContent = work.operation || "-";
  document.getElementById("stat-variant-id").textContent = work.variant_id || "-";
  document.getElementById("stat-active-work-updated").textContent = fmtAgo(work.updated_at);

  const unitRow = document.getElementById("active-work-unit-row");
  if (work.unit_index === null || work.unit_index === undefined) {
    unitRow.style.display = "none";
  } else {
    unitRow.style.display = "flex";
    document.getElementById("stat-unit-progress").textContent = fmtNum(work.unit_index) + " / " + (work.unit_total === null || work.unit_total === undefined ? "?" : fmtNum(work.unit_total));
    const unitPct = (work.unit_total && work.unit_total > 0) ? Math.min(100, (work.unit_index / work.unit_total) * 100) : 0;
    document.getElementById("unit-progress-bar-fill").style.width = unitPct + "%";
  }

  const barsRow = document.getElementById("active-work-bars-row");
  if (work.bars_scanned === null || work.bars_scanned === undefined) {
    barsRow.style.display = "none";
  } else {
    barsRow.style.display = "flex";
    document.getElementById("stat-bar-progress").textContent = fmtNum(work.bars_scanned) + " / " + (work.bars_total === null || work.bars_total === undefined ? "?" : fmtNum(work.bars_total));
    const barsPct = (work.bars_total && work.bars_total > 0) ? Math.min(100, (work.bars_scanned / work.bars_total) * 100) : 0;
    document.getElementById("bar-progress-bar-fill").style.width = barsPct + "%";
  }
}

function renderLeaders(leaderboard) {
  const empty = document.getElementById("leaders-empty");
  const table = document.getElementById("leaders-table");
  if (!leaderboard || !leaderboard.entries || leaderboard.entries.length === 0) {
    empty.style.display = "block"; table.style.display = "none";
    return;
  }
  empty.style.display = "none"; table.style.display = "table";
  document.getElementById("leaders-tbody").innerHTML = leaderboard.entries.map(e => `
    <tr>
      <td class="num">${e.rank}</td>
      <td>${e.strategy_id}</td>
      <td class="num ${e.pnl >= 0 ? "pnl-pos" : "pnl-neg"}">${fmtMoney(e.pnl)}</td>
      <td class="num">${fmtPct(e.roi)}</td>
      <td class="num">${fmtPct(e.win_rate)}</td>
      <td class="num">${fmtNum(e.trade_count)}</td>
      <td class="num">${fmtMoney(e.max_drawdown)}</td>
      <td class="num">${fmtNum(e.drawdowns_over_2000_count)}</td>
    </tr>
  `).join("");
}

function renderWarnings(status) {
  const warnings = status.warnings || [];
  const errors = status.errors || [];
  document.getElementById("warnings-empty").style.display = (warnings.length === 0 && errors.length === 0) ? "block" : "none";
  document.getElementById("warnings-list").innerHTML = warnings.map(w => `<div>&#9888; ${escapeHtml(w)}</div>`).join("");
  document.getElementById("errors-list").innerHTML = errors.map(e => `<div>&#10007; ${escapeHtml(e)}</div>`).join("");
}

function renderEvents(eventsData) {
  const events = (eventsData && eventsData.events) || [];
  const empty = document.getElementById("events-empty");
  const list = document.getElementById("events-list");
  if (events.length === 0) { empty.style.display = "block"; list.innerHTML = ""; return; }
  empty.style.display = "none";
  list.innerHTML = events.slice().reverse().map(ev => `<div><span class="ev-type">${ev.event_type}</span> &middot; ${ev.phase || "-"} &middot; ${ev.ts}</div>`).join("");
}

function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

// ---------- canvas chart helpers (no external library) ----------
function clearCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(rect.width, 200), h = 200;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return {ctx, w, h};
}
function emptyChart(canvas, message) {
  const {ctx, w, h} = clearCanvas(canvas);
  ctx.fillStyle = "#5b6377"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(message, w/2, h/2);
}
function drawLineChart(canvas, points, {color = "#4f8cff", fill = true} = {}) {
  if (!points || points.length === 0) { emptyChart(canvas, "No data yet"); return; }
  const {ctx, w, h} = clearCanvas(canvas);
  const pad = 24;
  const ys = points.map(p => p.y);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const range = (maxY - minY) || 1;
  const stepX = points.length > 1 ? (w - 2*pad) / (points.length - 1) : 0;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = pad + i * stepX;
    const y = h - pad - ((p.y - minY) / range) * (h - 2*pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  if (fill) {
    ctx.lineTo(pad + (points.length-1)*stepX, h-pad); ctx.lineTo(pad, h-pad); ctx.closePath();
    ctx.fillStyle = color + "22"; ctx.fill();
  }
  ctx.fillStyle = "#8b93a7"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(minY.toFixed(0), 2, h - 4);
  ctx.textAlign = "right"; ctx.fillText(maxY.toFixed(0), w - 2, 12);
}
function drawHistogram(canvas, values, {bins = 12, color = "#4f8cff"} = {}) {
  if (!values || values.length === 0) { emptyChart(canvas, "No data yet"); return; }
  const {ctx, w, h} = clearCanvas(canvas);
  const pad = 20;
  const minV = Math.min(...values), maxV = Math.max(...values);
  const range = (maxV - minV) || 1;
  const counts = new Array(bins).fill(0);
  values.forEach(v => { let idx = Math.floor(((v - minV) / range) * bins); if (idx >= bins) idx = bins - 1; if (idx < 0) idx = 0; counts[idx]++; });
  const maxCount = Math.max(...counts) || 1;
  const barW = (w - 2*pad) / bins;
  counts.forEach((c, i) => {
    const barH = (c / maxCount) * (h - 2*pad);
    ctx.fillStyle = color;
    ctx.fillRect(pad + i*barW + 1, h - pad - barH, barW - 2, barH);
  });
  ctx.fillStyle = "#8b93a7"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(minV.toFixed(0), pad, h - 4);
  ctx.textAlign = "right"; ctx.fillText(maxV.toFixed(0), w - pad, h - 4);
}
function drawScatter(canvas, points, {color = "#4f8cff"} = {}) {
  if (!points || points.length === 0) { emptyChart(canvas, "No data yet"); return; }
  const {ctx, w, h} = clearCanvas(canvas);
  const pad = 24;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), rangeX = (maxX - minX) || 1;
  const minY = Math.min(...ys), maxY = Math.max(...ys), rangeY = (maxY - minY) || 1;
  points.forEach(p => {
    const x = pad + ((p.x - minX) / rangeX) * (w - 2*pad);
    const y = h - pad - ((p.y - minY) / rangeY) * (h - 2*pad);
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 2*Math.PI); ctx.fillStyle = color + "cc"; ctx.fill();
  });
  ctx.fillStyle = "#8b93a7"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText("dd " + minX.toFixed(0) + " - " + maxX.toFixed(0), pad, h - 4);
}
function drawBarChartLabeled(canvas, labels, values, {color = "#4f8cff"} = {}) {
  if (!labels || labels.length === 0) { emptyChart(canvas, "No data yet"); return; }
  const {ctx, w, h} = clearCanvas(canvas);
  const pad = 24;
  const maxV = Math.max(...values) || 1;
  const barW = (w - 2*pad) / labels.length;
  values.forEach((v, i) => {
    const barH = (v / maxV) * (h - 2*pad - 14);
    ctx.fillStyle = color;
    ctx.fillRect(pad + i*barW + 1, h - pad - barH, Math.max(barW - 3, 1), barH);
  });
  ctx.fillStyle = "#8b93a7"; ctx.font = "9px monospace"; ctx.textAlign = "center";
  const step = Math.max(1, Math.ceil(labels.length / 8));
  labels.forEach((l, i) => { if (i % step === 0) ctx.fillText(l, pad + i*barW + barW/2, h - 4); });
}

function renderCharts(leaderboard, eventsData) {
  const entries = (leaderboard && leaderboard.entries) || [];

  const leader = entries.find(e => e.rank === 1) || entries[0];
  const equityPoints = (leader && leader.equity_curve || []).map((p, i) => ({x: i, y: p.equity}));
  drawLineChart(document.getElementById("chart-equity"), equityPoints, {color: "#34d399"});

  drawHistogram(document.getElementById("chart-pnl-dist"), entries.map(e => e.pnl), {color: "#4f8cff"});

  drawScatter(document.getElementById("chart-pnl-dd"), entries.map(e => ({x: e.max_drawdown, y: e.pnl})), {color: "#fbbf24"});

  drawHistogram(document.getElementById("chart-trade-count"), entries.map(e => e.trade_count), {color: "#a855f7"});

  drawHistogram(document.getElementById("chart-win-rate"), entries.map(e => e.win_rate * 100), {color: "#f87171"});

  // throughput over time, from progress events
  const events = (eventsData && eventsData.events) || [];
  const progressEvents = events.filter(e => e.event_type === "progress" && e.detail && e.detail.items_completed !== null && e.detail.items_completed !== undefined);
  const throughputPoints = [];
  for (let i = 1; i < progressEvents.length; i++) {
    const prev = progressEvents[i-1], cur = progressEvents[i];
    const dt = (new Date(cur.ts) - new Date(prev.ts)) / 1000;
    const dc = cur.detail.items_completed - prev.detail.items_completed;
    if (dt > 0 && dc >= 0) throughputPoints.push({x: i, y: dc / dt});
  }
  drawLineChart(document.getElementById("chart-throughput"), throughputPoints, {color: "#4f8cff", fill: false});

  // results by year/month, aggregated across all leaderboard entries
  const byMonth = {};
  entries.forEach(e => { Object.entries(e.trades_by_month || {}).forEach(([k, v]) => { byMonth[k] = (byMonth[k] || 0) + v; }); });
  const months = Object.keys(byMonth).sort();
  drawBarChartLabeled(document.getElementById("chart-by-month"), months, months.map(m => byMonth[m]), {color: "#34d399"});
}

async function refresh() {
  await loadRuns();
  if (!currentRunId) return;

  const [status, leaderboard, eventsData, manifest] = await Promise.all([
    fetchJson(`/api/runs/${currentRunId}/status`),
    fetchJson(`/api/runs/${currentRunId}/leaderboard`),
    fetchJson(`/api/runs/${currentRunId}/events?limit=100`),
    fetchJson(`/api/runs/${currentRunId}/manifest`),
  ]);

  if (!status) { showEmptyProgress(); return; }

  document.getElementById("progress-empty").style.display = "none";
  document.getElementById("progress-content").style.display = "block";

  const phaseBadge = document.getElementById("phase-badge");
  phaseBadge.textContent = PHASE_LABELS[status.phase] || status.phase;
  phaseBadge.className = "badge " + (status.phase === "completed" ? "badge-completed" : status.phase === "failed" ? "badge-failed" : "badge-phase");

  const demoBadge = document.getElementById("demo-badge");
  const demoBanner = document.getElementById("demo-banner");
  const isDemo = !!(status.is_demo || (manifest && manifest.is_demo));
  demoBadge.style.display = isDemo ? "inline-block" : "none";
  demoBanner.style.display = isDemo ? "block" : "none";

  document.getElementById("stat-elapsed").textContent = fmtSeconds(status.elapsed_seconds);
  document.getElementById("stat-items").textContent = (status.items_completed === null ? "-" : fmtNum(status.items_completed)) + " / " + (status.items_total === null || status.items_total === undefined ? "?" : fmtNum(status.items_total));
  document.getElementById("stat-throughput").textContent = status.throughput_per_second === null || status.throughput_per_second === undefined ? "-" : fmtNum(status.throughput_per_second) + "/s";
  document.getElementById("stat-eta").textContent = status.eta_seconds === null || status.eta_seconds === undefined ? "n/a" : fmtSeconds(status.eta_seconds);
  const usage = status.resource_usage || {};
  document.getElementById("stat-ram").textContent = fmtBytes(usage.rss_bytes);
  document.getElementById("stat-cpu").textContent = usage.cpu_seconds === null || usage.cpu_seconds === undefined ? "-" : fmtSeconds(usage.cpu_seconds);

  const pct = (status.items_total && status.items_total > 0 && status.items_completed !== null) ? Math.min(100, (status.items_completed / status.items_total) * 100) : (status.phase === "completed" ? 100 : 0);
  document.getElementById("progress-bar-fill").style.width = pct + "%";

  renderTimeline(status.phase);
  renderActiveWork(status);
  renderCards(status);
  renderLeaders(leaderboard);
  renderWarnings(status);
  renderEvents(eventsData);
  renderCharts(leaderboard, eventsData);

  const isTerminal = status.phase === "completed" || status.phase === "failed";
  if (isTerminal && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

document.getElementById("run-select").addEventListener("change", (e) => { currentRunId = e.target.value; refresh(); });
document.getElementById("refresh-btn").addEventListener("click", refresh);

(async () => {
  await refresh();
  pollTimer = setInterval(refresh, POLL_MS);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
