"""Minimal local server for Dominion. Zero external dependencies —
just the Python standard library, so this runs anywhere Python 3 runs.

Usage:
    python3 server.py
    then open http://localhost:8765 in a browser and click Start.
"""
from __future__ import annotations
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.events import EventLog
from engine.game import run_show, SCRIPTED_ONLY
from engine.history import HistoryRecorder, init_db, get_stats

WEB_DIR = Path(__file__).parent / "web"
PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/stats":
            # Aggregated across every show ever recorded to
            # engine/dominion_history.db, not just whatever's currently on
            # screen -- backs the Standings page (web/stats.html).
            body = json.dumps(get_stats()).encode("utf-8")
            self._send(200, body, "application/json")
            return
        if path == "/":
            path = "/index.html"
        file_path = (WEB_DIR / path.lstrip("/")).resolve()
        if WEB_DIR not in file_path.parents and file_path != WEB_DIR:
            self._send(403, b"forbidden", "text/plain")
            return
        if not file_path.exists():
            self._send(404, b"not found", "text/plain")
            return
        content_type = "text/html"
        if file_path.suffix == ".css":
            content_type = "text/css"
        elif file_path.suffix == ".js":
            content_type = "application/javascript"
        self._send(200, file_path.read_bytes(), content_type)

    def do_POST(self) -> None:
        if self.path.startswith("/api/run-show"):
            # Streamed as newline-delimited JSON rather than one big response:
            # once agents can make live Ollama calls (see engine/ollama_agent.py),
            # a show takes real, variable wall-clock time to compute, so the
            # client needs events as they're produced rather than waiting on
            # the whole thing to finish before seeing anything.
            seed = random.randint(0, 2_000_000_000)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Recorded to SQLite (engine/history.py) alongside streaming to
            # the browser -- every show gets captured automatically, not
            # just the one currently being watched, so patterns (which
            # model tends to win, whether the clock-fairness fix holds) can
            # be analyzed across many games later.
            recorder = HistoryRecorder(seed=seed, scripted_only=SCRIPTED_ONLY)

            def write_event(ev) -> None:
                line = json.dumps(ev.to_dict()).encode("utf-8") + b"\n"
                self.wfile.write(line)
                self.wfile.flush()
                recorder.on_event(ev)

            log = EventLog(on_emit=write_event)
            try:
                run_show(seed=seed, log=log)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                # The browser tab closed or navigated away mid-show: write_event's
                # wfile.write raises here (via emit()), and without this it
                # propagates all the way up to socketserver's default handler,
                # which both prints a full traceback for a completely normal
                # occurrence AND lets run_show keep burning real Ollama calls
                # for a client that's no longer listening. Nothing to recover --
                # just stop the show quietly, same as if it had finished.
                pass
            return
        self._send(404, b"not found", "text/plain")

    def log_message(self, format: str, *args) -> None:
        pass  # keep the console quiet; comment out to debug


if __name__ == "__main__":
    init_db()
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"Dominion running at http://localhost:{PORT}")
    server.serve_forever()
