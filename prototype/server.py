"""Minimal local server for Dominion. Zero external dependencies —
just the Python standard library, so this runs anywhere Python 3 runs.

Usage:
    python3 server.py
    then open http://localhost:8765 in a browser and click Start.
"""
from __future__ import annotations
import json
import os
import random
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.events import EventLog
from engine.game import run_show, SCRIPTED_ONLY
from engine.history import HistoryRecorder, init_db, get_stats
from engine.ollama_agent import TEXT_MODELS

WEB_DIR = Path(__file__).parent / "web"
# Overridable via DOMINION_PORT (README's Environment Variables section) --
# 8765 stays the default so the existing "open http://localhost:8765"
# instructions never go stale for anyone who doesn't set it, this just
# gives anyone with a port conflict (another local service already on
# 8765, running two shows side by side, etc.) an escape hatch that doesn't
# require editing this file.
PORT = int(os.environ.get("DOMINION_PORT", "8765"))

# Ollama's own list-installed-models endpoint (distinct from /api/generate,
# which engine/ollama_agent.py uses for the actual live calls) -- used to
# back the roster panel Scott asked for after running `ollama ps` himself
# and seeing nothing listed: "can we have list on the left, under the
# controls, of the agents/model running, parameters." `ollama ps` only
# shows models currently LOADED in memory, which unload after an idle
# timeout and are empty between shows -- not useful as a persistent status
# display. /api/tags instead lists everything actually INSTALLED, which is
# what "is this model available for the show to use" really means; a model
# being idle/unloaded between shows is normal, not an error state.
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_TAGS_TIMEOUT = 3.0

# Human-readable display names for the tag strings in TEXT_MODELS -- "one
# unique, perhaps English translated" per Scott's ask, since raw tags like
# "qwen2.5:3b" read as machine identifiers, not a roster a viewer would
# recognize at a glance.
MODEL_DISPLAY_NAMES = {
    "llama3.2:latest": "Llama 3.2",
    "qwen2.5:3b": "Qwen 2.5",
    "gemma2:2b": "Gemma 2",
    "phi3:mini": "Phi-3 Mini",
}
# Fallback parameter sizes if Ollama is unreachable (so the panel still
# shows something meaningful offline) -- overridden by the real
# details.parameter_size Ollama itself reports whenever it's reachable.
MODEL_FALLBACK_PARAMS = {
    "llama3.2:latest": "3.2B",
    "qwen2.5:3b": "3.09B",
    "gemma2:2b": "2.6B",
    "phi3:mini": "3.8B",
}


def get_model_roster() -> dict:
    """Cross-references TEXT_MODELS (the actual pool the show draws from,
    see engine/game.py's model_pool) against Ollama's own /api/tags -- one
    quick, short-timeout call, not per-model, since /api/tags always
    returns everything installed in one shot. ollama_reachable is False,
    and every model reports installed=False, if Ollama isn't running at
    all (connection refused) -- this is the "offline" state Scott asked
    to see reflected, distinct from a model simply not being installed
    while Ollama itself is up.
    """
    installed = {}
    reachable = False
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL)
        with urllib.request.urlopen(req, timeout=OLLAMA_TAGS_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        reachable = True
        for entry in data.get("models", []):
            name = entry.get("name") or entry.get("model")
            if name:
                installed[name] = entry.get("details", {}) or {}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        pass  # Ollama not running/unreachable -- roster still returns, all offline.

    models = []
    for tag in TEXT_MODELS:
        details = installed.get(tag)
        is_installed = details is not None
        param_size = (details or {}).get("parameter_size") or MODEL_FALLBACK_PARAMS.get(tag, "?")
        models.append({
            "tag": tag,
            "display_name": MODEL_DISPLAY_NAMES.get(tag, tag),
            "parameter_size": param_size,
            # A model only counts as genuinely online if Ollama itself is
            # reachable AND that specific model is actually installed --
            # either condition failing means the show can't actually call it
            # live (engine/ollama_agent.py's OllamaAgent falls back to
            # ScriptedAgent's canned behavior in that case).
            "online": reachable and is_installed,
        })
    return {"ollama_reachable": reachable, "scripted_only": SCRIPTED_ONLY, "models": models}


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
        if path == "/api/models":
            # Backs the model roster panel -- see get_model_roster above.
            # Polled periodically by the frontend, not just once, since
            # whether Ollama is up/down or a model is installed can change
            # while the page stays open.
            body = json.dumps(get_model_roster()).encode("utf-8")
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
        elif file_path.suffix == ".mp3":
            # Scott's own recorded sound files (web/sounds/) -- served
            # straight off disk like any other static asset here, just
            # needs the right content type so browsers actually play it
            # instead of trying to parse it as HTML (the default above).
            content_type = "audio/mpeg"
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
