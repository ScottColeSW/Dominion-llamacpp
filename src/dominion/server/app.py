"""FastAPI service for Dominion. Enterprise-service replacement for the
original stdlib-only `http.server` handler (see docs/ENGINE_NOTES.md) --
this is a deliberate dependency tradeoff (see pyproject.toml's comment),
not a change to the underlying game engine, which is untouched here.

Usage:
    uvicorn dominion.server.app:app --port 8765
    then open http://localhost:8765 in a browser and click Start.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import os
import random
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..engine.events import Event, EventLog
from ..engine.game import run_show, SCRIPTED_ONLY
from ..engine.history import HistoryRecorder, get_stats, init_db
from ..agents.llm_agent import TEXT_MODELS
from ..inference.base import InferenceClient
from ..inference.config import LLAMACPP_MODELS, settings
from ..inference.llamacpp_client import LlamaCppInferenceClient
from ..inference.ollama_client import OllamaInferenceClient
from ..observability.logging_config import configure_logging
from .model_catalog import get_model_catalog

# Windows' mimetypes database doesn't always know .mp3 -> audio/mpeg out of
# the box; the original Handler set this explicitly per-file, so pin it once
# here instead, ahead of StaticFiles doing its own guess_type lookups.
mimetypes.add_type("audio/mpeg", ".mp3")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PORT = int(os.environ.get("DOMINION_PORT", "8765"))

# Distinct from the actual generate/completion calls (which go through the
# async InferenceClient abstraction, inference/ollama_client.py and
# inference/llamacpp_client.py) -- these are one-off, short-timeout status
# checks, plain urllib deliberately (see get_model_roster's docstring).
OLLAMA_TAGS_TIMEOUT = 3.0
LLAMACPP_STATUS_TIMEOUT = 3.0

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


def _fetch_json(url: str, timeout: float) -> Optional[dict]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None  # Backend not running/unreachable -- caller treats this as offline, not an error.


def _ollama_roster() -> dict:
    """Cross-references TEXT_MODELS (the actual pool an Ollama-backend show
    draws from) against Ollama's own /api/tags -- one quick call, not
    per-model. reachable is False, and every model reports online=False,
    if Ollama isn't running at all (connection refused)."""
    data = _fetch_json(f"{settings.ollama_url}/api/tags", OLLAMA_TAGS_TIMEOUT)
    reachable = data is not None
    installed = {}
    for entry in (data or {}).get("models", []):
        name = entry.get("name") or entry.get("model")
        if name:
            installed[name] = entry.get("details", {}) or {}

    models = []
    for tag in TEXT_MODELS:
        details = installed.get(tag)
        is_installed = details is not None
        param_size = (details or {}).get("parameter_size") or MODEL_FALLBACK_PARAMS.get(tag, "?")
        models.append({
            "tag": tag,
            "display_name": MODEL_DISPLAY_NAMES.get(tag, tag),
            "parameter_size": param_size,
            "online": reachable and is_installed,
        })
    return {"reachable": reachable, "models": models}


def _llamacpp_roster() -> dict:
    """Same shape as _ollama_roster, for LLAMACPP_MODELS (inference/config.py)
    against a llama-server running in router mode. reachable comes from
    /health; per-model installed status from /v1/models' OpenAI-compatible
    {"data": [{"id": ...}, ...]} shape -- confirmed directly against a real
    ghcr.io/ggml-org/llama.cpp:server container in router mode
    (`{"data": [], "object": "list"}` with an empty models dir; loaded
    presets would appear as entries in that same "data" list). Matched
    against either the configured identifier or its GGUF basename (with or
    without extension), covering the ways a router might report it."""
    reachable = _fetch_json(f"{settings.llamacpp_url}/health", LLAMACPP_STATUS_TIMEOUT) is not None
    installed_ids = set()
    if reachable:
        models_data = _fetch_json(f"{settings.llamacpp_url}/v1/models", LLAMACPP_STATUS_TIMEOUT)
        for entry in (models_data or {}).get("data", []):
            model_id = entry.get("id")
            if model_id:
                installed_ids.add(model_id)

    models = []
    for identifier, gguf_basename in LLAMACPP_MODELS.items():
        gguf_stem = gguf_basename.rsplit(".", 1)[0]
        is_installed = bool(installed_ids & {identifier, gguf_basename, gguf_stem})
        models.append({
            "tag": identifier,
            "display_name": identifier,
            "parameter_size": "?",
            "online": reachable and is_installed,
        })
    return {"reachable": reachable, "models": models}


def get_model_roster() -> dict:
    """Backs the model roster panel. Top-level ollama_reachable/models
    fields are the same shape the frontend has always read; active_backend/
    llamacpp_reachable/llamacpp_models are additive so existing frontend
    code keeps working untouched, with the llama.cpp side layered on."""
    ollama = _ollama_roster()
    llamacpp = _llamacpp_roster()
    return {
        "ollama_reachable": ollama["reachable"],
        "scripted_only": SCRIPTED_ONLY,
        "models": ollama["models"],
        "active_backend": "scripted" if SCRIPTED_ONLY else settings.inference_backend,
        "llamacpp_reachable": llamacpp["reachable"],
        "llamacpp_models": llamacpp["models"],
    }


async def _stream_show(seed: int, client: Optional[InferenceClient], backend: str,
                        models: Optional[List[str]] = None) -> AsyncIterator[bytes]:
    """Runs run_show as a concurrent asyncio task and streams its events
    out as newline-delimited JSON, exactly the wire format the frontend
    already consumes. on_emit (a plain sync callback -- EventLog.emit
    calls it inline, not awaited) pushes onto an asyncio.Queue; this
    generator drains it.

    Client-disconnect handling: the original Handler relied on
    self.wfile.write raising ConnectionAbortedError/ConnectionResetError/
    BrokenPipeError, caught around run_show, to stop burning real model
    calls for a browser tab that already closed. Here, when the client
    disconnects, Starlette stops iterating this generator and its
    enclosing task is cancelled -- caught by the `finally` below, which
    cancels the still-running show task in turn. Because run_show is now
    genuinely async (M2) with a real `await` at every agent call, that
    cancellation actually interrupts a live model call in flight, not
    just some future check -- real cooperative cancellation, not the
    on_emit-raises-to-unwind trick the synchronous version needed.
    """
    q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
    recorder = HistoryRecorder(seed=seed, scripted_only=SCRIPTED_ONLY)

    def on_emit(ev: Event) -> None:
        q.put_nowait(json.dumps(ev.to_dict()).encode("utf-8") + b"\n")
        # The one potentially slow bit (a handful of INSERTs, only on the
        # finale event -- see HistoryRecorder._write) moves off the event
        # loop; every other event is fast, in-memory bookkeeping and stays
        # inline. Fire-and-forget, same "never let this block or crash a
        # live show" contract on_event already has -- the response
        # doesn't wait on this task either.
        if ev.type == "finale":
            asyncio.create_task(asyncio.to_thread(recorder.on_event, ev))
        else:
            recorder.on_event(ev)

    async def worker() -> None:
        try:
            await run_show(seed=seed, log=EventLog(on_emit=on_emit), client=client,
                            backend=backend, models=models)
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    try:
        while True:
            item = await q.get()
            if item is None:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    init_db()
    # One shared InferenceClient (connection pool) per backend for the
    # app's whole lifetime rather than one per show -- constructed
    # unconditionally (SCRIPTED_ONLY never touches either) so /api/models
    # can always report both backends' real reachability regardless of
    # which one, if any, a show would actually use.
    app.state.ollama_client = OllamaInferenceClient(
        settings.ollama_url,
        max_retry_attempts=settings.retry_max_attempts,
        circuit_breaker_failure_threshold=settings.circuit_breaker_failure_threshold,
        circuit_breaker_cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
    )
    app.state.llamacpp_client = LlamaCppInferenceClient(
        settings.llamacpp_url,
        max_retry_attempts=settings.retry_max_attempts,
        circuit_breaker_failure_threshold=settings.circuit_breaker_failure_threshold,
        circuit_breaker_cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
    )
    try:
        yield
    finally:
        await app.state.ollama_client.aclose()
        await app.state.llamacpp_client.aclose()


app = FastAPI(title="Dominion", lifespan=lifespan)
# Matches the original Handler's blanket Access-Control-Allow-Origin: * --
# same-origin today (frontend and API share this one server), kept for
# local-dev flexibility (e.g. hitting /api/* from a page served elsewhere).
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape target -- dominion_inference_requests_total,
    dominion_inference_latency_seconds, dominion_inference_fallback_total
    (see observability/metrics.py), recorded on every inference client
    call in ollama_client.py/llamacpp_client.py. No external Prometheus
    setup needed to just look at this directly."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/stats")
def api_stats() -> JSONResponse:
    """Aggregated across every show ever recorded to
    engine/dominion_history.db -- backs the Standings page."""
    return JSONResponse(get_stats())


@app.get("/api/models")
def api_models() -> JSONResponse:
    """Backs the model roster panel -- polled periodically by the
    frontend since Ollama's up/down state can change while the page
    stays open."""
    return JSONResponse(get_model_roster())


@app.get("/api/model-catalog")
def api_model_catalog() -> JSONResponse:
    """Backs the pre-show model picker: every installed model per backend
    with real on-disk size, plus system memory, so the frontend can budget
    picks live (see model_catalog.py and web/index.html's picker logic)."""
    return JSONResponse(get_model_catalog())


@app.post("/api/run-show")
async def api_run_show(
    backend: Optional[str] = None,
    models: List[str] = Query(default=[]),
) -> StreamingResponse:
    # Plain query params, not a JSON body -- startShow() sends no body
    # today, and query params sidestep FastAPI/pydantic's optional-body
    # edge cases entirely for what's just a string and a short list of
    # strings. backend falls back to settings.inference_backend on
    # anything not recognized (missing, misspelled, hand-edited) rather
    # than erroring -- this is a convenience default, not a security
    # boundary, so failing open to the configured default is the
    # friendlier behavior. models empty means "use the fixed
    # TEXT_MODELS/LLAMACPP_MODELS roster", exactly today's pre-M8 behavior
    # (run_show's own default) -- the picker's frontend only ever sends a
    # non-empty list when the user actually customized the roster.
    if backend not in ("ollama", "llamacpp"):
        backend = settings.inference_backend
    client = app.state.llamacpp_client if backend == "llamacpp" else app.state.ollama_client
    seed = random.randint(0, 2_000_000_000)
    return StreamingResponse(
        _stream_show(seed, client, backend, models or None),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/", include_in_schema=False)
def index() -> Response:
    return FileResponse(WEB_DIR / "index.html")


# Static assets last -- registered after the API routes above so /api/*
# is matched first; this mount is the fallback for everything else under
# web/ (index.html's own asset references, stats.html, about.html,
# books.html, sounds/*).
app.mount("/", StaticFiles(directory=WEB_DIR), name="web")


def main() -> None:
    import uvicorn

    print(f"Dominion running at http://localhost:{PORT}")
    uvicorn.run(app, host="localhost", port=PORT)


if __name__ == "__main__":
    main()
