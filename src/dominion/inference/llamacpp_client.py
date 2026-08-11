"""Async InferenceClient backed by llama.cpp's own server (llama-server),
talking to its native /completion endpoint rather than the OpenAI-compatible
/v1/chat/completions -- /completion is what accepts the "grammar" field
(GBNF, see grammars.py), which is the whole reason this backend exists
alongside Ollama (see docs/ENGINE_NOTES.md and README.md's "Two
implementations" section).

Run llama-server in router mode to serve every model this project needs
from one process/port, the direct analog of Ollama's own multi-model
swapping:

    llama-server --models-dir ./models --models-max 4

Each request's "model" field then selects which GGUF (by basename, no
extension needed) actually answers -- see inference/config.py's
LLAMACPP_MODELS for the identifier -> GGUF-basename mapping this project
uses, and README.md for where to get the actual model files (nothing here
auto-downloads a multi-gigabyte GGUF).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx
import structlog

from ..observability.metrics import record_inference_call
from .base import GenerationResult
from .config import LLAMACPP_MODELS
from .retry import CircuitBreaker, CircuitOpenError, call_with_retry

logger = structlog.get_logger(__name__)


def _server_alias(model: str) -> str:
    """Translate a LLAMACPP_MODELS registry identifier (e.g.
    "llama-3.2-3b-instruct", what every player draw and every log/metric/
    history record actually uses -- the friendly identifier, same role
    Ollama's own tags play) into the alias llama-server's router mode
    actually recognizes: the GGUF basename with the extension stripped
    (e.g. "Llama-3.2-3B-Instruct-Q4_K_M"). These two strings were never
    the same, at any point -- every real llama.cpp show has been sending
    the registry identifier straight through as the "model" field,
    which llama-server rejects with an HTTP 400 ("model '...' not
    found"), caught below by the exact same broad exception handler a
    genuine network failure hits -- so every single call silently fell
    back to ScriptedAgent, indistinguishable in logs from a normal
    fallback. Falls back to the input unchanged for any identifier not
    in the registry (a caller that already passes a real alias directly,
    e.g. an ad-hoc script, still works)."""
    gguf_name = LLAMACPP_MODELS.get(model, model)
    return gguf_name[:-len(".gguf")] if gguf_name.endswith(".gguf") else gguf_name


class LlamaCppInferenceClient:
    supports_grammar = True  # llama-server's /completion accepts a "grammar" (GBNF) field.
    backend_name = "llamacpp"

    def __init__(
        self,
        base_url: str,
        *,
        max_retry_attempts: int = 2,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_cooldown_seconds: float = 30.0,
    ) -> None:
        self._completion_url = f"{base_url.rstrip('/')}/completion"
        self._client = httpx.AsyncClient()
        self._max_retry_attempts = max_retry_attempts
        self._breaker = CircuitBreaker(
            circuit_breaker_failure_threshold, circuit_breaker_cooldown_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        timeout: float,
        num_predict: Optional[int] = None,
        grammar: Optional[str] = None,
    ) -> GenerationResult:
        # payload["model"] must be llama-server's own alias (the GGUF
        # basename), NOT the registry identifier every caller/log/metric
        # otherwise uses -- see _server_alias's docstring for why these
        # two were never the same string.
        payload: Dict[str, Any] = {"model": _server_alias(model), "prompt": prompt}
        if num_predict is not None:
            payload["n_predict"] = num_predict
        if grammar is not None:
            payload["grammar"] = grammar

        t0 = time.monotonic()
        attempts = 0

        async def _call() -> Dict[str, Any]:
            nonlocal attempts
            attempts += 1
            resp = await self._client.post(self._completion_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await call_with_retry(
                _call, max_attempts=self._max_retry_attempts, breaker=self._breaker)
        except (httpx.HTTPError, CircuitOpenError, ValueError) as exc:
            raw_latency = time.monotonic() - t0
            record_inference_call(backend=self.backend_name, model=model,
                                   succeeded=False, latency_seconds=raw_latency)
            # repr, not str -- some exceptions (a bare ConnectError wrapping
            # a Windows connection-refused OSError, observed directly)
            # stringify to an empty message, which would otherwise log a
            # useless empty error field with no information about what
            # actually failed.
            logger.warning("llamacpp_generate_failed", model=model, error=repr(exc),
                            retries=attempts - 1, latency_seconds=round(raw_latency, 3))
            return GenerationResult(
                text=None, think_seconds=None, raw_latency_seconds=raw_latency)

        raw_latency = time.monotonic() - t0
        text = (data.get("content") or "").strip()
        # llama-server's own timings.predicted_ms -- pure generation time,
        # already excluding prompt processing (timings.prompt_ms is
        # separate), the correct analog to Ollama's
        # total_duration-minus-load_duration: actual thinking time, not
        # infrastructure overhead. None if the field is missing/malformed.
        think_seconds = None
        timings = data.get("timings") or {}
        predicted_ms = timings.get("predicted_ms")
        if isinstance(predicted_ms, (int, float)):
            think_seconds = predicted_ms / 1000.0
        record_inference_call(backend=self.backend_name, model=model,
                               succeeded=True, latency_seconds=raw_latency)
        logger.info("llamacpp_generate_succeeded", model=model, retries=attempts - 1,
                    latency_seconds=round(raw_latency, 3),
                    think_seconds=round(think_seconds, 3) if think_seconds is not None else None)
        return GenerationResult(text=text, think_seconds=think_seconds, raw_latency_seconds=raw_latency)
