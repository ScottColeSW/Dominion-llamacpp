"""Async InferenceClient backed by Ollama's /api/generate. Faithful port
of the original, synchronous _ask_ollama (see docs/ENGINE_NOTES.md) --
same URL/timeout semantics, same think_seconds derivation -- onto httpx
and the shared retry/circuit-breaker helper (retry.py), so agents never
call urllib/httpx directly.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx
import structlog

from ..observability.metrics import record_inference_call
from .base import GenerationResult
from .retry import CircuitBreaker, CircuitOpenError, call_with_retry

logger = structlog.get_logger(__name__)


class OllamaInferenceClient:
    supports_grammar = False  # Ollama's public /api/generate has no GBNF equivalent.
    backend_name = "ollama"

    def __init__(
        self,
        base_url: str,
        *,
        max_retry_attempts: int = 2,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_cooldown_seconds: float = 30.0,
        keep_alive: str = "5m",
    ) -> None:
        # 127.0.0.1, not "localhost" -- on some machines "localhost"
        # resolves to the IPv6 loopback first, but Ollama only listens on
        # IPv4, so that connection hangs in SYN_SENT for minutes instead
        # of failing fast, silently defeating every timeout below. The
        # caller (inference/config.py's default) is expected to pass an
        # explicit 127.0.0.1 URL; this class doesn't rewrite it itself so
        # a deliberately different host (e.g. a remote Ollama) still works.
        self._generate_url = f"{base_url.rstrip('/')}/api/generate"
        self._client = httpx.AsyncClient()
        self._max_retry_attempts = max_retry_attempts
        self._breaker = CircuitBreaker(
            circuit_breaker_failure_threshold, circuit_breaker_cooldown_seconds)
        # See settings.ollama_keep_alive's own comment for the full "RAM
        # climbing every round" story -- sent on every /api/generate call
        # below so Ollama actually unloads an idle model instead of
        # falling back to its own 5-minute default, which let this show's
        # whole model pool pile up in memory at once.
        self._keep_alive = keep_alive

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
        if grammar is not None:
            logger.debug("ollama_grammar_unsupported", model=model)
        payload: Dict[str, Any] = {
            "model": model, "prompt": prompt, "stream": False, "keep_alive": self._keep_alive,
        }
        if num_predict is not None:
            payload["options"] = {"num_predict": num_predict}

        t0 = time.monotonic()
        attempts = 0

        # Scott's traces: a single duel_turn charging 41s, and separately a
        # liveTick that ran 57 real seconds before landing a turn charged
        # at only 1s ("clock... seems to fight itself"). Root cause: this
        # closure used to pass the SAME full `timeout` to every retry
        # attempt, so call_with_retry (retry.py) could genuinely spend up
        # to max_attempts * timeout of real wall-clock time -- e.g. one
        # attempt genuinely timing out around ~30s, then a retry that
        # falls back fast, landing a small charged seconds_used despite
        # the audience having just watched the live clock tick for nearly
        # a minute. duel.py's whole clock model (attempt_question's
        # call_timeout, derived from time_remaining) assumes ONE call
        # costs at most `timeout` -- that assumption only holds if retries
        # share the same budget instead of each getting a fresh one.
        # Shrinking `timeout` here by elapsed time on each attempt makes
        # the TOTAL time across every retry bounded by the original
        # `timeout`, matching what duel.py already assumed.
        async def _call() -> Dict[str, Any]:
            nonlocal attempts
            attempts += 1
            remaining = timeout - (time.monotonic() - t0)
            if remaining <= 0:
                raise httpx.TimeoutException(
                    "overall call budget exhausted before this retry attempt")
            resp = await self._client.post(self._generate_url, json=payload, timeout=remaining)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await call_with_retry(
                _call, max_attempts=self._max_retry_attempts, breaker=self._breaker)
        except (httpx.HTTPError, CircuitOpenError, ValueError) as exc:
            raw_latency = time.monotonic() - t0
            record_inference_call(backend=self.backend_name, model=model,
                                   succeeded=False, latency_seconds=raw_latency)
            # repr, not str -- see llamacpp_client.py's identical comment;
            # some exceptions stringify to an empty message on this
            # platform, which would log a useless empty error field otherwise.
            logger.warning("ollama_generate_failed", model=model, error=repr(exc),
                            retries=attempts - 1, latency_seconds=round(raw_latency, 3))
            return GenerationResult(
                text=None, think_seconds=None, raw_latency_seconds=raw_latency)

        raw_latency = time.monotonic() - t0
        text = (data.get("response") or "").strip()
        # Ollama's own total_duration minus load_duration (both
        # nanoseconds) -- actual generation time with model-loading
        # overhead excluded, so a player whose model happens to already
        # be resident vs. one that needs a cold load doesn't get a
        # different trivia-clock charge for the exact same quality of
        # answer. None if those fields are missing/malformed.
        think_seconds = None
        total_ns, load_ns = data.get("total_duration"), data.get("load_duration")
        if isinstance(total_ns, (int, float)) and isinstance(load_ns, (int, float)) and total_ns >= load_ns:
            think_seconds = (total_ns - load_ns) / 1e9
        record_inference_call(backend=self.backend_name, model=model,
                               succeeded=True, latency_seconds=raw_latency)
        logger.info("ollama_generate_succeeded", model=model, retries=attempts - 1,
                    latency_seconds=round(raw_latency, 3),
                    think_seconds=round(think_seconds, 3) if think_seconds is not None else None)
        return GenerationResult(text=text, think_seconds=think_seconds, raw_latency_seconds=raw_latency)
