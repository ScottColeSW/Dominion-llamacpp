"""Bounded retry + a simple per-endpoint circuit breaker, shared by every
InferenceClient backend. A live model call is worth one quick retry on a
transient failure, but not an unbounded one; once an endpoint has failed
enough times in a row, the breaker opens and fails fast for a cooldown
window instead of continuing to pile up slow, doomed calls against a
backend that's genuinely down (each of which would otherwise still pay
its own full timeout before failing).
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable, Optional, TypeVar

import httpx

T = TypeVar("T")


class CircuitOpenError(Exception):
    """Raised instead of attempting a call while the breaker is open."""


class CircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_seconds: float) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None

    def _cooldown_elapsed(self) -> bool:
        return self._opened_at is not None and (
            time.monotonic() - self._opened_at >= self.cooldown_seconds
        )

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._cooldown_elapsed():
            # Half-open: let the next call through as a probe, resetting
            # state so one success closes the breaker cleanly rather than
            # requiring failure_threshold successes to "fully" recover.
            self._opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = time.monotonic()


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    breaker: CircuitBreaker,
) -> T:
    """Runs fn() up to max_attempts times, tracking outcomes on breaker.
    Raises CircuitOpenError immediately (no attempt made at all) if the
    breaker is already open; raises the last real exception if every
    attempt fails. Callers (each InferenceClient's generate()) catch both
    and translate them into a GenerationResult(text=None, ...) -- this
    module only handles retry/breaker bookkeeping, not the
    never-raises-to-the-agent contract."""
    if breaker.is_open():
        raise CircuitOpenError()

    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            result = await fn()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see class docstring
            last_exc = exc
            # Scott: "there is a fallback trap... when we fallback once, we
            # decide to always fallback after once." Root cause: every
            # exception here used to count the same toward the breaker,
            # including a perfectly ordinary read timeout from a call that
            # simply had a tight budget (a low time_remaining, or a
            # cold-loading model) -- inference/config.py's own comment on
            # these settings already describes the INTENDED distinction
            # ("a transient failure (a dropped connection, not a real
            # timeout)"), just never actually enforced it. This game's own
            # call sites hand out short timeouts constantly by design
            # (attempt_question caps its wait to roughly what's left on
            # the clock), so hitting circuit_breaker_failure_threshold
            # consecutive read timeouts is a normal, frequent occurrence,
            # not evidence Ollama is actually down -- yet it tripped the
            # SAME breaker, which then force-failed EVERY subsequent call
            # (even from players with plenty of time left) for a full
            # circuit_breaker_cooldown_seconds, compounding one slow turn
            # into many forced fallbacks. A genuine inability to reach the
            # backend at all (ConnectError/ConnectTimeout) or a real error
            # response still counts; the response just being slow doesn't.
            is_mere_slowness = (
                isinstance(exc, httpx.TimeoutException)
                and not isinstance(exc, httpx.ConnectTimeout)
            )
            if not is_mere_slowness:
                breaker.record_failure()
                if breaker.is_open():
                    break
            continue
        breaker.record_success()
        return result
    assert last_exc is not None
    raise last_exc
