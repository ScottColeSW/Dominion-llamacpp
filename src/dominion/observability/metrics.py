"""Prometheus metrics for the inference layer -- exposed at /metrics
(server/app.py). No external Prometheus/Grafana stack is required to run
locally; this is just the endpoint, scrape it or don't.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

INFERENCE_REQUESTS = Counter(
    "dominion_inference_requests_total",
    "Inference calls by backend/model/outcome",
    ["backend", "model", "outcome"],
)
INFERENCE_LATENCY = Histogram(
    "dominion_inference_latency_seconds",
    "Real, uncharged wall-clock latency per inference call",
    ["backend", "model"],
)
INFERENCE_FALLBACKS = Counter(
    "dominion_inference_fallback_total",
    "Inference calls that returned no usable reply (client-level failure, "
    "before ScriptedAgent's own fallback logic ever runs)",
    ["backend", "model"],
)


def record_inference_call(*, backend: str, model: str, succeeded: bool,
                           latency_seconds: float) -> None:
    INFERENCE_LATENCY.labels(backend=backend, model=model).observe(latency_seconds)
    outcome = "success" if succeeded else "failure"
    INFERENCE_REQUESTS.labels(backend=backend, model=model, outcome=outcome).inc()
    if not succeeded:
        INFERENCE_FALLBACKS.labels(backend=backend, model=model).inc()
