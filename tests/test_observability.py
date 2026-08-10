"""observability/metrics.py and logging_config.py -- pure logic only
(recording into the Prometheus registry, configuring structlog); the
actual /metrics HTTP endpoint is a two-line wrapper around
prometheus_client.generate_latest() verified manually against a running
server rather than here, to avoid spinning up the full app lifespan
(real httpx clients) just to check a trivial route.
"""
from __future__ import annotations

import unittest

from dominion.observability.logging_config import configure_logging
from dominion.observability.metrics import (
    INFERENCE_FALLBACKS,
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    record_inference_call,
)


class MetricsTest(unittest.TestCase):
    def test_successful_call_increments_requests_and_latency_not_fallbacks(self) -> None:
        labels = {"backend": "testbackend", "model": "test-model-success"}
        before_requests = INFERENCE_REQUESTS.labels(**labels, outcome="success")._value.get()
        before_fallbacks = INFERENCE_FALLBACKS.labels(**labels)._value.get()
        before_latency_count = INFERENCE_LATENCY.labels(**labels)._sum.get()

        record_inference_call(backend="testbackend", model="test-model-success",
                               succeeded=True, latency_seconds=0.5)

        self.assertEqual(
            INFERENCE_REQUESTS.labels(**labels, outcome="success")._value.get(),
            before_requests + 1,
        )
        self.assertEqual(INFERENCE_FALLBACKS.labels(**labels)._value.get(), before_fallbacks)
        self.assertAlmostEqual(
            INFERENCE_LATENCY.labels(**labels)._sum.get(), before_latency_count + 0.5,
        )

    def test_failed_call_increments_fallbacks(self) -> None:
        labels = {"backend": "testbackend", "model": "test-model-failure"}
        before_fallbacks = INFERENCE_FALLBACKS.labels(**labels)._value.get()
        before_failures = INFERENCE_REQUESTS.labels(**labels, outcome="failure")._value.get()

        record_inference_call(backend="testbackend", model="test-model-failure",
                               succeeded=False, latency_seconds=1.0)

        self.assertEqual(INFERENCE_FALLBACKS.labels(**labels)._value.get(), before_fallbacks + 1)
        self.assertEqual(
            INFERENCE_REQUESTS.labels(**labels, outcome="failure")._value.get(),
            before_failures + 1,
        )


class LoggingConfigTest(unittest.TestCase):
    def test_configure_logging_does_not_raise(self) -> None:
        configure_logging()  # idempotent enough to call twice in a test session
        configure_logging()


if __name__ == "__main__":
    unittest.main()
