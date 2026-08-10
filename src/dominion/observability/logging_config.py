"""One-time structlog setup, called from server/app.py's lifespan.
Every inference client call gets logged as structured JSON (model,
backend, latency, retries, outcome) instead of free-text log lines --
the point being that these are meant to be queried/aggregated by a real
log pipeline, not just read by a human watching a terminal.
"""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
