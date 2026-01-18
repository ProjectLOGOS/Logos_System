# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: FORBIDDEN (DRY_RUN_ONLY)
# AUTHORITY: GOVERNED
# INSTALL_STATUS: DRY_RUN_ONLY
# SOURCE_LEGACY: system_utils.py

"""
DRY-RUN REWRITE

This file is a structural, governed rewrite candidate generated for
rewrite-system validation only. No execution, no side effects.
"""
"""
utils/system_utils.py

Shared utility functions for the UIP protocol.
Consolidated from common logging, timestamp, and error handling patterns across
legacy files. Provides consistent logging, event tracking, timestamps, and error
wrapping without introducing heavy dependencies.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

__all__ = [
    "logger",
    "structured_logger",
    "METRICS_REGISTRY",
    "get_current_timestamp",
    "log_uip_event",
    "handle_step_error",
    "calculate_average_processing_time",
    "generate_correlation_id",
    "record_request_outcome",
    "observe_request_latency",
    "export_metrics",
]


logger = logging.getLogger("UIP")
# Avoid duplicate handlers when module reloaded in tests / REPL.
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Structured logging configuration using structlog for richer audit trails.
structlog.configure_once(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
structured_logger = structlog.get_logger("UIP")


# Prometheus metrics registry and counters for UIP observability.
METRICS_REGISTRY = CollectorRegistry()
EVENT_COUNTER = Counter(
    "uip_events_total",
    "Total number of UIP events emitted via log_uip_event",
    labelnames=["event_type"],
    registry=METRICS_REGISTRY,
)
REQUEST_COUNTER = Counter(
    "uip_requests_total",
    "Total number of UIP routing outcomes",
    labelnames=["status"],
    registry=METRICS_REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "uip_request_latency_seconds",
    "UIP routing latency in seconds",
    registry=METRICS_REGISTRY,
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, float("inf")),
)


def get_current_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with trailing 'Z'."""
    return datetime.now(timezone.utc).isoformat()


def log_uip_event(event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Log a structured UIP event for auditing and diagnostics."""
    payload: Dict[str, Any] = {"event": event_type.upper()}
    if data:
        payload["data"] = data
    try:
        structured_logger.info("uip_event", **payload)
    except Exception:  # pragma: no cover - structlog fallback
        message = f"UIP Event [{event_type.upper()}]"
        if data:
            try:
                encoded = json.dumps(data, default=str, separators=(",", ":"))
                message = f"{message} | {encoded}"
            except Exception as exc:  # pragma: no cover - defensive guard
                message = f"{message} | JSON encode failed: {exc}"
        logger.info(message)
    else:
        message = f"UIP Event [{event_type.upper()}]"
        if data:
            try:
                encoded = json.dumps(data, default=str, separators=(",", ":"))
                logger.info("%s | %s", message, encoded)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.info("%s | JSON encode failed: %s", message, exc)
        else:
            logger.info(message)
    EVENT_COUNTER.labels(event_type=event_type.lower()).inc()


def handle_step_error(
    step: str, error: Exception, context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Wrap an exception from a UIP step into a structured payload."""
    payload: Dict[str, Any] = {
        "step": step,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "timestamp": get_current_timestamp(),
    }
    if context:
        payload["context_summary"] = str(context)[:200]
    log_uip_event("step_error", payload)
    return payload


def calculate_average_processing_time(
    current_avg: float, total_requests: int, new_time: float
) -> float:
    """Update running average processing time in milliseconds."""
    if total_requests <= 0:
        return 0.0
    return ((current_avg * (total_requests - 1)) + new_time) / total_requests


def generate_correlation_id() -> str:
    """Create a unique identifier for request correlation."""
    return str(uuid.uuid4())


def record_request_outcome(status: str) -> None:
    """Increment Prometheus counter for a routing outcome."""
    REQUEST_COUNTER.labels(status=status.lower()).inc()


def observe_request_latency(duration_ms: float) -> None:
    """Record routing latency in seconds for Prometheus histograms."""
    clamped_ms = max(duration_ms, 0.0)
    REQUEST_LATENCY.observe(clamped_ms / 1000.0)


def export_metrics() -> bytes:
    """Return the latest UIP metrics snapshot in Prometheus exposition format."""
    return generate_latest(METRICS_REGISTRY)