# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""
core_processing/registry.py

Central orchestration registry for the UIP pipeline. Collapses the legacy
registry implementation into a compact, auditable coordinator that tracks step
handlers, enforces dependency ordering, and records a detailed audit trail.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..utils.system_utils import handle_step_error, log_uip_event

LOGGER = logging.getLogger(__name__)


class UIPStep(Enum):
    STEP_0_PREPROCESSING = "preprocessing"
    STEP_1_LINGUISTIC = "linguistic_analysis"
    STEP_2_PXL_COMPLIANCE = "pxl_compliance"
    STEP_3_IEL_OVERLAY = "iel_overlay"
    STEP_4_TRINITY_INVOCATION = "trinity_invocation"
    STEP_5_ADAPTIVE = "adaptive_refinement"
    STEP_6_RESPONSE_SYNTHESIS = "response_synthesis"
    STEP_7_COMPLIANCE_RECHECK = "compliance_recheck"
    STEP_8_EGRESS_DELIVERY = "egress_delivery"


class UIPStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    DENIED = "denied"


@dataclass
class UIPContext:
    session_id: str
    correlation_id: str
    user_input: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[UIPStep, Any] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    current_step: Optional[UIPStep] = None
    status: UIPStatus = UIPStatus.PENDING
    timestamp: float = field(default_factory=time.time)


HandlerFunc = Callable[[UIPContext], Awaitable[Any]]


@dataclass
class StepHandler:
    step: UIPStep
    handler_func: HandlerFunc
    dependencies: List[UIPStep] = field(default_factory=list)
    timeout_seconds: int = 30
    retry_count: int = 0
    is_critical: bool = True


class UIPRegistry:
    """Register and execute ordered UIP pipeline steps."""

    def __init__(self) -> None:
        self.handlers: Dict[UIPStep, StepHandler] = {}
        self.pipeline_order: List[UIPStep] = list(UIPStep)
        self.active_contexts: Dict[str, UIPContext] = {}
        self.logger = LOGGER.getChild(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Registration and context creation.
    # ------------------------------------------------------------------
    def register_handler(self, handler: StepHandler) -> None:
        self.handlers[handler.step] = handler
        self.logger.info("Registered UIP handler for %s", handler.step.value)

    def create_context(
        self, session_id: str, user_input: str, metadata: Optional[Dict[str, Any]] = None
    ) -> UIPContext:
        correlation_id = str(uuid.uuid4())
        context = UIPContext(
            session_id=session_id,
            correlation_id=correlation_id,
            user_input=user_input,
            metadata=metadata or {},
        )
        self.active_contexts[correlation_id] = context
        audit_entry = {
            "timestamp": time.time(),
            "step": "INITIALISE",
            "action": "context_created",
            "session_id": session_id,
            "correlation_id": correlation_id,
        }
        context.audit_trail.append(audit_entry)
        log_uip_event("context_created", audit_entry)
        return context

    # ------------------------------------------------------------------
    # Pipeline execution.
    # ------------------------------------------------------------------
    async def process_pipeline(self, context: UIPContext) -> UIPContext:
        self.logger.info("Beginning UIP pipeline for %s", context.correlation_id)
        try:
            for step in self.pipeline_order:
                context = await self.process_step(context, step)
                if context.status in {UIPStatus.FAILED, UIPStatus.DENIED}:
                    self.logger.warning(
                        "Pipeline halted at %s for %s with status %s",
                        step.value,
                        context.correlation_id,
                        context.status.value,
                    )
                    break
        finally:
            self.active_contexts.pop(context.correlation_id, None)
        return context

    async def process_step(self, context: UIPContext, step: UIPStep) -> UIPContext:
        handler = self.handlers.get(step)
        if not handler:
            self.logger.debug("Skipping unregistered step %s", step.value)
            return context

        context.current_step = step
        context.status = UIPStatus.IN_PROGRESS

        audit_start = {
            "timestamp": time.time(),
            "step": step.value,
            "action": "step_started",
            "correlation_id": context.correlation_id,
        }
        context.audit_trail.append(audit_start)
        log_uip_event("step_started", audit_start)

        try:
            for dependency in handler.dependencies:
                if dependency not in context.step_results:
                    raise ValueError(f"Missing dependency {dependency.value} for {step.value}")

            attempts = max(handler.retry_count, 0) + 1
            retry_strategy = AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=0.5, min=0.1, max=max(handler.timeout_seconds, 1)),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            )

            result: Any = None
            async for attempt in retry_strategy:
                attempt_number = attempt.retry_state.attempt_number
                with attempt:
                    if attempt_number > 1:
                        retry_entry = {
                            "timestamp": time.time(),
                            "step": step.value,
                            "action": "step_retry",
                            "correlation_id": context.correlation_id,
                            "attempt": attempt_number,
                        }
                        context.audit_trail.append(retry_entry)
                        log_uip_event("step_retry", retry_entry)

                    result = await asyncio.wait_for(
                        handler.handler_func(context),
                        timeout=handler.timeout_seconds,
                    )
                break

            context.step_results[step] = result
            context.status = UIPStatus.COMPLETED

            audit_complete = {
                "timestamp": time.time(),
                "step": step.value,
                "action": "step_completed",
                "correlation_id": context.correlation_id,
                "result_summary": str(result)[:200] if result is not None else "none",
            }
            context.audit_trail.append(audit_complete)
            log_uip_event("step_completed", audit_complete)
        except asyncio.TimeoutError as exc:
            context.status = UIPStatus.FAILED
            payload = handle_step_error(step.value, exc, {"correlation_id": context.correlation_id})
            context.audit_trail.append(payload)
        except Exception as exc:  # pragma: no cover - defensive capture
            context.status = UIPStatus.FAILED
            payload = handle_step_error(step.value, exc, {"correlation_id": context.correlation_id})
            context.audit_trail.append(payload)
            if handler.is_critical:
                self.logger.error("Critical step %s failed: %s", step.value, exc)
        return context

    # ------------------------------------------------------------------
    # Registry diagnostics.
    # ------------------------------------------------------------------
    def get_pipeline_status(self) -> Dict[str, Any]:
        return {
            "registered_handlers": len(self.handlers),
            "active_contexts": len(self.active_contexts),
            "pipeline_steps": [step.value for step in self.pipeline_order],
        }


uip_registry = UIPRegistry()


def register_uip_handler(
    step: UIPStep,
    dependencies: Optional[List[UIPStep]] = None,
    timeout: int = 30,
    critical: bool = True,
) -> Callable[[HandlerFunc], HandlerFunc]:
    """Decorator for registering asynchronous UIP handlers."""

    def decorator(func: HandlerFunc) -> HandlerFunc:
        handler = StepHandler(
            step=step,
            handler_func=func,
            dependencies=dependencies or [],
            timeout_seconds=timeout,
            is_critical=critical,
        )
        uip_registry.register_handler(handler)
        return func

    return decorator


__all__ = [
    "UIPStep",
    "UIPStatus",
    "UIPContext",
    "StepHandler",
    "UIPRegistry",
    "uip_registry",
    "register_uip_handler",
]
