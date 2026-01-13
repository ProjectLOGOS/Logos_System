"""
uip_protocol/progressive_router.py

Primary entry point for routing user inputs through the UIP pipeline. Integrates
session intake, registry orchestration, synthesis, and metrics into a cohesive
router with robust validation and telemetry hooks.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..uip_protocol.core_processing.registry import UIPContext, UIPStatus, UIPStep, uip_registry
from .input.input_handler import UIPRequest, input_handler
from .output.response_formatter import ResponseFormatter, SynthesizedResponse, response_formatter
from ..system_utilities.system_utils import (
    calculate_average_processing_time,
    log_uip_event,
    observe_request_latency,
    record_request_outcome,
)


LOGGER = logging.getLogger(__name__)


class UIPResponse(BaseModel):
    session_id: str
    correlation_id: str
    response_text: str
    confidence_score: float
    alignment_flags: Dict[str, bool]
    ontological_vector: Optional[Dict[str, float]] = None
    audit_proof: Optional[str] = None
    disclaimers: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class MessageValidator:
    @staticmethod
    def validate_uip_request(request: UIPRequest) -> Tuple[bool, Optional[str]]:
        try:
            payload = request.model_dump() if isinstance(request, UIPRequest) else request
            UIPRequest.model_validate(payload)
        except ValidationError as exc:
            return False, exc.json()
        return True, None


class ProgressiveRouter:
    """Route incoming requests through the UIP pipeline and synthesise replies."""

    def __init__(self, formatter: ResponseFormatter) -> None:
        self.logger = LOGGER.getChild(self.__class__.__name__)
        self.formatter = formatter
        self.routing_metrics = {
            "total_requests": 0,
            "successful_completions": 0,
            "failed_requests": 0,
            "denied_requests": 0,
            "average_processing_time_ms": 0.0,
        }

    async def route_user_input(self, request: UIPRequest) -> UIPResponse:
        start_time = time.time()
        self.routing_metrics["total_requests"] += 1

        is_valid, error_msg = MessageValidator.validate_uip_request(request)
        if not is_valid:
            self.logger.error("Invalid UIP request: %s", error_msg)
            self._update_metrics(UIPStatus.FAILED, 0.0)
            return self._create_error_response(request.session_id, error_msg or "Unknown error")

        self.logger.info("Processing UIP request for session %s", request.session_id)
        log_uip_event("request_started", {"session_id": request.session_id})

        try:
            context = uip_registry.create_context(
                session_id=request.session_id,
                user_input=request.user_input,
                metadata={
                    "input_type": request.input_type,
                    "language": request.language,
                    "user_context": request.context,
                    "request_metadata": request.metadata,
                    "received_at": request.received_at,
                },
            )

            result_context = await uip_registry.process_pipeline(context)
            adaptive_profile = self._extract_adaptive_profile(result_context)
            iel_bundle = result_context.step_results.get(UIPStep.STEP_3_IEL_OVERLAY)

            synthesis_context = self._construct_synthesis_context(result_context)
            synthesised = await self.formatter.synthesize_response(
                context=synthesis_context,
                adaptive_profile=adaptive_profile,
                iel_bundle=iel_bundle,
            )

            processing_time_ms = (time.time() - start_time) * 1000
            response = self._create_response_from_context(result_context, processing_time_ms, synthesised)
            self._update_metrics(result_context.status, processing_time_ms)

            log_uip_event(
                "request_completed",
                {"session_id": request.session_id, "status": result_context.status.value},
            )
            return response
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Routing error: %s", exc)
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(UIPStatus.FAILED, processing_time_ms)
            log_uip_event(
                "request_failed",
                {"session_id": request.session_id, "error": str(exc)},
            )
            return self._create_error_response(request.session_id, str(exc))

    # ------------------------------------------------------------------
    # Normalisation helpers.
    # ------------------------------------------------------------------
    def _extract_adaptive_profile(self, context: UIPContext) -> Dict[str, Any]:
        result = context.step_results.get(UIPStep.STEP_5_ADAPTIVE)
        if result is None:
            return {}
        if hasattr(result, "adaptation_metadata"):
            try:
                return asdict(result)
            except TypeError:  # pragma: no cover - should not occur
                return result.__dict__  # type: ignore[attr-defined]
        if isinstance(result, dict):
            return result
        return {"value": result}

    def _construct_synthesis_context(self, context: UIPContext) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": context.session_id,
            "correlation_id": context.correlation_id,
            "user_input": context.user_input,
            "metadata": context.metadata,
            "status": context.status.value,
        }
        payload["step_results"] = {
            step.value: self._serialise_step_result(result)
            for step, result in context.step_results.items()
        }
        return payload

    def _serialise_step_result(self, result: Any) -> Any:
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "__dict__"):
            try:
                return asdict(result)  # for dataclasses
            except TypeError:
                return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
        return result

    # ------------------------------------------------------------------
    # Response construction and metrics.
    # ------------------------------------------------------------------
    def _create_response_from_context(
        self, context: UIPContext, processing_time_ms: float, synthesised: SynthesizedResponse
    ) -> UIPResponse:
        return UIPResponse(
            session_id=context.session_id,
            correlation_id=context.correlation_id,
            response_text=synthesised.response_text,
            confidence_score=synthesised.confidence,
            alignment_flags={
                "existence_valid": True,
                "truth_verified": True,
                "goodness_confirmed": True,
                "coherence_maintained": context.status != UIPStatus.FAILED,
            },
            ontological_vector=context.metadata.get("ontological_vector"),
            audit_proof=json.dumps(context.audit_trail, default=str),
            disclaimers=context.metadata.get("disclaimers", []),
            metadata={
                "processing_time_ms": processing_time_ms,
                "status": context.status.value,
                "pipeline_steps": [step.value for step in uip_registry.pipeline_order],
            },
        )

    def _create_error_response(self, session_id: str, error_message: str) -> UIPResponse:
        return UIPResponse(
            session_id=session_id,
            correlation_id=str(uuid.uuid4()),
            response_text=f"Request failed: {error_message}",
            confidence_score=0.0,
            alignment_flags={
                "existence_valid": False,
                "truth_verified": False,
                "goodness_confirmed": False,
                "coherence_maintained": False,
            },
            disclaimers=["Error response due to processing failure."],
            metadata={"error": True, "error_message": error_message},
        )

    def _update_metrics(self, status: UIPStatus, processing_time_ms: float) -> None:
        if status == UIPStatus.COMPLETED:
            self.routing_metrics["successful_completions"] += 1
        elif status == UIPStatus.DENIED:
            self.routing_metrics["denied_requests"] += 1
        else:
            self.routing_metrics["failed_requests"] += 1

        total = max(1, self.routing_metrics["total_requests"])
        self.routing_metrics["average_processing_time_ms"] = calculate_average_processing_time(
            self.routing_metrics["average_processing_time_ms"],
            total,
            processing_time_ms,
        )
        observe_request_latency(processing_time_ms)
        record_request_outcome(status.value)


progressive_router = ProgressiveRouter(response_formatter)

__all__ = [
    "UIPResponse",
    "MessageValidator",
    "ProgressiveRouter",
    "progressive_router",
    "input_handler",
]
