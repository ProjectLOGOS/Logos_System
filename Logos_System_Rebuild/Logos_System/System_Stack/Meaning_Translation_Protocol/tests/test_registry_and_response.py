# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Smoke tests for UIP registry and response formatter."""

from __future__ import annotations

import pytest

from User_Interaction_Protocol.uip_protocol.core_processing.registry import (
    StepHandler,
    UIPRegistry,
    UIPStatus,
    UIPStep,
)
from User_Interaction_Protocol.uip_protocol.output.response_formatter import (
    ResponseFormatter,
)


@pytest.mark.asyncio
async def test_registry_process_step_success() -> None:
    registry = UIPRegistry()

    async def dummy_handler(context):
        return {"ok": True}

    registry.register_handler(
        StepHandler(step=UIPStep.STEP_0_PREPROCESSING, handler_func=dummy_handler)
    )

    context = registry.create_context("session-test", "hello")
    context = await registry.process_step(context, UIPStep.STEP_0_PREPROCESSING)

    assert context.status == UIPStatus.COMPLETED
    assert context.step_results[UIPStep.STEP_0_PREPROCESSING] == {"ok": True}


@pytest.mark.asyncio
async def test_response_formatter_basic() -> None:
    formatter = ResponseFormatter()

    response = await formatter.synthesize_response(
        context={"user_input": "hello"},
        adaptive_profile={"confidence_level": 0.5},
        iel_bundle=None,
    )

    assert response.response_text
    assert 0.0 <= response.confidence <= 1.0
