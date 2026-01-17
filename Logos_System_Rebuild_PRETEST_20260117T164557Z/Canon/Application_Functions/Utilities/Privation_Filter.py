from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from Logos_System.System_Stack.Logos_Protocol.Runtime_Operations.Governance.Constraint_Result import ConstraintResult
from Logos_System.System_Stack.Logos_Protocol.Runtime_Operations.Governance.Constraint_Context import ConstraintContext

from Logos_System.System_Stack.Logos_Protocol.Logos_Agents.I2_Agent.protocol_operations.privation_handler.privation_classifier import (
    PrivationClassification,
    classify,
)
from Logos_System.System_Stack.Logos_Protocol.Logos_Agents.I2_Agent.protocol_operations.privation_handler.privation_analyst import (
    PrivationAnalysis,
    analyze,
)
from Logos_System.System_Stack.Logos_Protocol.Logos_Agents.I2_Agent.protocol_operations.privation_handler.privation_override import (
    PrivationOverride,
    PrivationOverrideDecision,
)
from Logos_System.System_Stack.Logos_Protocol.Logos_Agents.I2_Agent.protocol_operations.privation_handler.privation_transformer import (
    PrivationTransformer,
    TransformResult,
)


def _context_to_dict(context: ConstraintContext) -> Dict[str, Any]:
    return {
        "agent_id": context.agent_id,
        "session_id": context.session_id,
        "source": context.source,
        "tlm_token": context.tlm_token,
        "runtime_flags": dict(context.runtime_flags or {}),
    }


class Privation_Filter:
    """
    Application-neutral privation filter that wraps the I2 privation pipeline
    (classifier -> analyst -> override -> transformer) and emits a ConstraintResult.
    """

    @staticmethod
    def filter(
        payload: Any,
        *,
        context: ConstraintContext,
        override_requested: bool = False,
        override_token: Optional[Dict[str, Any]] = None,
        overlay_module: Optional[str] = None,
    ) -> ConstraintResult:
        tags: Dict[str, Any] = {
            "validator": "Privation_Filter",
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "source": context.source,
        }

        classification: PrivationClassification = classify(payload)
        classification_dict = asdict(classification)
        tags["classification"] = classification_dict

        analysis: PrivationAnalysis = analyze(
            classification=classification_dict,
            overlay_module=overlay_module,
        )
        analysis_dict = asdict(analysis)
        tags["analysis"] = analysis_dict

        override_decision: PrivationOverrideDecision = PrivationOverride.evaluate(
            override_requested=override_requested,
            override_token=override_token,
            context=_context_to_dict(context),
        )
        tags["override"] = {
            "requested": override_requested,
            "allowed": override_decision.allowed,
            "authority": override_decision.authority,
            "reason": override_decision.reason,
        }

        if override_decision.allowed:
            return ConstraintResult(
                ok=True,
                reason="Privation filter override authorized",
                tags=tags,
            )

        if analysis.action in {"quarantine", "escalate"}:
            tags["blocked"] = analysis.action
            return ConstraintResult(
                ok=False,
                reason=f"Privation filter: payload blocked ({analysis.action})",
                tags=tags,
            )

        if analysis.action == "transform":
            transform_result: TransformResult = PrivationTransformer.transform_input(
                payload=payload,
                context=_context_to_dict(context),
                classification=classification_dict,
                analysis=analysis_dict,
            )
            tags["transform"] = {
                "report": transform_result.transform_report,
                "exit_metadata": transform_result.exit_metadata,
            }
            tags["final_payload"] = transform_result.new_payload
            return ConstraintResult(
                ok=True,
                reason="Privation filter: payload transformed",
                tags=tags,
            )

        # Default allow path (analysis.action == "allow" or anything unrecognized but non-blocking)
        tags["final_payload"] = payload
        return ConstraintResult(
            ok=True,
            reason="Privation filter: payload allowed",
            tags=tags,
        )
