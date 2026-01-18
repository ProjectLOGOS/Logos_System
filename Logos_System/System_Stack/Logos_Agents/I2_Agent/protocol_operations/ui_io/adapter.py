# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from I2_Agent.config.constants import AGENT_I2
from I2_Agent.config.hashing import safe_hash
from I2_Agent.connections.id_handler import generate_packet_identity
from I2_Agent.connections.router import decide_route
from I2_Agent.protocol_operations.smp import build_smp


@dataclass(frozen=True)
class InboundResponse:
    route: str
    priority: str
    reason: str
    payload: Dict[str, Any]


def _normalize_inbound(inbound: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(inbound)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    # Fallback: wrap raw text into a dict for downstream stability.
    return {"input": inbound}


def route_input(*, inbound: str, default_route: str) -> InboundResponse:
    raw_payload = _normalize_inbound(inbound)
    identity = generate_packet_identity(origin=AGENT_I2, reference_obj=raw_payload)

    input_reference = {
        "input_hash": safe_hash(raw_payload),
        "original_input": raw_payload,
    }

    classification = {
        "tags": [],
        "domain": "unknown",
        "confidence": 0.0,
    }

    analysis = {
        "recommended_action": "allow",
        "summary": "UI ingress baseline",
    }

    transform_report: Dict[str, Any] = {
        "attempted": [],
        "succeeded": [],
        "failed": [],
        "status": "not_transformed",
    }

    benevolence = {"status": "unchecked"}

    triadic_scores: Dict[str, float] = {
        "existence": 0.0,
        "goodness": 0.0,
        "truth": 0.0,
    }

    provenance = {
        "ingress": "ui_io",
        "packet_identity": identity.to_dict(),
    }

    smp_obj = build_smp(
        origin_agent=AGENT_I2,
        input_reference=input_reference,
        classification=classification,
        analysis=analysis,
        transform_report=transform_report,
        bridge_passed=True,
        benevolence=benevolence,
        triadic_scores=triadic_scores,
        final_decision="allow",
        violations=[],
        route_to=default_route,
        triage_vector=None,
        delta_profile={},
        parent_id=identity.parent_id,
        provenance=provenance,
    )

    smp_dict = smp_obj.to_dict()
    decision = decide_route(smp=smp_dict, default_route=default_route)

    return InboundResponse(
        route=decision.route_to,
        priority=decision.priority,
        reason=decision.reason,
        payload=smp_dict,
    )


def handle_inbound(*, inbound: str, default_route: str) -> InboundResponse:
    return route_input(inbound=inbound, default_route=default_route)
