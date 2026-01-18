# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
Structured Meaning Packet (SMP) schema and builder for the I2 agent
termination stage.

This module is intentionally non-transformative. It only packages
already-evaluated metadata from the privation handler, bridge, benevolence
mediator, and Trinitarian mediator into a JSON-serializable record for
handoff to SCP (I1) or other agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class TriageVector:
    """
    Minimal, non-transformative IEL orientation overlay applied in I2.
    Serves only to record directional deltas; no remediation is performed here.
    """

    applied_iel: Optional[str] = None
    purpose: Optional[str] = None  # e.g. "orientation", "stabilization"
    overlay_type: str = "soft"
    delta_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied_iel": self.applied_iel,
            "purpose": self.purpose,
            "overlay_type": self.overlay_type,
            "delta_profile": self.delta_profile,
        }


@dataclass
class StructuredMeaningPacket:
    smp_id: str
    timestamp: float
    origin_agent: str
    parent_id: Optional[str]

    # References
    input_reference: Dict[str, Any]

    # Privation pipeline
    classification: Dict[str, Any]
    analysis: Dict[str, Any]
    transform_report: Dict[str, Any]

    # Constraint checks
    bridge_passed: bool
    benevolence: Dict[str, Any]

    # Trinitarian mediation
    triadic_scores: Dict[str, float]
    final_decision: str
    violations: List[str]
    route_to: str

    # Orientation overlay
    triage_vector: Optional[TriageVector] = None
    delta_profile: Dict[str, Any] = field(default_factory=dict)

    # Provenance
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        packet = {
            "smp_id": self.smp_id,
            "timestamp": self.timestamp,
            "origin_agent": self.origin_agent,
            "parent_id": self.parent_id,
            "input_reference": self.input_reference,
            "classification": self.classification,
            "analysis": self.analysis,
            "transform_report": self.transform_report,
            "bridge_passed": self.bridge_passed,
            "benevolence": self.benevolence,
            "triadic_scores": self.triadic_scores,
            "final_decision": self.final_decision,
            "violations": self.violations,
            "route_to": self.route_to,
            "delta_profile": self.delta_profile or {},
            "provenance": self.provenance,
        }

        if self.triage_vector:
            packet["triage_vector"] = self.triage_vector.to_dict()
            if not self.delta_profile:
                packet["delta_profile"] = self.triage_vector.delta_profile

        return packet


# Factory function

def build_smp(
    *,
    origin_agent: str,
    input_reference: Dict[str, Any],
    classification: Dict[str, Any],
    analysis: Dict[str, Any],
    transform_report: Dict[str, Any],
    bridge_passed: bool,
    benevolence: Dict[str, Any],
    triadic_scores: Dict[str, float],
    final_decision: str,
    violations: List[str],
    route_to: str,
    triage_vector: Optional[TriageVector] = None,
    delta_profile: Optional[Dict[str, Any]] = None,
    parent_id: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> StructuredMeaningPacket:
    """
    Package downstream-ready SMP metadata without altering or inferring content.
    """

    return StructuredMeaningPacket(
        smp_id=str(uuid.uuid4()),
        timestamp=time.time(),
        origin_agent=origin_agent,
        parent_id=parent_id,
        input_reference=input_reference,
        classification=classification,
        analysis=analysis,
        transform_report=transform_report,
        bridge_passed=bridge_passed,
        benevolence=benevolence,
        triadic_scores=triadic_scores,
        final_decision=final_decision,
        violations=violations,
        route_to=route_to,
        triage_vector=triage_vector,
        delta_profile=delta_profile or (triage_vector.delta_profile if triage_vector else {}),
        provenance=provenance or {},
    )
