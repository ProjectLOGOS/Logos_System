from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .smp_intake import SMPEnvelope


@dataclass(frozen=True)
class SCPWorkOrder:
    """
    SCP work order derived from SMP.
    This is the contract SCP uses to run deeper analysis loops (MVS/BDN, etc.).
    """
    smp_id: str
    input_hash: str
    priority: str
    objectives: List[str] = field(default_factory=list)
    selected_domains: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    hints: Dict[str, Any] = field(default_factory=dict)


def build_work_order(*, envelope: SMPEnvelope) -> SCPWorkOrder:
    """
    Convert SMPEnvelope -> SCPWorkOrder.
    No heavy reasoning: just mapping fields and honoring directives.
    """
    fd = (envelope.final_decision or "").lower().strip()
    priority = "high" if fd in {"escalate", "quarantine"} else "normal"

    analysis = envelope.raw.get("analysis")
    selected_domains: List[str] = []
    if isinstance(analysis, dict):
        v = analysis.get("selected_iel_domains") or analysis.get("selected_domains")
        if isinstance(v, list):
            selected_domains = [str(x) for x in v]

    constraints: List[str] = []
    classification = envelope.raw.get("classification")
    if isinstance(classification, dict):
        c = classification.get("constraints")
        if isinstance(c, list):
            constraints = [str(x) for x in c]

    objectives = [
        "analyze_privative_structure",
        "attempt_iterative_stabilization",
        "evaluate_risk_trajectory",
    ]

    hints: Dict[str, Any] = {}
    triage_vector = envelope.raw.get("triage_vector")
    if isinstance(triage_vector, dict):
        hints["triage_vector"] = triage_vector

    hints["triadic_scores"] = envelope.triadic_scores
    hints["violations"] = envelope.violations

    return SCPWorkOrder(
        smp_id=envelope.smp_id,
        input_hash=envelope.input_hash,
        priority=priority,
        objectives=objectives,
        selected_domains=selected_domains,
        constraints=constraints,
        hints=hints,
    )
