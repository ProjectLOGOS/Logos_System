# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass(frozen=True)
class PlanPacket:
    task_id: str
    created_at: float
    smp_id: Optional[str]
    plan: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    constraints: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "smp_id": self.smp_id,
            "plan": self.plan,
            "rationale": self.rationale,
            "constraints": self.constraints,
            "provenance": self.provenance,
        }


def emit_plan_packet(
    *,
    task_id: str,
    smp_id: Optional[str],
    plan: List[Dict[str, Any]],
    rationale: str,
    constraints: Optional[List[str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> PlanPacket:
    return PlanPacket(
        task_id=task_id,
        created_at=time.time(),
        smp_id=smp_id,
        plan=plan,
        rationale=rationale,
        constraints=constraints or [],
        provenance=provenance or {},
    )
