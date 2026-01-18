# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Principal operator for I3: Mind Principle.

Role: causal mechanism for planning/strategy structuring (ARP).
Constraints:
- Deterministic
- No belief formation; no autonomous goal selection
- Produces plan skeletons, step graphs, and trace metadata from explicit inputs

This operator should be invoked only with explicit objectives and constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from I3_Agent.diagnostics.errors import IntegrationError


@dataclass(frozen=True)
class PlanStep:
    sid: str
    description: str
    depends_on: List[str]


class MindPrincipalOperator:
    """Deterministic planner skeletonizer."""

    def __init__(self, *, max_steps: int = 12):
        if max_steps <= 0:
            raise IntegrationError("max_steps must be positive")
        self.max_steps = max_steps

    def build_plan_skeleton(
        self,
        *,
        objective: str,
        constraints: Optional[List[str]] = None,
        resources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Builds a minimal plan skeleton (no search, no inference).

        The caller must provide explicit objective/constraints/resources.
        The output is a structured placeholder plan that downstream ARP modules
        may refine.
        """
        if not isinstance(objective, str) or not objective.strip():
            raise IntegrationError("objective must be a non-empty string")

        constraints = constraints or []
        resources = resources or []

        # Deterministic skeleton: 4 canonical steps with room for refinement.
        steps: List[PlanStep] = [
            PlanStep(sid="s1", description="Restate objective and success criteria", depends_on=[]),
            PlanStep(sid="s2", description="Enumerate constraints and invariants", depends_on=["s1"]),
            PlanStep(sid="s3", description="Draft candidate action sequence", depends_on=["s2"]),
            PlanStep(sid="s4", description="Validate feasibility against constraints", depends_on=["s3"]),
        ]

        return {
            "objective": objective.strip(),
            "constraints": list(constraints),
            "resources": list(resources),
            "steps": [s.__dict__ for s in steps[: self.max_steps]],
            "notes": [
                "Deterministic plan skeleton only; requires downstream refinement.",
                "No autonomous goal selection performed.",
            ],
        }

    def apply_to_packet(
        self,
        *,
        packet: Dict[str, Any],
        objective_field: str = "objective",
        out_field: str = "mind_plan",
    ) -> Dict[str, Any]:
        if not isinstance(packet, dict):
            raise IntegrationError("packet must be a dict")

        objective = packet.get(objective_field, "")
        plan = self.build_plan_skeleton(objective=str(objective))

        out = dict(packet)
        out[out_field] = plan
        return out
