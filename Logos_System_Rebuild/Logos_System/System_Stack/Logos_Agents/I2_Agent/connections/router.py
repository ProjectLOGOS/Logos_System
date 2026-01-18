# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .constants import PRIORITY_HIGH, PRIORITY_NORMAL


@dataclass(frozen=True)
class RouteDecision:
    """Minimal routing directive. No reasoning; just dispatch."""
    route_to: str
    reason: str
    priority: str = PRIORITY_NORMAL  # normal | high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_to": self.route_to,
            "reason": self.reason,
            "priority": self.priority,
        }


def decide_route(*, smp: Dict[str, Any], default_route: str) -> RouteDecision:
    """Route based on explicit SMP directive.

    Expected SMP fields (best-effort):
      - route_to (top-level) OR
      - mediation.route_to OR
      - final_decision (top-level) with mapping

    This router does NOT interpret content. It only respects explicit metadata.
    """

    if not isinstance(smp, dict):
        return RouteDecision(route_to=default_route, reason="Invalid SMP type; default routing.")

    # Direct field
    rt = smp.get("route_to")
    if isinstance(rt, str) and rt.strip():
        return RouteDecision(route_to=rt.strip(), reason="Explicit SMP route_to.")

    # Nested mediation
    mediation = smp.get("mediation")
    if isinstance(mediation, dict):
        rt2 = mediation.get("route_to")
        if isinstance(rt2, str) and rt2.strip():
            return RouteDecision(route_to=rt2.strip(), reason="Explicit mediation.route_to.")

    # Fallback mapping from final_decision
    fd = smp.get("final_decision")
    if isinstance(fd, str):
        fd = fd.lower().strip()
        if fd in {"escalate", "quarantine"}:
            return RouteDecision(route_to="LOGOS", reason=f"final_decision={fd}", priority=PRIORITY_HIGH)
        if fd in {"allow", "annotate_only", "decompose_only"}:
            return RouteDecision(route_to=default_route, reason=f"final_decision={fd}")

    return RouteDecision(route_to=default_route, reason="No directive found; default routing.")
