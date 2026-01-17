from __future__ import annotations

from typing import Any, Dict, Tuple

from .risk_estimator import estimate_trajectory


def attach_trajectory_estimate(
    *,
    smp: Dict[str, Any],
    findings: Dict[str, Any],
    recommended_next: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attach non-tactical trajectory/risk estimate into findings and recommended_next.

    Returns updated (findings, recommended_next).
    """
    findings = dict(findings or {})
    recommended_next = dict(recommended_next or {})

    est = estimate_trajectory(smp=smp)
    findings["trajectory_estimate"] = est.to_dict()

    recommended_next.setdefault("route_to", est.recommended_route)
    recommended_next.setdefault("recommended_action", est.recommended_action)

    return findings, recommended_next
