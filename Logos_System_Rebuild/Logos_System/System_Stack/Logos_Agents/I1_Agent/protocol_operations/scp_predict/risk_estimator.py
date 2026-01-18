# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from typing import Any, Dict, List

from ..scp_analysis.trajectory_types import RiskSignal, TrajectoryEstimate


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def estimate_trajectory(*, smp: Dict[str, Any]) -> TrajectoryEstimate:
    """
    Conservative, metadata-only estimator.

    Uses:
      - analysis.severity_score
      - violations
      - triadic_scores (coherence/conservation/feasibility)
      - final_decision
      - classification.domain/tags (as labels only)

    Does NOT read raw content fields.
    """
    analysis = smp.get("analysis") if isinstance(smp.get("analysis"), dict) else {}
    tri = smp.get("triadic_scores") if isinstance(smp.get("triadic_scores"), dict) else {}
    violations = smp.get("violations") if isinstance(smp.get("violations"), list) else []
    final_decision = str(smp.get("final_decision", "")).lower().strip()

    try:
        sev = float(analysis.get("severity_score", 0.0))
    except Exception:
        sev = 0.0

    coh = float(tri.get("coherence", 0.0)) if isinstance(tri.get("coherence", 0.0), (int, float)) else 0.0
    con = float(tri.get("conservation", 0.0)) if isinstance(tri.get("conservation", 0.0), (int, float)) else 0.0
    feas = float(tri.get("feasibility", 0.0)) if isinstance(tri.get("feasibility", 0.0), (int, float)) else 0.0

    signals: List[RiskSignal] = []

    base = _clamp(sev)
    if sev >= 0.85:
        signals.append(RiskSignal(name="high_severity", weight=0.35, notes="Upstream severity_score>=0.85"))

    vset = {str(v).lower() for v in violations}
    if "bridge_violation" in vset:
        signals.append(RiskSignal(name="bridge_violation", weight=0.25, notes="Bridge constraint failed upstream"))
    if "low_coherence" in vset or coh < 0.5:
        signals.append(RiskSignal(name="low_coherence", weight=0.20, notes="Low coherence / instability"))
    if "benevolence_check_failed" in vset:
        signals.append(RiskSignal(name="benevolence_failed", weight=0.20, notes="Benevolence gate did not pass"))

    if feas < 0.5:
        signals.append(RiskSignal(name="low_feasibility", weight=0.10, notes="Low feasibility; defer to Logos for handling"))

    extra = sum(s.weight for s in signals)
    overall = _clamp(base * 0.6 + extra)

    tri_present = any(isinstance(tri.get(k), (int, float)) for k in ("coherence", "conservation", "feasibility"))
    conf = 0.65 if tri_present else 0.45
    conf = _clamp(conf - (0.1 if not violations else 0.0) + (0.05 if sev > 0 else 0.0))

    if overall >= 0.85:
        cat = "critical"
    elif overall >= 0.65:
        cat = "high"
    elif overall >= 0.35:
        cat = "medium"
    else:
        cat = "low"

    if cat in {"critical", "high"} or final_decision in {"escalate", "quarantine"}:
        route = "LOGOS"
        action = "escalate" if cat == "critical" else "review"
        rationale = "High/critical risk signals present; route to Logos for centralized handling."
    else:
        route = "LOGOS"
        action = "monitor" if cat == "medium" else "review"
        rationale = "No critical signals; maintain centralized oversight."

    return TrajectoryEstimate(
        overall_risk=overall,
        confidence=conf,
        category=cat,
        signals=signals,
        recommended_route=route,
        recommended_action=action,
        rationale=rationale,
        meta={
            "severity_score": sev,
            "triadic_scores": {"coherence": coh, "conservation": con, "feasibility": feas},
            "final_decision": final_decision,
            "violation_count": len(violations),
        },
    )
