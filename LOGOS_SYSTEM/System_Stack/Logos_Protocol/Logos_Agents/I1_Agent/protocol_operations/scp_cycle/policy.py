from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SCPPolicyDecision:
    run_loop: bool
    reason: str


def decide_policy(*, smp: Dict[str, Any]) -> SCPPolicyDecision:
    """
    Minimal policy for whether SCP should run the stabilization loop.
    This is NOT moral/policy enforcement; it's a compute routing decision.
    """
    fd = str(smp.get("final_decision", "")).lower().strip()
    if fd in {"escalate", "quarantine"}:
        return SCPPolicyDecision(run_loop=True, reason=f"final_decision={fd}")
    analysis = smp.get("analysis")
    sev = 0.0
    if isinstance(analysis, dict):
        try:
            sev = float(analysis.get("severity_score", 0.0))
        except Exception:
            sev = 0.0
    if sev >= 0.85:
        return SCPPolicyDecision(run_loop=True, reason="severity>=0.85")
    return SCPPolicyDecision(run_loop=False, reason="No trigger for SCP loop.")
