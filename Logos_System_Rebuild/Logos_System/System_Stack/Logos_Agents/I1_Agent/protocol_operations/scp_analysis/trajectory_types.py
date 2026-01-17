from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class RiskSignal:
    name: str
    weight: float
    notes: str = ""

@dataclass(frozen=True)
class TrajectoryEstimate:
    """
    Non-tactical risk/trajectory estimate. This is NOT a prediction of specific acts.
    It is a conservative outcome-risk summary used to decide routing and escalation.
    """
    overall_risk: float  # 0..1
    confidence: float    # 0..1 (confidence in the estimate, not in any act)
    category: str        # "low" | "medium" | "high" | "critical"
    signals: List[RiskSignal] = field(default_factory=list)
    recommended_route: str = "LOGOS"
    recommended_action: str = "review"  # review | escalate | quarantine | monitor
    rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_risk": self.overall_risk,
            "confidence": self.confidence,
            "category": self.category,
            "signals": [s.__dict__ for s in self.signals],
            "recommended_route": self.recommended_route,
            "recommended_action": self.recommended_action,
            "rationale": self.rationale,
            "meta": self.meta,
        }
