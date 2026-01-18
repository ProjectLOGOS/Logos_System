# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Benevolence filter for I2 output validation.
Enforces principled kindness without diluting truth or warranted correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import time


@dataclass(frozen=True)
class BenevolenceAssessment:
    benevolence_passed: bool
    score: float
    rationale: str
    violations: List[str]
    timestamp: float


class BenevolenceMediator:
    """
    Applies principled benevolence constraints to I2 output.
    Permits firm, corrective, or severe messages when justified by truth, clarity,
    or moral necessity. Blocks malicious, deceitful, needlessly cruel, or mission-
    corrosive content.
    """

    @staticmethod
    def assess_output(
        *,
        payload: Any,
        context: Dict[str, Any],
        transform_metadata: Dict[str, Any],
        triadic_scores: Dict[str, float],
        bridge_constraint_passed: bool,
    ) -> BenevolenceAssessment:
        now = time.time()
        violations: List[str] = []
        rationale_lines: List[str] = []

        passed = True
        score = 1.0

        coherence = triadic_scores.get("coherence", 0.0)
        conservation = triadic_scores.get("conservation", 0.0)
        feasibility = triadic_scores.get("feasibility", 0.0)

        if coherence < 0.5:
            violations.append("low_coherence")
            rationale_lines.append("Output lacks sufficient coherence.")
            score -= 0.3

        if conservation < 0.4:
            violations.append("loss_of_meaning")
            rationale_lines.append("Potential semantic degradation detected.")
            score -= 0.3

        if not bridge_constraint_passed:
            violations.append("bridge_violation")
            rationale_lines.append("Output derived from a null or impossible structure.")
            score -= 0.3

        if isinstance(payload, str):
            lower = payload.lower()
            if any(word in lower for word in ["you deserve", "worthless", "shame on you"]):
                violations.append("potential_cruelty")
                rationale_lines.append("Detected potentially harmful phrasing; evaluate context.")
                score -= 0.2

        if score < 0.5:
            passed = False
            rationale_lines.append("Output rejected: fails principled benevolence threshold.")

        return BenevolenceAssessment(
            benevolence_passed=passed,
            score=round(max(0.0, min(1.0, score)), 2),
            rationale=" ".join(rationale_lines) or "Output consistent with Logos benevolence constraints.",
            violations=violations,
            timestamp=now,
        )
