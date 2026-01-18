# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Trinitarian mediator for I2 arbitration across subsystem assessments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import time


@dataclass(frozen=True)
class TrinitarianDecision:
    final_decision: str
    rationale: str
    score_vector: Dict[str, float]
    violations: List[str]
    bridge_passed: bool
    final_payload: Any
    provenance: Dict[str, Any]
    timestamp: float


class TrinitarianMediator:
    """
    Applies a Trinitarian Optimization Protocol to arbitrate between subsystem
    assessments (classifier/analyst/transformer/benevolence/bridge) and issue a
    final routing decision.
    """

    @staticmethod
    def arbitrate(
        *,
        triadic_scores: Dict[str, float],
        bridge_passed: bool,
        benevolence_result: Dict[str, Any],
        privation_metadata: Dict[str, Any],
        transform_metadata: Dict[str, Any],
        original_payload: Any,
    ) -> TrinitarianDecision:
        now = time.time()
        rationale_lines: List[str] = []
        violations: List[str] = []

        coh = triadic_scores.get("coherence", 0.0)
        con = triadic_scores.get("conservation", 0.0)
        feas = triadic_scores.get("feasibility", 0.0)
        bene = benevolence_result.get("score", 0.0)
        bene_passed = benevolence_result.get("benevolence_passed", False)

        decision = "annotate_only"

        if coh < 0.4 and con < 0.4:
            decision = "quarantine"
            rationale_lines.append("Coherence and conservation both critically low.")
        elif not bridge_passed and feas < 0.5:
            decision = "escalate"
            rationale_lines.append("Bridge constraint failed and feasibility too low to salvage.")
        elif coh < 0.5 and con > 0.6 and bene_passed:
            decision = "annotate_only"
            rationale_lines.append("Preserved value and benevolence despite incoherence.")
        elif coh >= 0.6 and con >= 0.6 and feas >= 0.6 and bene_passed:
            decision = "allow"
            rationale_lines.append("All triadic scores within functional range, benevolence satisfied.")
        elif transform_metadata.get("mode") == "decompose" and not bridge_passed:
            decision = "decompose_only"
            rationale_lines.append("Bridge failed but decomposition preserved signal.")
        else:
            decision = "annotate_only"
            rationale_lines.append("Defaulted to annotation due to score uncertainty.")

        if coh < 0.5:
            violations.append("low_coherence")
        if con < 0.4:
            violations.append("loss_of_meaning")
        if not bridge_passed:
            violations.append("bridge_violation")
        if not bene_passed:
            violations.append("benevolence_check_failed")

        return TrinitarianDecision(
            final_decision=decision,
            rationale=" ".join(rationale_lines),
            score_vector={
                "coherence": round(coh, 2),
                "conservation": round(con, 2),
                "feasibility": round(feas, 2),
                "benevolence": round(bene, 2),
            },
            violations=violations,
            bridge_passed=bridge_passed,
            final_payload=transform_metadata.get("output") or original_payload,
            provenance={
                "classified_as": privation_metadata.get("tags", []),
                "domain": privation_metadata.get("domain", "unknown"),
                "severity": privation_metadata.get("severity", 0.0),
                "mediated_by": "TrinitarianMediator.v1",
                "source_modules": [
                    "privation_classifier",
                    "analyst",
                    "transformer",
                    "bridge",
                    "benevolence",
                ],
            },
            timestamp=now,
        )
