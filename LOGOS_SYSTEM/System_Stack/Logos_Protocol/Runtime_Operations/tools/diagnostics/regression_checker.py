#!/usr/bin/env python3
"""
Regression Checker
==================

Derive Cycle Regression Checker to satisfy tool optimizer gap regression_checker

Compares baseline and candidate tool outputs for drift detection.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict


class RegressionChecker:
    """Detect regressions between tool output snapshots."""

    def compare(
        self,
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
        tolerance: float = 0.0,
    ) -> Dict[str, Any]:
        deltas: Dict[str, Any] = {}
        keys = set(baseline.keys()) | set(candidate.keys())
        for key in sorted(keys):
            base_val = baseline.get(key)
            cand_val = candidate.get(key)
            if base_val == cand_val:
                continue
            if isinstance(base_val, (int, float)) and isinstance(cand_val, (int, float)):
                diff = cand_val - base_val
                if abs(diff) <= tolerance:
                    continue
                deltas[key] = {"baseline": base_val, "candidate": cand_val, "delta": diff}
            else:
                deltas[key] = {"baseline": base_val, "candidate": cand_val}
        status = "ok" if not deltas else "regression_detected"
        return {
            "status": status,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "deltas": deltas,
        }

    def report(
        self,
        baseline_label: str,
        candidate_label: str,
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
        tolerance: float = 0.0,
    ) -> Dict[str, Any]:
        comparison = self.compare(baseline, candidate, tolerance=tolerance)
        comparison.update(
            {
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
            }
        )
        return comparison


CHECKER = RegressionChecker()


def compare_tool_outputs(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Compare two tool outputs and surface deltas."""

    return CHECKER.compare(baseline, candidate, tolerance=tolerance)


if __name__ == "__main__":
    result = compare_tool_outputs({"value": 1}, {"value": 2})
    print(json.dumps(result))
