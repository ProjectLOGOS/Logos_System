#!/usr/bin/env python3
"""Test evaluator ranking with truth categories."""

import sys
from pathlib import Path

# Add paths
scripts_dir = Path(__file__).parent
repo_root = scripts_dir.parent
logos_dir = repo_root / "logos"
if str(logos_dir) not in sys.path:
    sys.path.insert(0, str(logos_dir))

from evaluator import choose_best


def main():
    # Mock metrics state
    metrics_state = {
        "metrics": {
            "GENERAL": {
                "mission.status": {
                    "attempts": 10,
                    "successes": 8,
                    "denies": 1,
                    "errors": 1,
                    "last_outcome": "SUCCESS",
                    "last_updated_at": "2023-01-01T00:00:00Z",
                }
            }
        }
    }

    # Two proposals with identical tool/metrics but different truth
    proposals = [
        {
            "tool": "mission.status",
            "args": "",
            "rationale": "Test proposal 1",
            "confidence": 0.8,
            "truth_annotation": {
                "truth": "HEURISTIC",
                "evidence": {
                    "type": "inference",
                    "ref": None,
                    "details": "Pattern match",
                },
            },
        },
        {
            "tool": "mission.status",
            "args": "",
            "rationale": "Test proposal 2",
            "confidence": 0.8,
            "truth_annotation": {
                "truth": "VERIFIED",
                "evidence": {
                    "type": "hash",
                    "ref": "abcd1234",
                    "details": "Validated state",
                },
            },
        },
    ]

    objective_class = "GENERAL"
    ranked = choose_best(proposals, objective_class, metrics_state)

    # Assert VERIFIED is ranked first
    if ranked[0]["truth_annotation"]["truth"] != "VERIFIED":
        print(
            f"FAIL: VERIFIED not ranked first. Ranked: {[p['truth_annotation']['truth'] for p in ranked]}"
        )
        return False

    # Assert evaluator_reason includes truth adjustment
    for proposal in ranked:
        if "Truth adjustment:" not in proposal["evaluator_reason"]:
            print(
                f"FAIL: evaluator_reason missing truth adjustment: {proposal['evaluator_reason']}"
            )
            return False

    print("PASS: Evaluator correctly ranks VERIFIED over HEURISTIC")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
