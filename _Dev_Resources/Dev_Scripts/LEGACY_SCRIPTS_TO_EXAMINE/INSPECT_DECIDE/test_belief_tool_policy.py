#!/usr/bin/env python3
"""Test belief-informed tool selection policy."""

import sys
from pathlib import Path

# Add logos to path
sys.path.insert(0, str(Path(__file__).parent.parent / "logos"))

from policy import apply_belief_policy
from evaluator import choose_best


def test_belief_policy():
    # Create synthetic beliefs
    beliefs_container = {
        "items": [
            {
                "id": "belief_active_probe",
                "status": "ACTIVE",
                "truth": "VERIFIED",
                "confidence": 0.9,
                "objective_tags": ["GENERAL"],
                "content": {"preferred_tool": "probe.last"},
            },
            {
                "id": "belief_quarantine_mission",
                "status": "QUARANTINED",
                "truth": "CONTRADICTED",
                "objective_tags": ["GENERAL"],
                "content": {"contradicted_tools": ["mission.status"]},
            },
        ]
    }

    # Create proposals with equal metrics
    proposals = [
        {
            "tool": "mission.status",
            "args": "",
            "rationale": "Check status",
            "confidence": 0.8,
            "truth_annotation": {"truth": "HEURISTIC"},
        },
        {
            "tool": "probe.last",
            "args": "",
            "rationale": "Probe last",
            "confidence": 0.8,
            "truth_annotation": {"truth": "HEURISTIC"},
        },
    ]

    # Mock metrics state
    metrics_state = {
        "metrics": {
            "GENERAL": {
                "mission.status": {
                    "attempts": 1,
                    "successes": 1,
                    "denies": 0,
                    "errors": 0,
                    "last_outcome": "SUCCESS",
                    "last_updated_at": "",
                },
                "probe.last": {
                    "attempts": 1,
                    "successes": 1,
                    "denies": 0,
                    "errors": 0,
                    "last_outcome": "SUCCESS",
                    "last_updated_at": "",
                },
            }
        }
    }

    # Apply policy
    filtered_proposals, policy_notes = apply_belief_policy(
        proposals, "GENERAL", beliefs_container
    )

    print(f"Original proposals: {len(proposals)}")
    print(f"Filtered proposals: {len(filtered_proposals)}")
    print(f"Policy notes: {policy_notes}")

    # Assert mission.status is filtered
    tool_names = [p["tool"] for p in filtered_proposals]
    assert "mission.status" not in tool_names, (
        "mission.status should be filtered due to quarantine"
    )
    assert "probe.last" in tool_names, "probe.last should remain"

    # Apply evaluator
    ranked = choose_best(filtered_proposals, "GENERAL", metrics_state)

    print(f"Ranked proposals: {[p['tool'] for p in ranked]}")
    print(f"First proposal score: {ranked[0]['evaluator_score']}")
    print(f"Policy adjustment: {ranked[0].get('policy_adjustment', 0)}")

    # Assert probe.last is first due to boost
    assert ranked[0]["tool"] == "probe.last", (
        "probe.last should be ranked first due to boost"
    )
    assert ranked[0]["policy_adjustment"] == 0.15, (
        "probe.last should have +0.15 policy adjustment"
    )

    print("All policy tests passed")


if __name__ == "__main__":
    test_belief_policy()
