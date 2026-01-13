#!/usr/bin/env python3
"""Test belief quarantine persistence across consolidation cycles."""

import sys
from pathlib import Path

# Add logos to path
sys.path.insert(0, str(Path(__file__).parent.parent / "logos"))

from beliefs import consolidate_beliefs, init_beliefs_container


def main():
    # Create a mock SCP state with a QUARANTINED belief
    scp_state = {
        "schema_version": 1,
        "working_memory": {"short_term": [], "long_term": []},
        "plans": {"active": [], "history": []},
        "truth_events": [],
    }
    init_beliefs_container(scp_state)

    # Manually add a QUARANTINED belief
    belief = {
        "belief_id": "test_belief_123",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "objective_tags": ["GENERAL"],
        "content": {"tool": "test.tool", "outcome": "SUCCESS"},
        "truth": "CONTRADICTED",
        "confidence": 0.5,
        "supporting_refs": [],
        "contradicting_refs": ["contradict_ref"],
        "status": "QUARANTINED",
        "notes": {"test": True},
    }
    scp_state["beliefs"]["items"].append(belief)

    original_belief_id = belief["belief_id"]
    original_status = belief["status"]

    print(f"Initial belief: {original_belief_id}, status: {original_status}")

    # First consolidation (no new evidence)
    consolidated1 = consolidate_beliefs(scp_state, run_id="run1")
    scp_state["beliefs"] = consolidated1

    beliefs1 = consolidated1["items"]
    if not beliefs1:
        print("FAIL: No beliefs after first consolidation")
        return False

    belief1 = beliefs1[0]
    print(
        f"After first consolidation: {belief1['belief_id']}, status: {belief1['status']}"
    )

    if belief1["belief_id"] != original_belief_id:
        print("FAIL: Belief ID changed")
        return False

    if belief1["status"] != "QUARANTINED":
        print("FAIL: Status not preserved in first consolidation")
        return False

    # Second consolidation (no new evidence)
    consolidated2 = consolidate_beliefs(scp_state, run_id="run2")
    scp_state["beliefs"] = consolidated2

    beliefs2 = consolidated2["items"]
    if not beliefs2:
        print("FAIL: No beliefs after second consolidation")
        return False

    belief2 = beliefs2[0]
    print(
        f"After second consolidation: {belief2['belief_id']}, status: {belief2['status']}"
    )

    if belief2["belief_id"] != original_belief_id:
        print("FAIL: Belief ID changed in second consolidation")
        return False

    if belief2["status"] != "QUARANTINED":
        print("FAIL: Status not preserved in second consolidation")
        return False

    print("PASS: Quarantine persistence works")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
