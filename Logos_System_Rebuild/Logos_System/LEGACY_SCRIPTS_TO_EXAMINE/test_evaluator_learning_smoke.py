# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_evaluator_learning_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for evaluator learning: demonstrates deterministic improvement in proposal selection."""

from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For logos module

from logos.evaluator import load_metrics, update_metrics, choose_best


def test_learning():
    """Test that evaluator learns from outcomes."""
    print("=== Evaluator Learning Test ===")

    # Create temp metrics
    metrics = load_metrics(Path("/tmp/nonexistent.json"))  # Gets default

    # Initial proposals
    proposals = [
        {
            "tool": "mission.status",
            "args": "",
            "rationale": "Status check",
            "confidence": 0.9,
        },
        {
            "tool": "probe.last",
            "args": "",
            "rationale": "Probe recent",
            "confidence": 0.8,
        },
    ]

    # Choose best initially - should prefer mission.status (higher confidence, no history)
    best = choose_best(proposals, "STATUS", metrics)
    assert best[0]["tool"] == "mission.status", (
        f"Expected mission.status first, got {best[0]['tool']}"
    )
    print("✓ Initial selection: mission.status preferred")

    # Simulate deny for mission.status
    metrics = update_metrics(metrics, "STATUS", "mission.status", "DENY")
    print("✓ Simulated DENY for mission.status")

    # Choose best again - should prefer probe.last due to penalty
    best = choose_best(proposals, "STATUS", metrics)
    assert best[0]["tool"] == "probe.last", (
        f"Expected probe.last first after deny, got {best[0]['tool']}"
    )
    print("✓ After deny: probe.last preferred")

    # Simulate success for probe.last
    metrics = update_metrics(metrics, "STATUS", "probe.last", "SUCCESS")
    print("✓ Simulated SUCCESS for probe.last")

    # Choose best - should still prefer probe.last
    best = choose_best(proposals, "STATUS", metrics)
    assert best[0]["tool"] == "probe.last", (
        f"Expected probe.last first after success, got {best[0]['tool']}"
    )
    print("✓ After success: probe.last remains preferred")

    print("\n=== Test Passed: Evaluator demonstrates learning ===")


if __name__ == "__main__":
    test_learning()
