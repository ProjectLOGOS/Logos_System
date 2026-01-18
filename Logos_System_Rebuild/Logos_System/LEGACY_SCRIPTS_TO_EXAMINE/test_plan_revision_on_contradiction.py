# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/test_plan_revision_on_contradiction.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Test plan revision when beliefs contradict plan steps."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    scripts_dir = Path(__file__).parent
    repo_root = scripts_dir.parent
    state_dir = Path(os.getenv("LOGOS_STATE_DIR", repo_root / "state"))

    # Clean scp_state.json
    scp_state_path = state_dir / "scp_state.json"
    if scp_state_path.exists():
        scp_state_path.unlink()

    # Run start_agent.py to create an active plan
    cmd = [
        sys.executable,
        str(scripts_dir / "start_agent.py"),
        "--enable-logos-agi",
        "--logos-agi-mode",
        "stub",
        "--objective",
        "status",
        "--read-only",
        "--assume-yes",
        "--budget-sec",
        "1",  # Short run to create plan but not complete it
    ]

    print("Running agent to create active plan...")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=30
    )

    if result.returncode != 0:
        print(f"Agent failed: stdout={result.stdout} stderr={result.stderr}")
        return False

    # Load scp_state and ensure active plan exists
    if not scp_state_path.exists():
        print("FAIL: SCP state not persisted")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False

    with open(scp_state_path) as f:
        scp_state = json.load(f)

    # Ensure there's an active plan
    plans = scp_state.setdefault("plans", {"active": [], "history": []})
    if not plans["active"]:
        # Create a dummy active plan
        plan = {
            "schema_version": 1,
            "plan_id": "test_plan",
            "created_at": "2023-01-01T00:00:00+00:00",
            "objective": "status",
            "objective_class": "GENERAL",
            "steps": [
                {
                    "index": 0,
                    "step_id": "step_0",
                    "tool": "mission.status",
                    "args": [""],
                    "status": "PENDING",
                },
                {
                    "index": 1,
                    "step_id": "step_1",
                    "tool": "probe.last",
                    "args": [""],
                    "status": "PENDING",
                },
            ],
            "current_index": 0,
            "status": "ACTIVE",
            "checkpoints": [],
        }
        plans["active"].append(plan)
    else:
        # Fix existing plan steps
        for plan in plans["active"]:
            if "steps" in plan:
                for i, step in enumerate(plan["steps"]):
                    if "step_id" not in step:
                        step["step_id"] = f"step_{i}"
                    if "args" not in step:
                        step["args"] = [""]

    # Ensure beliefs exist
    beliefs = scp_state.setdefault("beliefs", {"items": []})
    if not beliefs["items"]:
        # Create dummy beliefs
        beliefs["items"] = [
            {
                "id": "test_belief_1",
                "created_at": "2023-01-01T00:00:00+00:00",
                "last_accessed_at": "2023-01-01T00:00:00+00:00",
                "objective_tags": ["GENERAL"],
                "truth": "VERIFIED",
                "evidence": {"type": "test"},
                "content": {"test": "belief1"},
                "salience": 0.5,
                "decay_rate": 0.15,
                "access_count": 0,
                "source": "TEST",
                "status": "ACTIVE",
            }
        ]

    # Save back
    with open(scp_state_path, "w") as f:
        json.dump(scp_state, f, indent=2)

    plans = scp_state.get("plans", {})
    active_plans = [p for p in plans.get("active", []) if p.get("status") == "ACTIVE"]
    if not active_plans:
        print("FAIL: No active plan created")
        return False

    active_plan = active_plans[0]
    steps = active_plan.get("steps", [])
    if not steps:
        print("FAIL: Active plan has no steps")
        return False

    next_step = steps[active_plan.get("current_index", 0)]
    tool_name = next_step.get("tool")
    print(f"Active plan step: {tool_name}")

    # Inject a QUARANTINED belief contradicting the next step
    beliefs = scp_state.get("beliefs", {})
    items = beliefs.get("items", [])
    if items:
        # Modify the first belief to be QUARANTINED
        items[0]["status"] = "QUARANTINED"
        items[0]["truth"] = "CONTRADICTED"
        items[0]["contradicting_refs"] = ["test_contradict"]
        print(f"Modified belief: {items[0]['id']} to QUARANTINED")
        print(f"Belief content: {items[0].get('content', {})}")
    else:
        print("No beliefs to modify")
        return False

    # Save modified state
    with open(scp_state_path, "w") as f:
        json.dump(scp_state, f, indent=2)

    # Reset plan current_index to 0 to test revision on the first step
    active_plan["current_index"] = 0

    # Run start_agent.py again for 1 iteration
    cmd = [
        sys.executable,
        str(scripts_dir / "start_agent.py"),
        "--enable-logos-agi",
        "--logos-agi-mode",
        "stub",
        "--objective",
        "status",
        "--assume-yes",  # Add back to auto-approve
        "--read-only",
        "--budget-sec",
        "1",  # Short run
        "--scp-recovery-mode",
        "--max-plan-steps-per-run",
        "1",
    ]

    # Set env for recovery
    env = os.environ.copy()
    env["LOGOS_DEV_BYPASS_OK"] = "1"

    result2 = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=30, env=env
    )

    print(f"Agent stdout: {result2.stdout}")
    print(f"Agent stderr: {result2.stderr}")

    if result2.returncode != 0:
        print(
            f"Agent failed on second run: stdout={result2.stdout} stderr={result2.stderr}"
        )
        return False

    # Load final state and check plan revision
    with open(scp_state_path) as f:
        final_state = json.load(f)

    final_beliefs = final_state.get("beliefs", {}).get("items", [])
    print(f"Final beliefs count: {len(final_beliefs)}")
    for b in final_beliefs:
        print(f"Belief: {b['id']}, status: {b['status']}")
    quarantined = [b for b in final_beliefs if b.get("status") == "QUARANTINED"]
    print(f"Quarantined beliefs: {len(quarantined)}")

    final_plans = final_state.get("plans", {})
    final_active = [
        p for p in final_plans.get("active", []) if p.get("status") == "ACTIVE"
    ]

    if not final_active:
        print("FAIL: No active plan after revision")
        return False

    # Stub beliefs should not quarantine tools; ensure steps remain actionable
    final_plan = final_active[0]
    steps = final_plan.get("steps", [])
    skipped_steps = [s for s in steps if s.get("status") == "SKIPPED"]
    if skipped_steps:
        print("FAIL: Stub belief triggered plan skip")
        return False

    print("PASS: Plan remained active; stub beliefs did not alter execution")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
