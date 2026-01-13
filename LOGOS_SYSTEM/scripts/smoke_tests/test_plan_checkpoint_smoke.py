#!/usr/bin/env python3
"""Smoke test for plan checkpoint execution."""

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

    # Run start_agent.py in stub Logos_AGI mode with objective "status" and budget 2 iterations
    cmd_first = [
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
        "2",
        "--max-plan-steps-per-run",
        "1",
    ]

    cmd_second = [
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
        "2",
    ]

    print("Running agent with plan execution...")
    result = subprocess.run(
        cmd_first, capture_output=True, text=True, cwd=repo_root, timeout=30
    )

    if result.returncode != 0:
        print(f"Agent failed: stdout={result.stdout} stderr={result.stderr}")
        return False

    output = result.stdout + result.stderr
    print("Agent output:")
    print(output)

    # Assert scp_state.json contains plans.active with one plan
    if not scp_state_path.exists():
        print("FAIL: SCP state not persisted")
        return False

    with open(scp_state_path) as f:
        scp_state = json.load(f)

    plans = scp_state.get("plans", {})
    active_plans = plans.get("active", [])
    if not active_plans:
        print("FAIL: No active plans")
        return False

    plan = active_plans[0]
    if plan.get("objective_class") != "STATUS":
        print(
            f"FAIL: Plan objective_class is {plan.get('objective_class')}, expected STATUS"
        )
        return False

    print("PASS: Active plan created")

    # Assert at least 1 step marked DONE or DENIED
    steps = plan.get("steps", [])
    completed_steps = [s for s in steps if s.get("status") in ["DONE", "DENIED"]]
    if not completed_steps:
        print("FAIL: No steps completed")
        return False

    print(f"PASS: {len(completed_steps)} steps completed")

    # Assert at least 1 checkpoint exists
    checkpoints = plan.get("checkpoints", [])
    if not checkpoints:
        print("FAIL: No checkpoints")
        return False

    print(f"PASS: {len(checkpoints)} checkpoints recorded")

    # Run again and assert resumption
    result2 = subprocess.run(
        cmd_second, capture_output=True, text=True, cwd=repo_root, timeout=30
    )
    if result2.returncode != 0:
        print(f"Second run failed: stdout={result2.stdout} stderr={result2.stderr}")
        return False

    with open(scp_state_path) as f:
        scp_state2 = json.load(f)

    plans2 = scp_state2.get("plans", {})
    active_plans2 = plans2.get("active", [])
    history_plans2 = plans2.get("history", [])

    # Check if plan resumed or completed
    resumed = any(p.get("plan_id") == plan["plan_id"] for p in active_plans2)
    completed = any(p.get("plan_id") == plan["plan_id"] for p in history_plans2)

    if not (resumed or completed):
        print("FAIL: Plan not resumed or completed")
        return False

    print(f"PASS: Plan {'resumed' if resumed else 'completed'}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
