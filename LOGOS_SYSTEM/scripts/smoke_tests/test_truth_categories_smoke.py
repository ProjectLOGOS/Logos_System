#!/usr/bin/env python3
"""Smoke test for truth categories emission and persistence."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    scripts_dir = Path(__file__).parent
    repo_root = scripts_dir.parent
    state_dir = Path(os.getenv("LOGOS_STATE_DIR", repo_root / "state"))

    # Run agent with Logos_AGI enabled, read-only, short budget
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
        "1",
    ]

    print("Running agent with Logos_AGI...")
    print(f"Cmd: {cmd}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=10
    )

    if result.returncode != 0:
        print(f"Agent failed: stdout={result.stdout} stderr={result.stderr}")
        return False

    output = result.stdout + result.stderr
    print("Agent output:")
    print(output)

    # Check persisted SCP state
    scp_state_path = state_dir / "scp_state.json"
    if not scp_state_path.exists():
        print("FAIL: SCP state not persisted")
        return False

    with open(scp_state_path) as f:
        scp_state = json.load(f)

    if "truth_events" not in scp_state or not scp_state["truth_events"]:
        print("FAIL: No truth_events in SCP state")
        return False

    if "working_memory" not in scp_state:
        print("FAIL: No working_memory in SCP state")
        return False

    wm = scp_state["working_memory"]
    if "short_term" not in wm or not isinstance(wm["short_term"], list):
        print("FAIL: working_memory missing short_term list")
        return False

    print("PASS: Working memory initialized")

    # Check at least one event has truth_annotation
    has_truth = False
    for event in scp_state["truth_events"]:
        if "truth_annotation" in event:
            has_truth = True
            break

    if not has_truth:
        print("FAIL: No truth_annotation in truth_events")
        return False

    print("PASS: Truth categories emitted and persisted")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
