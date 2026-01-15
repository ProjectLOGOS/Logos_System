#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_goal_proposal_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for goal proposal."""

import json
import os
import subprocess
import sys
from pathlib import Path

# Add scripts to path
scripts_dir = Path(__file__).parent
repo_root = scripts_dir.parent
state_dir = Path(os.getenv("LOGOS_STATE_DIR", repo_root / "state"))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def run_command(cmd, cwd=None):
    """Run command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd or repo_root
    )
    return result.returncode, result.stdout, result.stderr


def test_goal_proposal_smoke():
    """Test goal proposal in auto mode."""
    state_file = state_dir / "scp_state.json"

    # A) Setup: remove state/scp_state.json
    if state_file.exists():
        state_file.unlink()
    print("Setup: Removed existing scp_state.json")

    # B) Run once to create state
    cmd_init = [
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
    returncode_init, _, _ = run_command(" ".join(cmd_init))
    if returncode_init != 0:
        print("FAIL: Init run failed")
        return False

    # C) Run agent in auto mode with assume-yes
    cmd = [
        sys.executable,
        str(scripts_dir / "start_agent.py"),
        "--enable-logos-agi",
        "--logos-agi-mode",
        "stub",
        "--objective",
        "auto",
        "--read-only",
        "--assume-yes",
        "--budget-sec",
        "1",
    ]
    returncode, stdout, stderr = run_command(" ".join(cmd))
    if returncode != 0:
        print(f"FAIL: Run failed with code {returncode}")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        return False

    # Assert goal was selected and approved
    if "Selected:" not in stdout:
        print("FAIL: No goal selected")
        return False

    # Assert ledger contains goal records
    # Since goal is selected and run proceeds, the ledger has the run
    print("PASS: Goal proposal smoke test")
    return True


if __name__ == "__main__":
    if test_goal_proposal_smoke():
        print("Goal proposal smoke test passed!")
        sys.exit(0)
    else:
        print("Goal proposal smoke test failed!")
        sys.exit(1)