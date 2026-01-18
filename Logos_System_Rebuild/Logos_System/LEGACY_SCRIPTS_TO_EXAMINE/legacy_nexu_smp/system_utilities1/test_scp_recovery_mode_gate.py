# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""Test SCP recovery mode gating."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    scripts_dir = Path(__file__).parent
    repo_root = scripts_dir.parent
    state_dir = Path(os.getenv("LOGOS_STATE_DIR", repo_root / "state"))
    scp_state_path = state_dir / "scp_state.json"
    logos_agi_state_path = state_dir / "logos_agi_scp_state.json"

    # Clean scp_state.json
    if scp_state_path.exists():
        scp_state_path.unlink()

    # Create a valid scp_state.json first
    cmd = [
        sys.executable,
        str(scripts_dir / "start_agent.py"),
        "--enable-logos-agi",
        "--logos-agi-mode",
        "stub",
        "--objective",
        "status",
        "--read-only",
        "--budget-sec",
        "1",
    ]

    print("Creating valid SCP state...")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=30
    )
    if result.returncode != 0:
        print(f"Failed to create state: {result.stderr}")
        return False

    if not scp_state_path.exists():
        print("FAIL: SCP state not created")
        return False

    # Load and corrupt the state_hash to make it invalid
    with open(scp_state_path) as f:
        state = json.load(f)
    state["state_hash"] = "invalid_hash"

    with open(scp_state_path, "w") as f:
        json.dump(state, f, indent=2)

    if logos_agi_state_path.exists():
        with open(logos_agi_state_path) as f:
            logos_state = json.load(f)
        logos_state["state_hash"] = "invalid_hash"
        with open(logos_agi_state_path, "w") as f:
            json.dump(logos_state, f, indent=2)

    print("Corrupted state_hash to simulate invalid state")

    # Test 1: Run without recovery mode -> should not load state
    print("Test 1: Running without --scp-recovery-mode")
    result1 = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=30
    )
    if result1.returncode != 0:
        print(f"Unexpected failure: {result1.stderr}")
        return False

    # Check audit log for scp_state_valid=false
    output = result1.stdout + result1.stderr
    if '"scp_state_valid": false' not in output:
        print("FAIL: scp_state_valid should be false without recovery mode")
        return False
    print("PASS: scp_state_valid=false without recovery mode")

    # Test 2: Run with --scp-recovery-mode but without LOGOS_DEV_BYPASS_OK -> should fail
    print("Test 2: Running with --scp-recovery-mode but without LOGOS_DEV_BYPASS_OK")
    cmd_recovery = cmd + ["--scp-recovery-mode"]
    result2 = subprocess.run(
        cmd_recovery, capture_output=True, text=True, cwd=repo_root, timeout=30
    )
    if result2.returncode == 0:
        print("FAIL: Should have failed without LOGOS_DEV_BYPASS_OK")
        return False
    if (
        "ERROR: --scp-recovery-mode requires LOGOS_DEV_BYPASS_OK=1"
        not in result2.stdout
    ):
        print(f"FAIL: Wrong error message: {result2.stdout}")
        return False
    print("PASS: Correctly failed without LOGOS_DEV_BYPASS_OK")

    # Re-corrupt state before recovery test
    with open(scp_state_path) as f:
        state = json.load(f)
    state["state_hash"] = "invalid_hash"
    with open(scp_state_path, "w") as f:
        json.dump(state, f, indent=2)

    if logos_agi_state_path.exists():
        with open(logos_agi_state_path) as f:
            logos_state = json.load(f)
        logos_state["state_hash"] = "invalid_hash"
        with open(logos_agi_state_path, "w") as f:
            json.dump(logos_state, f, indent=2)

    # Test 3: Run with LOGOS_DEV_BYPASS_OK=1 and --scp-recovery-mode -> should load and mark NON-PRODUCTION
    print("Test 3: Running with LOGOS_DEV_BYPASS_OK=1 and --scp-recovery-mode")
    env = os.environ.copy()
    env["LOGOS_DEV_BYPASS_OK"] = "1"
    result3 = subprocess.run(
        cmd_recovery, capture_output=True, text=True, cwd=repo_root, timeout=30, env=env
    )
    if result3.returncode != 0:
        print(f"Unexpected failure: {result3.stderr}")
        return False

    output3 = result3.stdout + result3.stderr
    if (
        '"scp_state_valid": false' not in output3
        or '"scp_recovery_mode": true' not in output3
    ):
        print("FAIL: Should have scp_state_valid=false and scp_recovery_mode=true")
        return False
    print("PASS: Loaded invalid state in recovery mode")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
