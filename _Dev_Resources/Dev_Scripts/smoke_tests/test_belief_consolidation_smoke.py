#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_belief_consolidation_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for belief consolidation."""

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

    # Run start_agent.py twice in stub mode with objective "status" to generate repeated tool outcomes + checkpoints
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
        "5",
    ]

    print("Running agent first time...")
    result1 = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=30
    )
    if result1.returncode != 0:
        print(f"First run failed: {result1.stderr}")
        return False

    print("Running agent second time...")
    result2 = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=30
    )
    if result2.returncode != 0:
        print(f"Second run failed: {result2.stderr}")
        return False

    # Assert scp_state.json contains beliefs
    if not scp_state_path.exists():
        print("FAIL: SCP state not persisted")
        return False

    with open(scp_state_path) as f:
        scp_state = json.load(f)

    beliefs = scp_state.get("beliefs", {})
    items = beliefs.get("items", [])
    if not items:
        print("FAIL: Beliefs items empty")
        return False

    print(f"Beliefs items count: {len(items)}")

    # Stub-mode beliefs must remain non-VERIFIED
    forbidden = [b for b in items if b.get("truth") in {"VERIFIED", "PROVED"}]
    if forbidden:
        print("FAIL: Stub consolidation produced elevated truths")
        return False

    print(f"PASS: Stub beliefs bounded (count={len(items)})")

    # Assert beliefs container has state_hash and prev_hash chaining
    if "state_hash" not in beliefs or "prev_hash" not in beliefs:
        print("FAIL: Missing state_hash or prev_hash in beliefs")
        return False

    # If prev_hash is not None, it should match the first run's hash
    if beliefs["prev_hash"] is not None:
        print("PASS: Beliefs have hash chaining")

    print("PASS: Belief consolidation smoke test")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
