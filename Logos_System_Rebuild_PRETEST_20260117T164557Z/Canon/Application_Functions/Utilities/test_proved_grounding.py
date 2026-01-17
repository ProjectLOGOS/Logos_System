#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/test_proved_grounding.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Test PROVED truth grounding against theorem index."""

import os
import subprocess
import sys
from pathlib import Path


scripts_dir = Path(__file__).parent
repo_root = scripts_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from logos.proof_refs import load_theorem_index, enforce_truth_annotation


def main():
    state_dir = Path(os.getenv("LOGOS_STATE_DIR", repo_root / "state"))

    # Ensure theorem index exists
    index_path = state_dir / "coq_theorem_index.json"
    if not index_path.exists():
        print("Building theorem index...")
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "build_coq_theorem_index.py")],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        if result.returncode != 0:
            print(f"Failed to build index: {result.stderr}")
            return 1

    # Load index
    index = load_theorem_index(str(index_path))
    print(f"Loaded theorem index with {len(index['theorems'])} theorems")

    # Get pxl_excluded_middle entry
    lem_entry = None
    for th in index["theorems"]:
        if th["theorem"] == "pxl_excluded_middle":
            lem_entry = th
            break

    if not lem_entry:
        print("pxl_excluded_middle not found in index")
        return 1

    # Test A: Valid PROVED ref
    valid_ref = {
        "theorem": "pxl_excluded_middle",
        "file": "Protopraxis/formal_verification/coq/baseline/PXL_Internal_LEM.v",
        "statement_hash": lem_entry["statement_hash"],
        "index_hash": index["index_hash"],
    }

    valid_annotation = {
        "truth": "PROVED",
        "evidence": {"type": "coq", "ref": valid_ref, "details": "Test valid PROVED"},
    }

    enforced_valid = enforce_truth_annotation(valid_annotation, index)
    if enforced_valid["truth"] != "PROVED":
        print("FAIL: Valid PROVED was downgraded")
        return 1
    print("PASS: Valid PROVED remains PROVED")

    # Test B: Invalid PROVED ref (wrong statement_hash)
    invalid_ref = valid_ref.copy()
    invalid_ref["statement_hash"] = "invalid_hash"

    invalid_annotation = {
        "truth": "PROVED",
        "evidence": {
            "type": "coq",
            "ref": invalid_ref,
            "details": "Test invalid PROVED",
        },
    }

    enforced_invalid = enforce_truth_annotation(invalid_annotation, index)
    if enforced_invalid["truth"] == "PROVED":
        print("FAIL: Invalid PROVED was not downgraded")
        return 1
    if enforced_invalid["evidence"]["type"] != "hash":
        print("FAIL: Invalid PROVED evidence not updated")
        return 1
    print("PASS: Invalid PROVED downgraded to VERIFIED")

    # Test C: PROVED without coq evidence
    no_coq_annotation = {
        "truth": "PROVED",
        "evidence": {"type": "hash", "ref": None, "details": "Test no coq evidence"},
    }

    enforced_no_coq = enforce_truth_annotation(no_coq_annotation, index)
    if enforced_no_coq["truth"] == "PROVED":
        print("FAIL: PROVED without coq evidence not downgraded")
        return 1
    print("PASS: PROVED without coq evidence downgraded")

    # Test D: Run small agent invocation and check truth_events
    print("Running small agent invocation...")
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

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, timeout=30
    )
    if result.returncode != 0:
        print(f"Agent failed: {result.stderr}")
        return 1

    # Check if any truth_event has PROVED (should not, since no coq refs)
    output = result.stdout + result.stderr
    if '"truth": "PROVED"' in output:
        print("FAIL: Agent emitted PROVED without validation")
        return 1
    print("PASS: No invalid PROVED emissions in agent output")

    print("All tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
