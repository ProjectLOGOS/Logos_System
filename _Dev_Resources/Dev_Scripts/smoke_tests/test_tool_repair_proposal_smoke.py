#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_tool_repair_proposal_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for UIP-gated tool repair proposal generation."""

import json
import subprocess
import sys
from pathlib import Path


def main() -> bool:
    repo_root = Path(__file__).parent.parent
    health_report = {
        "timestamp": "2025-01-01T00:00:00Z",
        "overall_health": "BROKEN",
        "tools": {
            "demo_tool": {
                "health": "BROKEN",
                "stats": {"total": 3, "errors": 2, "denies": 0, "successes": 1},
                "issues": ["import failure"],
                "explanation": "import failure",
                "evidence_refs": ["ledger_hash_123"],
            }
        },
    }
    report_path = repo_root / "sandbox" / "health_report_demo.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(health_report), encoding="utf-8")

    output_dir = repo_root / "sandbox" / "tool_repair_proposals_smoke"
    if output_dir.exists():
        for child in output_dir.iterdir():
            try:
                child.unlink()
            except OSError:
                pass
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "tool_repair_proposal.py"),
        "--health-report",
        str(report_path),
        "--uip-decision",
        "approve",
        "--operator",
        "smoke_test",
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        print(f"FAIL: proposal script errored: {result.stderr}")
        return False

    # Ensure proposal file exists
    proposals = list(output_dir.glob("repair_*.json"))
    if not proposals:
        print("FAIL: no proposals generated")
        return False

    proposal = json.loads(proposals[0].read_text())
    if proposal.get("tool_name") != "demo_tool":
        print("FAIL: proposal tool mismatch")
        return False
    if not proposal.get("governance", {}).get("requires_uip"):
        print("FAIL: governance gating missing")
        return False
    if "No execution" not in proposal.get("description", ""):
        print("FAIL: missing non-execution notice")
        return False

    print("PASS: repair proposal generated under UIP gate")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
