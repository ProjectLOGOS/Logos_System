#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_tool_introspection_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for deterministic tool introspection."""

import json
import subprocess
import sys
from pathlib import Path


def main() -> bool:
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "tool_introspection.py")],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"FAIL: introspection script errored: {result.stderr}")
        return False

    try:
        records = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"FAIL: could not parse introspection output: {exc}")
        return False

    if not isinstance(records, list) or not records:
        print("FAIL: introspection returned no records")
        return False

    mission_records = [r for r in records if r.get("tool_name") == "mission.status"]
    if not mission_records:
        print("FAIL: mission.status capability missing")
        return False

    required_fields = {
        "tool_name",
        "objective_classes",
        "input_shape",
        "output_shape",
        "side_effects",
        "risk_level",
        "truth_dependencies",
        "introduced_by",
        "approval_hash",
        "last_used",
        "success_rate",
    }
    for record in records:
        missing = [f for f in required_fields if f not in record]
        if missing:
            print(f"FAIL: missing fields {missing} in {record.get('tool_name')}")
            return False
        if not (0.0 <= float(record.get("success_rate", 0.0)) <= 1.0):
            print("FAIL: success_rate out of bounds")
            return False
        intro = record.get("introduced_by")
        if intro not in {"builtin", "pipeline"}:
            print("FAIL: invalid introduced_by")
            return False

    print("PASS: tool introspection deterministic and complete")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
