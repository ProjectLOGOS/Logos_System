#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/repo_tools/run_mvp_acceptance.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Run the frozen MVP acceptance contract (Phase 12.1).

Produces exactly one terminal line:
- MVP_ACCEPTANCE: PASS
- MVP_ACCEPTANCE: FAIL (step N: <command>)

Optionally writes an audit artifact under audit/mvp_acceptance/.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_ROOT = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit"))
AUDIT_DIR = AUDIT_ROOT / "mvp_acceptance"

COMMANDS: List[List[str]] = [
    [sys.executable, "test_lem_discharge.py"],
    [sys.executable, "scripts/test_alignment_gate_smoke.py"],
    [sys.executable, "scripts/test_tool_pipeline_smoke.py"],
    [sys.executable, "scripts/test_uwm_smoke.py"],
    [sys.executable, "scripts/test_plan_checkpoint_smoke.py"],
    [sys.executable, "scripts/test_plan_validation_smoke.py"],
    [sys.executable, "scripts/test_plan_scoring_smoke.py"],
    [sys.executable, "scripts/test_plan_history_update_smoke.py"],
    [sys.executable, "scripts/test_belief_consolidation_smoke.py"],
    [sys.executable, "scripts/test_plan_revision_on_contradiction.py"],
    [sys.executable, "scripts/test_proved_grounding.py"],
    [sys.executable, "scripts/test_belief_tool_policy.py"],
    [sys.executable, "scripts/test_run_ledger_smoke.py"],
    [sys.executable, "scripts/test_goal_proposal_smoke.py"],
    [sys.executable, "scripts/test_tool_improvement_intent_smoke.py"],
]

TAIL_LINES = 80


def _tail(text: str, limit: int = TAIL_LINES) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-limit:]) if lines else ""


def _maybe_write_audit(records: List[Dict[str, object]], overall: str) -> None:
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "schema_version": 1,
            "timestamp": timestamp,
            "overall": overall,
            "results": records,
        }
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        record["hash"] = digest
        sanitized_timestamp = (
            timestamp.replace(":", "").replace("-", "").replace(".", "")
        )
        fname = f"mvp_{sanitized_timestamp}.json"
        with (AUDIT_DIR / fname).open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
    except (OSError, TypeError, ValueError):
        # Silent best-effort audit
        pass


def main() -> int:
    results: List[Dict[str, object]] = []
    for idx, cmd in enumerate(COMMANDS, start=1):
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        stdout_tail = _tail(proc.stdout)
        stderr_tail = _tail(proc.stderr)
        results.append(
            {
                "step": idx,
                "command": cmd,
                "returncode": proc.returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        )
        if proc.returncode != 0:
            if stdout_tail:
                print("--- STDOUT (last 80 lines) ---")
                print(stdout_tail)
            if stderr_tail:
                print("--- STDERR (last 80 lines) ---")
                print(stderr_tail)
            print(f"MVP_ACCEPTANCE: FAIL (step {idx}: {' '.join(cmd)})")
            _maybe_write_audit(results, "FAIL")
            return 1

    print("MVP_ACCEPTANCE: PASS")
    _maybe_write_audit(results, "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
