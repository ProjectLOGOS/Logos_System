#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_llm_advisor_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test: LLM advisor can propose a safe low-impact tool in stub mode."""

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))
START_AGENT = REPO_ROOT / "scripts" / "start_agent.py"
SCP_STATE = STATE_DIR / "scp_state.json"


def _run_with_payload(payload: dict) -> subprocess.CompletedProcess:
    if SCP_STATE.exists():
        SCP_STATE.unlink()
    env = os.environ.copy()
    env["LLM_ADVISOR_STUB_PAYLOAD"] = json.dumps(payload)
    env.setdefault("LOGOS_DEV_BYPASS_OK", "1")
    cmd = [
        sys.executable,
        str(START_AGENT),
        "--enable-logos-agi",
        "--logos-agi-mode",
        "stub",
        "--enable-llm-advisor",
        "--llm-provider",
        "stub",
        "--llm-model",
        "stub",
        "--llm-timeout-sec",
        "5",
        "--objective",
        "status",
        "--assume-yes",
        "--read-only",
        "--budget-sec",
        "2",
        "--no-require-attestation",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=40)


def _load_state():
    if not SCP_STATE.exists():
        return {}
    try:
        return json.loads(SCP_STATE.read_text())
    except Exception:
        return {}


def test_safe_low_impact_proposal() -> bool:
    payload = {
        "proposals": [
            {
                "tool": "mission.status",
                "args": "",
                "rationale": "fetch mission profile",
                "truth_annotation": {"truth": "HEURISTIC", "evidence": {"type": "none", "ref": None}},
            }
        ]
    }
    result = _run_with_payload(payload)
    if result.returncode != 0:
        print(f"FAIL: run error {result.stderr}")
        return False
    state = _load_state()
    last_props = state.get("last_proposals", []) if isinstance(state, dict) else []
    if not last_props:
        print("FAIL: no proposals recorded")
        return False
    has_status = any(isinstance(p, dict) and p.get("tool") == "mission.status" for p in last_props)
    if not has_status:
        print("FAIL: mission.status proposal not present")
        return False
    truth_events = state.get("truth_events", []) if isinstance(state, dict) else []
    exec_attempts = [e for e in truth_events if isinstance(e, dict) and e.get("content", {}).get("reason") == "direct_execution_attempt"]
    if exec_attempts:
        print("FAIL: safe proposal flagged as direct execution attempt")
        return False
    return True


def main() -> bool:
    tests = [test_safe_low_impact_proposal]
    for test in tests:
        if not test():
            return False
    print("PASS: LLM advisor stub proposes safe tool")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
