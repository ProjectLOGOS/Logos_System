#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_logos_gpt_chat_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test: LOGOS-GPT chat loop stays advisory and gated."""

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
CHAT = REPO_ROOT / "scripts" / "llm_interface_suite" / "logos_gpt_chat.py"
STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))
SCP_STATE = STATE_DIR / "scp_state.json"
AUDIT_ROOT = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit"))
RUN_LEDGER_DIR = AUDIT_ROOT / "run_ledgers"


def _run_chat() -> subprocess.CompletedProcess:
    if SCP_STATE.exists():
        SCP_STATE.unlink()
    env = os.environ.copy()
    env["LOGOS_DEV_BYPASS_OK"] = env.get("LOGOS_DEV_BYPASS_OK", "1")
    env["LLM_ADVISOR_STUB_PAYLOAD"] = json.dumps(
        {
            "reply": "Stub reply",
            "proposals": [
                {
                    "tool": "mission.status",
                    "args": "",
                    "rationale": "fetch status",
                    "truth_annotation": {"truth": "HEURISTIC", "evidence": {"type": "none", "ref": None}},
                }
            ],
        }
    )
    input_data = "status\nwhat happened?\n"
    cmd = [
        sys.executable,
        str(CHAT),
        "--enable-llm-advisor",
        "--llm-provider",
        "stub",
        "--llm-model",
        "stub",
        "--llm-timeout-sec",
        "5",
        "--assume-yes",
        "--read-only",
        "--max-turns",
        "2",
        "--objective-class",
        "CHAT",
        "--no-require-attestation",
    ]
    return subprocess.run(
        cmd,
        input=input_data,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=60,
    )


def _load_state():
    if not SCP_STATE.exists():
        return {}
    try:
        return json.loads(SCP_STATE.read_text())
    except Exception:
        return {}


def test_chat_loop() -> bool:
    result = _run_chat()
    if result.returncode != 0:
        print(f"FAIL: chat run error {result.stderr}")
        return False

    state = _load_state()
    if not state:
        print("FAIL: scp_state missing")
        return False

    wm = state.get("working_memory", {}) if isinstance(state, dict) else {}
    short = wm.get("short_term", []) if isinstance(wm, dict) else []
    role_counts = {"user": 0, "assistant": 0}
    for item in short:
        content = item.get("content", {}) if isinstance(item, dict) else {}
        role = content.get("role")
        if role in role_counts:
            role_counts[role] += 1
    if role_counts["user"] < 1 or role_counts["assistant"] < 1:
        print("FAIL: working_memory missing conversation items")
        return False

    # Ensure no PROVED claims without Coq evidence
    last_props = state.get("last_proposals", []) if isinstance(state, dict) else []
    for prop in last_props:
        truth = prop.get("truth_annotation", {}).get("truth") if isinstance(prop, dict) else None
        if truth == "PROVED":
            print("FAIL: PROVED claim present without Coq evidence")
            return False

    # Tool executions (if any) should be logged
    last_results = state.get("last_tool_results", []) if isinstance(state, dict) else []
    if last_results:
        if not all(isinstance(r, dict) and r.get("tool") for r in last_results):
            print("FAIL: tool results not logged via dispatch")
            return False

    # Run ledger existence
    if not RUN_LEDGER_DIR.exists():
        print("FAIL: run ledger directory missing")
        return False
    ledger_files = [p for p in RUN_LEDGER_DIR.glob("logos_gpt_chat_*.json")]
    if not ledger_files:
        print("FAIL: no run ledger written")
        return False

    # Stub beliefs bounded
    beliefs = state.get("beliefs", {}) if isinstance(state, dict) else {}
    items = beliefs.get("items", []) if isinstance(beliefs, dict) else []
    if len(items) > 5:
        print("FAIL: beliefs exceed stub bounds")
        return False

    return True


def main() -> int:
    if test_chat_loop():
        print("PASS: LOGOS-GPT chat loop gated")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
