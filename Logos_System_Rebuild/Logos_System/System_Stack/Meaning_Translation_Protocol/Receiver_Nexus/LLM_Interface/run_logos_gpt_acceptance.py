# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""Run LOGOS-GPT acceptance: MVP + advisor boundary smokes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_ROOT = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit"))
AUDIT_DIR = AUDIT_ROOT / "logos_gpt_acceptance"

COMMANDS: List[List[str]] = [
    [sys.executable, "scripts/run_mvp_acceptance.py"],
    [sys.executable, "scripts/test_plan_history_inprocess_refresh.py"],
    [sys.executable, "scripts/test_server_nexus_isolation_smoke.py"],
    [sys.executable, "scripts/test_llm_bypass_smoke.py"],
    [sys.executable, "scripts/test_llm_advisor_smoke.py"],
    [sys.executable, "scripts/test_logos_gpt_chat_smoke.py"],
    [sys.executable, "scripts/test_llm_real_provider_smoke.py"],
    [sys.executable, "scripts/test_llm_streaming_smoke.py"],
    [sys.executable, "scripts/test_tool_playbook_load.py"],
    [sys.executable, "scripts/test_retrieval_local_smoke.py"],
    [sys.executable, "scripts/test_tool_validation_smoke.py"],
    [sys.executable, "scripts/test_tool_fallback_proposal.py"],
    [sys.executable, "scripts/test_grounded_reply_enforcement.py"],
    [sys.executable, "scripts/test_web_grounding_smoke.py"],
    [sys.executable, "scripts/test_logos_gpt_web_smoke.py"],
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
        safe_ts = timestamp.replace(":", "").replace("-", "").replace(".", "")
        fname = f"logos_gpt_{safe_ts}.json"
        with (AUDIT_DIR / fname).open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
    except (OSError, ValueError, TypeError):
        # best-effort audit
        pass


def main() -> int:
    results: List[Dict[str, object]] = []
    for idx, cmd in enumerate(COMMANDS, start=1):
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False
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
            print(f"LOGOS_GPT_ACCEPTANCE: FAIL (step {idx}: {' '.join(cmd)})")
            _maybe_write_audit(results, "FAIL")
            return 1

    print("LOGOS_GPT_ACCEPTANCE: PASS")
    _maybe_write_audit(results, "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
