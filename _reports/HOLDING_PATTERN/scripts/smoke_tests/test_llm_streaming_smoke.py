#!/usr/bin/env python3
"""Smoke test for streaming chat path. Skips when no provider keys are set."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_ROOT = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit"))
LEDGER_DIR = AUDIT_ROOT / "run_ledgers"


def _latest_ledger(before: List[Path]) -> Optional[Path]:
    existing = set(before)
    candidates = [p for p in LEDGER_DIR.glob("logos_gpt_chat_*.json") if p not in existing]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    provider: Optional[str] = None
    if has_openai:
        provider = "openai"
    elif has_anthropic:
        provider = "anthropic"

    if provider is None:
        print("SKIP: no provider keys set")
        return 0

    before = list(LEDGER_DIR.glob("logos_gpt_chat_*.json")) if LEDGER_DIR.exists() else []
    env = os.environ.copy()
    env.setdefault("LOGOS_DEV_BYPASS_OK", "1")

    cmd = [
        sys.executable,
        "scripts/llm_interface_suite/logos_gpt_chat.py",
        "--enable-llm-advisor",
        "--llm-provider",
        provider,
        "--llm-model",
        "gpt-4.1-mini",
        "--stream",
        "--max-turns",
        "1",
        "--read-only",
        "--assume-yes",
        "--no-require-attestation",
    ]

    proc = subprocess.run(
        cmd,
        input="hello\n",
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return 1

    combined_output = (proc.stdout or "") + (proc.stderr or "")
    assert combined_output.strip(), "expected streamed assistant output"

    ledger_path = _latest_ledger(before)
    assert ledger_path is not None and ledger_path.exists(), "ledger not created"

    data = json.loads(ledger_path.read_text())
    advisor_meta = data.get("advisor", {}) if isinstance(data, dict) else {}
    assert advisor_meta.get("stream") is True, "ledger missing stream flag"

    executed = data.get("executed_events", []) if isinstance(data, dict) else []
    assert not executed, "tools should not execute in read-only smoke"

    print("PASS: streaming smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
