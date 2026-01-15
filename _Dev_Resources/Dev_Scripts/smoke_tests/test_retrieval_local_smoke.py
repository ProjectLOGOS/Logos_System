#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_retrieval_local_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for retrieve.local tool over docs/README.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent import RuntimeContext, dispatch_tool


def main() -> int:
    ctx = RuntimeContext(attestation_hash="dev", mission_profile_hash="dev")
    args = {
        "query": "Documentation Index",
        "max_files": 5,
        "max_snippets": 5,
        "root": "docs",
    }
    raw = dispatch_tool("retrieve.local", json.dumps(args), ctx=ctx)
    data = json.loads(raw)
    snippets = data.get("snippets", []) if isinstance(data, dict) else []
    if not snippets:
        print("FAIL: no snippets returned")
        return 1
    first = snippets[0]
    if first.get("path") != "docs/README.md":
        print(f"FAIL: expected docs/README.md, got {first.get('path')}")
        return 1
    if int(first.get("start_line", 0)) <= 0 or int(first.get("end_line", 0)) < int(first.get("start_line", 0)):
        print("FAIL: invalid line range")
        return 1
    ordered = [(s.get("path"), int(s.get("start_line", 0))) for s in snippets]
    if ordered != sorted(ordered):
        print("FAIL: snippets not in deterministic order")
        return 1
    print("PASS: retrieve.local smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
