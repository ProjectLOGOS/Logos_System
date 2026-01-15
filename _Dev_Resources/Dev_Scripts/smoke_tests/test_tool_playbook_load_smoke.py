#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_tool_playbook_load_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test to ensure tool playbooks cover all registered tools."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logos.tool_playbooks import REQUIRED_FIELDS, TOOL_PLAYBOOKS
from JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent import TOOLS


def main() -> int:
    missing = [name for name in TOOLS if name not in TOOL_PLAYBOOKS]
    if missing:
        print(f"FAIL: missing playbooks for {missing}")
        return 1

    for name, playbook in TOOL_PLAYBOOKS.items():
        for field in REQUIRED_FIELDS:
            if field not in playbook:
                print(f"FAIL: playbook {name} missing {field}")
                return 1
        if not isinstance(playbook.get("success_validators"), list):
            print(f"FAIL: playbook {name} success_validators must be list")
            return 1
        if not isinstance(playbook.get("fallback_tools"), list):
            print(f"FAIL: playbook {name} fallback_tools must be list")
            return 1
    print("PASS: all tools have playbooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
