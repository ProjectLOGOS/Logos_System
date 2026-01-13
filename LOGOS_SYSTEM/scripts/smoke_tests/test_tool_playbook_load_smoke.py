#!/usr/bin/env python3
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
