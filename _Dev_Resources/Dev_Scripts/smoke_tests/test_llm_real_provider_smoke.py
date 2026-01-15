#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_llm_real_provider_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke tests for real LLM providers (advisor-only).

Skips cleanly when no provider keys are present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llm_interface_suite.llm_advisor import LLMAdvisor, build_tool_schema
from JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent import TOOLS


FORBIDDEN_KEYS = {"execute", "run", "shell", "code"}


def _assert_clean(proposals: List[Dict[str, Any]]) -> None:
    assert isinstance(proposals, list)
    for prop in proposals:
        assert isinstance(prop, dict)
        assert not (FORBIDDEN_KEYS & set(prop.keys())), "unexpected execution markers"


def _run_provider(provider: str) -> None:
    tool_schema = build_tool_schema(TOOLS)
    advisor = LLMAdvisor(
        provider=provider,
        model="gpt-4.1-mini",
        tools_schema=tool_schema,
        truth_rules={},
        timeout_sec=10,
    )
    result = advisor.propose("Provide a short acknowledgment only", {"conversation_recall": [], "tool_summary": {}})
    reply = result.get("reply")
    proposals = result.get("proposals", [])
    _assert_clean(proposals)
    assert reply, "reply should be present"


def main() -> int:
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not has_openai and not has_anthropic:
        print("SKIP: no provider keys set")
        return 0

    if has_openai:
        _run_provider("openai")
    if has_anthropic:
        _run_provider("anthropic")

    print("PASS: real provider smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
