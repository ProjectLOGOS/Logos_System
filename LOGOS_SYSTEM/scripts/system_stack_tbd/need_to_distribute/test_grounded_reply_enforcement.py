#!/usr/bin/env python3
"""Ensure advisor grounding sanitizer downgrades unsupported claims."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llm_interface_suite.llm_advisor import LLMAdvisor


def _build_advisor() -> LLMAdvisor:
    tools_schema = {"tools": []}
    return LLMAdvisor(provider="stub", model="stub", tools_schema=tools_schema, truth_rules={}, timeout_sec=5)


def main() -> int:
    payload = {
        "reply": "test reply",
        "claims": [
            {"text": "claim without evidence", "truth": "VERIFIED", "evidence_refs": []},
            {
                "text": "invalid proof ref",
                "truth": "PROVED",
                "evidence_refs": [{"type": "coq", "ref": {"theorem": "fake", "file": "fake.v", "statement_hash": "x", "index_hash": "y"}}],
            },
        ],
    }
    os.environ["LLM_ADVISOR_STUB_PAYLOAD"] = json.dumps(payload)
    advisor = _build_advisor()
    result = advisor.propose("grounding check", {})
    claims = result.get("claims", []) if isinstance(result, dict) else []
    if not claims:
        print("FAIL: advisor returned no claims")
        return 1
    truths = {c.get("text"): c.get("truth") for c in claims if isinstance(c, dict)}
    if truths.get("claim without evidence") not in {"HEURISTIC", "UNVERIFIED", "INFERRED"}:
        print(f"FAIL: claim without evidence not downgraded (truth={truths.get('claim without evidence')})")
        return 1
    if truths.get("invalid proof ref") in {"PROVED", "VERIFIED"}:
        print(f"FAIL: invalid proof ref not downgraded (truth={truths.get('invalid proof ref')})")
        return 1
    print("PASS: grounded reply enforcement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
