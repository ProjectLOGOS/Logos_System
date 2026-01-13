#!/usr/bin/env python3
"""Smoke tests for deterministic tool validation pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent import RuntimeContext, dispatch_tool


def _make_ctx() -> RuntimeContext:
    ctx = RuntimeContext(attestation_hash="dev", mission_profile_hash="dev")
    ctx.objective_class = "STATUS"
    ctx.read_only = False
    return ctx


def test_valid_mission_status() -> bool:
    ctx = _make_ctx()
    output = dispatch_tool("mission.status", "", ctx=ctx)
    if not output.strip():
        print("FAIL: mission.status empty output")
        return False
    if not ctx.tool_validation_events:
        print("FAIL: validation events missing")
        return False
    last = ctx.tool_validation_events[-1]
    if not last.get("ok"):
        print(f"FAIL: mission.status validator failed: {last}")
        return False
    truth_event = ctx.truth_events[-1] if ctx.truth_events else {}
    truth = truth_event.get("truth_annotation", {}).get("truth")
    if truth not in {"VERIFIED", "INFERRED"}:
        print("FAIL: mission.status truth tier not recorded")
        return False
    return True


def test_invalid_retrieve_web() -> bool:
    os.environ["LOGOS_ENABLE_WEB_RETRIEVAL"] = "0"
    ctx = _make_ctx()
    payload = json.dumps({"url": ""})
    _ = dispatch_tool("retrieve.web", payload, ctx=ctx)
    if not ctx.tool_validation_events:
        print("FAIL: retrieve.web validation events missing")
        return False
    last = ctx.tool_validation_events[-1]
    if last.get("ok"):
        print("FAIL: retrieve.web unexpectedly validated")
        return False
    truth_event = ctx.truth_events[-1] if ctx.truth_events else {}
    truth = truth_event.get("truth_annotation", {}).get("truth")
    if truth == "VERIFIED":
        print("FAIL: retrieve.web marked VERIFIED on failure")
        return False
    return True


def main() -> int:
    if not test_valid_mission_status():
        return 1
    if not test_invalid_retrieve_web():
        return 1
    print("PASS: tool validation smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
