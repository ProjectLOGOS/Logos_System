# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""Ensure failed tools surface fallback proposals without execution."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.start_agent import RuntimeContext, dispatch_tool


def main() -> int:
    os.environ["LOGOS_ENABLE_WEB_RETRIEVAL"] = "0"
    ctx = RuntimeContext(attestation_hash="dev", mission_profile_hash="dev")
    ctx.objective_class = "STATUS"
    payload = json.dumps({"url": ""})
    _ = dispatch_tool("retrieve.web", payload, ctx=ctx)

    proposals = ctx.fallback_proposals or []
    if not proposals:
        print("FAIL: no fallback proposal surfaced")
        return 1
    latest = proposals[-1]
    if latest.get("fallback_from") != "retrieve.web":
        print(f"FAIL: fallback_from mismatch: {latest}")
        return 1
    if len(ctx.tool_validation_events or []) != 1:
        print("FAIL: fallback triggered extra executions")
        return 1
    print("PASS: fallback proposal recorded without execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
