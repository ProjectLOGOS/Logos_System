# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from datetime import datetime, timezone

def utc_now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
