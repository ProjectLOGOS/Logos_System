# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

from datetime import datetime, timezone

def utc_now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
