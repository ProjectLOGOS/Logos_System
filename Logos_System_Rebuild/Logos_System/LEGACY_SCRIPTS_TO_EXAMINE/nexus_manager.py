# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/nexus_manager.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Bounded in-memory manager for LogosAgiNexus instances."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

from JUNK_DRAWER.scripts.runtime.need_to_distribute.logos_agi_adapter import LogosAgiNexus
from JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent import REPO_ROOT, STATE_DIR

MAX_SESSIONS = 100
IDLE_TTL_SECONDS = 1800  # 30 minutes


def _repo_sha() -> str:
    try:
        from subprocess import check_output

        sha = (
            check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], timeout=1)
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return "unknown"


class NexusManager:
    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl_seconds: int = IDLE_TTL_SECONDS):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        # session_id -> (nexus, last_access_ts, initialized)
        self._items: Dict[str, Tuple[LogosAgiNexus, float, bool]] = {}

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, (_, ts, _) in self._items.items() if now - ts > self.ttl_seconds]
        for sid in expired:
            self._items.pop(sid, None)
        # Capacity eviction (oldest first)
        while len(self._items) > self.max_sessions:
            oldest = min(self._items.items(), key=lambda kv: kv[1][1])[0]
            self._items.pop(oldest, None)

    def get(self, session_id: str, mode: str, max_compute_ms: int, audit_logger) -> LogosAgiNexus:
        self._evict_expired()
        now = time.monotonic()
        if session_id in self._items:
            nexus, _, initialized = self._items[session_id]
            self._items[session_id] = (nexus, now, initialized)
            return nexus
        nexus = LogosAgiNexus(
            enable=True,
            audit_logger=audit_logger,
            max_compute_ms=max_compute_ms,
            state_dir=str(STATE_DIR),
            repo_sha=_repo_sha(),
            mode=mode,
            scp_recovery_mode=False,
        )
        nexus.bootstrap()
        self._items[session_id] = (nexus, now, True)
        return nexus

    def debug_snapshot(self) -> Dict[str, Any]:
        return {"sessions": list(self._items.keys()), "count": len(self._items)}


NEXUS_MANAGER = NexusManager()
