# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional


def make_sample_task(*, kind: str = "analysis", priority: str = "normal", smp_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Minimal task dict for testing I3 planning cycle.
    """
    task_id = str(uuid.uuid4())
    return {
        "task_id": task_id,
        "timestamp": time.time(),
        "origin": "LOGOS",
        "kind": kind,
        "priority": priority,
        "smp_id": smp_id,
        "constraints": ["no_memory_writes", "append_only_packets"],
        "run_evaluation": True,
        "payload": {"goal": "Generate a baseline plan skeleton for testing."},
    }
