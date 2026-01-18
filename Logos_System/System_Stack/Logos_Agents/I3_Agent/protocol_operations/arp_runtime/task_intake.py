# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from I3_Agent.config.schema_utils import require_dict, get_str
from I3_Agent.diagnostics.errors import SchemaError


@dataclass(frozen=True)
class TaskEnvelope:
    task_id: str
    timestamp: float
    origin: str
    kind: str
    smp_id: Optional[str]
    raw: Dict[str, Any]


def load_task(*, task: Dict[str, Any]) -> TaskEnvelope:
    task = require_dict(task, "task")
    task_id = get_str(task, "task_id") or get_str(task, "id")
    if not task_id:
        raise SchemaError("Task missing task_id")

    origin = get_str(task, "origin") or "LOGOS"
    kind = get_str(task, "kind") or "generic"
    smp_id = get_str(task, "smp_id") or None

    ts = task.get("timestamp")
    if not isinstance(ts, (int, float)):
        ts = 0.0

    return TaskEnvelope(
        task_id=task_id,
        timestamp=float(ts),
        origin=origin,
        kind=kind,
        smp_id=smp_id,
        raw=task,
    )
