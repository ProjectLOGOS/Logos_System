"""
LOGOS / Audit-Normalize Automation
Phase 5 (Execute) — Step 1 Action Interfaces (Stubs)

This module defines the execution interface for Phase 5 actions WITHOUT implementing mutation.
Actual mutation logic is introduced in Step 2 (runner integration) and Step 3 (safety layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .phase_5_types import Phase5Task, Phase5Action


@dataclass(frozen=True)
class Phase5PlannedOperation:
    """
    A normalized, deterministic plan for a single operation.
    This is still 'evidence-only' until Step 2 executes it.
    """
    task_id: str
    action: Phase5Action
    src_path: str
    dest_path: Optional[str]
    reason: str


def normalize_task_to_operation(task: Phase5Task) -> Phase5PlannedOperation:
    """
    Deterministically convert a Phase5Task into a planned operation.
    No filesystem reads/writes are permitted here.
    """
    return Phase5PlannedOperation(
        task_id=task.task_id,
        action=task.action,
        src_path=task.src_path,
        dest_path=task.dest_path,
        reason=task.reason,
    )
