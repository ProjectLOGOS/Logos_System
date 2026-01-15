"""
LOGOS / Audit-Normalize Automation
Phase 5 (Execute) — Step 1 Core Types

This module defines the Phase 5 queue contract and validation helpers.
It MUST remain side-effect free: no filesystem mutation, no network, no subprocess.

Naming: Title_Case_With_Underscores for files/dirs; Python identifiers follow PEP8.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Phase5Action(str, Enum):
    NOOP = "NOOP"
    MOVE = "MOVE"
    RENAME = "RENAME"
    MERGE = "MERGE"
    DELETE = "DELETE"


@dataclass(frozen=True)
class Phase5Evidence:
    """Pointers back to Phase 4 evidence (paths/keys), not embedded blobs."""
    artifacts: List[str]
    notes: Optional[str] = None


@dataclass(frozen=True)
class Phase5Task:
    task_id: str
    action: Phase5Action
    src_path: str
    dest_path: Optional[str]
    reason: str
    evidence: Phase5Evidence


@dataclass(frozen=True)
class Phase5Queue:
    version: str
    generated_utc: str
    source_phase_4: Dict[str, Any]
    tasks: List[Phase5Task]


class Phase5ContractError(RuntimeError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise Phase5ContractError(msg)


def _is_repo_relative_path(p: str) -> bool:
    if not isinstance(p, str) or not p:
        return False
    if p.startswith("/") or p.startswith("~") or ":\\" in p:
        return False
    if ".." in p.split("/"):
        return False
    return True


def validate_phase5_queue_obj(obj: Dict[str, Any]) -> Phase5Queue:
    """
    Validate a Phase 5 queue dict and return a strongly typed Phase5Queue.

    Fail-closed:
    - unknown keys are tolerated at top-level only if they do not affect execution
    - unknown actions are hard-fail
    - any non-repo-relative path is hard-fail
    """
    _require(isinstance(obj, dict), "Phase 5 queue must be a JSON object (dict).")

    for k in ("version", "generated_utc", "source_phase_4", "tasks"):
        _require(k in obj, f"Missing required key: {k}")

    version = obj["version"]
    generated_utc = obj["generated_utc"]
    source_phase_4 = obj["source_phase_4"]
    tasks_raw = obj["tasks"]

    _require(isinstance(version, str) and version.strip(), "version must be a non-empty string.")
    _require(isinstance(generated_utc, str) and generated_utc.strip(), "generated_utc must be a non-empty string.")
    _require(isinstance(source_phase_4, dict), "source_phase_4 must be an object (dict).")
    _require(isinstance(tasks_raw, list), "tasks must be a list.")

    tasks: List[Phase5Task] = []
    seen_ids = set()

    for i, t in enumerate(tasks_raw):
        _require(isinstance(t, dict), f"Task[{i}] must be an object (dict).")

        for k in ("task_id", "action", "src_path", "dest_path", "reason", "evidence"):
            _require(k in t, f"Task[{i}] missing required key: {k}")

        task_id = t["task_id"]
        _require(isinstance(task_id, str) and task_id.strip(), f"Task[{i}].task_id must be a non-empty string.")
        _require(task_id not in seen_ids, f"Duplicate task_id: {task_id}")
        seen_ids.add(task_id)

        action_raw = t["action"]
        _require(isinstance(action_raw, str), f"Task[{i}].action must be a string.")
        try:
            action = Phase5Action(action_raw)
        except ValueError as e:
            raise Phase5ContractError(f"Task[{i}] has unknown action '{action_raw}' (fail-closed).") from e

        src_path = t["src_path"]
        _require(_is_repo_relative_path(src_path), f"Task[{i}].src_path must be repo-relative and non-traversing.")

        dest_path = t["dest_path"]
        if dest_path is not None:
            _require(_is_repo_relative_path(dest_path), f"Task[{i}].dest_path must be repo-relative and non-traversing.")

        reason = t["reason"]
        _require(isinstance(reason, str) and reason.strip(), f"Task[{i}].reason must be a non-empty string.")

        ev = t["evidence"]
        _require(isinstance(ev, dict), f"Task[{i}].evidence must be an object (dict).")
        _require("artifacts" in ev, f"Task[{i}].evidence.artifacts is required.")
        artifacts = ev["artifacts"]
        _require(isinstance(artifacts, list) and artifacts, f"Task[{i}].evidence.artifacts must be a non-empty list.")
        for a in artifacts:
            _require(isinstance(a, str) and _is_repo_relative_path(a), f"Task[{i}].evidence.artifacts entries must be repo-relative strings.")

        notes = ev.get("notes", None)
        if notes is not None:
            _require(isinstance(notes, str), f"Task[{i}].evidence.notes must be a string when present.")

        tasks.append(
            Phase5Task(
                task_id=task_id,
                action=action,
                src_path=src_path,
                dest_path=dest_path,
                reason=reason,
                evidence=Phase5Evidence(artifacts=artifacts, notes=notes),
            )
        )

    return Phase5Queue(
        version=version,
        generated_utc=generated_utc,
        source_phase_4=source_phase_4,
        tasks=tasks,
    )
