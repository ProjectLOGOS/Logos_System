"""
Phase 5 — Controlled Execution (Step 3)

Default mode is DRY_RUN: no filesystem mutation.
EXECUTE mode requires explicit env PHASE5_MODE=EXECUTE.

Allowed actions (initial):
- NOOP (no changes)
- MOVE/RENAME (file moves only; no directory moves)
- DELETE (file delete only; no directory delete)
MERGE is declared but NOT executed in Step 3 (left as FAIL-CLOSED).

Coq immutability guard:
- refuses any path touching PXL_Gate or Runtime_Compiler Coq stacks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .phase_5_types import validate_phase5_queue_obj, Phase5Action, Phase5Queue
from .phase_5_actions import normalize_task_to_operation
from .phase_5_safety import capture_pre_state, capture_post_state
from .phase_5_verify import run_compileall, import_probe

GUARDED_PREFIXES = (
    "PXL_Gate/",
    "Logos_System/System_Entry_Point/Runtime_Compiler/",
    "Logos_System/System_Entry_Point/Runtime_Compiler/coq/",
)


def _guard_path(repo_rel: str) -> None:
    rr = repo_rel.replace("\\", "/")
    for pfx in GUARDED_PREFIXES:
        if rr.startswith(pfx):
            raise RuntimeError(f"FAIL-CLOSED: refusing to mutate guarded path: {repo_rel}")
    if rr.startswith("/"):
        raise RuntimeError("FAIL-CLOSED: absolute paths are not permitted in Phase 5 execution")


def _mode() -> str:
    return os.environ.get("PHASE5_MODE", "DRY_RUN").strip().upper()


def _ensure_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"FAIL-CLOSED: expected file exists: {path}")


def _move_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dest)


def _delete_file(src: Path) -> None:
    _ensure_file(src)
    src.unlink()


def execute_phase5_queue(
    repo_root: Path,
    phase5_queue_obj: Dict[str, Any] | Phase5Queue,
    report_dir: Path,
    strict_per_op_verify: bool = False,
) -> Dict[str, Any]:
    q = phase5_queue_obj if isinstance(phase5_queue_obj, Phase5Queue) else validate_phase5_queue_obj(phase5_queue_obj)
    mode = _mode()

    phase5_dir = report_dir / "Phase_5"
    backup_jsonl = phase5_dir / "Phase_5_Backup_Log.jsonl"
    exec_jsonl = phase5_dir / "Phase_5_Execution_Log.jsonl"

    def log_exec(obj: Dict[str, Any]) -> None:
        import json

        exec_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with exec_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")

    applied = 0
    skipped = 0

    for task in q.tasks:
        op = normalize_task_to_operation(task)
        action = op.action.value if hasattr(op.action, "value") else str(op.action)

        _guard_path(op.src_path)
        if op.dest_path:
            _guard_path(op.dest_path)

        src = repo_root / Path(op.src_path)
        dest = (repo_root / Path(op.dest_path)) if op.dest_path else None

        pre = capture_pre_state(op.task_id, action, src, dest, backup_jsonl)

        if action == Phase5Action.NOOP.value:
            skipped += 1
            log_exec({"task_id": op.task_id, "action": action, "mode": mode, "result": "SKIP", "reason": "NOOP"})
            continue

        if mode != "EXECUTE":
            skipped += 1
            log_exec({"task_id": op.task_id, "action": action, "mode": mode, "result": "DRY_RUN"})
            continue

        if action in (Phase5Action.MOVE.value, Phase5Action.RENAME.value):
            if dest is None:
                raise RuntimeError(f"FAIL-CLOSED: {action} requires dest_path for task {op.task_id}")
            _ensure_file(src)
            _move_file(src, dest)
            applied += 1
            capture_post_state(op.task_id, action, src, dest, backup_jsonl)
            log_exec({"task_id": op.task_id, "action": action, "mode": mode, "result": "APPLIED"})

        elif action == Phase5Action.DELETE.value:
            _delete_file(src)
            applied += 1
            capture_post_state(op.task_id, action, src, dest, backup_jsonl)
            log_exec({"task_id": op.task_id, "action": action, "mode": mode, "result": "APPLIED"})

        elif action == Phase5Action.MERGE.value:
            raise RuntimeError("FAIL-CLOSED: MERGE not implemented in Step 3 executor (by design).")

        else:
            raise RuntimeError(f"FAIL-CLOSED: Unknown action at execution: {action}")

        if strict_per_op_verify:
            run_compileall(repo_root, [repo_root])
            import_probe(repo_root, "audit_normalize_automation")

    run_compileall(repo_root, [repo_root])
    import_probe(repo_root, "audit_normalize_automation")

    return {
        "mode": mode,
        "applied": applied,
        "skipped": skipped,
        "backup_log": str(backup_jsonl.as_posix()),
        "execution_log": str(exec_jsonl.as_posix()),
    }
