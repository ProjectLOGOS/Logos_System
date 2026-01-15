"""
LOGOS / Audit-Normalize Automation
Phase 5 (Execute) — Step 2/3 Runner (Queue Emission + Optional Controlled Execution)

This runner:
- Requires Phase 4 PASS and consumes Phase 4 crawl outputs as evidence inputs.
- Emits Phase 5 queue + status artifacts deterministically.
- Optionally performs controlled execution (Step 3) when PHASE5_ENABLE_STEP3=1.

Fail-closed posture:
- Missing Phase 4 artifacts => FAIL
- Phase 4 not PASS => FAIL
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .phase_5_types import Phase5Queue, Phase5Task, Phase5Evidence, Phase5Action, validate_phase5_queue_obj
from .phase_5_policy import phase5_step3_should_run, phase5_step3_run


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _repo_root_hint() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "_Dev_Resources").exists():
            return parent
    return here.parents[5]


def build_phase5_queue_from_phase4(
    report_dir: Path,
    phase4_status_path: Path,
    phase4_crawl_queue_path: Path,
) -> Dict[str, Any]:
    """
    Step 2 policy:
    - Generate Phase 5 tasks as NOOP placeholders for each Phase 4 task item.
    - Evidence points back to Phase 4 artifacts.
    - Sorting/action assignment is introduced in Step 3.
    """
    phase4_status = _read_json(phase4_status_path)
    if phase4_status.get("status") != "PASS":
        raise RuntimeError(f"Phase 4 status not PASS; refusing Phase 5. status={phase4_status.get('status')}")

    crawl_queue = _read_json(phase4_crawl_queue_path)

    items = crawl_queue.get("tasks") or crawl_queue.get("items") or crawl_queue.get("queue") or []
    if not isinstance(items, list):
        raise RuntimeError("Phase 4 Crawl_Queue.json: expected a list at tasks/items/queue.")

    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        task_id = f"P5T{idx+1:05d}"

        src = None
        if isinstance(item, dict):
            for k in ("path", "src_path", "file", "relpath", "repo_relpath"):
                if k in item and isinstance(item[k], str) and item[k].strip():
                    src = item[k].strip()
                    break
        if not src:
            raise RuntimeError(f"Phase 4 Crawl_Queue task[{idx}] missing recognizable source path key.")

        tasks.append(
            {
                "task_id": task_id,
                "action": Phase5Action.NOOP.value,
                "src_path": src,
                "dest_path": None,
                "reason": "Step 2 placeholder: action assignment occurs in Step 3 sorting logic.",
                "evidence": {
                    "artifacts": [
                        str(phase4_status_path.as_posix()),
                        str(phase4_crawl_queue_path.as_posix()),
                    ],
                    "notes": "Derived from Phase 4 crawl queue; no mutation in Step 2.",
                },
            }
        )

    queue_obj: Dict[str, Any] = {
        "version": "audit_normalize/phase_5_queue/1",
        "generated_utc": _utc_now(),
        "source_phase_4": {
            "status_path": str(phase4_status_path.as_posix()),
            "crawl_queue_path": str(phase4_crawl_queue_path.as_posix()),
            "report_dir": str(report_dir.as_posix()),
        },
        "tasks": tasks,
    }

    _ = validate_phase5_queue_obj(queue_obj)
    return queue_obj


def run_phase5_step2(report_dir: Path) -> Dict[str, Any]:
    """
    Orchestrate Phase 5 Step 2 in a deterministic, side-effect-limited manner:
    - Write Phase_5_Queue.json and Phase_5_Status.json
    """
    phase4_status_path = report_dir / "Phase_4_Status.json"
    phase4_crawl_queue_path = report_dir / "Phase_4" / "Crawl_Queue.json"

    queue_obj = build_phase5_queue_from_phase4(
        report_dir=report_dir,
        phase4_status_path=phase4_status_path,
        phase4_crawl_queue_path=phase4_crawl_queue_path,
    )

    phase5_dir = report_dir / "Phase_5"
    queue_path = phase5_dir / "Phase_5_Queue.json"
    status_path = report_dir / "Phase_5_Status.json"

    _write_json(queue_path, queue_obj)

    status = {
        "phase": 5,
        "status": "UNKNOWN",
        "note": "Phase 5 Step 2 emits queue + wiring only. No filesystem mutation until Step 3 safety layer is implemented.",
        "tasks_count": len(queue_obj["tasks"]),
        "phase_5_queue_path": str(queue_path.as_posix()),
        "generated_utc": _utc_now(),
    }
    _write_json(status_path, status)

    return {"queue_path": str(queue_path.as_posix()), "status_path": str(status_path.as_posix())}


def run_phase5_step3(report_dir: Path, strict_per_op_verify: bool = False, repo_root: Path | None = None) -> Dict[str, Any]:
    """
    Controlled execution layer for Phase 5.
    - Enabled only when PHASE5_ENABLE_STEP3=1.
    - Defaults to DRY_RUN unless PHASE5_MODE=EXECUTE.
    - Fail-closed on missing queue or validation errors.
    """
    if not phase5_step3_should_run():
        return {"status": "SKIPPED", "reason": "PHASE5_ENABLE_STEP3 not set"}

    phase5_dir = report_dir / "Phase_5"
    queue_path = phase5_dir / "Phase_5_Queue.json"
    if not queue_path.exists():
        raise RuntimeError(f"Phase 5 queue missing for Step 3: {queue_path}")

    queue_obj = _read_json(queue_path)
    repo_root = repo_root or _repo_root_hint()
    result = phase5_step3_run(repo_root, report_dir, queue_obj, strict_per_op_verify=strict_per_op_verify)

    status_path = report_dir / "Phase_5_Status.json"
    status = _read_json(status_path) if status_path.exists() else {}
    status.update(
        {
            "phase": 5,
            "status": result.get("mode", "UNKNOWN"),
            "note": "Phase 5 Step 3 executed with controlled executor.",
            "generated_utc": _utc_now(),
        }
    )
    _write_json(status_path, status)

    return result
