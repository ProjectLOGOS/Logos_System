"""
Phase 5 — Policy and wiring for Step 3 execution.

- Default is DISABLED unless env PHASE5_ENABLE_STEP3=1
- Mode controlled by PHASE5_MODE (DRY_RUN or EXECUTE)
- Merge remains unimplemented in Step 3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .phase_5_execute import execute_phase5_queue
from .phase_5_types import validate_phase5_queue_obj


def phase5_step3_should_run() -> bool:
    return os.environ.get("PHASE5_ENABLE_STEP3", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}


def phase5_step3_run(
    repo_root: Path,
    report_dir: Path,
    phase5_queue_obj: Dict[str, Any],
    strict_per_op_verify: bool = False,
) -> Dict[str, Any]:
    q = validate_phase5_queue_obj(phase5_queue_obj)
    return execute_phase5_queue(repo_root, q, report_dir, strict_per_op_verify=strict_per_op_verify)
