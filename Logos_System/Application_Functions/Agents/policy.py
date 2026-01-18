# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ARPPolicyDecision:
    priority: str  # normal | high
    run_evaluation: bool
    reason: str


def decide_policy(*, task: Dict[str, Any]) -> ARPPolicyDecision:
    """
    Minimal metadata-only policy:
    - If task.kind is safety/alignment or explicit priority high -> high priority
    - Default run_evaluation=True unless disabled explicitly
    """
    kind = str(task.get("kind", "generic")).lower().strip()
    requested_priority = str(task.get("priority", "")).lower().strip()

    priority = "high" if requested_priority == "high" or kind in {"safety", "alignment"} else "normal"

    run_eval = task.get("run_evaluation")
    if isinstance(run_eval, bool):
        return ARPPolicyDecision(priority=priority, run_evaluation=run_eval, reason="Explicit run_evaluation flag.")
    return ARPPolicyDecision(priority=priority, run_evaluation=True, reason="Default evaluation enabled.")
