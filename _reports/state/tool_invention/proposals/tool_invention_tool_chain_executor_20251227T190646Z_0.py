# Tool invention draft (proposal-only)
# improvement_id: tool_invention_tool_chain_executor_20251227T190646Z_0
# gap_id: tool_chain_executor
# target_module: logos_core.tools.tool_chain_executor
# origin: tool_optimizer
# policy_class: enhancement
# stage_ok: True
# policy_reasoning: Enhancement allowed by policy configuration | deployment disabled (tool_invention_proposal_only)
# entry_id: cded2224877a9a853b3788dd1c1b3253
# timestamp_utc: 2025-12-27T19:06:46.804421+00:00

#!/usr/bin/env python3
"""
Tool Chain Executor
===================

Derive Deterministic Tool Chain Executor to satisfy tool optimizer gap tool_chain_executor

Provides deterministic orchestration across validated tool steps.
"""

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


class ToolChainExecutor:
    """Execute ordered tool callables with audit history."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def execute(
        self,
        steps: Iterable[Tuple[str, Callable[[Any], Any]]],
        payload: Any = None,
        allow_partial: bool = False,
    ) -> Dict[str, Any]:
        timeline: List[Dict[str, Any]] = []
        current = payload
        for index, (name, func) in enumerate(list(steps)):
            record = {
                "step": index,
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                current = func(current)
                record["status"] = "ok"
            except Exception as exc:  # noqa: BLE001 - surface failure detail to caller
                record["status"] = "error"
                record["reason"] = str(exc)
                if not allow_partial:
                    timeline.append(record)
                    break
            timeline.append(record)
        outcome = {
            "status": "ok",
            "results": timeline,
            "final_payload": current,
        }
        if any(entry.get("status") == "error" for entry in timeline):
            outcome["status"] = "partial" if allow_partial else "error"
        self.history.append(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "timeline": timeline,
                "status": outcome["status"],
            }
        )
        return outcome

    def last_run(self) -> Optional[Dict[str, Any]]:
        """Return the most recent execution record if available."""
        return self.history[-1] if self.history else None


EXECUTOR = ToolChainExecutor()


def run_chain(
    step_functions: Iterable[Tuple[str, Callable[[Any], Any]]],
    payload: Any = None,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """Execute a deterministic chain of tool callables."""

    return EXECUTOR.execute(step_functions, payload, allow_partial=allow_partial)


if __name__ == "__main__":

    def _echo(value: Any) -> Any:
        return value

    report = run_chain([("echo", _echo)], {"demo": True})
    print(json.dumps(report))
