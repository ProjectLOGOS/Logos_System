from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..arp_runtime.task_intake import TaskEnvelope
from ..arp_runtime.plan_packet import emit_plan_packet


def build_baseline_plan(*, env: TaskEnvelope) -> List[Dict[str, Any]]:
    """
    Deterministic baseline plan skeleton based on task kind.
    No tool execution; no heavy reasoning.
    """
    kind = (env.kind or "generic").lower().strip()

    # Default plan: intake -> analyze constraints -> propose actions -> output packet
    plan: List[Dict[str, Any]] = [
        {"step": 1, "action": "normalize_task", "notes": "Ensure task fields are complete."},
        {"step": 2, "action": "extract_constraints", "notes": "Identify explicit constraints from task payload."},
        {"step": 3, "action": "propose_candidate_actions", "notes": "Generate minimal action candidates (no execution)."},
        {"step": 4, "action": "evaluate_plan_quality", "notes": "Run lightweight checks for completeness/consistency."},
        {"step": 5, "action": "emit_plan_packet", "notes": "Return plan for Logos approval/execution."},
    ]

    # Specialize a little by kind (still generic)
    if kind in {"analysis", "research"}:
        plan.insert(3, {"step": 3, "action": "outline_information_needs", "notes": "List missing info and assumptions."})
    elif kind in {"routing", "orchestration"}:
        plan.insert(3, {"step": 3, "action": "identify_dependencies", "notes": "List agent/protocol dependencies and order."})
    elif kind in {"safety", "alignment"}:
        plan.insert(3, {"step": 3, "action": "enumerate_risks", "notes": "List likely failure modes and mitigations."})

    # Re-number steps cleanly
    for idx, item in enumerate(plan, start=1):
        item["step"] = idx

    return plan


def plan_task(*, env: TaskEnvelope) -> Any:
    """
    TaskEnvelope -> PlanPacket (append-only).
    """
    plan = build_baseline_plan(env=env)

    # Pull constraints from task raw if provided
    raw = env.raw or {}
    constraints = raw.get("constraints") if isinstance(raw.get("constraints"), list) else []

    rationale = "Baseline plan generated deterministically from task kind; no execution performed."
    return emit_plan_packet(
        task_id=env.task_id,
        smp_id=env.smp_id,
        plan=plan,
        rationale=rationale,
        constraints=[str(x) for x in constraints],
        provenance={"planner": "baseline_planner.v1", "kind": env.kind},
    )
