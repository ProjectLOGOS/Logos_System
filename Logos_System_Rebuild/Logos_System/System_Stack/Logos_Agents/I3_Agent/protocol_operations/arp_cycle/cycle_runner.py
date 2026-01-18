# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from typing import Any, Dict

from ..arp_runtime.task_intake import load_task
from ..arp_planner.baseline_planner import plan_task
from ..arp_planner.plan_evaluator import evaluate_plan_packet
from .policy import decide_policy


def run_arp_cycle(*, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    One ARP cycle:
      - Validate/normalize task
      - Generate baseline plan packet
      - Optionally evaluate plan packet
      - Return JSON-serializable dict bundle
    """
    env = load_task(task=task)
    pol = decide_policy(task=env.raw)

    plan_pkt = plan_task(env=env)
    plan_dict = plan_pkt.to_dict() if hasattr(plan_pkt, "to_dict") else plan_pkt

    if pol.run_evaluation:
        eval_pkt = evaluate_plan_packet(plan_packet=plan_dict)
        eval_dict = eval_pkt.to_dict() if hasattr(eval_pkt, "to_dict") else eval_pkt
    else:
        eval_dict = {
            "task_id": env.task_id,
            "smp_id": env.smp_id,
            "status": "skipped",
            "scores": {},
            "issues": [],
            "summary": "Evaluation skipped by policy.",
            "provenance": {"evaluator": "none"},
        }

    return {
        "policy": {"priority": pol.priority, "run_evaluation": pol.run_evaluation, "reason": pol.reason},
        "plan": plan_dict,
        "evaluation": eval_dict,
    }
