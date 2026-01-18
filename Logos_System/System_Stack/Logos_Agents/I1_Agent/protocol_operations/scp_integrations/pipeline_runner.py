# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

from typing import Any, Dict

from ..scp_runtime.smp_intake import load_smp
from ..scp_runtime.work_order import build_work_order
from ..scp_runtime.result_packet import emit_result_packet

from ..scp_analysis.analysis_runner import run_analysis
from I1_Agent.protocol_operations.scp_predict.risk_estimator import estimate_trajectory
from ..scp_transform.iterative_loop import run_iterative_stabilization, LoopConfig
from ..scp_cycle.policy import decide_policy


def run_scp_pipeline(*, smp: Dict[str, Any], payload_ref: Any = None) -> Any:
    """
    Full SCP v1 pipeline:
      - Validate SMP
      - Build work order
      - Run adapter-driven analysis (MVS/BDN stubs OK)
      - Run non-tactical trajectory estimate (metadata-only)
      - Optionally run bounded stabilization loop (policy-driven)
      - Emit append-only SCPResultPacket
    """
    env = load_smp(smp=smp)
    wo = build_work_order(envelope=env)

    analysis_bundle = run_analysis(
        smp_id=env.smp_id,
        input_hash=env.input_hash,
        selected_domains=wo.selected_domains,
        hints=wo.hints,
        payload_ref=payload_ref,
    )

    traj = estimate_trajectory(smp=env.raw)

    pol = decide_policy(smp=env.raw)

    findings = {
        "work_order": {
            "priority": wo.priority,
            "objectives": wo.objectives,
            "selected_domains": wo.selected_domains,
            "constraints": wo.constraints,
        },
        "analysis_bundle": analysis_bundle.to_dict(),
        "trajectory_estimate": traj.to_dict(),
    }

    recommended_next = {
        "route_to": "LOGOS",
        "recommended_action": traj.recommended_action,
        "policy_reason": pol.reason,
    }

    if pol.run_loop:
        outcome = run_iterative_stabilization(
            payload=payload_ref if payload_ref is not None else {"smp_id": env.smp_id, "input_hash": env.input_hash},
            context={"work_order": wo.__dict__, "smp_id": env.smp_id},
            config=LoopConfig(max_iters=3, stop_on_no_change=True),
        )
        findings["transform"] = {
            "status": outcome.status,
            "summary": outcome.summary,
            "steps": [{"name": s.name, "applied": s.applied, "notes": s.notes} for s in outcome.steps],
        }
        score_vector = outcome.score_vector or env.triadic_scores
        summary = f"SCP pipeline ran loop ({pol.reason}); {analysis_bundle.summary}; traj={traj.category}"
        status = outcome.status
    else:
        score_vector = env.triadic_scores
        summary = f"SCP pipeline skipped loop ({pol.reason}); {analysis_bundle.summary}; traj={traj.category}"
        status = "ok"

    return emit_result_packet(
        smp_id=env.smp_id,
        status=status,
        summary=summary,
        score_vector=score_vector,
        findings=findings,
        recommended_next=recommended_next,
        reference_obj=env.input_hash,
    )
