# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

from typing import Any, Dict, Optional

from .types import LogosBundle
from .dispatch import dispatch_to_scp, dispatch_to_arp


def _should_call_scp(smp: Dict[str, Any]) -> bool:
    """
    Minimal heuristic:
    - If route_to indicates SCP, call SCP.
    - If final_decision is escalate/quarantine, call SCP.
    """
    rt = str(smp.get("route_to", "")).strip()
    fd = str(smp.get("final_decision", "")).lower().strip()
    return rt.upper() in {"SCP", "I1"} or fd in {"escalate", "quarantine"}


def _should_call_arp(smp: Dict[str, Any]) -> bool:
    """
    Minimal heuristic:
    - If SMP requests ARP explicitly or task kind indicates planning.
    - Default: call ARP only when requested (keeps it conservative).
    """
    rt = str(smp.get("route_to", "")).strip()
    return rt.upper() in {"ARP", "I3"}


def run_logos_cycle(
    *,
    smp: Dict[str, Any],
    payload_ref: Any = None,
    arp_task_override: Optional[Dict[str, Any]] = None,
) -> LogosBundle:
    """
    Minimal Logos orchestration cycle.
    Input: an SMP dict (already produced by I2).
    Output: LogosBundle with optional SCP and ARP results.

    NOTE: Later we will add:
      - I2 end-to-end intake that produces the SMP
      - UWM integration + SOP governance
    """
    route_summary = {
        "called_scp": False,
        "called_arp": False,
        "reasons": [],
    }

    scp_result = None
    if _should_call_scp(smp):
        scp_result = dispatch_to_scp(smp=smp, payload_ref=payload_ref)
        route_summary["called_scp"] = True
        route_summary["reasons"].append("SCP selected by route_to/final_decision.")

    arp_result = None
    if _should_call_arp(smp):
        # Build a minimal ARP task if none provided
        if arp_task_override is not None:
            task = arp_task_override
        else:
            task = {
                "task_id": f"arp-from-smp-{str(smp.get('smp_id',''))}",
                "timestamp": smp.get("timestamp", 0.0),
                "origin": "LOGOS",
                "kind": "routing",
                "priority": "normal",
                "smp_id": smp.get("smp_id"),
                "constraints": ["append_only_packets", "no_memory_writes"],
                "run_evaluation": True,
                "payload": {"goal": "Produce a baseline plan for handling this SMP."},
            }
        arp_result = dispatch_to_arp(task=task)
        route_summary["called_arp"] = True
        route_summary["reasons"].append("ARP selected by route_to.")

    return LogosBundle(
        smp=smp,
        scp_result=scp_result,
        arp_result=arp_result,
        route_summary=route_summary,
    )
