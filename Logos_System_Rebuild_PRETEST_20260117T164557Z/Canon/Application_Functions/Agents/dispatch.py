from __future__ import annotations

from typing import Any, Dict, Optional

# I1 SCP pipeline
from ..I1.scp_pipeline.pipeline_runner import run_scp_pipeline
# I3 ARP cycle
from ..I3.arp_cycle.cycle_runner import run_arp_cycle


def dispatch_to_scp(*, smp: Dict[str, Any], payload_ref: Any = None) -> Dict[str, Any]:
    """
    Call I1 SCP pipeline and return JSON-serializable dict.
    """
    pkt = run_scp_pipeline(smp=smp, payload_ref=payload_ref)
    return pkt.to_dict() if hasattr(pkt, "to_dict") else pkt


def dispatch_to_arp(*, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call I3 ARP cycle and return JSON-serializable dict.
    """
    out = run_arp_cycle(task=task)
    return out
