from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..scp_mvs_adapter.mvs_types import MVSRequest
from ..scp_mvs_adapter.mvs_adapter import IMVSAdapter, StubMVSAdapter

from ..scp_bdn_adapter.bdn_types import BDNRequest
from ..scp_bdn_adapter.bdn_adapter import IBDNAdapter, StubBDNAdapter


@dataclass(frozen=True)
class SCPAnalysisBundle:
    """
    Aggregate analysis bundle produced by SCP from pluggable engines.
    """
    mvs: Dict[str, Any]
    bdn: Dict[str, Any]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {"mvs": self.mvs, "bdn": self.bdn, "summary": self.summary}


def run_analysis(
    *,
    smp_id: str,
    input_hash: str,
    selected_domains: Optional[list] = None,
    hints: Optional[Dict[str, Any]] = None,
    mvs_adapter: Optional[IMVSAdapter] = None,
    bdn_adapter: Optional[IBDNAdapter] = None,
    payload_ref: Any = None,
) -> SCPAnalysisBundle:
    """
    Run SCP analysis via adapters and return a unified bundle.

    - selected_domains and hints should come from SCPWorkOrder.
    - payload_ref should be an opaque handle; avoid raw content when possible.
    """
    selected_domains = selected_domains or []
    hints = hints or {}

    mvs = mvs_adapter or StubMVSAdapter()
    bdn = bdn_adapter or StubBDNAdapter()

    mvs_req = MVSRequest(
        smp_id=smp_id,
        input_hash=input_hash,
        payload_ref=payload_ref,
        selected_domains=[str(x) for x in selected_domains],
        hints=hints,
    )
    bdn_req = BDNRequest(
        smp_id=smp_id,
        input_hash=input_hash,
        payload_ref=payload_ref,
        selected_domains=[str(x) for x in selected_domains],
        hints=hints,
    )

    mvs_res = mvs.analyze(mvs_req).to_dict()
    bdn_res = bdn.analyze(bdn_req).to_dict()

    parts = []
    parts.append("MVS: available" if mvs_res.get("available") else "MVS: stub")
    parts.append("BDN: available" if bdn_res.get("available") else "BDN: stub")
    summary = "; ".join(parts)

    return SCPAnalysisBundle(mvs=mvs_res, bdn=bdn_res, summary=summary)
