from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LogosBundle:
    """
    Aggregate output from Logos orchestration.
    - smp: Structured Meaning Packet from I2 (input to this bundle)
    - scp_result: append-only packet from I1 (optional)
    - arp_result: plan/eval bundle from I3 (optional)
    """

    smp: Dict[str, Any]
    scp_result: Optional[Dict[str, Any]] = None
    arp_result: Optional[Dict[str, Any]] = None
    route_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smp": self.smp,
            "scp_result": self.scp_result,
            "arp_result": self.arp_result,
            "route_summary": self.route_summary,
        }
