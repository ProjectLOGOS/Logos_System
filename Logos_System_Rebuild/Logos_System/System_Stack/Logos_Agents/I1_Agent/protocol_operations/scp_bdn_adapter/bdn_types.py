# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class BDNRequest:
    """
    Request for BDN analysis. Avoid raw unsafe content; use references/hashes when possible.
    """
    smp_id: str
    input_hash: str
    payload_ref: Any = None  # optional opaque handle (NOT required)
    selected_domains: List[str] = field(default_factory=list)
    hints: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BDNResult:
    """
    Result of BDN analysis, intended for SCP findings.
    """
    available: bool
    summary: str
    stability_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "summary": self.summary,
            "stability_score": self.stability_score,
            "meta": self.meta,
        }
