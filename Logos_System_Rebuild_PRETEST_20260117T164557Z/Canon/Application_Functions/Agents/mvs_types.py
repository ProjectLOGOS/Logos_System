from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class MVSRequest:
    """
    Request for MVS analysis. Avoid raw unsafe content; use references/hashes when possible.
    """
    smp_id: str
    input_hash: str
    payload_ref: Any = None  # optional opaque handle (NOT required)
    selected_domains: List[str] = field(default_factory=list)
    hints: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class MVSResult:
    """
    Result of MVS analysis, intended for SCP findings.
    """
    available: bool
    summary: str
    coherence_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "summary": self.summary,
            "coherence_score": self.coherence_score,
            "meta": self.meta,
        }
