from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class InputReference:
    """A safe reference to input payload (hash + minimal metadata)."""
    input_hash: str
    preview: str = ""
    kind: str = "opaque"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "preview": self.preview,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class MediationSummary:
    """Minimal mediation summary for routing."""
    final_decision: str
    route_to: str
    violations: Optional[list] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_decision": self.final_decision,
            "route_to": self.route_to,
            "violations": self.violations or [],
        }
