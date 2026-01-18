# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class TransformStep:
    name: str
    applied: bool
    notes: str = ""
    delta: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TransformOutcome:
    # Payload is treated as opaque; SCP should avoid emitting raw harmful content.
    payload: Any
    steps: List[TransformStep]
    score_vector: Dict[str, float] = field(default_factory=dict)
    status: str = "partial"  # ok | partial | blocked | error
    summary: str = ""
