# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

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
