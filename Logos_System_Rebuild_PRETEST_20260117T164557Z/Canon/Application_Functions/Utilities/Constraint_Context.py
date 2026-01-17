from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConstraintContext:
    agent_id: str
    session_id: str
    source: Optional[str] = None
    tlm_token: Optional[str] = None
    runtime_flags: dict[str, bool] = field(default_factory=dict)
