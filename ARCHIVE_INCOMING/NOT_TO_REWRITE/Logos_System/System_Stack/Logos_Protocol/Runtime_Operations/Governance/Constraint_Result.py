from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConstraintResult:
    ok: bool
    reason: str
    tags: dict[str, Any] = field(default_factory=dict)
