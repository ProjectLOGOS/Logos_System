from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .hashing import safe_hash


@dataclass(frozen=True)
class PacketIdentity:
    """Stateless, structured identity block for packets and SMP handoffs."""
    packet_id: str
    origin: str
    created_at: float
    parent_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "origin": self.origin,
            "created_at": self.created_at,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
        }


def generate_packet_identity(
    *,
    origin: str,
    parent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    reference_obj: Any = None,
) -> PacketIdentity:
    """Create a unique packet identity.

    - origin: e.g., "I1", "I2", "I3", "LOGOS"
    - parent_id: upstream packet id if chaining
    - session_id: optional run/session marker
    - reference_obj: optional object to derive an additional stable reference hash
    """
    now = time.time()
    rand = uuid.uuid4().hex[:8]
    ref = safe_hash(reference_obj)[:8] if reference_obj is not None else ""
    suffix = f"{rand}{('-' + ref) if ref else ''}"
    packet_id = f"{origin}-{int(now)}-{suffix}"
    return PacketIdentity(
        packet_id=packet_id,
        origin=origin,
        created_at=now,
        parent_id=parent_id,
        session_id=session_id,
    )
