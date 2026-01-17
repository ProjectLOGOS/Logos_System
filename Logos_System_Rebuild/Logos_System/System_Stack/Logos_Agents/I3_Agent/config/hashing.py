from __future__ import annotations

import hashlib
from typing import Any

def safe_hash(obj: Any) -> str:
    """Stable hash for references; avoids persisting raw content."""
    try:
        s = str(obj)
    except Exception:
        s = repr(obj)
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
