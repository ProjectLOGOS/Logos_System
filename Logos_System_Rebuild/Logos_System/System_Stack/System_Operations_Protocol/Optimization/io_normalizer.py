# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""
IO Normalizer
=============

Derive Standardized Tool IO Normalizer to satisfy tool optimizer gap io_normalizer

Transforms heterogeneous tool payloads into a canonical structure.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


class IONormalizer:
    """Normalize tool IO payloads while preserving audit metadata."""

    def normalize(self, payload: Any) -> Dict[str, Any]:
        envelope: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "metadata": {"raw_type": type(payload).__name__},
        }
        if payload is None:
            envelope["data"] = {}
        elif isinstance(payload, dict):
            envelope["data"] = payload
        elif isinstance(payload, (list, tuple, set)):
            envelope["data"] = {"items": list(payload)}
        else:
            envelope["data"] = {"value": payload}
        return envelope

    def batch_normalize(self, payloads: Iterable[Any]) -> Dict[str, Any]:
        normalized: List[Dict[str, Any]] = [self.normalize(item) for item in payloads]
        return {"status": "ok", "count": len(normalized), "items": normalized}


NORMALIZER = IONormalizer()


def normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize a single payload into canonical schema."""

    return NORMALIZER.normalize(payload)


if __name__ == "__main__":
    result = normalize_payload({"value": 42})
    print(json.dumps(result))
