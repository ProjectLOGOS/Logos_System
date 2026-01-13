# Tool invention draft (proposal-only)
# improvement_id: tool_invention_io_normalizer_20251227T190646Z_1
# gap_id: io_normalizer
# target_module: logos_core.tools.io_normalizer
# origin: tool_optimizer
# policy_class: enhancement
# stage_ok: True
# policy_reasoning: Enhancement allowed by policy configuration | deployment disabled (tool_invention_proposal_only)
# entry_id: 4a869f1a7dd9ff6277797ca51d186c2c
# timestamp_utc: 2025-12-27T19:06:46.830928+00:00

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
