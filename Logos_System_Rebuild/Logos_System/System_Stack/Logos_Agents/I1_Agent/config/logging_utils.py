from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


def log_event(
    *,
    event: str,
    level: str = "INFO",
    context: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Lightweight structured log record generator.

    Returns a dict; caller decides where to emit it.
    """
    record = {
        "ts": time.time(),
        "level": level,
        "event": event,
        "context": context or {},
        "data": data or {},
    }
    return record


def dumps_log(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=False)
