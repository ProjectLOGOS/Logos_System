# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from typing import Any, Dict, List, Optional

from I3_Agent.diagnostics.errors import SchemaError


def require_dict(obj: Any, name: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise SchemaError(f"{name} must be a dict")
    return obj


def require_str(obj: Any, name: str) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return obj


def get_str(d: Dict[str, Any], key: str, default: str = "") -> str:
    v = d.get(key, default)
    return v if isinstance(v, str) else default


def get_dict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def get_list(d: Dict[str, Any], key: str) -> List[Any]:
    v = d.get(key)
    return v if isinstance(v, list) else []
