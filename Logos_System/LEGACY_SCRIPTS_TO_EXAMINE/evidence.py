# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED


"""Evidence reference helpers for grounded replies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from logos import proof_refs


_ALLOWED_TYPES = {"file", "url", "coq", "schema", "hash"}


def _as_posix(path: str) -> str:
    return Path(path).as_posix()


def validate_evidence_ref(ref: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single evidence reference and return normalized copy."""
    if not isinstance(ref, dict):
        raise ValueError("evidence ref must be dict")
    ref_type = ref.get("type")
    if ref_type not in _ALLOWED_TYPES:
        raise ValueError(f"unsupported evidence type: {ref_type}")

    if ref_type == "file":
        path = str(ref.get("path", "")).strip()
        start = int(ref.get("start_line", 0))
        end = int(ref.get("end_line", 0))
        if not path:
            raise ValueError("file evidence missing path")
        if start <= 0 or end <= 0 or end < start:
            raise ValueError("file evidence invalid line range")
        return {
            "type": "file",
            "path": _as_posix(path),
            "start_line": start,
            "end_line": end,
        }

    if ref_type == "url":
        url = str(ref.get("url", "")).strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("url evidence must be http(s)")
        return {"type": "url", "url": url}

    if ref_type == "coq":
        inner = ref.get("ref")
        if not isinstance(inner, dict):
            raise ValueError("coq evidence requires ref object")
        # Do not validate against index here; delegate to proof_refs when available.
        return {"type": "coq", "ref": dict(inner)}

    if ref_type == "schema":
        schema_id = (
            str(ref.get("schema", "")).strip() or str(ref.get("ref", "")).strip()
        )
        if not schema_id:
            raise ValueError("schema evidence requires identifier")
        return {"type": "schema", "schema": schema_id}

    if ref_type == "hash":
        hash_value = str(ref.get("hash", "")).strip() or str(ref.get("ref", "")).strip()
        if not hash_value:
            raise ValueError("hash evidence requires value")
        return {"type": "hash", "hash": hash_value}

    raise ValueError(f"unsupported evidence type: {ref_type}")


def _sort_key(ref: Dict[str, Any]) -> Tuple:
    ref_type = ref.get("type", "")
    if ref_type == "file":
        return (
            ref_type,
            ref.get("path", ""),
            int(ref.get("start_line", 0)),
            int(ref.get("end_line", 0)),
        )
    if ref_type == "url":
        return (ref_type, ref.get("url", ""))
    if ref_type == "coq":
        inner = ref.get("ref", {}) if isinstance(ref.get("ref"), dict) else {}
        return (ref_type, inner.get("theorem", ""), inner.get("file", ""))
    if ref_type == "schema":
        return (ref_type, ref.get("schema", ""))
    if ref_type == "hash":
        return (ref_type, ref.get("hash", ""))
    return (ref_type, "")


def normalize_evidence_refs(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe and deterministically order evidence references."""
    normalized: List[Dict[str, Any]] = []
    seen: set[Tuple] = set()
    for raw in refs or []:
        try:
            valid = validate_evidence_ref(raw)
        except ValueError:
            continue
        key = _sort_key(valid)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(valid)
    normalized.sort(key=_sort_key)
    return normalized


def evidence_to_citation_string(ref: Dict[str, Any]) -> str:
    ref_type = ref.get("type")
    if ref_type == "file":
        start = int(ref.get("start_line", 0))
        end = int(ref.get("end_line", 0))
        suffix = f"L{start}-L{end}" if start != end else f"L{start}"
        return f"[file:{ref.get('path', '')}:{suffix}]"
    if ref_type == "url":
        return f"[url:{ref.get('url', '')}]"
    if ref_type == "coq":
        inner = ref.get("ref", {}) if isinstance(ref.get("ref"), dict) else {}
        theorem = inner.get("theorem") or inner.get("file") or "coq"
        return f"[coq:{theorem}]"
    if ref_type == "schema":
        return f"[schema:{ref.get('schema', '')}]"
    if ref_type == "hash":
        return f"[hash:{ref.get('hash', '')}]"
    return "[evidence]"


def is_proved_reference(ref: Dict[str, Any], theorem_index: Dict[str, Any]) -> bool:
    """Return True if ref is a proved Coq reference against index."""
    if not theorem_index:
        return False
    try:
        valid = validate_evidence_ref(ref)
    except Exception:
        return False
    if valid.get("type") != "coq":
        return False
    inner = valid.get("ref") or {}
    return proof_refs.is_proved_ref(inner, theorem_index)
