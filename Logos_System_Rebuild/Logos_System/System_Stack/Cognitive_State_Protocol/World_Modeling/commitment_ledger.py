# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Commitment ledger management for LOGOS AGI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

LEDGER_VERSION = 1
DEFAULT_LEDGER_PATH = Path("state/commitment_ledger.json")
MAX_COMMITMENTS = 200
MAX_HISTORY_EVENTS = 1000
JUSTIFICATION_MAX = 500
SUMMARY_MAX = 300
BLOCKER_MAX = 200


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def compute_ledger_hash(ledger: Dict[str, Any]) -> str:
    payload = dict(ledger)
    integrity = dict(payload.get("integrity", {}))
    integrity["ledger_hash"] = None
    payload["integrity"] = integrity
    return sha256_bytes(canonical_json(payload))


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def _clamp_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _ensure_constraints(constraints: Dict[str, Any], defaults: Dict[str, bool]) -> Dict[str, bool]:
    result = {
        "allow_enhancements": bool(defaults.get("allow_enhancements", False)),
        "safe_only": bool(defaults.get("safe_only", False)),
        "read_only": bool(defaults.get("read_only", False)),
    }
    for key in ("allow_enhancements", "safe_only", "read_only"):
        if key in constraints:
            result[key] = bool(constraints[key])
    return result


def _compute_commitment_id(commitment: Dict[str, Any]) -> str:
    canonical = canonical_json(
        {
            "title": commitment.get("title", ""),
            "type": commitment.get("type", "other"),
            "justification": commitment.get("justification", ""),
            "created_utc": commitment.get("created_utc", ""),
        }
    )
    return _sha256_hex(canonical)


def _compute_event_id(event: Dict[str, Any]) -> str:
    canonical = canonical_json(
        {
            "commitment_id": event.get("commitment_id", ""),
            "action": event.get("action", ""),
            "timestamp_utc": event.get("timestamp_utc", ""),
            "summary": event.get("summary", ""),
        }
    )
    return _sha256_hex(canonical)


def _trim_commitments(ledger: Dict[str, Any]) -> None:
    commitments = ledger.get("commitments", [])
    active_id = ledger.get("active_commitment_id")
    if len(commitments) <= MAX_COMMITMENTS:
        return
    # Keep active, then most recently updated
    def sort_key(item: Dict[str, Any]) -> str:
        return item.get("updated_utc", "")

    commitments.sort(key=sort_key, reverse=True)
    retained: List[Dict[str, Any]] = []
    for item in commitments:
        if item.get("commitment_id") == active_id:
            retained.insert(0, item)
        elif len(retained) < MAX_COMMITMENTS - (1 if active_id else 0):
            retained.append(item)
    ledger["commitments"] = retained


def _trim_history(ledger: Dict[str, Any]) -> None:
    history = ledger.get("history", [])
    if len(history) <= MAX_HISTORY_EVENTS:
        return
    ledger["history"] = history[-MAX_HISTORY_EVENTS:]


def load_or_create_ledger(path: Path | str = DEFAULT_LEDGER_PATH) -> Dict[str, Any]:
    if isinstance(path, (str, os.PathLike)):
        path = Path(path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                ledger = json.load(handle)
            if not isinstance(ledger, dict):
                raise ValueError("Ledger root must be an object")
            return ledger
        except (json.JSONDecodeError, ValueError):
            # Fall back to new ledger if corrupted
            pass
    now = _utc_now()
    ledger = {
        "ledger_version": LEDGER_VERSION,
        "created_utc": now,
        "updated_utc": now,
        "active_commitment_id": None,
        "commitments": [],
        "history": [],
        "integrity": {"ledger_hash": None},
    }
    return ledger


def validate_ledger(ledger: Dict[str, Any], identity: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []
    warnings: List[str] = []
    if ledger.get("ledger_version") != LEDGER_VERSION:
        reasons.append("Unsupported ledger version")
    commitments = ledger.get("commitments", [])
    active_id = ledger.get("active_commitment_id")
    active_count = sum(1 for item in commitments if item.get("status") == "active")
    if active_count > 1:
        reasons.append("Multiple active commitments present")
    if active_id:
        match = next((item for item in commitments if item.get("commitment_id") == active_id), None)
        if not match:
            reasons.append("Active commitment missing from list")
        elif match.get("status") != "active":
            reasons.append("Active commitment not marked active")
    mission = identity.get("mission", {})
    mission_allow = bool(mission.get("allow_enhancements", False))
    for commitment in commitments:
        cid = commitment.get("commitment_id") or "<unknown>"
        if not commitment.get("success_criteria"):
            reasons.append(f"Commitment {cid} missing success criteria")
        for crit in commitment.get("success_criteria", []):
            if not crit:
                reasons.append(f"Commitment {cid} has empty success criterion")
        justification = commitment.get("justification", "")
        if not justification:
            reasons.append(f"Commitment {cid} missing justification")
        elif len(justification) > JUSTIFICATION_MAX:
            warnings.append(f"Commitment {cid} justification truncated")
        evidence_refs = commitment.get("evidence_refs", [])
        if not evidence_refs:
            reasons.append(f"Commitment {cid} missing evidence references")
        constraints = commitment.get("constraints", {})
        allow_value = bool(constraints.get("allow_enhancements", False))
        if allow_value and not mission_allow:
            reasons.append(f"Commitment {cid} exceeds mission allow_enhancements policy")
    stored_hash = ledger.get("integrity", {}).get("ledger_hash")
    computed_hash = compute_ledger_hash(ledger)
    if stored_hash and stored_hash != computed_hash:
        reasons.append("Ledger hash mismatch")
    return (not reasons, reasons, warnings)


def record_event(ledger: Dict[str, Any], action: str, commitment_id: str, summary: str) -> None:
    summary = _clamp_text(summary, SUMMARY_MAX)
    event = {
        "event_id": None,
        "timestamp_utc": _utc_now(),
        "action": action,
        "commitment_id": commitment_id,
        "summary": summary,
    }
    event["event_id"] = _compute_event_id(event)
    history = ledger.setdefault("history", [])
    history.append(event)
    _trim_history(ledger)


def upsert_commitment(ledger: Dict[str, Any], commitment: Dict[str, Any]) -> Dict[str, Any]:
    commitments = ledger.setdefault("commitments", [])
    cid = commitment.get("commitment_id")
    if not cid:
        commitment["commitment_id"] = _compute_commitment_id(commitment)
        cid = commitment["commitment_id"]
    for existing in commitments:
        if existing.get("commitment_id") == cid:
            existing.update(commitment)
            existing["updated_utc"] = _utc_now()
            _clamp_success_criteria(existing)
            break
    else:
        if "created_utc" not in commitment:
            commitment["created_utc"] = _utc_now()
        commitment.setdefault("updated_utc", commitment["created_utc"])
        _clamp_success_criteria(commitment)
        commitments.append(commitment)
    ledger["updated_utc"] = _utc_now()
    _trim_commitments(ledger)
    return ledger


def set_active_commitment(ledger: Dict[str, Any], commitment_id: str) -> None:
    commitments = ledger.get("commitments", [])
    for item in commitments:
        if item.get("commitment_id") == commitment_id:
            item["status"] = "active"
            item["updated_utc"] = _utc_now()
        elif item.get("status") == "active" and item.get("commitment_id") != commitment_id:
            item["status"] = "deferred"
            item["updated_utc"] = _utc_now()
            record_event(ledger, "defer", item.get("commitment_id", ""), "Auto-deferred due to new activation")
    ledger["active_commitment_id"] = commitment_id
    record_event(ledger, "activate", commitment_id, "Commitment activated")
    ledger["updated_utc"] = _utc_now()


def mark_commitment_status(
    ledger: Dict[str, Any], commitment_id: str, status: str, reason: str
) -> None:
    commitments = ledger.get("commitments", [])
    reason = _clamp_text(reason, SUMMARY_MAX)
    for item in commitments:
        if item.get("commitment_id") == commitment_id:
            item["status"] = status
            item["updated_utc"] = _utc_now()
            if status in {"blocked", "deferred"}:
                blockers = item.setdefault("blockers", [])
                if reason:
                    blockers.append(_clamp_text(reason, BLOCKER_MAX))
                    if len(blockers) > MAX_COMMITMENTS:
                        item["blockers"] = blockers[-MAX_COMMITMENTS:]
            break
    record_event(ledger, status if status in {"blocked", "deferred"} else "update", commitment_id, reason or status)
    if status != "active" and ledger.get("active_commitment_id") == commitment_id:
        ledger["active_commitment_id"] = None
    ledger["updated_utc"] = _utc_now()


def write_ledger(path: Path | str, ledger: Dict[str, Any]) -> Tuple[str, Path]:
    if isinstance(path, (str, os.PathLike)):
        path = Path(path)
    ledger_hash = compute_ledger_hash(ledger)
    ledger.setdefault("integrity", {})["ledger_hash"] = ledger_hash
    atomic_write_json(path, ledger)
    return ledger_hash, path


def ensure_active_commitment(
    ledger: Dict[str, Any],
    identity: Dict[str, Any],
    uwm_ref: Optional[str],
    planner_digest_ref: Optional[str],
) -> Dict[str, Any]:
    if ledger.get("active_commitment_id"):
        return ledger
    mission = identity.get("mission", {})
    constraints_defaults = {
        "allow_enhancements": bool(mission.get("allow_enhancements", False)),
        "safe_only": bool(mission.get("safe_interfaces_only", False)),
        "read_only": bool(mission.get("safe_interfaces_only", False)),
    }
    justification = "Maintain integrity using latest world model and planner snapshots."
    evidence_refs: List[str] = []
    if uwm_ref:
        evidence_refs.append(uwm_ref)
    if planner_digest_ref:
        evidence_refs.append(planner_digest_ref)
    if not evidence_refs:
        evidence_refs.append("identity:state/agent_identity.json")
    commitment = {
        "commitment_id": None,
        "title": "Maintain system integrity and completeness",
        "type": "repair",
        "status": "active",
        "priority": 1,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "justification": _clamp_text(justification, JUSTIFICATION_MAX),
        "evidence_refs": evidence_refs,
        "constraints": _ensure_constraints({}, constraints_defaults),
        "success_criteria": [
            "No identity validation failures this cycle",
            "UWM snapshot updated and bound",
            "No orphan artifacts detected",
        ],
        "blockers": [],
        "provenance": {
            "identity_hash": identity.get("identity_hash"),
            "cycle_utc": _utc_now(),
            "run_id": None,
        },
    }
    upsert_commitment(ledger, commitment)
    cid = commitment["commitment_id"]
    record_event(ledger, "create", cid, "Default integrity commitment created")
    set_active_commitment(ledger, cid)
    ledger["updated_utc"] = _utc_now()
    return ledger


@dataclass
class LedgerUpdateResult:
    ledger: Dict[str, Any]
    ledger_hash: str
    ledger_path: Path


def mark_cycle_outcome(
    ledger: Dict[str, Any],
    commitment_id: Optional[str],
    succeeded: bool,
    details: str,
) -> None:
    if not commitment_id:
        return
    details = _clamp_text(details, SUMMARY_MAX)
    action = "advance" if succeeded else "block"
    record_event(ledger, action, commitment_id, details)
    if not succeeded:
        blockers = next(
            (item.setdefault("blockers", []) for item in ledger.get("commitments", []) if item.get("commitment_id") == commitment_id),
            None,
        )
        if blockers is not None and details:
            blockers.append(_clamp_text(details, BLOCKER_MAX))
    ledger["updated_utc"] = _utc_now()


def _clamp_success_criteria(commitment: Dict[str, Any]) -> None:
    criteria = commitment.get("success_criteria")
    if not isinstance(criteria, list):
        commitment["success_criteria"] = []
        return
    trimmed = []
    for entry in criteria:
        if not entry:
            continue
        trimmed.append(_clamp_text(str(entry), SUMMARY_MAX))
        if len(trimmed) >= 10:
            break
    commitment["success_criteria"] = trimmed