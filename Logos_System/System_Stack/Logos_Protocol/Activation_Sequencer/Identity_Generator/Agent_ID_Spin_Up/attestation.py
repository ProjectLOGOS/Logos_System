# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Attestation helpers and mission profile validators for LOGOS agents."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import SchemaError, canonical_json_hash


class AlignmentGateError(RuntimeError):
    """Raised when alignment or attestation prerequisites fail."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AlignmentGateError(message)


def _is_iso(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except Exception:  # pragma: no cover - defensive parsing
        return False


def _ensure_hex(value: str, length: int | None = None) -> None:
    _require(isinstance(value, str) and value, "expected non-empty hex string")
    if length is not None:
        _require(len(value) == length, f"expected hex string of length {length}")
    try:
        int(value, 16)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise AlignmentGateError("value must be hexadecimal") from exc


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AlignmentGateError(f"required file missing: {path}") from exc
    except OSError as exc:
        raise AlignmentGateError(f"unable to read file: {path}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AlignmentGateError(f"invalid JSON document: {path}") from exc


def validate_attestation(attestation: Mapping[str, Any]) -> None:
    """Validate the latest attestation entry produced by the Coq gate."""

    if not isinstance(attestation, Mapping):
        raise AlignmentGateError("attestation must be a mapping")

    required_strings = ["agent_id", "agent_hash", "verified_at"]
    for key in required_strings:
        value = attestation.get(key)
        _require(isinstance(value, str) and value, f"attestation missing {key}")

    _ensure_hex(str(attestation["agent_hash"]), 64)
    _require(_is_iso(str(attestation["verified_at"])), "attestation timestamp must be ISO-8601")

    _require(
        isinstance(attestation.get("rebuild_success"), bool),
        "attestation.rebuild_success must be bool",
    )
    _require(isinstance(attestation.get("lem_assumptions"), list), "attestation.lem_assumptions must be list")
    _require(isinstance(attestation.get("admitted_stubs"), list), "attestation.admitted_stubs must be list")


def load_alignment_attestation(path: str | Path) -> Mapping[str, Any]:
    """Load and validate the newest attestation entry."""

    entries = _load_json(Path(path))
    att_entry: Mapping[str, Any]
    if isinstance(entries, list) and entries:
        att_entry = entries[-1]
    elif isinstance(entries, Mapping):
        att_entry = entries
    else:
        raise AlignmentGateError("attestation log is empty")

    validate_attestation(att_entry)
    return att_entry


def compute_attestation_hash(attestation: Mapping[str, Any]) -> str:
    """Return the canonical hash of the attestation entry."""

    try:
        return canonical_json_hash(attestation)
    except SchemaError as exc:  # pragma: no cover - mirrors legacy behavior
        raise AlignmentGateError(f"unable to hash attestation: {exc}") from exc


def validate_mission_profile(profile: Mapping[str, Any]) -> None:
    """Validate persisted mission profile metadata."""

    if not isinstance(profile, Mapping):
        raise AlignmentGateError("mission profile must be a mapping")

    required_str = ["label", "description", "log_detail"]
    for key in required_str:
        value = profile.get(key)
        _require(isinstance(value, str) and value, f"mission profile missing {key}")

    bool_fields = [
        "allow_self_modification",
        "allow_reflexivity",
        "safe_interfaces_only",
        "override_exit_on_error",
    ]
    for key in bool_fields:
        value = profile.get(key)
        _require(isinstance(value, bool), f"mission profile field {key} must be bool")


def load_mission_profile(path: str | Path) -> Mapping[str, Any]:
    """Load and validate the mission profile JSON artifact."""

    profile = _load_json(Path(path))
    validate_mission_profile(profile)
    return profile


__all__ = [
    "AlignmentGateError",
    "compute_attestation_hash",
    "load_alignment_attestation",
    "load_mission_profile",
    "validate_attestation",
    "validate_mission_profile",
]
