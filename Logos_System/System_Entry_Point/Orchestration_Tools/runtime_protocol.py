# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Minimal runtime protocol hooks for LOGOS bootstrapping.

This module records boot phases, modality transitions, and basic runtime
contract enforcement. It intentionally stays stdlib-only so it can be imported
from the earliest startup code.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Runtime phases (public constants)
PHASE_0_PATH_SETUP = "PHASE_0_PATH_SETUP"
PHASE_1_PROOF_GATE = "PHASE_1_PROOF_GATE"
PHASE_2_IDENTITY_AUDIT = "PHASE_2_IDENTITY_AUDIT"
PHASE_3_TELEMETRY_DASHBOARD = "PHASE_3_TELEMETRY_DASHBOARD"
PHASE_4_UI_SERVICES = "PHASE_4_UI_SERVICES"
PHASE_5_STACK_LOAD = "PHASE_5_STACK_LOAD"
PHASE_6_SIGNAL_LOOP = "PHASE_6_SIGNAL_LOOP"

# Modalities
MODALITY_PASSIVE = "PASSIVE"
MODALITY_ACTIVE = "ACTIVE"

# Runtime roots allowed by contract
ALLOWED_RUNTIME_ROOTS = ("START_LOGOS.py", "System_Stack", "PXL_Gate")

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = REPO_ROOT / "state" / "alignment" / "runtime_protocol.jsonl"

_state: Dict[str, Any] = {
    "current_phase": None,
    "modality": None,
    "proof_gate_passed": False,
    "identity_attested": False,
}
_runtime_roots_logged = False


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_log_dir() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_event(event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Append a structured event to the runtime protocol JSONL log."""
    record = {
        "ts": _timestamp(),
        "event": event_type,
        "payload": payload or {},
    }
    _ensure_log_dir()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def ensure_runtime_roots(repo_root: str | Path) -> Dict[str, bool]:
    """Record the presence of all allowed runtime roots (idempotent)."""
    global _runtime_roots_logged
    root_path = Path(repo_root).resolve()
    results = {name: (root_path / name).exists() for name in ALLOWED_RUNTIME_ROOTS}
    if not _runtime_roots_logged:
        log_event("runtime_roots_checked", {"repo_root": str(root_path), "roots": results})
        _runtime_roots_logged = True
    return results


def mark_phase(phase_name: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Update the current phase and emit a log entry."""
    _state["current_phase"] = phase_name
    if phase_name == PHASE_2_IDENTITY_AUDIT:
        _state["proof_gate_passed"] = True
        _state["identity_attested"] = True
    log_event("phase", {"name": phase_name, **(extra or {})})


def mark_modality(modality_name: str) -> None:
    """Record the current modality (PASSIVE/ACTIVE)."""
    _state["modality"] = modality_name
    log_event("modality", {"name": modality_name})


def assert_can_enter_active(current_state: Optional[Dict[str, Any]] = None) -> None:
    """Ensure proof + audit completed before activating user endpoints."""
    state = current_state or _state
    if not state.get("proof_gate_passed") or not state.get("identity_attested"):
        log_event(
            "active_violation",
            {
                "proof_gate_passed": bool(state.get("proof_gate_passed")),
                "identity_attested": bool(state.get("identity_attested")),
            },
        )
        raise RuntimeError(
            "Runtime protocol violation: ACTIVE modality requires proof gate and identity audit"
        )


def current_state() -> Dict[str, Any]:
    """Return a shallow copy of the in-memory runtime protocol state."""
    return dict(_state)
