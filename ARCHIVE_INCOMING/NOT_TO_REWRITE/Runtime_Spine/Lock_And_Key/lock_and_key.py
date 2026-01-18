"""
===============================================================================
FILE: lock_and_key.py
PATH: Logos_System_Rebuild/Runtime_Spine/Lock_And_Key/lock_and_key.py
PROJECT: LOGOS System
PHASE: Phase-D
STEP: 2.3 — Runtime Spine / Lock-and-Key
STATUS: GOVERNED — NON-BYPASSABLE

CLASSIFICATION:
- Runtime Spine Core Module
- First Executable Layer After System Entry Point
- Fail-Closed Authorization Gate

GOVERNANCE:
- System_Entry_Point_Execution_Contract.md
- Runtime_Spine_Lock_And_Key_Execution_Contract.md

ROLE:
This module implements the Lock-and-Key mechanism that authorizes
existence within a LOGOS runtime session.

It performs dual-site compile commutation between:
- External PXL Gate (LEM admitted)
- Internal Runtime Compiler (LEM admitted)

On success, it issues the Universal Session ID.
On failure, it halts execution immediately.

ORDERING GUARANTEE:
No module, protocol, agent, or orchestration layer may execute
before this module completes successfully.

PROHIBITIONS:
- No agent creation
- No protocol activation
- No LEM discharge
- No SOP access
- No external library access
- No recovery or degraded modes

FAILURE SEMANTICS:
Any failure results in immediate halt.
No retries. No bypass. No continuation.

===============================================================================
"""
from typing import Dict, Any
import hashlib
import json
import time


class LockAndKeyFailure(Exception):
    """Raised when Lock-and-Key validation fails."""
    pass


def _hash_compile_artifact(artifact: bytes) -> str:
    """
    Deterministically hash a compile artifact.
    Stub implementation — real compile artifacts wired later.
    """
    return hashlib.sha256(artifact).hexdigest()


def execute_lock_and_key(
    external_compile_artifact: bytes,
    internal_compile_artifact: bytes,
    audit_log_path: str = "boot_sequence_log.json",
) -> Dict[str, Any]:
    """
    Execute Lock-and-Key dual-site validation.

    Inputs:
    - external_compile_artifact: output of PXL Gate compile (LEM admitted)
    - internal_compile_artifact: output of Runtime Compiler compile (LEM admitted)

    Output:
    - Universal Session ID (dict)

    Failure:
    - Any mismatch halts execution immediately.
    """

    external_hash = _hash_compile_artifact(external_compile_artifact)
    internal_hash = _hash_compile_artifact(internal_compile_artifact)

    if external_hash != internal_hash:
        raise LockAndKeyFailure("Compile hash commutation failed.")

    session_id = external_hash  # canonical universal session ID

    audit_entry = {
        "event": "LOCK_AND_KEY_SUCCESS",
        "session_id": session_id,
        "timestamp": time.time(),
    }

    try:
        with open(audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")
    except Exception as e:
        raise LockAndKeyFailure(f"Audit logging failed: {e}")

    return {
        "session_id": session_id,
        "status": "AUTHORIZED",
    }
