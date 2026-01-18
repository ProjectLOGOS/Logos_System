"""
===============================================================================
FILE: lock_and_key.py
PATH: Logos_System/Runtime_Governance/lock_and_key.py
PROJECT: LOGOS System
PHASE: Phase-F (Prelude)
STEP: Runtime Governance Bridge — Lock-and-Key
STATUS: GOVERNED - NON-BYPASSABLE

ROLE:
Bridges to the Phase-D Lock-and-Key implementation for Constructive compile
and LEM discharge pathways. Provides the governed alias for
execute_lock_and_key.

ORDERING GUARANTEE:
Invoked immediately after System Entry Point and before any Constructive
compile or LEM discharge routines.

PROHIBITIONS:
- No agent logic
- No protocol logic
- No execution on import

FAILURE SEMANTICS:
Delegates fail-closed behavior to the underlying module.
===============================================================================
"""

from typing import Dict, Any

from Logos_System_Rebuild.Runtime_Spine.Lock_And_Key.lock_and_key import (
    execute_lock_and_key as _execute_lock_and_key,
    LockAndKeyFailure,
)


def execute_lock_and_key(
    external_compile_artifact: bytes,
    internal_compile_artifact: bytes,
    audit_log_path: str = "boot_sequence_log.json",
) -> Dict[str, Any]:
    """Governed alias for Lock-and-Key (Constructive compile / LEM pathway)."""
    return _execute_lock_and_key(
        external_compile_artifact=external_compile_artifact,
        internal_compile_artifact=internal_compile_artifact,
        audit_log_path=audit_log_path,
    )
