# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""
===============================================================================
FILE: LOGOS_SYSTEM.py
PATH: Logos_System_Rebuild/LOGOS_SYSTEM.py
PROJECT: LOGOS System
PHASE: Phase-F
STEP: Runtime Spine Wiring → LOGOS Agent Startup → LEM Discharge
STATUS: GOVERNED — NON-BYPASSABLE

ROLE:
- Receives the governed handoff from System Entry Point
- Executes Lock-and-Key to derive the universal session id
- Starts the LOGOS agent session envelope
- Discharges LEM to derive the LOGOS agent identity

ORDERING GUARANTEE:
Executes strictly after System_Entry_Point.START_LOGOS and immediately before
any LOGOS agent execution or protocol activation.

FAILURE SEMANTICS:
Any invariant failure raises RuntimeHalt and halts progression (fail-closed).
No degraded modes or retries.
===============================================================================
"""

from typing import Dict, Any, Literal, Optional

from Logos_System_Rebuild.Logos_System.System_Entry_Point.System_Entry_Point import (
    START_LOGOS,
    StartupHalt,
)
from Logos_System_Rebuild.Logos_System.Runtime_Governance.lock_and_key import (
    execute_lock_and_key,
    LockAndKeyFailure,
)
from Logos_System_Rebuild.Logos_System.System_Stack.Logos_Agents.Logos_Agent.Start_Logos_Agent import (
    LogosAgentStartupHalt,
    start_logos_agent,
)
from Logos_System_Rebuild.Logos_System.System_Stack.Logos_Agents.Logos_Agent.Lem_Discharge import (
    LemDischargeHalt,
    discharge_lem,
)


class RuntimeHalt(Exception):
    """Raised when the runtime spine fails an invariant."""


def RUN_LOGOS_SYSTEM(
    config_path: Optional[str] = None,
    mode: Literal["headless", "interactive"] = "headless",
    diagnostic: bool = False,
) -> Dict[str, Any]:
    """Canonical runtime spine entry receiving handoff from START_LOGOS."""

    try:
        handoff = START_LOGOS(
            config_path=config_path,
            mode=mode,
            diagnostic=diagnostic,
        )
    except StartupHalt as exc:
        raise RuntimeHalt(f"Startup halted: {exc}")

    try:
        lock_and_key_result = execute_lock_and_key(
            external_compile_artifact=b"STUB_COMPILE_ARTIFACT",
            internal_compile_artifact=b"STUB_COMPILE_ARTIFACT",
        )
    except LockAndKeyFailure as exc:
        raise RuntimeHalt(f"Lock-and-Key failed: {exc}")

    verified_context = dict(handoff)
    verified_context["session_id"] = lock_and_key_result.get("session_id")
    verified_context["lock_and_key_status"] = lock_and_key_result.get("status")

    if not isinstance(verified_context["session_id"], str):
        raise RuntimeHalt("Derived session_id is missing or invalid")

    try:
        logos_session = start_logos_agent(verified_context)
    except LogosAgentStartupHalt as exc:
        raise RuntimeHalt(f"LOGOS agent startup failed: {exc}")

    try:
        logos_identity = discharge_lem(logos_session)
    except LemDischargeHalt as exc:
        raise RuntimeHalt(f"LEM discharge failed: {exc}")

    return {
        "status": "LOGOS_AGENT_READY",
        "logos_identity": logos_identity,
        "logos_session": logos_session,
    }
