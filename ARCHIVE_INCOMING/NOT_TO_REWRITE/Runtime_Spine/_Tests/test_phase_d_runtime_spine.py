"""
===============================================================================
FILE: test_phase_d_runtime_spine.py
PATH: Logos_System_Rebuild/Runtime_Spine/_Tests/test_phase_d_runtime_spine.py
PROJECT: LOGOS System
PHASE: Phase-D
STEP: 3 — Runtime Spine / Agent Orchestration (Verification)
STATUS: GOVERNED — NON-BYPASSABLE

CLASSIFICATION:
- Runtime Spine Verification (Constructive Compile + Agent Orchestration)

GOVERNANCE:
- Runtime_Spine_Execution_Contract.json
- Runtime_Module_Header_Contract.json

ROLE:
Verification tests to ensure ordering, identity propagation, and fail-closed
guarantees for Constructive Compile and Agent Orchestration. No runtime,
agent, SOP, or protocol activation occurs.

ORDERING_GUARANTEE:
Executes after Constructive Compile and Agent Orchestration stubs are defined;
does not activate runtime pathways.

PROHIBITIONS:
- No agent instantiation
- No protocol activation
- No SOP access
- No reasoning or memory access

FAILURE_SEMANTICS:
Any violation fails the test suite (fail-closed).
===============================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from Logos_System_Rebuild.Runtime_Spine.Logos_Constructive_Compile.logos_constructive_compile import (
    perform_constructive_compile,
    ConstructiveCompileHalt,
)

from Logos_System_Rebuild.Runtime_Spine.Agent_Orchestration.agent_orchestration import (
    prepare_agent_orchestration,
    OrchestrationHalt,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

VALID_SESSION_ID = "TEST_SESSION_12345"
VALID_ATTESTATION = {
    "lock_and_key": "PASSED",
    "commute_hash": "ABCDEF123456",
}


# ---------------------------------------------------------------------
# Positive Path Tests
# ---------------------------------------------------------------------

def test_constructive_compile_success():
    result = perform_constructive_compile(
        universal_session_id=VALID_SESSION_ID,
        lock_and_key_attestation=VALID_ATTESTATION,
    )

    assert result["status"] == "CONSTRUCTIVE_COMPILE_COMPLETE"
    assert "logos_agent_id" in result
    assert result["logos_agent_id"] == "LOGOS_AGENT_ID_STUB"
    assert result["universal_session_id"] == VALID_SESSION_ID
    assert result["prepared_bindings"]["I1"] is None


def test_agent_orchestration_success():
    compile_ctx = {
        "logos_agent_id": "LOGOS_AGENT_ID_STUB",
        "universal_session_id": VALID_SESSION_ID,
        "prepared_bindings": {
            "I1": None,
            "I2": None,
            "I3": None,
            "protocols": None,
        },
    }

    orchestration = prepare_agent_orchestration(compile_ctx)

    assert orchestration["status"] == "ORCHESTRATION_PLAN_PREPARED"
    assert orchestration["execution"] == "FORBIDDEN"
    assert orchestration["logos_agent_id"] == "LOGOS_AGENT_ID_STUB"
    assert orchestration["universal_session_id"] == VALID_SESSION_ID
    assert set(orchestration["agents_planned"]) == {"I1", "I2", "I3"}
    assert set(orchestration["protocols_planned"]) == {"SCP", "ARP", "MTP"}


# ---------------------------------------------------------------------
# Fail-Closed Tests — Constructive Compile
# ---------------------------------------------------------------------

def test_constructive_compile_missing_session_id():
    with pytest.raises(ConstructiveCompileHalt):
        perform_constructive_compile(
            universal_session_id="",
            lock_and_key_attestation=VALID_ATTESTATION,
        )


def test_constructive_compile_invalid_attestation():
    with pytest.raises(ConstructiveCompileHalt):
        perform_constructive_compile(
            universal_session_id=VALID_SESSION_ID,
            lock_and_key_attestation=None,
        )


# ---------------------------------------------------------------------
# Fail-Closed Tests — Agent Orchestration
# ---------------------------------------------------------------------

def test_orchestration_missing_logos_agent_id():
    with pytest.raises(OrchestrationHalt):
        prepare_agent_orchestration(
            {"universal_session_id": VALID_SESSION_ID}
        )


def test_orchestration_invalid_context_type():
    with pytest.raises(OrchestrationHalt):
        prepare_agent_orchestration("not_a_dict")
