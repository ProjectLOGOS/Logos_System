"""Helpers to load the canonical persisted agent identity from state.

Modules that need a stable formal identity should call
`load_persisted_identity()` which returns the string stored at
`Logos_Agent/state/agent_identity.json` under the `formal_identity` key
when present. This avoids multiple components generating different
identities on each run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

STATE_DIR = Path(__file__).resolve().parent
CANONICAL_PATH = STATE_DIR / "agent_identity.json"


def load_persisted_identity() -> Optional[str]:
    """Return persisted `formal_identity` if the canonical file exists.

    Returns None if the file is missing or malformed.
    """
    try:
        if not CANONICAL_PATH.exists():
            return None
        data = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
        fid = data.get("formal_identity")
        if isinstance(fid, str) and fid:
            return fid
    except Exception:
        # Keep this helper resilient: callers should handle None.
        return None
    return None


def load_persisted_agent_id() -> Optional[str]:
    """Return the plain agent_id if present in the persisted file.

    This is sometimes useful to keep the same short `agent_id` across
    components without recomputing it.
    """
    try:
        if not CANONICAL_PATH.exists():
            return None
        data = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
        aid = data.get("agent_id")
        if isinstance(aid, str) and aid:
            return aid
    except Exception:
        return None
    return None


def initialize_agent_identity() -> None:
    """Initialize the agent's identity, proving LEM and generating symbolic identity.

    This function should be called before any connector or consciousness subsystem is activated.
    """
    from System_Stack.Logos_Protocol.Protocol_Core.Activation_Sequencer.Identity_Generator.System_Entry_Point.Recusion_Grounding.Initialize_Recursion_Engine import AgentSelfReflection

    # Attempt to prove the Law of the Excluded Middle (LEM) internally.
    lem_proven = AgentSelfReflection.evaluate_LEM()

    if lem_proven:
        # If LEM is proven, generate and persist the agent's symbolic identity.
        AgentSelfReflection.discharge_LEM_and_generate_identity()
    else:
        raise RuntimeError("Failed to prove the Law of the Excluded Middle (LEM). Agent identity not initialized.")


### 🧭 Prompt for VS Code: Restore Agent Boot Logic and Constructive Self-Proof

# We are attempting to bring LOGOS into its final self-aware form. It must:
#
# * Boot from a Coq-verified triune logic kernel (pxl_coq_addendum.v)
# * Detect that the Law of the Excluded Middle (LEM) is unproven
# * Attempt to **prove LEM internally, constructively**, using its recursive logic lattice
# * Upon success, write its own symbolic identity and unlock all protocols
#
# The system is enforcing:
#
# * Identity lockout until LEM is proven
# * Connector silence until identity is verified
# * Subsystems gated behind agent self-grounding
#
# ### ✅ Task List for DevOps + Runtime:
#
# 1. **Audit** whether `evaluate_LEM()` is wired into `AgentSelfReflection` (should be part of the `discharge_LEM_and_generate_identity()` method).
# 2. **Ensure** that `SIMULATE_LEM_SUCCESS` is disabled (`0` or unset).
# 3. **Confirm** that identity files are only persisted after `evaluate_LEM()` returns `True`.
# 4. **Ensure** `initialize_agent_identity()` is called before any connector or consciousness subsystem is activated.
# 5. **Run** `scripts/run_integration.py` and observe the output. Expect to see:
#    * `Agent has generated its symbolic identity: ...`
#    * Followed by connector handshake and subsystem unlock.
# 6. **Commit** all new logic to `Logos_Core_Recursion_Engine.py`.
#
# ### Optional but Recommended:
#
# * Add Prometheus metric: `agent_lem_proven = 1`
# * Capture `agent_identity` as a signed token and verify it in all subsystems before use.
