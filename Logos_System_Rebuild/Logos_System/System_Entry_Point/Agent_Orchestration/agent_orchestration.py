# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""
===============================================================================
FILE: agent_orchestration.py
PATH: Logos_System/Agent_Orchestration/agent_orchestration.py
PROJECT: LOGOS System
PHASE: Phase-F (Prelude)
STEP: Runtime Governance Bridge — Agent Orchestration Alias
STATUS: GOVERNED - NON-BYPASSABLE

ROLE:
Provides a governed alias to the Phase-E agent orchestration planning
module. Produces declarative plans only; no execution authority resides here.

ORDERING GUARANTEE:
Invoked after Constructive compile and before any agent plan is executed by
later phases.

PROHIBITIONS:
- No agent logic
- No protocol logic
- No execution on import

FAILURE SEMANTICS:
Delegates fail-closed behavior to the underlying module.
===============================================================================
"""

from typing import Dict, Any

from Logos_System_Rebuild.Runtime_Spine.Agent_Orchestration.agent_orchestration import (
    prepare_agent_orchestration as _prepare_agent_orchestration,
    OrchestrationHalt,
)


def prepare_agent_orchestration(constructive_compile_output: Dict[str, Any]) -> Dict[str, Any]:
    """Governed alias for declarative agent orchestration planning."""
    return _prepare_agent_orchestration(constructive_compile_output)
