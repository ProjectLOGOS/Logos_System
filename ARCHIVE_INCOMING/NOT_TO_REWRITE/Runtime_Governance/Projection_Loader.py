"""
===============================================================================
FILE: Projection_Loader.py
PATH: Logos_System/System_Stack/Runtime_Governance/Projection_Loader.py
PROJECT: LOGOS System
PHASE: Phase-F
STEP: Phase-F-3 — Projection Loader / Validator (Fail-Closed)
STATUS: GOVERNED — NON-BYPASSABLE

CLASSIFICATION:
- Runtime Governance Layer (Projection Loader)

GOVERNANCE:
- Agent_Grounding_Projection_Contract.json
- Projection_Loader_Validator_Contract.json
- Agent_Projection_Schema.json
- Agent_Projection_Integrity_Index.json

ROLE:
Enforces mandatory projection loading during agent boot. Invokes the
Projection_Validator with fail-closed semantics and provides the validated
projection to the runtime without mutation authority.

ORDERING_GUARANTEE:
Executes during agent boot before activation, after projection artifacts and
integrity indices are available.

PROHIBITIONS:
- No validation bypass
- No projection mutation
- No silent failures

FAILURE_SEMANTICS:
Any loader or validation failure halts execution immediately (fail-closed).
===============================================================================
"""

from pathlib import Path
from .Projection_Validator import validate_projection, ProjectionValidationError

def load_agent_projection(agent_identity: str) -> dict:
    base = Path("Logos_System/System_Stack/Logos_Agents/Agent_Resources/Projections")
    projection_path = base / f"{agent_identity}_Grounding_Projection.json"

    schema_path = Path("Logos_System/_Governance/Agent_Projection_Schema.json")
    integrity_index_path = Path("Logos_System/_reports/Agent_Projection_Integrity_Index.json")

    return validate_projection(
        agent_identity=agent_identity,
        projection_path=projection_path,
        schema_path=schema_path,
        integrity_index_path=integrity_index_path
    )
