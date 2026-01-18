"""
===============================================================================
FILE: Projection_Validator.py
PATH: Logos_System/System_Stack/Runtime_Governance/Projection_Validator.py
PROJECT: LOGOS System
PHASE: Phase-F
STEP: Phase-F-3 — Projection Loader / Validator (Fail-Closed)
STATUS: GOVERNED — NON-BYPASSABLE

CLASSIFICATION:
- Runtime Governance Layer (Projection Validation)

GOVERNANCE:
- Agent_Grounding_Projection_Contract.json
- Projection_Loader_Validator_Contract.json
- Agent_Projection_Schema.json
- Agent_Projection_Integrity_Index.json

ROLE:
Validates agent grounding projections before agent activation. Enforces schema
compliance, hash verification, and authority rules. Fails closed on any
violation.

ORDERING_GUARANTEE:
Executes before agent activation and after projection artifacts are published.

PROHIBITIONS:
- No projection mutation
- No bypass of hash or schema checks
- No dynamic schema changes

FAILURE_SEMANTICS:
Any validation failure halts execution immediately (fail-closed).
===============================================================================
"""

import json
import hashlib
from pathlib import Path

class ProjectionValidationError(Exception):
    pass

def validate_projection(
    agent_identity: str,
    projection_path: Path,
    schema_path: Path,
    integrity_index_path: Path
) -> dict:
    # Load projection
    try:
        projection = json.loads(projection_path.read_text())
    except Exception as e:
        raise ProjectionValidationError(f"Projection load failed: {e}")

    # Agent identity check
    if projection.get("agent_identity") != agent_identity:
        raise ProjectionValidationError("Agent identity mismatch")

    # Schema validation (structural presence only; deep validation later)
    required_fields = json.loads(schema_path.read_text()).get("required_fields", [])
    for field in required_fields:
        if field not in projection:
            raise ProjectionValidationError(f"Missing required field: {field}")

    # Hash verification
    data = projection_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    index = json.loads(integrity_index_path.read_text())
    entry = index["projections"].get(projection_path.name)

    if not entry or entry["sha256"] != digest:
        raise ProjectionValidationError("Projection hash mismatch")

    return projection
