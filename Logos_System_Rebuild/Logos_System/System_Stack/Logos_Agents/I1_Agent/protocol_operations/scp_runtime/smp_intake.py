# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from I1_Agent.config.hashing import safe_hash
from I1_Agent.config.schema_utils import require_dict, get_dict, get_list, get_str
from I1_Agent.diagnostics.errors import SchemaError


@dataclass(frozen=True)
class SMPEnvelope:
    """
    Normalized, minimal SMP view for SCP intake.
    SCP should treat SMP as immutable; this is a read-only normalization.
    """
    smp_id: str
    origin_agent: str
    timestamp: float
    input_hash: str
    route_to: str
    final_decision: str
    triadic_scores: Dict[str, float]
    violations: List[str]
    raw: Dict[str, Any]


def load_smp(*, smp: Dict[str, Any]) -> SMPEnvelope:
    """
    Validate + normalize an SMP dict.
    Does not require full schema, only what SCP needs to begin.
    """
    smp = require_dict(smp, "smp")

    smp_id = get_str(smp, "smp_id") or get_str(smp, "id")  # tolerate legacy naming
    if not smp_id:
        raise SchemaError("SMP missing smp_id")

    origin = get_str(smp, "origin_agent") or "I2"
    ts = smp.get("timestamp")
    if not isinstance(ts, (int, float)):
        ts = 0.0

    input_ref = get_dict(smp, "input_reference")
    input_hash = get_str(input_ref, "input_hash")
    if not input_hash:
        input_hash = safe_hash(input_ref or smp.get("original_input") or smp.get("input"))

    route_to = get_str(smp, "route_to") or "SCP"
    final_decision = get_str(smp, "final_decision") or ""

    triadic_scores = smp.get("triadic_scores")
    if not isinstance(triadic_scores, dict):
        triadic_scores = {}

    violations = get_list(smp, "violations")

    return SMPEnvelope(
        smp_id=smp_id,
        origin_agent=origin,
        timestamp=float(ts),
        input_hash=input_hash,
        route_to=route_to,
        final_decision=final_decision,
        triadic_scores={k: float(v) for k, v in triadic_scores.items() if isinstance(v, (int, float))},
        violations=[str(x) for x in violations],
        raw=smp,
    )
