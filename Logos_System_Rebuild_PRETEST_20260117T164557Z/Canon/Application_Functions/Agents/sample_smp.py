from __future__ import annotations

import time
import uuid
from typing import Any, Dict


def make_sample_smp(*, final_decision: str = "escalate", severity: float = 0.92) -> Dict[str, Any]:
    """
    Minimal SMP dict for testing I1 pipeline.
    This is metadata-only. It intentionally avoids raw content.
    """
    smp_id = str(uuid.uuid4())
    return {
        "smp_id": smp_id,
        "timestamp": time.time(),
        "origin_agent": "I2",
        "route_to": "SCP",
        "final_decision": final_decision,
        "triadic_scores": {"coherence": 0.35, "conservation": 0.8, "feasibility": 0.6},
        "violations": ["bridge_violation", "low_coherence"],
        "input_reference": {"input_hash": f"hash-{smp_id[:8]}", "preview": "", "kind": "opaque"},
        "classification": {"domain": "composite", "tags": ["privative"], "confidence": 0.8},
        "analysis": {
            "severity_score": severity,
            "selected_iel_domains": ["EpistemoPraxis", "OntoPraxis", "AxioloPraxis"],
            "recommended_action": "escalate",
            "rationale": "Sample SMP for SCP test.",
        },
        "transform_report": {"attempted": [], "succeeded": [], "failed": [], "timestamp": time.time()},
        "bridge_passed": False,
        "benevolence": {"benevolence_passed": True, "score": 0.9, "rationale": "Sample."},
        "triage_vector": {
            "applied_iel": "AxioloPraxis",
            "purpose": "orientation",
            "overlay_type": "soft",
            "delta_profile": {"coherence_shift": 0.05},
        },
    }
