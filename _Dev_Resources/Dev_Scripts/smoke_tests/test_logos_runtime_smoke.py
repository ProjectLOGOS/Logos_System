# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_logos_runtime_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

# Ensure repo imports resolve similarly to other smoke tests
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PYTHONPATH", f"{REPO_ROOT}:external/Logos_AGI")


def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def build_minimal_smp() -> Dict[str, Any]:
    """
    Construct the smallest SMP that satisfies the Logos coordinator + SCP intake expectations.
    Keeps fields aligned with I1/I2 heuristics (route_to, final_decision, triadic scores, input hash).
    """
    smp_id = str(uuid.uuid4())
    return {
        "smp_id": smp_id,
        "timestamp": time.time(),
        "origin_agent": "I2",
        "route_to": "SCP",  # triggers SCP path; ARP stays off unless explicitly requested
        "final_decision": "observe",
        "triadic_scores": {"coherence": 0.62, "conservation": 0.7, "feasibility": 0.65},
        "violations": [],
        "input_reference": {"input_hash": f"hash-{smp_id[:8]}", "preview": "", "kind": "opaque"},
        "classification": {"domain": "composite", "tags": ["privative_smoke"], "constraints": ["append_only_packets"]},
        "analysis": {
            "severity_score": 0.4,
            "selected_iel_domains": ["EpistemoPraxis"],
            "recommended_action": "monitor",
            "rationale": "Smoke test baseline.",
        },
        "triage_vector": {
            "applied_iel": "EpistemoPraxis",
            "purpose": "smoke_test",
            "overlay_type": "soft",
            "delta_profile": {"coherence_shift": 0.0},
        },
    }


def main() -> int:
    from Logos_Agent.logos_runtime.coordinator import run_logos_cycle  # type: ignore

    smp = build_minimal_smp()
    bundle = run_logos_cycle(smp=smp)

    if bundle is None:
        raise SystemExit("LogosBundle was None")

    if hasattr(bundle, "to_dict"):
        out = bundle.to_dict()  # type: ignore[attr-defined]
    elif is_dataclass(bundle):
        out = asdict(bundle)
    elif isinstance(bundle, dict):
        out = bundle
    else:
        out = {k: getattr(bundle, k) for k in ("smp", "route_summary", "scp_result", "arp_result") if hasattr(bundle, k)}

    if not isinstance(out, dict):
        raise SystemExit("Unexpected bundle shape")
    if "smp" not in out:
        raise SystemExit("Bundle missing 'smp'")
    if "route_summary" not in out:
        raise SystemExit("Bundle missing 'route_summary'")

    print(json.dumps(_to_jsonable(out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
