# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED


from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x

def main() -> int:
    # 1) Build an SMP via I2 using the canonical builder.
    from Logos_Agent.I2_Agent.I2_Core.smp import build_smp

    raw = "privation: smoke test input"
    smp = build_smp(
        origin_agent="I2-smoke",
        input_reference={"raw_text": raw},
        classification={"label": "privative-test", "confidence": 0.5},
        analysis={"notes": "smoke"},
        transform_report={"steps": []},
        bridge_passed=True,
        benevolence={"score": 0.0, "flags": []},
        triadic_scores={"logos": 0.5, "ethos": 0.5, "pathos": 0.5},
        final_decision="route",
        violations=[],
        route_to="NONE",  # keep neutral so runtime does not dispatch SCP/ARP
    )

    # 2) Run Logos runtime orchestration
    from Logos_Agent.logos_runtime.coordinator import run_logos_cycle

    smp_dict = smp.to_dict() if hasattr(smp, "to_dict") else smp  # coordinator expects a mapping

    # Optional: force dispatch path via env without changing coordinator logic
    import os
    force = os.environ.get("FORCE_ROUTE_TO")
    if force and isinstance(smp_dict, dict):
        if "route_to" in smp_dict:
            smp_dict["route_to"] = force
        elif "final_decision" in smp_dict:
            smp_dict["final_decision"] = force
        else:
            smp_dict["route_to"] = force

    bundle = run_logos_cycle(smp=smp_dict)

    # 3) Validate minimal bundle shape
    if bundle is None:
        raise SystemExit("LogosBundle was None")

    if is_dataclass(bundle):
        d = asdict(bundle)
    elif isinstance(bundle, dict):
        d = bundle
    else:
        d = {}
        for k in ("smp", "route_summary", "scp_output", "arp_output"):
            if hasattr(bundle, k):
                d[k] = getattr(bundle, k)

    if "smp" not in d:
        raise SystemExit("Bundle missing 'smp'")
    if not any(k in d for k in ("route_summary", "route", "summary")):
        raise SystemExit("Bundle missing route summary (expected route_summary/route/summary)")

    # Lightweight assertions when forcing routes (avoid brittle schema checks)
    import os
    force = os.environ.get("FORCE_ROUTE_TO")
    if force == "SCP":
        rs = d.get("route_summary") or d.get("route") or d.get("summary") or {}
        if not (isinstance(rs, dict) and rs.get("called_scp") is True):
            raise SystemExit("Expected called_scp=true in route summary when FORCE_ROUTE_TO=SCP")
        scp = d.get("scp_output") or d.get("scp_result")
        if scp in (None, {}, []):
            raise SystemExit("Expected non-empty scp output when FORCE_ROUTE_TO=SCP")
    if force == "ARP":
        rs = d.get("route_summary") or d.get("route") or d.get("summary") or {}
        if not (isinstance(rs, dict) and rs.get("called_arp") is True):
            raise SystemExit("Expected called_arp=true in route summary when FORCE_ROUTE_TO=ARP")
        arp = d.get("arp_output") or d.get("arp_result")
        if arp in (None, {}, []):
            raise SystemExit("Expected non-empty arp output when FORCE_ROUTE_TO=ARP")

    # Optional: SOP stub evaluation (no enforcement), controlled by env flag
    import os
    if os.environ.get("RUN_SOP_STUB") == "1":
        from Logos_Agent.sop_runtime import DefaultSOPPolicyEngine, SOPContext
        engine = DefaultSOPPolicyEngine()
        env = engine.evaluate(bundle, SOPContext(channel="smoke", tags=["sop-stub"]))
        # Attach minimal SOP fields into printed output without mutating the bundle
        d["sop_stub"] = {
            "decision": getattr(env.decision, "value", str(env.decision)),
            "reasons": list(env.reasons),
            "policy_version": env.policy_version,
        }

    print(json.dumps(_jsonable(d), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
