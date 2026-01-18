# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Deterministic, UIP-gated tool repair proposal generator.

Reads a tool health report and emits repair proposals (schema-compatible) for
DEGRADED/BROKEN tools. No code execution is performed. UIP approval is
explicitly required via CLI decision flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import canonical_json_hash, validate_tool_proposal
DEFAULT_OUTPUT = REPO_ROOT / "sandbox" / "tool_repair_proposals"
AUDIT_ROOT = Path(os.environ.get("LOGOS_AUDIT_DIR") or (REPO_ROOT / "audit"))
AUDIT_LOG = AUDIT_ROOT / "tool_repair_proposals.jsonl"


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_health_report(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _base_provenance() -> Dict[str, Any]:
    # Lightweight provenance; deterministic and read-only
    prov: Dict[str, Any] = {"repo_sha": "unknown", "logos_agi_pinned_sha": "unknown", "attestation_hash": "pending"}
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        )
        if result.returncode == 0:
            prov["repo_sha"] = result.stdout.strip()
    except Exception:
        pass
    return prov


def _build_proposal(tool_name: str, health_entry: Dict[str, Any]) -> Dict[str, Any]:
    ts = _timestamp()
    proposal = {
        "schema_version": 1,
        "proposal_id": f"repair_{tool_name}_{int(time.time())}",
        "created_at": ts,
        "created_by": "tool_repair_proposal",
        "objective_class": "GENERAL",
        "tool_name": tool_name,
        "description": (
            "Repair proposal for tool '{tool}' (health={health}). Issues: {issues}. "
            "No execution without UIP approval."
        ).format(tool=tool_name, health=health_entry.get("health"), issues=", ".join(health_entry.get("issues", []))),
        "inputs_schema": {"args": "string"},
        "outputs_schema": {"result": "string"},
        "code": "# Repair implementation to be authored post-approval\n",
        "safety": {"no_network": True, "sandbox_only": True, "max_runtime_ms": 200},
        "provenance": _base_provenance(),
        "failure_evidence": health_entry.get("evidence_refs", []),
        "governance": {
            "requires_uip": True,
            "decision": None,
            "constraints": ["No autonomous execution", "UIP approval required"],
        },
    }
    # Validate shape deterministically (raises on failure)
    validate_tool_proposal(proposal)
    proposal["proposal_hash"] = canonical_json_hash(proposal)
    return proposal


def _write_audit(entry: Dict[str, Any]) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_repair_proposals(health_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    for tool_name, entry in health_report.get("tools", {}).items():
        if entry.get("health") in {"BROKEN", "DEGRADED"}:
            proposals.append(_build_proposal(tool_name, entry))
    return proposals


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UIP-gated tool repair proposal generator")
    parser.add_argument("--health-report", required=True, help="Path to tool health report JSON")
    parser.add_argument(
        "--uip-decision",
        choices=["approve", "reject", "defer"],
        required=True,
        help="UIP operator decision (no silent approvals)",
    )
    parser.add_argument("--operator", default="unknown", help="Operator id for audit")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory to write proposals (on approve)")
    parser.add_argument(
        "--allow-audit-write",
        action="store_true",
        help="Acknowledge this script will write repair artifacts and append audit logs.",
    )
    args = parser.parse_args(argv)

    if os.environ.get("LOGOS_OPERATOR_OK", "").strip() != "1":
        print(
            "ERROR: operator ack required. Set LOGOS_OPERATOR_OK=1 to run this script.",
            file=sys.stderr,
        )
        return 2
    if not args.allow_audit_write:
        print(
            "ERROR: --allow-audit-write is required to write audit/proposal artifacts.",
            file=sys.stderr,
        )
        return 2

    report_path = Path(args.health_report)
    health_report = _load_health_report(report_path)
    proposals = generate_repair_proposals(health_report)

    audit_entry = {
        "timestamp": _timestamp(),
        "decision": args.uip_decision,
        "operator": args.operator,
        "health_report": report_path.as_posix(),
        "tools_considered": [p["tool_name"] for p in proposals],
    }

    if args.uip_decision != "approve":
        audit_entry["note"] = "No proposals written (decision={})".format(args.uip_decision)
        _write_audit(audit_entry)
        print(f"UIP decision={args.uip_decision}; proposals not persisted")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for proposal in proposals:
        outfile = output_dir / f"{proposal['proposal_id']}.json"
        with open(outfile, "w") as f:
            json.dump(proposal, f, indent=2)
        written.append(outfile.as_posix())

    audit_entry["proposals_written"] = written
    _write_audit(audit_entry)
    print(json.dumps({"written": written}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
