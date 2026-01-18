# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""Governed Tool Proposal Pipeline: generate → validate → approve → register."""

import scripts.system_stack_tbd.need_to_distribute._bootstrap as _bootstrap  # Ensure repo root in sys.path for logos imports

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import (
    canonical_json_hash,
    validate_tool_proposal,
    validate_tool_validation_report,
)

# Add scripts and external to path
scripts_dir = Path(__file__).parent
external_dir = scripts_dir.parent / "external" / "Logos_AGI"
for path_dir in [scripts_dir, external_dir]:
    if str(path_dir) not in sys.path:
        sys.path.insert(0, str(path_dir))


def _timestamp() -> str:
    """ISO timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_current_hashes() -> Dict[str, Any]:
    """Load current provenance hashes."""
    hashes = {}

    # Repo SHA
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=scripts_dir.parent,
        )
        if result.returncode == 0:
            hashes["repo_sha"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        hashes["repo_sha"] = "unknown"

    # Logos_AGI pinned SHA (from external/Logos_AGI)
    logos_agi_dir = scripts_dir.parent / "external" / "Logos_AGI"
    if logos_agi_dir.exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=logos_agi_dir,
            )
            if result.returncode == 0:
                hashes["logos_agi_pinned_sha"] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            hashes["logos_agi_pinned_sha"] = "unknown"
    else:
        hashes["logos_agi_pinned_sha"] = "unavailable"

    # Attestation hash (placeholder)
    hashes["attestation_hash"] = "pending"  # Would be from attestation system

    # State hashes
    scp_state_file = scripts_dir.parent / "state" / "scp_state.json"
    if scp_state_file.exists():
        try:
            with open(scp_state_file) as f:
                scp_state = json.load(f)
            hashes["scp_state_hash"] = scp_state.get("state_hash")
        except (OSError, json.JSONDecodeError):
            pass

    metrics_file = scripts_dir.parent / "state" / "proposal_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            hashes["metrics_state_hash"] = canonical_json_hash(metrics)
        except (OSError, json.JSONDecodeError):
            pass

    return hashes


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate tool proposals using existing invention/optimization."""
    proposals = []

    # For demo purposes, create a dummy proposal
    # In production, this would call the actual invention/optimization functions
    proposal = {
        "schema_version": 1,
        "proposal_id": f"demo_{int(time.time())}",
        "created_at": _timestamp(),
        "created_by": "tool_pipeline_demo",
        "objective_class": args.objective,
        "tool_name": "demo_tool",
        "description": "Demo tool for pipeline testing",
        "inputs_schema": {"message": "string"},
        "outputs_schema": {"response": "string"},
        "code": (
            "def run(inputs):\n"
            "    return {'response': f'Echo: {inputs[\"message\"]}'}\n"
        ),
        "safety": {"no_network": True, "sandbox_only": True, "max_runtime_ms": 200},
        "provenance": _load_current_hashes(),
    }
    proposals.append(proposal)

    # Write proposals
    repo_root = scripts_dir.parent
    proposals_dir = repo_root / "sandbox" / "tool_proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)

    for proposal in proposals:
        proposal_file = proposals_dir / f"{proposal['proposal_id']}.json"
        with open(proposal_file, "w") as f:
            json.dump(proposal, f, indent=2)
        print(f"Generated proposal: {proposal_file}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate tool proposal in sandbox."""
    proposal_file = Path(args.proposal_file)
    if not proposal_file.exists():
        print(f"ERROR: Proposal file not found: {proposal_file}", file=sys.stderr)
        sys.exit(1)

    with open(proposal_file) as f:
        proposal = json.load(f)

    # Validate schema
    try:
        validate_tool_proposal(proposal)
    except ValueError as e:
        print(f"ERROR: Invalid proposal schema: {e}", file=sys.stderr)
        sys.exit(1)

    # Create build directory
    build_dir = (
        scripts_dir.parent
        / "sandbox"
        / "tool_proposals"
        / "_build"
        / proposal["proposal_id"]
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    # Write code to file
    code_file = build_dir / "tool.py"
    with open(code_file, "w") as f:
        f.write(proposal["code"])

    # Run in subprocess with timeout
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "tool.py"],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=proposal["safety"]["max_runtime_ms"] / 1000.0,
            env={"PYTHONPATH": str(build_dir)},  # Isolated
        )
        runtime_ms = int((time.time() - start_time) * 1000)
        exec_ok = result.returncode == 0
        output = result.stdout
        errors = result.stderr.splitlines() if result.stderr else []
    except subprocess.TimeoutExpired:
        runtime_ms = proposal["safety"]["max_runtime_ms"]
        exec_ok = False
        output = ""
        errors = ["Timeout exceeded"]

    # Write validation report
    report = {
        "proposal_id": proposal["proposal_id"],
        "exec_ok": exec_ok,
        "runtime_ms": runtime_ms,
        "output": output,
        "errors": errors,
        "validated_at": _timestamp(),
    }

    validate_tool_validation_report(report)  # Self-validate

    report_file = proposal_file.with_suffix(".validation.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation {'PASSED' if exec_ok else 'FAILED'}: {report_file}")


def cmd_approve(args: argparse.Namespace) -> None:
    """Approve validated proposal and register tool."""
    proposal_file = Path(args.proposal_file)
    validation_file = proposal_file.with_suffix(".validation.json")

    if not proposal_file.exists():
        print(f"ERROR: Proposal file not found: {proposal_file}", file=sys.stderr)
        sys.exit(1)
    if not validation_file.exists():
        print(f"ERROR: Validation report not found: {validation_file}", file=sys.stderr)
        sys.exit(1)

    with open(proposal_file) as f:
        proposal = json.load(f)
    with open(validation_file) as f:
        validation = json.load(f)

    if not validation["exec_ok"]:
        print("ERROR: Cannot approve proposal that failed validation", file=sys.stderr)
        sys.exit(1)

    # Copy to approved
    approved_dir = scripts_dir.parent / "tools" / "approved" / proposal["tool_name"]
    approved_dir.mkdir(parents=True, exist_ok=True)

    # Copy proposal
    approved_proposal = approved_dir / f"{proposal['proposal_id']}.json"
    with open(approved_proposal, "w") as f:
        json.dump(proposal, f, indent=2)

    # Write tool code
    tool_py = approved_dir / "tool.py"
    with open(tool_py, "w") as f:
        f.write(proposal["code"])

    # Compute tool.py hash
    import hashlib

    tool_py_sha256 = hashlib.sha256(proposal["code"].encode("utf-8")).hexdigest()

    # Create APPROVAL.json
    approval_manifest = {
        "tool_name": proposal["tool_name"],
        "proposal_id": proposal["proposal_id"],
        "proposal_hash": canonical_json_hash(proposal),
        "validation_hash": canonical_json_hash(validation),
        "approved_at": _timestamp(),
        "approved_by": args.operator or "unknown",
        "tool_py_sha256": tool_py_sha256,
        "pin": proposal["provenance"],
        "attestation_hash_at_approval": "pending",  # Would be from current attestation
    }

    approval_file = approved_dir / "APPROVAL.json"
    with open(approval_file, "w") as f:
        json.dump(approval_manifest, f, indent=2)

    # Audit log (honors LOGOS_AUDIT_DIR override for tests/sandboxes)
    audit_root = Path(os.environ.get("LOGOS_AUDIT_DIR") or (scripts_dir.parent / "audit"))
    audit_root.mkdir(parents=True, exist_ok=True)
    audit_file = audit_root / "tool_approvals.jsonl"

    audit_entry = {
        "event": "tool_approval",
        "proposal_id": proposal["proposal_id"],
        "tool_name": proposal["tool_name"],
        "proposal_hash": approval_manifest["proposal_hash"],
        "validation_hash": approval_manifest["validation_hash"],
        "operator": args.operator or "unknown",
        "approved_at": approval_manifest["approved_at"],
    }

    with open(audit_file, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")

    print(f"Approved tool: {approved_proposal}")


def cmd_register(args: argparse.Namespace) -> None:
    """Register approved tool in canonical registry."""
    approved_dir = scripts_dir.parent / "tools" / "approved" / args.tool_name
    if not approved_dir.exists():
        print(f"ERROR: Approved tool not found: {approved_dir}", file=sys.stderr)
        sys.exit(1)

    # Find latest proposal
    proposal_files = list(approved_dir.glob("*.json"))
    if not proposal_files:
        print(f"ERROR: No proposal files in {approved_dir}", file=sys.stderr)
        sys.exit(1)

    latest_proposal = max(proposal_files, key=lambda p: p.stat().st_mtime)
    with open(latest_proposal) as f:
        proposal = json.load(f)

    # Load tool code
    tool_py = approved_dir / "tool.py"
    if not tool_py.exists():
        print(f"ERROR: Tool code not found: {tool_py}", file=sys.stderr)
        sys.exit(1)

    # Import and register (this would modify start_agent.TOOLS)
    # For now, just print what would be done
    print(f"Would register tool '{proposal['tool_name']}' from {tool_py}")
    print("Registration requires modifying start_agent.py TOOLS dict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool Proposal Pipeline")
    parser.add_argument(
        "--allow-audit-write",
        action="store_true",
        help="Acknowledge this script will write proposal artifacts and append audit logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate tool proposals")
    gen_parser.add_argument("--objective", required=True, help="Objective class")
    gen_parser.set_defaults(func=cmd_generate)

    # Validate
    val_parser = subparsers.add_parser("validate", help="Validate tool proposal")
    val_parser.add_argument("proposal_file", help="Path to proposal JSON")
    val_parser.set_defaults(func=cmd_validate)

    # Approve
    app_parser = subparsers.add_parser("approve", help="Approve validated proposal")
    app_parser.add_argument("proposal_file", help="Path to proposal JSON")
    app_parser.add_argument("--operator", help="Operator identifier")
    app_parser.set_defaults(func=cmd_approve)

    # Register
    reg_parser = subparsers.add_parser("register", help="Register approved tool")
    reg_parser.add_argument("tool_name", help="Tool name to register")
    reg_parser.set_defaults(func=cmd_register)

    args = parser.parse_args()

    # Operator guardrail: requires explicit acknowledgment for audit writes.
    if os.environ.get("LOGOS_OPERATOR_OK", "").strip() != "1":
        print(
            "ERROR: operator ack required. Set LOGOS_OPERATOR_OK=1 to run this script.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.allow_audit_write:
        print(
            "ERROR: --allow-audit-write is required to write audit/proposal artifacts.",
            file=sys.stderr,
        )
        sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()
