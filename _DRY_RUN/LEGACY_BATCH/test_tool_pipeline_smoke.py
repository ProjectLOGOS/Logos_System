# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: FORBIDDEN (DRY_RUN_ONLY)
# AUTHORITY: GOVERNED
# INSTALL_STATUS: DRY_RUN_ONLY
# SOURCE_LEGACY: test_tool_pipeline_smoke.py

"""
DRY-RUN REWRITE

This file is a structural, governed rewrite candidate generated for
rewrite-system validation only. No execution, no side effects.
"""
"""Smoke test for governed tool proposal pipeline."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add scripts and repo root to path
scripts_dir = Path(__file__).parent
repo_root = scripts_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import validate_tool_proposal, validate_tool_validation_report


def run_pipeline_cmd(cmd_args):
    """Run pipeline command and return result."""
    cmd = [
        sys.executable,
        str(scripts_dir / "tool_proposal_pipeline.py"),
        "--allow-audit-write",
    ] + cmd_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=scripts_dir.parent)
    return result


def test_pipeline():
    """Test the full pipeline."""
    repo_root = scripts_dir.parent
    proposals_dir = repo_root / "sandbox" / "tool_proposals"

    if "LOGOS_AUDIT_DIR" not in os.environ:
        os.environ["LOGOS_AUDIT_DIR"] = tempfile.mkdtemp(prefix="logos_smoke_audit_")
    os.environ.setdefault("LOGOS_OPERATOR_OK", "1")
    audit_root = Path(os.environ["LOGOS_AUDIT_DIR"])
    audit_root.mkdir(parents=True, exist_ok=True)

    print("1. Generate proposals...")
    result = run_pipeline_cmd(["generate", "--objective", "test"])
    if result.returncode != 0:
        print(f"Generate failed: {result.stderr}")
        return False
    print("Generate OK")

    # Find generated proposals
    if not proposals_dir.exists():
        print("No proposals directory created")
        return False

    proposal_files = [
        f
        for f in proposals_dir.glob("*.json")
        if not f.name.endswith(".validation.json")
    ]
    if not proposal_files:
        print("No proposal files generated")
        return False

    proposal_file = proposal_files[0]  # Use first one
    print(f"Using proposal: {proposal_file}")

    # Validate proposal schema
    with open(proposal_file) as f:
        proposal = json.load(f)
    try:
        validate_tool_proposal(proposal)
        print("Proposal schema valid")
    except Exception as e:
        print(f"Invalid proposal: {e}")
        return False

    print("2. Validate proposal...")
    result = run_pipeline_cmd(["validate", str(proposal_file)])
    if result.returncode != 0:
        print(f"Validate failed: {result.stderr}")
        return False

    validation_file = proposal_file.with_suffix(".validation.json")
    if not validation_file.exists():
        print("Validation report not created")
        return False

    with open(validation_file) as f:
        validation = json.load(f)
    try:
        validate_tool_validation_report(validation)
        if not validation["exec_ok"]:
            print(f"Validation failed: {validation['errors']}")
            return False
        print("Validation passed")
    except Exception as e:
        print(f"Invalid validation report: {e}")
        return False

    print("3. Attempt execution before approval (should fail)...")
    # Try to execute via start_agent (mock)
    # Since we can't easily run start_agent, check if tool is in TOOLS
    from scripts.system_stack_tbd.could_be_dev.start_agent import TOOLS

    tool_name = proposal["tool_name"]
    if tool_name in TOOLS:
        print(f"Tool {tool_name} already registered - test invalid")
        return False
    print("Tool not registered (good)")

    print("4. Approve proposal...")
    result = run_pipeline_cmd(
        ["approve", str(proposal_file), "--operator", "test_operator"]
    )
    if result.returncode != 0:
        print(f"Approve failed: {result.stderr}")
        return False

    approved_dir = repo_root / "tools" / "approved" / tool_name
    if not approved_dir.exists():
        print("Approved tool directory not created")
        return False

    approved_proposal = approved_dir / f"{proposal['proposal_id']}.json"
    if not approved_proposal.exists():
        print("Approved proposal not copied")
        return False

    tool_py = approved_dir / "tool.py"
    if not tool_py.exists():
        print("Tool code not copied")
        return False

    approval_file = approved_dir / "APPROVAL.json"
    if not approval_file.exists():
        print("APPROVAL.json not created")
        return False

    with open(approval_file) as f:
        manifest = json.load(f)
    if manifest["tool_name"] != tool_name:
        print("APPROVAL.json tool_name mismatch")
        return False

    audit_file = audit_root / "tool_approvals.jsonl"
    if not audit_file.exists():
        print("Audit log not created")
        return False

    print("Approval OK")

    print("5. Check tool loads correctly...")
    # Simulate loading
    from logos.tool_registry_loader import load_approved_tools

    load_approved_tools(TOOLS)

    from scripts.system_stack_tbd.could_be_dev.start_agent import TOOLS

    print(f"TOOLS keys: {list(TOOLS.keys())}")
    if tool_name not in TOOLS:
        print(f"Tool {tool_name} not in TOOLS after load")
        return False
    print("Tool loaded")

    print("6. Check audit logs...")
    with open(audit_file) as f:
        lines = f.readlines()
    if not lines:
        print("No audit entries")
        return False

    audit_entry = json.loads(lines[-1])
    if audit_entry["proposal_id"] != proposal["proposal_id"]:
        print("Audit entry mismatch")
        return False

    required_hashes = ["proposal_hash", "validation_hash"]
    for h in required_hashes:
        if h not in audit_entry:
            print(f"Missing {h} in audit")
            return False

    print("Audit OK")

    print("All tests passed!")
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)