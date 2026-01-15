#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_tool_approval_tamper_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Test tampering detection for approved tools."""

import json
import subprocess
import sys
from pathlib import Path

# Add scripts and repo root to path
scripts_dir = Path(__file__).parent
repo_root = scripts_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def run_pipeline_cmd(cmd_args):
    """Run pipeline command."""
    cmd = [sys.executable, str(scripts_dir / "tool_proposal_pipeline.py")] + cmd_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=scripts_dir.parent)
    return result


def run_agent_cmd(cmd_args):
    """Run start_agent command."""
    cmd = [sys.executable, str(scripts_dir / "start_agent.py")] + cmd_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=scripts_dir.parent)
    return result


def test_tamper_detection():
    """Test that tampering is detected."""
    repo_root = scripts_dir.parent
    proposals_dir = repo_root / "sandbox" / "tool_proposals"
    approved_dir = repo_root / "tools" / "approved"

    # Clean up
    import shutil

    if proposals_dir.exists():
        shutil.rmtree(proposals_dir)
    if approved_dir.exists():
        shutil.rmtree(approved_dir)

    print("1. Generate and approve a tool...")
    result = run_pipeline_cmd(["generate", "--objective", "tamper_test"])
    if result.returncode != 0:
        print(f"Generate failed: {result.stderr}")
        return False

    proposal_files = [
        f
        for f in proposals_dir.glob("*.json")
        if not f.name.endswith(".validation.json")
    ]
    if not proposal_files:
        print("No proposal generated")
        return False

    proposal_file = proposal_files[0]

    result = run_pipeline_cmd(["validate", str(proposal_file)])
    if result.returncode != 0:
        print(f"Validate failed: {result.stderr}")
        return False

    result = run_pipeline_cmd(["approve", str(proposal_file), "--operator", "test"])
    if result.returncode != 0:
        print(f"Approve failed: {result.stderr}")
        return False

    # Get tool name
    with open(proposal_file) as f:
        proposal = json.load(f)
    tool_name = proposal["tool_name"]

    print(f"Approved tool: {tool_name}")

    print("2. Run agent to load tools (should succeed)...")
    result = run_agent_cmd(
        [
            "--objective",
            "status",
            "--no-require-attestation",
            "--read-only",
            "--budget-sec",
            "1",
        ]
    )
    if "Loaded 1 approved tools" not in result.stdout:
        print(f"Tool load failed: {result.stdout} {result.stderr}")
        return False

    print("Tool loaded successfully")

    print("3. Tamper with tool.py...")
    tool_py = approved_dir / tool_name / "tool.py"
    if not tool_py.exists():
        print("tool.py not found")
        return False

    with open(tool_py, "a") as f:
        f.write("\n# tampered\n")

    print("Tampered with tool.py")

    print("4. Run agent again (should fail to load)...")
    result = run_agent_cmd(
        [
            "--objective",
            "status",
            "--no-require-attestation",
            "--read-only",
            "--budget-sec",
            "1",
        ]
    )
    if "tool.py hash mismatch" not in result.stdout:
        print(f"Tampering not detected: {result.stdout} {result.stderr}")
        return False

    print("Tampering detected correctly")

    print("5. Restore tool.py and run again (should succeed)...")
    # Restore from proposal
    with open(tool_py, "w") as f:
        f.write(proposal["code"])

    result = run_agent_cmd(
        [
            "--objective",
            "status",
            "--no-require-attestation",
            "--read-only",
            "--budget-sec",
            "1",
        ]
    )
    if "Loaded 1 approved tools" not in result.stdout:
        print(f"Restore failed: {result.stdout} {result.stderr}")
        return False

    print("Tool restored and loaded successfully")

    print("All tamper tests passed!")
    return True


if __name__ == "__main__":
    success = test_tamper_detection()
    sys.exit(0 if success else 1)
