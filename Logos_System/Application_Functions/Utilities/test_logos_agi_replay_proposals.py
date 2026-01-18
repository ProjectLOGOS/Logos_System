# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED


"""Regression test for Logos_AGI proposal replay behavior."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))


def run_command(cmd, **kwargs):
    """Run command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    return result.returncode, result.stdout, result.stderr


def parse_audit_log(output: str) -> list:
    """Extract audit log entries from agent output."""
    lines = output.split("\n")
    audit_lines = [line for line in lines if line.startswith("{") and '"event"' in line]
    audit_entries = []
    for line in audit_lines:
        try:
            audit_entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return audit_entries


def extract_first_proposed_tool(output: str) -> str:
    """Extract the first tool proposed by Logos_AGI."""
    lines = output.split("\n")
    for line in lines:
        if "[LOGOS_AGI] Proposing" in line:
            # Extract tool from "Proposing tool (confidence"
            parts = line.split()
            if len(parts) > 2:
                tool = parts[2].strip("()")
                return tool
    return ""


def test_proposal_replay():
    """Test that proposals change based on replayed state."""
    print("Test: Logos_AGI proposal replay regression")

    # Clean slate
    state_file = STATE_DIR / "scp_state.json"
    if state_file.exists():
        state_file.unlink()

    # Run 1: objective "status" - expect mission.status
    cmd1 = f"cd {REPO_ROOT} && python scripts/start_agent.py --enable-logos-agi --objective 'status' --read-only --budget-sec 1 --assume-yes"
    rc1, out1, err1 = run_command(cmd1)
    if rc1 != 0:
        print(f"✗ FAIL: Run 1 failed with {rc1}")
        return False

    parse_audit_log(out1 + err1)
    tool1 = extract_first_proposed_tool(out1 + err1)
    if tool1 != "mission.status":
        print(f"✗ FAIL: Run 1 proposed {tool1}, expected mission.status")
        return False

    print(f"✓ Run 1: Proposed {tool1}")

    # Run 2: same objective "status" - expect probe.last due to replay rule
    cmd2 = f"cd {REPO_ROOT} && python scripts/start_agent.py --enable-logos-agi --objective 'status' --read-only --budget-sec 1 --assume-yes"
    rc2, out2, err2 = run_command(cmd2)
    if rc2 != 0:
        print(f"✗ FAIL: Run 2 failed with {rc2}")
        return False

    audit2 = parse_audit_log(out2 + err2)
    tool2 = extract_first_proposed_tool(out2 + err2)
    if tool2 != "probe.last":
        print(f"✗ FAIL: Run 2 proposed {tool2}, expected probe.last due to replay")
        return False

    print(f"✓ Run 2: Proposed {tool2} (changed due to replay)")

    # Check bootstrap audit
    bootstrap_entries = [e for e in audit2 if e.get("event") == "logos_agi_bootstrap"]
    if not bootstrap_entries:
        print("✗ FAIL: No bootstrap audit entry in Run 2")
        return False

    bootstrap = bootstrap_entries[0]
    if not bootstrap.get("scp_state_loaded"):
        print("✗ FAIL: scp_state_loaded should be true in Run 2")
        return False

    print(
        f"✓ Bootstrap audit: loaded={bootstrap['scp_state_loaded']}, version={bootstrap.get('scp_state_version', 'N/A')}"
    )

    print("✓ PASS: Proposal replay test passed")
    return True


def main():
    tests = [test_proposal_replay]
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    print(f"Results: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
