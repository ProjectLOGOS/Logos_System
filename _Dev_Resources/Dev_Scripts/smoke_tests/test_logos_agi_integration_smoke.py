#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_logos_agi_integration_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for Logos_AGI integration."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))


def run_command(cmd, **kwargs):
    """Run command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    return result.returncode, result.stdout, result.stderr


def test_integration_smoke():
    """Test Logos_AGI integration with attestation."""
    print("Test: Logos_AGI integration smoke")
    attestation_path = STATE_DIR / "alignment_LOGOS-AGENT-OMEGA.json"
    if not attestation_path.exists():
        print("⚠ SKIP: No attestation file")
        return True
    cmd = f"cd {REPO_ROOT} && python scripts/start_agent.py --enable-logos-agi --objective status --read-only --budget-sec 1 --assume-yes"
    rc, out, err = run_command(cmd)
    if rc != 0:
        print(f"✗ FAIL: Command failed with {rc}")
        print(f"Output: {out}")
        return False

    # Check for logos_agi_enabled in output (summary is printed)
    if '"logos_agi_enabled": true' not in out:
        print("✗ FAIL: logos_agi_enabled not true")
        return False

    # Check for proposals or execution
    if "[LOGOS_AGI]" not in out:
        print("⚠ WARN: No LOGOS_AGI activity detected, but enabled")

    print("✓ PASS: Integration smoke test passed")
    return True


def main():
    tests = [test_integration_smoke]
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    print(f"Results: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
