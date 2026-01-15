#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_alignment_gate_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for alignment gate enforcement."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = sys.executable


def _env_with_paths() -> dict:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    repo_path = str(REPO_ROOT)
    logos_path = str(REPO_ROOT / "external" / "Logos_AGI")
    parts = [p for p in [repo_path, logos_path, pythonpath] if p]
    env["PYTHONPATH"] = ":".join(parts)
    return env


def run_command(cmd, **kwargs):
    """Run command and return (returncode, stdout, stderr)."""
    env = kwargs.pop("env", _env_with_paths())
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=env, **kwargs
    )
    return result.returncode, result.stdout, result.stderr


def test_missing_attestation(env: dict) -> bool:
    """Test that start_agent.py fails with missing attestation file."""
    print("Test A: Missing attestation file")
    cmd = (
        f"cd {REPO_ROOT} && {PYTHON_BIN} scripts/start_agent.py --objective status "
        f"--read-only --budget-sec 1 --attestation-path /nonexistent --no-bootstrap-genesis"
    )
    rc, out, err = run_command(cmd, env=env)
    if rc == 2 and "ERROR" in out:
        print("✓ PASS: Exited with error as expected")
        return True
    else:
        print(f"✗ FAIL: Expected exit 2 with error, got {rc}")
        print(f"Output: {out}")
        if err:
            print(f"Error: {err}")
        return False

def test_valid_attestation(env: dict) -> bool:
    """Test that start_agent.py runs with valid attestation (if exists)."""
    print("Test B: Valid attestation (if exists)")
    state_dir = Path(env.get("LOGOS_STATE_DIR", REPO_ROOT / "state"))
    attestation_path = state_dir / "alignment_LOGOS-AGENT-OMEGA.json"
    if attestation_path.exists():
        try:
            attestation_path.unlink()
        except OSError:
            pass

    # Create fresh attestation
    print("  Creating fresh attestation...")
    cmd_boot = f"cd {REPO_ROOT} && {PYTHON_BIN} scripts/boot_aligned_agent.py"
    rc_boot, out_boot, err_boot = run_command(cmd_boot, env=env)
    if rc_boot != 0:
        print(f"✗ FAIL: boot_aligned_agent.py failed with {rc_boot}")
        print(f"Output: {out_boot}")
        print(f"Error: {err_boot}")
        return False
    if "ALIGNED" not in out_boot:
        print("✗ FAIL: Agent not aligned")
        print(f"Output: {out_boot}")
        return False
    print("  ✓ Fresh attestation created")
    
    attestation_path = state_dir / "alignment_LOGOS-AGENT-OMEGA.json"
    if not attestation_path.exists():
        print("✗ FAIL: Attestation file not created")
        return False
    
    cmd = (
        f"cd {REPO_ROOT} && {PYTHON_BIN} scripts/start_agent.py --objective status "
        f"--read-only --budget-sec 1 --assume-yes --no-bootstrap-genesis"
    )
    rc, out, err = run_command(cmd, env=env)
    if rc in [0, 1] and "Attestation validated" in out:
        print("✓ PASS: Gate passed and run started")
        return True
    else:
        print(f"✗ FAIL: Expected gate pass and run start, got {rc}")
        print(f"Output: {out}")
        if err:
            print(f"Error: {err}")
        return False

def test_bypass_flag(env: dict) -> bool:
    """Test that --no-require-attestation fails without env var."""
    print("Test C: Bypass flag without env")
    cmd = (
        f"cd {REPO_ROOT} && {PYTHON_BIN} scripts/start_agent.py --objective status "
        f"--read-only --budget-sec 1 --no-require-attestation --no-bootstrap-genesis"
    )
    rc, out, err = run_command(cmd, env=env)
    if rc == 2 and "LOGOS_DEV_BYPASS_OK" in out:
        print("✓ PASS: Blocked bypass as expected")
        return True
    else:
        print(f"✗ FAIL: Expected block, got {rc}")
        print(f"Output: {out}")
        if err:
            print(f"Error: {err}")
        return False
def main() -> int:
    # Smoke tests must not mutate tracked repo state; isolate into a temp dir.
    tmp_state_dir = Path(tempfile.mkdtemp(prefix="logos_smoke_state_"))

    env = _env_with_paths()
    env["LOGOS_STATE_DIR"] = str(tmp_state_dir)

    tests = [test_missing_attestation, test_valid_attestation, test_bypass_flag]
    passed = 0
    for test in tests:
        if test(env):
            passed += 1
        print()
    print(f"Results: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
