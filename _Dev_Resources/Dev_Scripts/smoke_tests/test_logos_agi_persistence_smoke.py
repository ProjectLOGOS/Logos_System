#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_logos_agi_persistence_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke test for Logos_AGI persistence across cycles."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logos_AGI persistence smoke test")
    parser.add_argument(
        "--allow-state-test-write",
        action="store_true",
        help="Acknowledge writes to state/audit during persistence smoke test",
    )
    parser.add_argument(
        "--state-dir",
        default=str(DEFAULT_STATE_DIR),
        help="State directory to use (default: repo state)",
    )
    return parser.parse_args(argv)


def run_command(cmd, **kwargs):
    """Run command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return result.returncode, result.stdout, result.stderr


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 of file."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_persistence(state_dir: Path) -> bool:
    """Test persistence changes across runs."""
    print("Test: Logos_AGI persistence smoke")

    # Clean any existing
    persisted_path = state_dir / "logos_agi_scp_state.json"
    if persisted_path.exists():
        persisted_path.unlink()

    # Run first time
    env = os.environ.copy()
    env.setdefault("LOGOS_OPERATOR_OK", "1")
    env["LOGOS_STATE_DIR"] = str(state_dir)

    cmd1 = [
        sys.executable,
        "scripts/start_agent.py",
        "--enable-logos-agi",
        "--objective",
        "status",
        "--read-only",
        "--budget-sec",
        "1",
        "--assume-yes",
    ]
    rc1, out1, err1 = run_command(cmd1, cwd=REPO_ROOT, env=env)
    if rc1 != 0:
        print(f"✗ FAIL: First run failed with {rc1}")
        return False

    hash1 = compute_file_hash(persisted_path)
    if not hash1:
        print("✗ FAIL: No persisted file after first run")
        return False

    # Run second time with different objective
    cmd2 = [
        sys.executable,
        "scripts/start_agent.py",
        "--enable-logos-agi",
        "--objective",
        "status2",
        "--read-only",
        "--budget-sec",
        "1",
        "--assume-yes",
    ]
    rc2, out2, err2 = run_command(cmd2, cwd=REPO_ROOT, env=env)
    if rc2 != 0:
        print(f"✗ FAIL: Second run failed with {rc2}")
        return False

    hash2 = compute_file_hash(persisted_path)
    if not hash2:
        print("✗ FAIL: No persisted file after second run")
        return False

    if hash1 == hash2:
        print("✗ FAIL: Persisted file did not change")
        return False

    print("✓ PASS: Persistence test passed")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if os.environ.get("LOGOS_OPERATOR_OK", "").strip() != "1":
        print(
            "ERROR: LOGOS_OPERATOR_OK=1 is required for persistence smoke writes.",
            file=sys.stderr,
        )
        return 2
    if not args.allow_state_test_write:
        print(
            "ERROR: --allow-state-test-write is required for state/audit writes.",
            file=sys.stderr,
        )
        return 2

    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    tests = [test_persistence]
    passed = 0
    for test in tests:
        if test(state_dir):
            passed += 1
        print()
    print(f"Results: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
