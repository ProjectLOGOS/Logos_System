#!/usr/bin/env python3
"""Smoke test for run ledger generation."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))
AUDIT_DIR = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit"))


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


def test_run_ledger_smoke():
    """Test that ledger is created and valid."""
    print("Test: Run ledger smoke test")

    # Clean any existing ledgers
    ledger_dir = AUDIT_DIR / "run_ledgers"
    if ledger_dir.exists():
        for f in ledger_dir.glob("*.json"):
            f.unlink()

    # Run start_agent in stub mode
    cmd = f"cd {REPO_ROOT} && python scripts/start_agent.py --no-require-attestation --enable-logos-agi --logos-agi-mode stub --objective 'status' --read-only --budget-sec 10 --assume-yes"
    env = _env_with_paths()
    env["LOGOS_DEV_BYPASS_OK"] = "1"
    rc, out, err = run_command(cmd, env=env)
    if rc != 0:
        print(f"✗ FAIL: start_agent failed with {rc}")
        print("STDOUT:", out)
        print("STDERR:", err)
        return False

    # Find the newest ledger file
    if not ledger_dir.exists():
        print("✗ FAIL: Ledger directory not created")
        return False

    ledger_files = list(ledger_dir.glob("*.json"))
    if not ledger_files:
        print("✗ FAIL: No ledger files created")
        return False

    ledger_file = max(ledger_files, key=lambda f: f.stat().st_mtime)
    print(f"✓ Found ledger: {ledger_file}")

    # Load and validate JSON
    try:
        with open(ledger_file) as f:
            ledger = json.load(f)
    except Exception as e:
        print(f"✗ FAIL: Invalid JSON: {e}")
        return False

    # Check required keys
    required_keys = [
        "schema_version",
        "run_start_ts",
        "run_end_ts",
        "hashes",
        "truth_summary",
        "belief_usage",
        "policy_interventions",
        "execution_trace",
        "governance_flags",
        "ledger_hash",
    ]
    for key in required_keys:
        if key not in ledger:
            print(f"✗ FAIL: Missing key {key}")
            return False

    # Check hashes section
    hashes = ledger["hashes"]
    if not isinstance(hashes, dict):
        print("✗ FAIL: hashes not dict")
        return False

    # Check truth_summary structure
    truth_summary = ledger["truth_summary"]
    expected_truth_keys = [
        "proposals_generated",
        "proposals_executed",
        "plan_steps",
        "truth_events",
    ]
    for key in expected_truth_keys:
        if key not in truth_summary:
            print(f"✗ FAIL: Missing truth_summary key {key}")
            return False

    # Check belief_usage
    belief_usage = ledger["belief_usage"]
    if "referenced_belief_ids" not in belief_usage:
        print("✗ FAIL: Missing belief_usage.referenced_belief_ids")
        return False

    # Check policy_interventions (may be empty)
    policy = ledger["policy_interventions"]
    expected_policy_keys = [
        "boosted_tools",
        "filtered_tools",
        "boosted_belief_ids",
        "filtered_belief_ids",
    ]
    for key in expected_policy_keys:
        if key not in policy:
            print(f"✗ FAIL: Missing policy key {key}")
            return False

    # Check execution_trace
    execution_trace = ledger["execution_trace"]
    if not isinstance(execution_trace, list):
        print("✗ FAIL: execution_trace not list")
        return False

    # Check governance_flags
    governance = ledger["governance_flags"]
    if "logos_agi_mode" not in governance:
        print("✗ FAIL: Missing governance logos_agi_mode")
        return False

    # Check ledger_hash
    from logos.ledger import compute_ledger_hash

    computed_hash = compute_ledger_hash(ledger)
    if computed_hash != ledger["ledger_hash"]:
        print(
            f"✗ FAIL: ledger_hash mismatch: {computed_hash} != {ledger['ledger_hash']}"
        )
        return False

    print("✓ Ledger JSON valid")
    print("✓ Ledger hash field present")
    print("✓ All required sections present")

    # Test with policy trigger
    print("\nTest: Ledger with policy intervention")

    # Clean ledgers
    for f in ledger_dir.glob("*.json"):
        f.unlink()

    # Inject belief state to trigger policy
    scp_state_path = STATE_DIR / "scp_state.json"
    if scp_state_path.exists():
        with open(scp_state_path) as f:
            scp_state = json.load(f)

        # Clear test beliefs and add fresh one to trigger filtering
        if "beliefs" not in scp_state:
            scp_state["beliefs"] = {"items": []}
        # Remove any previous test beliefs
        scp_state["beliefs"]["items"] = [
            b for b in scp_state["beliefs"].get("items", [])
            if b.get("id") != "test_belief_filter"
        ]
        # Add fresh test belief
        scp_state["beliefs"]["items"].append(
            {
                "id": "test_belief_filter",
                "status": "QUARANTINED",
                "truth": "VERIFIED",
                "confidence": 0.9,
                "objective_tags": ["STATUS"],
                "content": {"contradicted_tools": ["mission.status"]},
            }
        )

        # Clear active plans to force new plan creation with fresh policy evaluation
        if "plans" in scp_state:
            scp_state["plans"]["active"] = []

        with open(scp_state_path, "w") as f:
            json.dump(scp_state, f, indent=2)

    # Run again
    env2 = _env_with_paths()
    env2["LOGOS_DEV_BYPASS_OK"] = "1"
    rc2, out2, err2 = run_command(cmd, env=env2)
    if rc2 != 0:
        print(f"✗ FAIL: Second run failed with {rc2}")
        return False

    # Check new ledger
    ledger_files2 = list(ledger_dir.glob("*.json"))
    if not ledger_files2:
        print("✗ FAIL: No ledger after second run")
        return False

    ledger_file2 = max(ledger_files2, key=lambda f: f.stat().st_mtime)
    with open(ledger_file2) as f:
        ledger2 = json.load(f)

    policy2 = ledger2["policy_interventions"]
    if "mission.status" not in policy2.get("filtered_tools", []):
        print("✗ FAIL: Policy did not record filtered tool")
        return False

    print("✓ Policy intervention recorded in ledger")

    print("✓ PASS: Run ledger smoke test passed")
    return True


def main():
    tests = [test_run_ledger_smoke]
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    print(f"Results: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
