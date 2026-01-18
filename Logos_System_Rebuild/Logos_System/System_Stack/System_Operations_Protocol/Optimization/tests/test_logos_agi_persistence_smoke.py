# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""Smoke test for Logos_AGI persistence across runs."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_test():
    """Run persistence test."""
    # Create temp dir for test state
    test_state_dir = Path(tempfile.mkdtemp(prefix="logos_agi_test_"))
    persisted_path = test_state_dir / "logos_agi_scp_state.json"

    try:
        # First run: bootstrap and observe
        print("=== First run: bootstrap and observe ===")
        cmd1 = [
            sys.executable,
            "-c",
            """
import sys
sys.path.insert(0, '/workspaces/pxl_demo_wcoq_proofs')
from scripts.logos_agi_adapter import LogosAgiNexus

# Mock audit logger
class MockLogger:
    def log(self, *args): pass

nexus = LogosAgiNexus(
    enable=True,
    audit_logger=MockLogger(),
    max_compute_ms=1000,
    state_dir='"""
            + str(test_state_dir)
            + """',
    repo_sha='test_sha_123'
)
nexus.bootstrap()
print(f"Bootstrap available: {nexus.available}")
nexus.observe({"event": "test_observation_1", "data": "first_run"})
nexus.observe({"event": "test_observation_2", "data": "second_obs"})
nexus.persist()
print(f"Persisted path exists: {nexus.persisted_path.exists()}")
if nexus.persisted_path.exists():
    with open(nexus.persisted_path) as f:
        state1 = json.load(f)
    print(f"First run observations: {len(state1.get('observations', []))}")
""",
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        print("STDOUT:", result1.stdout)
        if result1.stderr:
            print("STDERR:", result1.stderr)
        print(f"Return code: {result1.returncode}")

        # Check if persisted
        if persisted_path.exists():
            with open(persisted_path) as f:
                state1 = json.load(f)
            print(
                f"Persisted observations count: {len(state1.get('observations', []))}"
            )
            print(f"Persisted repo_sha: {state1.get('repo_sha')}")
        else:
            print("ERROR: No persistence file after first run")
            return False

        # Second run: load and add more observations
        print("\n=== Second run: load and add observations ===")
        cmd2 = [
            sys.executable,
            "-c",
            """
import sys
sys.path.insert(0, '/workspaces/pxl_demo_wcoq_proofs')
from scripts.logos_agi_adapter import LogosAgiNexus

class MockLogger:
    def log(self, *args): pass

nexus = LogosAgiNexus(
    enable=True,
    audit_logger=MockLogger(),
    max_compute_ms=1000,
    state_dir='"""
            + str(test_state_dir)
            + """',
    repo_sha='test_sha_123'
)
nexus.bootstrap()
print(f"Bootstrap available: {nexus.available}")
# Observations should persist in memory, but for test we add more
nexus.observe({"event": "test_observation_3", "data": "third_obs"})
nexus.persist()
print(f"Persisted path exists: {nexus.persisted_path.exists()}")
if nexus.persisted_path.exists():
    with open(nexus.persisted_path) as f:
        state2 = json.load(f)
    print(f"Second run observations: {len(state2.get('observations', []))}")
""",
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        print("STDOUT:", result2.stdout)
        if result2.stderr:
            print("STDERR:", result2.stderr)
        print(f"Return code: {result2.returncode}")

        # Verify persistence
        if persisted_path.exists():
            with open(persisted_path) as f:
                state2 = json.load(f)
            obs_count = len(state2.get("observations", []))
            print(f"Final persisted observations: {obs_count}")
            if obs_count >= 3:  # At least the new one
                print("SUCCESS: Persistence working - observations accumulated")
                return True
            else:
                print(f"ERROR: Expected at least 3 observations, got {obs_count}")
                return False
        else:
            print("ERROR: No persistence file after second run")
            return False

    except Exception as e:
        print(f"Test error: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(test_state_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_test()
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
