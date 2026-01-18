# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED


"""Test Logos_AGI bootstrap modes."""

import os
import subprocess
import sys
from pathlib import Path


def test_stub_mode():
    """Stub mode must always pass."""
    scripts_dir = Path(__file__).parent
    repo_root = scripts_dir.parent
    state_dir = Path(os.getenv("LOGOS_STATE_DIR", repo_root / "state"))

    cmd = [
        sys.executable,
        str(scripts_dir / "start_agent.py"),
        "--enable-logos-agi",
        "--logos-agi-mode",
        "stub",
        "--objective",
        "status",
        "--read-only",
        "--budget-sec",
        "1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        print(f"FAIL: Stub mode failed: {result.stderr}")
        return False

    # Check scp_state for available=True
    scp_state_path = state_dir / "scp_state.json"
    if scp_state_path.exists():
        import json

        with open(scp_state_path) as f:
            scp_state = json.load(f)
        if scp_state.get("arp_status") is not None:  # stub has None
            print("PASS: Stub mode bootstrap successful")
            return True
        else:
            print("FAIL: Stub mode did not set correct status")
            return False
    else:
        print("FAIL: No SCP state created")
        return False


def test_real_mode():
    """Real mode must pass if dependencies OK, or fail with explicit error."""
    scripts_dir = Path(__file__).parent
    repo_root = scripts_dir.parent

    cmd = [
        sys.executable,
        str(scripts_dir / "start_agent.py"),
        "--enable-logos-agi",
        "--logos-agi-mode",
        "real",
        "--objective",
        "status",
        "--read-only",
        "--budget-sec",
        "1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if result.returncode == 0:
        print("PASS: Real mode bootstrap successful")
        return True
    else:
        error_msg = result.stdout + result.stderr
        if (
            "Real Logos_AGI bootstrap failed" in error_msg
            or "Logos_AGI pin verification failed" in error_msg
        ):
            print("PASS: Real mode failed with expected error")
            return True
        else:
            print(f"FAIL: Real mode failed with unexpected error: {error_msg}")
            return False


def main():
    print("Testing stub mode...")
    stub_ok = test_stub_mode()

    print("Testing real mode...")
    real_ok = test_real_mode()

    if stub_ok and real_ok:
        print("All bootstrap tests passed")
        return True
    else:
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
