#!/usr/bin/env python3
"""Smoke test for tool improvement intent."""

import json
import subprocess
import sys
from pathlib import Path

# Add scripts to path
scripts_dir = Path(__file__).parent
repo_root = scripts_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def run_command(cmd, cwd=None, env=None):
    """Run command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd or repo_root,
        env=env,
        check=False
    )
    return result.returncode, result.stdout, result.stderr


def test_tool_improvement_intent_smoke():
    """Test tool improvement intent."""
    import tempfile
    from logos.tool_health import analyze_tool_health
    from logos.tool_improvement import propose_tool_improvements

    # Create temp dir for isolated test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ledger_dir = temp_path / "ledgers"
        ledger_dir.mkdir()

        # Write synthetic ledger
        synthetic_ledger = {
            "execution_trace": [
                {"tool_name": "broken_tool", "outcome": "error"},
                {"tool_name": "broken_tool", "outcome": "error"},
                {"tool_name": "broken_tool", "outcome": "error"},
            ]
        }
        with open(ledger_dir / "synthetic.json", "w", encoding='utf-8') as f:
            json.dump(synthetic_ledger, f)

        # Analyze health
        scp_state = {}
        beliefs = {}
        metrics = {}
        health_report = analyze_tool_health(ledger_dir, scp_state, beliefs, metrics)

        # Assert health is broken
        if health_report["overall_health"] != "BROKEN":
            print("FAIL: Health not broken")
            return False

        # Assert improvement proposed
        intents = propose_tool_improvements(health_report)
        if not intents:
            print("FAIL: No improvement intents")
            return False

        if intents[0]["tool"] != "broken_tool":
            print("FAIL: Wrong tool in intent")
            return False

    print("PASS: Tool improvement intent smoke test")
    return True


if __name__ == "__main__":
    if test_tool_improvement_intent_smoke():
        print("Tool improvement intent smoke test passed!")
        sys.exit(0)
    else:
        print("Tool improvement intent smoke test failed!")
        sys.exit(1)