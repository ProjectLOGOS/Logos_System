# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Smoke tests for run_simulation_stub CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_simulation_stub.py"


def test_run_simulation_stub_writes_to_custom_log_dir() -> None:
    with TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--steps",
                "2",
                "--no-ingest",
                "--log-dir",
                str(log_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        stdout = result.stdout
        assert "[sim] wrote" in stdout
        log_file = log_dir / "gridworld_events.jsonl"
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
