"""
Phase 5 — Verification Gates (Fail-Closed)

These checks run after each batch (or each op in strict mode):
- python -m compileall on target roots (best-effort but treated as gate)
- optional import probe for audit_normalize_automation

No network. No Coq compilation. No external side effects.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, List


def run_compileall(repo_root: Path, targets: Iterable[Path]) -> None:
    args: List[str] = ["python3", "-m", "compileall", "-q"]
    args.extend([str(t) for t in targets])
    p = subprocess.run(args, cwd=str(repo_root))
    if p.returncode != 0:
        raise RuntimeError(f"compileall gate failed rc={p.returncode}")


def import_probe(repo_root: Path, module: str) -> None:
    repo_tools = repo_root / "_Dev_Resources" / "Dev_Scripts" / "repo_tools"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    prepend = str(repo_tools)
    env["PYTHONPATH"] = prepend if not existing else f"{prepend}:{existing}"
    cmd = ["python3", "-c", f"import {module}; print('OK')"]
    p = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"import probe failed for {module}: {p.stderr.strip()}")
