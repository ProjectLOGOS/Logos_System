#!/usr/bin/env python3
"""Provenance utilities for Logos_AGI pinning and drift detection."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


class DriftError(Exception):
    """Raised when Logos_AGI repository has drifted from pinned commit."""

    pass


class PinConfigError(Exception):
    """Raised when pin configuration is invalid."""

    pass


def run_git(args: list[str], cwd: str) -> str:
    """Run git command and return stdout, raise on error."""
    result = subprocess.run(
        ["git"] + args, cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def get_git_head_sha(repo_dir: str) -> str:
    """Get current HEAD SHA."""
    return run_git(["rev-parse", "HEAD"], repo_dir)


def is_git_dirty(repo_dir: str) -> bool:
    """Check if repository has uncommitted changes."""
    output = run_git(["status", "--porcelain"], repo_dir)
    return bool(output.strip())


def resolve_ref(repo_dir: str, ref: str) -> str:
    """Resolve a ref (tag, branch, SHA) to full SHA."""
    return run_git(["rev-parse", ref], repo_dir)


def load_pin(pin_path: str) -> Dict[str, Any]:
    """Load and validate pin file."""
    if not Path(pin_path).exists():
        raise PinConfigError(f"Pin file not found: {pin_path}")

    with open(pin_path) as f:
        pin = json.load(f)

    # Validate schema
    required = ["repo", "pinned_sha", "pinned_at", "pinned_by", "note", "allow_dirty"]
    for key in required:
        if key not in pin:
            raise PinConfigError(f"Missing pin key: {key}")

    if not isinstance(pin["pinned_sha"], str) or len(pin["pinned_sha"]) != 40:
        raise PinConfigError("pinned_sha must be 40-character hex string")

    if not all(c in "0123456789abcdef" for c in pin["pinned_sha"]):
        raise PinConfigError("pinned_sha must be valid hex")

    return pin


def write_pin(
    pin_path: str, pinned_sha: str, note: str, allow_dirty: bool = False
) -> None:
    """Write pin file."""
    pin = {
        "repo": "ProjectLOGOS/Logos_AGI",
        "pinned_sha": pinned_sha,
        "pinned_at": datetime.now(timezone.utc).isoformat(),
        "pinned_by": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        "note": note,
        "allow_dirty": allow_dirty,
    }

    Path(pin_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pin_path, "w") as f:
        json.dump(pin, f, indent=2)


def verify_pinned_repo(
    repo_dir: str,
    pin: Dict[str, Any],
    *,
    require_clean: bool = True,
    allow_drift: bool = False,
) -> Dict[str, Any]:
    """Verify repository matches pin, return provenance dict."""
    head_sha = get_git_head_sha(repo_dir)
    dirty = is_git_dirty(repo_dir)

    provenance = {
        "repo_dir": repo_dir,
        "head_sha": head_sha,
        "pinned_sha": pin["pinned_sha"],
        "dirty": dirty,
        "match": head_sha == pin["pinned_sha"],
    }

    if require_clean and dirty and not pin.get("allow_dirty", False):
        raise DriftError(f"Repository is dirty and allow_dirty=false: {repo_dir}")

    if not allow_drift and not provenance["match"]:
        raise DriftError(f"SHA mismatch: pinned={pin['pinned_sha']}, head={head_sha}")

    return provenance
