#!/usr/bin/env python3
"""Compute or verify a canonical identity hash over gated artifacts.

Hash input ordering and formatting are fixed:
  PATH=<relative path>\n
  LEN=<byte length>\n
  <file bytes>
  \n--\n
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent.parent
IDENTITY_PATH = ROOT / "state" / "IDENTITY_HASH.txt"
FILE_LIST: List[str] = [
    "state/golden_run_fingerprint.txt",
    "scripts/test_lem_discharge.py",
    "_CoqProject",
    "AUDIT/STRUCTURE.md",
    "release/TARBALL_SHA256SUM.txt",
]


def _current_parent_sha() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip()
    )


def _compute_digest() -> str:
    h = hashlib.sha256()
    for rel in FILE_LIST:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing required file: {rel}")
        data = path.read_bytes()
        h.update(f"PATH={rel}\n".encode())
        h.update(f"LEN={len(data)}\n".encode())
        h.update(data)
        h.update(b"\n--\n")
    return h.hexdigest()


def _write_identity(parent_sha: str, digest: str) -> None:
    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    contents = [
        f"parent_sha={parent_sha}",
        f"identity_sha256={digest}",
        "included_files=" + ",".join(FILE_LIST),
        "",
    ]
    IDENTITY_PATH.write_text("\n".join(contents), encoding="utf-8")


def _load_identity() -> dict:
    if not IDENTITY_PATH.is_file():
        raise FileNotFoundError("IDENTITY_HASH.txt is missing; run with --init-identity")
    meta: dict[str, str] = {}
    for line in IDENTITY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute or verify identity hash")
    parser.add_argument(
        "--init-identity",
        action="store_true",
        help="write IDENTITY_HASH.txt instead of verifying",
    )
    args = parser.parse_args()

    parent_sha = _current_parent_sha()
    digest = _compute_digest()

    if args.init_identity:
        _write_identity(parent_sha, digest)
        print(f"identity_sha256={digest}")
        return 0

    try:
        meta = _load_identity()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    recorded_sha = meta.get("identity_sha256")
    recorded_parent = meta.get("parent_sha")

    errors = []
    if recorded_sha != digest:
        errors.append("digest mismatch")

    if recorded_parent and recorded_parent != parent_sha:
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", recorded_parent, parent_sha],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            errors.append("parent_sha mismatch")

    if errors:
        print(
            "Identity hash verification failed: " + ", ".join(errors),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())