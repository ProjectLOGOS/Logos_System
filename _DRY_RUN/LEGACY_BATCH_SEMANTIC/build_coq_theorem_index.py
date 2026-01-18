# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# INSTALL_STATUS: SEMANTIC_REWRITE
# SOURCE_LEGACY: build_coq_theorem_index.py

"""
SEMANTIC REWRITE

This module has been rewritten for governed integration into the
LOGOS System Rebuild. Its runtime scope and protocol role have been
normalized, but its original logical structure has been preserved.
"""

"""Build Coq theorem index from existing proof artifacts."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def canonical_json_hash(data):
    """Compute SHA256 of canonical JSON."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_statement(statement):
    """Normalize theorem statement for hashing."""
    # Strip leading/trailing whitespace, collapse multiple spaces
    return " ".join(statement.split())


def extract_theorem_statement(file_path, theorem_name):
    """Extract theorem statement from Coq file."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Find the theorem line
        theorem_line_idx = None
        for i, line in enumerate(lines):
            if f"Theorem {theorem_name}" in line or f"Lemma {theorem_name}" in line:
                theorem_line_idx = i
                break

        if theorem_line_idx is None:
            return None

        # Collect statement lines until '.'
        statement_lines = []
        for line in lines[theorem_line_idx:]:
            statement_lines.append(line.strip())
            if "." in line:
                break

        statement = " ".join(statement_lines)
        # Extract from ':' to '.'
        if ":" in statement and "." in statement:
            start = statement.find(":") + 1
            end = statement.rfind(".")
            statement = statement[start:end].strip()

        return normalize_statement(statement)

    except Exception as e:
        return None


    repo_root = Path(__file__).parent.parent
    state_dir = repo_root / "state"
    state_dir.mkdir(exist_ok=True)

    # Get repo SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        return 1
    repo_sha = result.stdout.strip()

    # Coq version (hardcoded for now)
    coq_version = "8.16"

    # Foundational theorems
    theorems = [
        {
            "theorem": "pxl_excluded_middle",
            "file": "Protopraxis/formal_verification/coq/baseline/PXL_Internal_LEM.v",
        }
    ]

    theorem_entries = []
    for th in theorems:
        file_path = repo_root / th["file"]
        statement = extract_theorem_statement(file_path, th["theorem"])
        if statement:
            statement_hash = hashlib.sha256(statement.encode("utf-8")).hexdigest()
            theorem_entries.append(
                {
                    "theorem": th["theorem"],
                    "file": th["file"],
                    "statement": statement,
                    "statement_hash": statement_hash,
                }
            )
        else:

    # Build index
    index = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_sha": repo_sha,
        "coq_version": coq_version,
        "theorems": theorem_entries,
    }

    # Compute index_hash
    temp = index.copy()
    index["index_hash"] = canonical_json_hash(temp)

    # Write to file
    index_path = state_dir / "coq_theorem_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return 0
