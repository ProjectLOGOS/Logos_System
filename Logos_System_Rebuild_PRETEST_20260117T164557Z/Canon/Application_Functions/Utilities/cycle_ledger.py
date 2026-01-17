# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/cycle_ledger.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Ledger utilities for supervised promotion cycles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

LEDGER_SCHEMA_VERSION = "1.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_path(raw_path: str, repo_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _collect_bundle_hashes(
    promotion_outcomes: Iterable[Dict[str, Any]],
    repo_root: Path,
) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for entry in promotion_outcomes:
        outcome = entry.get("outcome")
        if not isinstance(outcome, dict):
            continue
        bundle_paths = outcome.get("bundle_paths") or {}
        if not isinstance(bundle_paths, dict):
            continue
        for _, raw_path in bundle_paths.items():
            if not raw_path:
                continue
            resolved = _normalize_path(raw_path, repo_root)
            if not resolved.exists() or not resolved.is_file():
                continue
            path_obj = Path(raw_path)
            if path_obj.is_absolute():
                try:
                    key = path_obj.relative_to(repo_root).as_posix()
                except ValueError:
                    key = path_obj.as_posix()
            else:
                key = path_obj.as_posix()
            hashes[key] = _sha256_file(resolved)
    return hashes


def _sanitize_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for step in steps:
        sanitized.append(
            {
                "step": step.get("step"),
                "tool": step.get("tool"),
                "status": step.get("status"),
                "output": step.get("output", ""),
            }
        )
    return sanitized


def _sanitize_outcomes(outcomes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for entry in outcomes:
        sanitized.append(
            {
                "tool": entry.get("tool"),
                "status": entry.get("status"),
                "config": entry.get("config"),
                "outcome": entry.get("outcome"),
                "reason": entry.get("reason"),
                "timestamp": entry.get("timestamp"),
                "run_id": entry.get("run_id"),
                "returncode": entry.get("returncode"),
                "raw_status": entry.get("raw_status"),
            }
        )
    return sanitized


def write_cycle_ledger(
    *,
    run_id: str,
    objective: str,
    mission: str,
    timestamp_utc: str,
    steps: List[Dict[str, Any]],
    promotion_outcomes: List[Dict[str, Any]],
    tests_required: Iterable[str],
    verification_steps: Iterable[str],
    rollback_steps: Iterable[str],
    sandbox_root: Path,
    repo_root: Path,
) -> Path:
    sandbox_root.mkdir(parents=True, exist_ok=True)

    entry = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "objective": objective,
        "mission": mission,
        "steps": _sanitize_steps(steps),
        "promotion_outcomes": _sanitize_outcomes(promotion_outcomes),
        "bundle_hashes": _collect_bundle_hashes(promotion_outcomes, repo_root),
        "tests_required": sorted({item for item in tests_required if item}),
        "verification_steps": sorted({item for item in verification_steps if item}),
        "rollback_steps": sorted({item for item in rollback_steps if item}),
        "operator_decision": None,
    }

    ledger_path = sandbox_root / f"cycle_ledger_{run_id}.json"
    latest_path = sandbox_root / "cycle_ledger_latest.json"

    payload = json.dumps(entry, indent=2, sort_keys=True)
    ledger_path.write_text(payload + "\n", encoding="utf-8")
    latest_path.write_text(payload + "\n", encoding="utf-8")
    return ledger_path
