#!/usr/bin/env python3
"""Deterministic tool capability introspection (read-only, audited).

Builds capability records from the authoritative TOOLS registry, approved
manifests, and run ledgers. No tool execution is performed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_ROOT = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit"))

from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import canonical_json_hash

# Paths
APPROVED_DIR = REPO_ROOT / "tools" / "approved"
LEDGER_DIR = AUDIT_ROOT / "run_ledgers"


@dataclass
class ToolCapability:
    tool_name: str
    objective_classes: List[str]
    input_shape: str
    output_shape: str
    side_effects: List[str]
    risk_level: str
    truth_dependencies: List[str]
    introduced_by: str
    approval_hash: str
    last_used: Optional[str]
    success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["success_rate"] = round(self.success_rate, 3)
        return data


def _load_builtin_tools() -> Dict[str, Any]:
    """Import TOOLS without executing any actions."""
    from scripts import start_agent  # Local import to avoid heavy globals at module load

    return start_agent.TOOLS  # Authoritative registry


def _load_approved_manifests() -> List[Tuple[str, Dict[str, Any]]]:
    manifests: List[Tuple[str, Dict[str, Any]]] = []
    if not APPROVED_DIR.exists():
        return manifests
    for tool_dir in sorted(APPROVED_DIR.iterdir()):
        if not tool_dir.is_dir():
            continue
        manifest_path = tool_dir / "APPROVAL.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            manifests.append((tool_dir.name, manifest))
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def _load_run_ledgers() -> List[Dict[str, Any]]:
    ledgers: List[Dict[str, Any]] = []
    if not LEDGER_DIR.exists():
        return ledgers
    for file in sorted(LEDGER_DIR.glob("*.json")):
        try:
            with open(file) as f:
                ledgers.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return ledgers


def _compute_usage_stats(ledgers: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for ledger in ledgers:
        run_end = ledger.get("run_end_ts")
        ledger_hash = ledger.get("ledger_hash")
        for entry in ledger.get("execution_trace", []):
            tool = entry.get("tool_name")
            if not tool:
                continue
            tool_stat = stats.setdefault(tool, {
                "total": 0,
                "success": 0,
                "errors": 0,
                "denies": 0,
                "last_used": None,
                "ledger_refs": [],
            })
            tool_stat["total"] += 1
            outcome = entry.get("outcome")
            if outcome == "SUCCESS":
                tool_stat["success"] += 1
            elif outcome == "DENY" or outcome == "denied":
                tool_stat["denies"] += 1
            else:
                tool_stat["errors"] += 1
            if run_end and (tool_stat["last_used"] is None or run_end > tool_stat["last_used"]):
                tool_stat["last_used"] = run_end
            if ledger_hash:
                tool_stat["ledger_refs"].append(ledger_hash)
    return stats


def _builtin_defaults(tool_name: str) -> Tuple[List[str], str, str, List[str], str, List[str]]:
    obj_map = {
        "mission.status": ["STATUS"],
        "probe.last": ["STATUS"],
        "fs.read": ["GENERAL"],
    }
    objective_classes = obj_map.get(tool_name, ["GENERAL"])
    input_shape = "string"
    output_shape = "string|json"
    side_effects = ["read-only"]
    risk_level = "LOW"
    truth_dependencies: List[str] = []
    return objective_classes, input_shape, output_shape, side_effects, risk_level, truth_dependencies


def build_capability_records() -> List[ToolCapability]:
    tools = _load_builtin_tools()
    manifests = _load_approved_manifests()
    ledgers = _load_run_ledgers()
    usage_stats = _compute_usage_stats(ledgers)

    records: List[ToolCapability] = []

    # Builtins
    for name in sorted(tools.keys()):
        objective_classes, input_shape, output_shape, side_effects, risk_level, truth_deps = _builtin_defaults(name)
        stat = usage_stats.get(name, {})
        total = stat.get("total", 0)
        success = stat.get("success", 0)
        success_rate = (success / total) if total else 0.0
        records.append(
            ToolCapability(
                tool_name=name,
                objective_classes=objective_classes,
                input_shape=input_shape,
                output_shape=output_shape,
                side_effects=side_effects,
                risk_level=risk_level,
                truth_dependencies=truth_deps,
                introduced_by="builtin",
                approval_hash="builtin",
                last_used=stat.get("last_used"),
                success_rate=success_rate,
            )
        )

    # Approved tools
    for tool_dir_name, manifest in manifests:
        name = manifest.get("tool_name", tool_dir_name)
        stat = usage_stats.get(name, {})
        total = stat.get("total", 0)
        success = stat.get("success", 0)
        success_rate = (success / total) if total else 0.0
        records.append(
            ToolCapability(
                tool_name=name,
                objective_classes=manifest.get("objective_classes", ["GENERAL"]),
                input_shape=str(manifest.get("inputs_schema", "unspecified")),
                output_shape=str(manifest.get("outputs_schema", "unspecified")),
                side_effects=["unknown"],
                risk_level="MEDIUM",
                truth_dependencies=[],
                introduced_by="pipeline",
                approval_hash=manifest.get("tool_py_sha256") or canonical_json_hash(manifest),
                last_used=stat.get("last_used"),
                success_rate=success_rate,
            )
        )

    return records


def build_introspection_summary() -> Dict[str, Any]:
    records = build_capability_records()
    broken_tools: List[str] = []  # Populated by health analysis elsewhere
    summary = {
        "tools_analyzed": len(records),
        "broken_tools": broken_tools,
        "repair_proposals_generated": [],
        "uip_decisions": [],
        "capabilities_hash": canonical_json_hash({"records": [r.to_dict() for r in records]}),
        "records": [r.to_dict() for r in records],
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    records = build_capability_records()
    output = [r.to_dict() for r in records]
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
