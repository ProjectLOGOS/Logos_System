# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# INSTALL_STATUS: SEMANTIC_REWRITE
# SOURCE_LEGACY: export_tool_registry.py

"""
SEMANTIC REWRITE

This module has been rewritten for governed integration into the
LOGOS System Rebuild. Its runtime scope and protocol role have been
normalized, but its original logical structure has been preserved.
"""

# ============================
# LOGOS FILE HEADER (STANDARD)
# ============================
# File: _Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/export_tool_registry.py
# Generated: 2026-01-14T19:54:10Z (UTC)
#
# SUMMARY:
#   Legacy dev/runtime-adjacent utility script captured during Audit & Normalize.
#   Header-standardized under Rewrite_Sub_Batch_1A.
#
# PURPOSE:
#   Preserve existing behavior; standardize documentation + naming/abbreviation policy markers.
#
# EXECUTION PATH:
#   Developer-invoked utility (audit/registry/diagnostic); not a critical runtime entrypoint.
#
# SIDE EFFECTS:
#   Review below; do not expand side effects. Fail-closed on unsafe operations.
#
# GOVERNANCE:
#   - Fail-closed required.
#   - Do not touch Coq stacks.
#   - Canon abbreviations: ETGC, SMP, MVS, BDN, PXL, IEL, SOP, SCP, ARP, MTP, TLM, UWM, LEM.
#   - Alias forbidden: use ETGC only; legacy variant disallowed.
# ============================

"""
Snapshot the native tool catalog from scripts.start_agent
into docs/agent_tool_registry.json/README.
"""

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_scripts_importable() -> None:
    parent_dir = Path(__file__).parent.parent
    parent_str = str(parent_dir)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)


def _load_tools() -> Dict[str, Any]:
    _ensure_scripts_importable()
    from JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent import TOOLS as START_AGENT_TOOLS  # type: ignore
    from logos.tool_registry_loader import load_approved_tools

    load_approved_tools(START_AGENT_TOOLS)

    return START_AGENT_TOOLS


def _tool_metadata(tool_id: str, tool: Any) -> Dict[str, Any]:
    def _normalize(value: Any) -> Any:
        if hasattr(value, "__name__"):
            return value.__name__
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    handler = getattr(tool, "handler", None)
    entry: Dict[str, Any] = {
        "id": tool_id,
        "description": getattr(tool, "description", ""),
        "requires_safe_interfaces": getattr(tool, "require_safe_interfaces", None),
        "restrict_writes_to": getattr(tool, "restrict_writes_to", None),
        "module": getattr(handler, "__module__", None),
        "callable": getattr(handler, "__name__", None),
    }

    if is_dataclass(tool):
        entry["raw"] = asdict(tool)
    else:
        entry["raw"] = {
            key: _normalize(value)
            for key, value in vars(tool).items()
            if not key.startswith("_")
        }

    return entry


def _write_json(entries: List[Dict[str, Any]], target: Path) -> None:
    target.write_text(json.dumps(entries, indent=2, sort_keys=False), encoding="utf-8")


def _write_markdown(entries: List[Dict[str, Any]], target: Path) -> None:
    lines = [
        "# LOGOS Native Tool Registry",
        "",
        "Automatically generated snapshot of `scripts.start_agent.TOOLS`.",
        "",
        "| Tool ID | Description | Handler | Safe Interfaces | Sandbox Writes |",
        "| ------- | ----------- | ------- | --------------- | -------------- |",
    ]
    for entry in entries:
        handler = (
            f"{entry['module']}.{entry['callable']}"
            if entry["module"] and entry["callable"]
            else "—"
        )
        lines.append(
            f"| `{entry['id']}` | {entry['description']} | `{handler}` | "
            f"{entry['requires_safe_interfaces']} | {entry['restrict_writes_to']} |"
        )

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Raw metadata</summary>",
            "",
            "```json",
            json.dumps(entries, indent=2),
            "```",
            "",
            "</details>",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Directory to receive the generated registry files (default: docs/).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tools = _load_tools()

    entries = [
        _tool_metadata(tool_id, tool)
        for tool_id, tool in sorted(tools.items(), key=lambda item: item[0])
    ]

    _write_json(entries, output_dir / "agent_tool_registry.json")
    _write_markdown(entries, output_dir / "agent_tool_registry.md")

    return 0
