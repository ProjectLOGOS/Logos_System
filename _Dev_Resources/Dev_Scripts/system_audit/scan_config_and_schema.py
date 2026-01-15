#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/repo_tools/system_audit/scan_config_and_schema.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from _common import iter_files, write_json

CONFIG_SUFFIXES = (".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".env")
SCHEMA_HINTS = ("schema", "jsonschema", "$schema")

def scan(repo: Path) -> Dict[str, Any]:
    config_files = []
    schema_defs = []

    for p in iter_files(repo):
        rel = str(p.relative_to(repo))
        if p.suffix.lower() in CONFIG_SUFFIXES:
            config_files.append(rel)

            if p.suffix.lower() == ".json":
                try:
                    obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                    # heuristic schema detection
                    if isinstance(obj, dict) and ("$schema" in obj or "properties" in obj or "definitions" in obj):
                        schema_defs.append(rel)
                except Exception:
                    pass

    return {
        "config_files": sorted(config_files),
        "schema_definitions": sorted(schema_defs),
        "schema_drift": {"note": "schema drift requires pairing schema defs with runtime validation points; add once validators are centralized"}
    }

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/11_config_and_schema")
    r = scan(repo)
    write_json(base / "config_files.json", r["config_files"])
    write_json(base / "schema_definitions.json", r["schema_definitions"])
    write_json(base / "schema_drift.json", r["schema_drift"])
    print(base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
