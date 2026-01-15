#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/repo_tools/system_audit/scan_runtime_surface.py
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

from pathlib import Path
from typing import Any, Dict, List

from _common import iter_files, read_text, write_json

ENTRYPOINT_NAMES = {"START_LOGOS.py", "__main__.py"}
SERVER_HINTS = ("flask", "fastapi", "uvicorn", "gunicorn", "app.run", "@app.", "Blueprint")

def scan(repo: Path) -> Dict[str, Any]:
    entrypoints = []
    servers = []
    cli_tools = []

    for p in iter_files(repo, suffixes=(".py",)):
        rel = str(p.relative_to(repo))
        txt = read_text(p)

        if p.name in ENTRYPOINT_NAMES:
            entrypoints.append({"path": rel})

        if any(h in txt.lower() for h in SERVER_HINTS):
            servers.append({"path": rel})

        # crude CLI detection
        if "if __name__ == \"__main__\"" in txt or "argparse" in txt or "click" in txt.lower():
            cli_tools.append({"path": rel})

    return {
        "entrypoints": sorted(entrypoints, key=lambda x: x["path"]),
        "servers": sorted(servers, key=lambda x: x["path"]),
        "cli_tools": sorted(cli_tools, key=lambda x: x["path"]),
        "notes": "heuristic scan; pair with import_graph for reachability"
    }

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/06_runtime_surface")
    r = scan(repo)
    write_json(base / "entrypoints.json", r["entrypoints"])
    write_json(base / "servers.json", r["servers"])
    write_json(base / "cli_tools.json", r["cli_tools"])
    print(base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
