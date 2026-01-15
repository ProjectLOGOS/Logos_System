#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/repo_tools/system_audit/scan_side_effects.py
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

WRITE_HINTS = ("open(", ".write(", "mkdir(", "rmdir(", "unlink(", "rename(", "replace(", "shutil.", "Path(", ".touch(")
SUBPROCESS_HINTS = ("subprocess.", "Popen(", "check_call(", "check_output(", "run(")
NETWORK_HINTS = ("requests.", "httpx.", "urllib", "socket", "flask", "fastapi")

def scan(repo: Path) -> Dict[str, Any]:
    writes = []
    procs = []
    net = []

    for p in iter_files(repo, suffixes=(".py",)):
        rel = str(p.relative_to(repo))
        txt = read_text(p)
        low = txt.lower()

        if any(h in txt for h in WRITE_HINTS):
            writes.append({"path": rel})
        if any(h.lower() in low for h in SUBPROCESS_HINTS):
            procs.append({"path": rel})
        if any(h.lower() in low for h in NETWORK_HINTS):
            net.append({"path": rel})

    return {
        "filesystem_writes": sorted(writes, key=lambda x: x["path"]),
        "subprocess_calls": sorted(procs, key=lambda x: x["path"]),
        "network_surfaces": sorted(net, key=lambda x: x["path"]),
        "notes": "heuristic scan; use for triage before deeper taint analysis"
    }

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/07_side_effects")
    r = scan(repo)
    write_json(base / "filesystem_writes.json", r["filesystem_writes"])
    write_json(base / "state_mutations.json", {"subprocess_calls": r["subprocess_calls"]})
    write_json(base / "logging_emitters.json", {"network_surfaces": r["network_surfaces"]})
    print(base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
