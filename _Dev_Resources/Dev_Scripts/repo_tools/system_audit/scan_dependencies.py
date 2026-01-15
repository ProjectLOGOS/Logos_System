#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from _common import iter_files, parse_python_imports, looks_like_stdlib, write_json

def scan_python_dependencies(repo: Path) -> Dict[str, Any]:
    third_party: Set[str] = set()
    stdlib: Set[str] = set()
    local_like: Set[str] = set()

    for p in iter_files(repo, suffix=".py"):
        for imp in parse_python_imports(p, repo):
            if imp.level:
                continue
            mod = (imp.module.split(".")[0] if imp.module else "")
            if not mod:
                continue
            if looks_like_stdlib(mod):
                stdlib.add(mod)
            else:
                # Heuristic: if top-level module name matches a first-party top directory, mark local-like
                local_like.add(mod)

    # We cannot reliably disambiguate third-party vs local without import resolution + sys.path.
    # Provide both sets; automation can refine later.
    return {
        "stdlib_candidates": sorted(stdlib),
        "top_level_non_stdlib_candidates": sorted(local_like),
        "notes": {
            "classification": "stdlib is heuristic; non-stdlib includes both first-party and third-party. Refine via runtime import resolution if needed."
        }
    }

def scan_toolchain(repo: Path) -> Dict[str, Any]:
    # Collect common dependency surfaces (requirements, pyproject, environment hints)
    candidates = []
    for p in iter_files(repo):
        rel = str(p.relative_to(repo))
        if p.name in {"requirements.txt", "requirements-dev.txt", "pyproject.toml", "Pipfile", "environment.yml"}:
            candidates.append(rel)
    return {"files": sorted(candidates)}

def scan_inter_system(repo: Path) -> Dict[str, Any]:
    # Capture known cross-system dependencies based on repository structure
    # This is structural; import_graph covers code-level coupling.
    surfaces = []
    for d in ["PXL_Gate", "System_Stack", "LOGOS_SYSTEM", "System_Entry_Point", "System_Audit_Logs"]:
        p = repo / d
        if p.exists():
            surfaces.append(d)
    return {"top_level_system_surfaces": surfaces}

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    out_base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/02_dependencies")
    write_json(out_base / "python_dependencies.json", scan_python_dependencies(repo))
    write_json(out_base / "toolchain_dependencies.json", scan_toolchain(repo))
    write_json(out_base / "inter_system_dependencies.json", scan_inter_system(repo))
    print(out_base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
