#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

from _common import iter_files, read_text, write_json

DEV_SCRIPTS_DIR = Path("/workspaces/Logos_System/_Dev_Resources/Dev_Scripts").resolve()

def scan_scripts(repo: Path) -> Dict[str, Any]:
    scripts = []
    function_index = []
    call_graph = []

    if not DEV_SCRIPTS_DIR.exists():
        return {"error": f"missing {DEV_SCRIPTS_DIR}"}

    for p in DEV_SCRIPTS_DIR.rglob("*.py"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(repo))
        scripts.append({"path": rel})

        src = read_text(p)
        try:
            tree = ast.parse(src, filename=str(p))
        except SyntaxError:
            continue

        defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        for d in defs:
            function_index.append({
                "script": rel,
                "function": d.name,
                "line": getattr(d, "lineno", 0),
                "args": [a.arg for a in d.args.args],
            })

        # heuristic call graph: function calls by name
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = None
                if isinstance(n.func, ast.Name):
                    fn = n.func.id
                elif isinstance(n.func, ast.Attribute):
                    fn = n.func.attr
                if fn:
                    call_graph.append({"script": rel, "calls": fn, "line": getattr(n, "lineno", 0)})

    return {
        "script_inventory": sorted(scripts, key=lambda x: x["path"]),
        "function_index": sorted(function_index, key=lambda x: (x["script"], x["line"], x["function"])),
        "call_graph": sorted(call_graph, key=lambda x: (x["script"], x["line"], x["calls"])),
    }

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/09_dev_scripts")
    r = scan_scripts(repo)
    write_json(base / "script_inventory.json", r.get("script_inventory", r))
    write_json(base / "function_index.json", r.get("function_index", r))
    write_json(base / "call_graph.json", r.get("call_graph", r))
    write_json(base / "side_effects.json", {"note": "use 07_side_effects for repo-wide side effect surfaces; dev-scripts can be cross-referenced by path"})
    print(base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
