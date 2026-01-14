#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from _common import iter_files, parse_python_imports, write_json, normalize_module_to_path

def build_import_graph(repo: Path) -> Dict[str, Any]:
    imports: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []
    unresolved: List[Dict[str, Any]] = []

    # map file-> module-ish path for local resolution heuristics
    py_files = list(iter_files(repo, suffixes=(".py",)))
    file_set = {str(p.relative_to(repo)) for p in py_files}

    def resolve_local(mod: str) -> str | None:
        # Try resolve "a.b.c" -> "a/b/c.py" or "a/b/c/__init__.py"
        base = normalize_module_to_path(mod)
        cand1 = f"{base}.py"
        cand2 = f"{base}/__init__.py"
        if cand1 in file_set:
            return cand1
        if cand2 in file_set:
            return cand2
        return None

    for p in py_files:
        for imp in parse_python_imports(p, repo):
            rec = {
                "importer_file": imp.importer_file,
                "line": imp.line,
                "col": imp.col,
                "kind": imp.kind,
                "module": imp.module,
                "name": imp.name,
                "level": imp.level,
            }
            imports.append(rec)

            # Edge target: attempt local resolve, else keep module string
            target = resolve_local(imp.module) if imp.level == 0 else None
            # Relative imports require package context; record as unresolved if relative
            if imp.level and imp.kind == "from":
                unresolved.append({**rec, "reason": "relative_import_requires_package_context"})
            else:
                edges.append((imp.importer_file, target or imp.module))
                if target is None and imp.level == 0:
                    # unresolved local resolution doesn't mean invalid; still report
                    unresolved.append({**rec, "reason": "unresolved_to_local_path"})

    # Cycle detection over file->file edges only (local-resolved)
    adj: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edges:
        if isinstance(b, str) and (b.endswith(".py") or b.endswith("/__init__.py")):
            adj[a].add(b)

    # Tarjan SCC (simple)
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in adj.get(v, set()):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                sccs.append(sorted(comp))

    for v in sorted(adj.keys()):
        if v not in indices:
            strongconnect(v)

    return {
        "imports": imports,
        "edges": [{"from": a, "to": b} for a, b in edges],
        "cycles": sccs,
        "unresolved": unresolved,
        "notes": {
            "relative_imports": "reported as unresolved; require package context + PYTHONPATH for exact resolution",
            "edge_to": "local file path when resolvable; else module string"
        }
    }

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/01_import_graph")
    graph = build_import_graph(repo)
    write_json(base / "import_graph.json", graph)
    write_json(base / "import_cycles.json", {"cycles": graph["cycles"]})
    write_json(base / "unresolved_imports.json", {"unresolved": graph["unresolved"]})
    print(base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
