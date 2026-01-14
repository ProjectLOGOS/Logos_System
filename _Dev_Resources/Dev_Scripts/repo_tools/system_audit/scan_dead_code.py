#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Set

from _common import iter_files, parse_python_imports, write_json, normalize_module_to_path

def build_reverse_importers(repo: Path) -> Dict[str, Set[str]]:
    py_files = list(iter_files(repo, suffixes=(".py",)))
    file_set = {str(p.relative_to(repo)) for p in py_files}

    def resolve_local(mod: str) -> str | None:
        base = normalize_module_to_path(mod)
        cand1 = f"{base}.py"
        cand2 = f"{base}/__init__.py"
        if cand1 in file_set:
            return cand1
        if cand2 in file_set:
            return cand2
        return None

    rev = defaultdict(set)
    for p in py_files:
        src = str(p.relative_to(repo))
        for imp in parse_python_imports(p, repo):
            if imp.level:
                continue
            tgt = resolve_local(imp.module)
            if tgt:
                rev[tgt].add(src)
    return rev

def find_orphans(repo: Path) -> Dict[str, Any]:
    py_files = sorted([str(p.relative_to(repo)) for p in iter_files(repo, suffixes=(".py",))])
    rev = build_reverse_importers(repo)

    # Seeds: entrypoints + any file named __main__.py or START_LOGOS.py
    seeds = [p for p in py_files if p.endswith("/__main__.py") or p.endswith("START_LOGOS.py") or p == "__main__.py"]
    reachable: Set[str] = set(seeds)
    q = deque(seeds)

    # Forward graph inferred from reverse: to walk forward we need adjacency; approximate by scanning imports again
    file_set = set(py_files)
    adj = defaultdict(set)
    for p in iter_files(repo, suffixes=(".py",)):
        src = str(p.relative_to(repo))
        for imp in parse_python_imports(p, repo):
            if imp.level:
                continue
            base = normalize_module_to_path(imp.module)
            cand1 = f"{base}.py"
            cand2 = f"{base}/__init__.py"
            if cand1 in file_set:
                adj[src].add(cand1)
            if cand2 in file_set:
                adj[src].add(cand2)

    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, set()):
            if nxt not in reachable:
                reachable.add(nxt)
                q.append(nxt)

    orphaned = [p for p in py_files if p not in reachable and not p.endswith("__init__.py")]
    return {
        "reachability_seeds": seeds,
        "reachable_count": len(reachable),
        "total_py_files": len(py_files),
        "orphaned_files": orphaned,
        "notes": "heuristic reachability; relative imports + runtime PYTHONPATH can change true reachability"
    }

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/10_dead_code")
    r = find_orphans(repo)
    write_json(base / "orphaned_files.json", r)
    write_json(base / "unreachable_modules.json", {"note": "module-level unreachable analysis can be derived from orphaned_files + naming scan"})
    print(base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
