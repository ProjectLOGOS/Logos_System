#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _common import iter_files, read_text, write_json

def scan_symbols(repo: Path) -> Dict[str, Any]:
    classes: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    globals_: List[Dict[str, Any]] = []

    for p in iter_files(repo, suffix=".py"):
        src = read_text(p)
        try:
            tree = ast.parse(src, filename=str(p))
        except SyntaxError:
            continue

        rel = str(p.relative_to(repo))

        # globals: top-level assignments
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = []
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            targets.append(t.id)
                else:
                    if isinstance(node.target, ast.Name):
                        targets.append(node.target.id)
                for name in targets:
                    globals_.append({
                        "file": rel,
                        "line": getattr(node, "lineno", 0),
                        "name": name,
                        "kind": type(node).__name__,
                    })

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(getattr(b, "attr", ""))
                    else:
                        bases.append(type(b).__name__)
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({
                    "file": rel,
                    "line": getattr(node, "lineno", 0),
                    "class_name": node.name,
                    "bases": bases,
                    "methods": methods,
                    "decorators": [getattr(d, "id", getattr(d, "attr", type(d).__name__)) for d in node.decorator_list],
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only count top-level functions
                if isinstance(getattr(node, "parent", None), ast.ClassDef):
                    continue
                functions.append({
                    "file": rel,
                    "line": getattr(node, "lineno", 0),
                    "function_name": node.name,
                    "args": [a.arg for a in node.args.args],
                    "decorators": [getattr(d, "id", getattr(d, "attr", type(d).__name__)) for d in node.decorator_list],
                    "async": isinstance(node, ast.AsyncFunctionDef),
                })

        # annotate parents to distinguish methods
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, "parent", parent)

    return {"classes": classes, "functions": functions, "globals": globals_}

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/03_symbol_inventory")
    sym = scan_symbols(repo)
    write_json(base / "classes.json", sym["classes"])
    write_json(base / "functions.json", sym["functions"])
    write_json(base / "globals.json", sym["globals"])
    # placeholders (populated later by refactor tooling)
    write_json(base / "methods.json", {"note": "methods are included per-class in classes.json"})
    write_json(base / "deprecated_candidates.json", {"note": "populate after dead-code + import reachability analysis"})
    print(base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
