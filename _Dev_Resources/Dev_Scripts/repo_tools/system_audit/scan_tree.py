#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from _common import is_excluded_path, write_json

def build_tree(repo_root: Path) -> Dict[str, Any]:
    def node_for(path: Path) -> Dict[str, Any]:
        return {"name": path.name, "type": "dir" if path.is_dir() else "file", "children": []}

    root = {"name": repo_root.name, "type": "dir", "children": []}
    index = {repo_root: root}

    for p in sorted(repo_root.rglob("*")):
        if is_excluded_path(p, repo_root):
            continue
        parent = p.parent
        if is_excluded_path(parent, repo_root):
            continue
        if parent not in index:
            # Ensure parents exist
            cur = parent
            stack = []
            while cur not in index and cur != repo_root:
                stack.append(cur)
                cur = cur.parent
            while stack:
                d = stack.pop()
                nd = node_for(d)
                index[d] = nd
                index[d.parent]["children"].append(nd)

        n = node_for(p)
        index[p] = n
        index[parent]["children"].append(n)

    return root

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    out = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/08_directory_tree/tree.json")
    write_json(out, build_tree(repo))
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
