#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from _common import iter_files, sha256_file, write_json

def scan_duplicates(repo: Path) -> Dict[str, Any]:
    by_hash = defaultdict(list)
    by_name = defaultdict(list)

    for p in iter_files(repo):
        rel = str(p.relative_to(repo))
        by_name[p.name].append(rel)
        try:
            h = sha256_file(p)
        except Exception:
            continue
        by_hash[h].append(rel)

    dup_files = [{"hash": h, "files": paths} for h, paths in by_hash.items() if len(paths) > 1]
    dup_names = [{"name": n, "paths": paths} for n, paths in by_name.items() if len(paths) > 1]

    # directory duplicates by basename
    dir_by_name = defaultdict(list)
    for d in repo.rglob("*"):
        if not d.is_dir():
            continue
        rel = str(d.relative_to(repo))
        dir_by_name[d.name].append(rel)
    dup_dirs = [{"name": n, "paths": paths} for n, paths in dir_by_name.items() if len(paths) > 1]

    return {
        "duplicate_files_by_hash": sorted(dup_files, key=lambda x: (-len(x["files"]), x["hash"])),
        "duplicate_file_names": sorted(dup_names, key=lambda x: (-len(x["paths"]), x["name"])),
        "duplicate_directories": sorted(dup_dirs, key=lambda x: (-len(x["paths"]), x["name"])),
    }

def main() -> int:
    repo = Path("/workspaces/Logos_System").resolve()
    base = Path("/workspaces/Logos_System/_Reports/SYSTEM_AUDIT/04_duplicates")
    dup = scan_duplicates(repo)
    write_json(base / "semantic_duplicates.json", dup["duplicate_files_by_hash"])
    write_json(base / "duplicate_files.json", dup["duplicate_file_names"])
    write_json(base / "duplicate_directories.json", dup["duplicate_directories"])
    print(base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
