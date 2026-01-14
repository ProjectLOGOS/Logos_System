#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TITLE_CASE_WITH_UNDERSCORES_RE = re.compile(r"^[A-Z][A-Za-z0-9]*(?:_[A-Z][A-Za-z0-9]*)*$")

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "env",
    "node_modules",
    "_build", "dist", "build",
}

DEFAULT_EXCLUDE_PREFIXES = (
    str(Path("_Reports")),
    str(Path("_reports")),
)

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def is_excluded_path(path: Path, repo_root: Path, extra_excludes: Optional[Iterable[str]] = None) -> bool:
    rel = path.relative_to(repo_root)
    parts = rel.parts
    if any(p in DEFAULT_EXCLUDE_DIRS for p in parts):
        return True
    rel_str = str(rel)
    if rel_str.startswith(DEFAULT_EXCLUDE_PREFIXES):
        return True
    if extra_excludes:
        for ex in extra_excludes:
            if rel_str.startswith(ex):
                return True
    return False

def iter_files(repo_root: Path, suffixes: Optional[Tuple[str, ...]] = None, extra_excludes: Optional[Iterable[str]] = None) -> Iterable[Path]:
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if is_excluded_path(p, repo_root, extra_excludes=extra_excludes):
            continue
        if suffixes and p.suffix not in suffixes:
            continue
        yield p

def title_case_violation(name: str) -> bool:
    # Ignore dotfiles and some known build artifacts
    if name.startswith("."):
        return False
    return TITLE_CASE_WITH_UNDERSCORES_RE.match(name) is None

@dataclass
class PyImport:
    importer_file: str
    line: int
    col: int
    kind: str  # "import" | "from"
    module: str
    name: Optional[str]
    level: int  # relative import level

def parse_python_imports(py_path: Path, repo_root: Path) -> List[PyImport]:
    src = read_text(py_path)
    out: List[PyImport] = []
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return out
    rel_file = str(py_path.relative_to(repo_root))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(PyImport(
                    importer_file=rel_file,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    kind="import",
                    module=alias.name,
                    name=alias.asname,
                    level=0
                ))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                out.append(PyImport(
                    importer_file=rel_file,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    kind="from",
                    module=mod,
                    name=alias.name,
                    level=int(getattr(node, "level", 0) or 0)
                ))
    return out

def normalize_module_to_path(module: str) -> str:
    return module.replace(".", "/")

def looks_like_stdlib(mod: str) -> bool:
    # Heuristic only; final classification uses importlib metadata where possible.
    return mod in {
        "json","datetime","pathlib","typing","re","os","sys","hashlib","subprocess","time","math",
        "itertools","functools","collections","logging","asyncio","dataclasses","enum","uuid",
        "http","urllib","socket","threading","multiprocessing","queue","statistics","fractions",
        "traceback","contextlib","copy","pprint","csv","gzip","bz2","lzma","shutil","tempfile",
    }
