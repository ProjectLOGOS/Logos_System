from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import json
import re
from typing import Union, Iterable, Optional, Tuple


@dataclass(frozen=True, slots=True)
class ImportRecord:
    """
    Minimal record shape expected by scan_imports.py:
      importer_file, line, col, kind, module, name, level
    """

    importer_file: str
    line: int
    col: int
    kind: str  # "import" | "from"
    module: str  # e.g. "os.path" for `from os import path`; "" if unknown
    name: str  # imported symbol/module name (alias.name)
    level: int  # relative import level for ImportFrom; 0 otherwise


# Exclusion defaults mirror the legacy system_audit helpers
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "_build",
    "dist",
    "build",
}

DEFAULT_EXCLUDE_PREFIXES = (
    str(Path("_Reports")),
    str(Path("_reports")),
)


def iter_files(root: Path, suffix: str = ".py", suffixes=None):
    """Yield files matching suffix or suffixes (tuple) under root."""
    patterns = []
    if suffixes:
        for s in suffixes:
            patterns.append(f"*{s}")
    else:
        patterns.append(f"*{suffix}")

    seen = set()
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


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


TITLE_CASE_WITH_UNDERSCORES_RE = re.compile(r"^[A-Z][A-Za-z0-9]*(?:_[A-Z][A-Za-z0-9]*)*$")


def title_case_violation(name: str) -> bool:
    # Ignore dotfiles and some known build artifacts
    if name.startswith("."):
        return False
    return TITLE_CASE_WITH_UNDERSCORES_RE.match(name) is None


def read_text(path: Union[str, Path]) -> str:
    """Read text with utf-8 and fallback replacement on errors."""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="replace")

def parse_python_imports(py_file: Union[str, Path], repo: Path | None = None) -> list[ImportRecord]:
    """
    Parse a Python file and return a list of ImportRecord objects.
    This function is intentionally "shape-stable" for scan_imports.py.
    Fail-closed: syntax errors return empty list (caller logs per-file failures).
    """

    p = Path(py_file)
    try:
        src = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = p.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(src, filename=str(p))
    except SyntaxError:
        return []

    out: list[ImportRecord] = []
    importer_file = str(p)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(
                    ImportRecord(
                        importer_file=importer_file,
                        line=int(getattr(node, "lineno", 0) or 0),
                        col=int(getattr(node, "col_offset", 0) or 0),
                        kind="import",
                        module="",
                        name=str(alias.name),
                        level=0,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            lvl = int(getattr(node, "level", 0) or 0)
            for alias in node.names:
                out.append(
                    ImportRecord(
                        importer_file=importer_file,
                        line=int(getattr(node, "lineno", 0) or 0),
                        col=int(getattr(node, "col_offset", 0) or 0),
                        kind="from",
                        module=str(mod),
                        name=str(alias.name),
                        level=lvl,
                    )
                )

    return out

def normalize_module_to_path(module: str):
    return module.replace(".", "/") + ".py"

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
