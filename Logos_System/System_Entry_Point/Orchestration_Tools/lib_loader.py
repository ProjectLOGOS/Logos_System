"""Passive external library loader scaffold for LOGOS orchestration tools.

Features:
- Central registry for imported modules.
- Regulated access gate to avoid silent side effects.
- Optional auto-install (disabled by default).
- Preload helper to import a set of libraries defined in runtime_requirements.json.

Usage:
    from lib_loader import preload_libraries, require, get, enable
    preload_libraries()  # imports according to runtime_requirements.json
    nltk = get("nltk")
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

# Configuration flags. Keep defaults conservative for safety.
AUTO_INSTALL = False  # When True, missing packages will be pip-installed.
REGULATE_ACCESS = True  # When True, access goes through RegulatedLibrary wrappers.

# Registry maps library name to imported module (or regulated wrapper).
REGISTRY: Dict[str, object] = {}
# Tracks which libraries have been explicitly enabled for access.
ENABLED: set[str] = set()

REQUIREMENTS_FILE = Path(__file__).with_name("runtime_requirements.json")


class RegulatedLibrary:
    """Simple gate that prevents access until explicitly enabled."""

    def __init__(self, name: str, module: object):
        self._name = name
        self._module = module

    def enable(self) -> None:
        ENABLED.add(self._name)

    def __getattr__(self, item):
        if self._name not in ENABLED:
            raise PermissionError(
                f"Library '{self._name}' is not enabled; call enable('{self._name}') first."
            )
        return getattr(self._module, item)


def _install(lib_name: str) -> None:
    """Install a library via pip. Honors AUTO_INSTALL flag."""
    if not AUTO_INSTALL:
        raise RuntimeError(f"Missing library '{lib_name}' and AUTO_INSTALL is disabled.")

    cmd = [sys.executable, "-m", "pip", "install", lib_name]
    subprocess.run(cmd, check=True)


def require(lib_name: str) -> object:
    """Import a library, optionally installing if missing, and register it."""
    if lib_name in REGISTRY:
        return REGISTRY[lib_name]

    try:
        module = importlib.import_module(lib_name)
    except ImportError:
        _install(lib_name)
        module = importlib.import_module(lib_name)

    wrapped = RegulatedLibrary(lib_name, module) if REGULATE_ACCESS else module
    REGISTRY[lib_name] = wrapped
    return wrapped


def get(lib_name: str) -> object:
    """Retrieve a library from the registry (must be preloaded or required)."""
    if lib_name not in REGISTRY:
        raise KeyError(f"Library '{lib_name}' not found in registry. Call require() or preload_libraries().")
    return REGISTRY[lib_name]


def enable(lib_name: str) -> None:
    """Enable access for a regulated library."""
    if lib_name not in REGISTRY:
        raise KeyError(f"Library '{lib_name}' not loaded; cannot enable.")
    ENABLED.add(lib_name)


def enable_all() -> None:
    """Enable access for all loaded libraries."""
    ENABLED.update(REGISTRY.keys())


def _load_requirements(requirements_path: Path) -> dict:
    with requirements_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def preload_libraries(
    requirements_path: Optional[Path | str] = None,
    categories: Optional[Iterable[str]] = None,
) -> None:
    """Preload libraries from requirements JSON. If categories is provided, only preload those groups."""
    req_path = Path(requirements_path) if requirements_path else REQUIREMENTS_FILE
    data = _load_requirements(req_path)

    selected = data
    if categories is not None:
        categories_set = set(categories)
        selected = {k: v for k, v in data.items() if k in categories_set}

    for group, libs in selected.items():
        for lib in libs:
            try:
                require(lib)
            except Exception as exc:  # noqa: BLE001
                # Keep preload non-fatal; report and continue.
                print(f"[lib_loader] Failed to load '{lib}' from group '{group}': {exc}")


__all__ = [
    "AUTO_INSTALL",
    "REGULATE_ACCESS",
    "REGISTRY",
    "ENABLED",
    "preload_libraries",
    "require",
    "get",
    "enable",
    "enable_all",
    "RegulatedLibrary",
]
