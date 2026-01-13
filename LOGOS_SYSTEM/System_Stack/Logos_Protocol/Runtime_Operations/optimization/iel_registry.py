"""Compatibility proxy for the historical IEL registry implementation."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional

_TARGET_MODULE = (
    "System_Operations_Protocol.code_generator.meta_reasoning.iel_registry"
)
_delegate: ModuleType | None = None
__all__: List[str] = []


def _load_delegate() -> ModuleType:
    global _delegate, __all__
    if _delegate is None:
        _delegate = importlib.import_module(_TARGET_MODULE)
        exported = getattr(_delegate, "__all__", None)
        __all__ = list(exported) if exported is not None else [
            name for name in dir(_delegate) if not name.startswith("_")
        ]
    return _delegate


def __getattr__(name: str) -> Any:  # pragma: no cover - thin wrapper
    return getattr(_load_delegate(), name)


def __dir__() -> List[str]:  # pragma: no cover - thin wrapper
    loaded = _load_delegate()
    return sorted(set(globals()) | set(dir(loaded)))


def _load_registry(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return []


def _write_registry(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def _touch_db(path: Path) -> None:
    db_path = path.with_suffix(".db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch(exist_ok=True)


def _add_entry(registry_path: Path, artifact: Path, signature_path: Path) -> None:
    entries = _load_registry(registry_path)
    signature: dict[str, Any] = {}
    if signature_path.exists():
        try:
            signature = json.loads(signature_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            signature = {"signature": signature_path.read_text(encoding="utf-8")}  # type: ignore[assignment]

    entry = {
        "artifact": str(artifact),
        "signature": signature.get("signature", "unknown"),
        "algorithm": signature.get("algorithm", "SHA256"),
        "registered_at": time.time(),
    }
    entries.append(entry)
    _write_registry(registry_path, entries)
    _touch_db(registry_path)
    print(f"Registered IEL artifact {artifact} -> {registry_path}")


def _list_entries(registry_path: Path) -> None:
    entries = _load_registry(registry_path)
    print(f"Registry contains {len(entries)} entries")
    for entry in entries:
        print(f" - {entry['artifact']} ({entry['algorithm']})")


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - exercised by tests
    parser = argparse.ArgumentParser(description="Minimal IEL registry manager")
    parser.add_argument("--registry", type=Path, required=True, help="Registry JSON file")
    parser.add_argument("--add", type=Path, help="Add an IEL artifact to the registry")
    parser.add_argument("--sig", type=Path, help="Signature JSON file")
    parser.add_argument("--list", action="store_true", help="List registry contents")
    args = parser.parse_args(argv)

    if args.add:
        if not args.sig:
            raise SystemExit("--sig is required when using --add")
        _add_entry(args.registry, args.add, args.sig)
    if args.list:
        _list_entries(args.registry)
    if not args.add and not args.list:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    raise SystemExit(main())
