"""Compatibility proxy for the historical IEL signer implementation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional

_TARGET_MODULE = (
    "System_Operations_Protocol.code_generator.meta_reasoning.iel_signer"
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


def _compute_signature(artifact: Path, key_path: Path) -> str:
    payload = artifact.read_bytes() if artifact.exists() else b""
    key_material = key_path.read_bytes() if key_path.exists() else b""
    digest = hashlib.sha256()
    digest.update(payload)
    digest.update(key_material)
    return digest.hexdigest()


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - exercised by tests
    parser = argparse.ArgumentParser(description="Minimal IEL signer")
    parser.add_argument("--sign", type=Path, help="Candidate IEL file")
    parser.add_argument("--key", type=Path, help="Signing key path")
    parser.add_argument("--out", type=Path, help="Output signature json file")
    args = parser.parse_args(argv)

    if not args.sign or not args.out:
        parser.error("--sign and --out are required to create a signature")

    key_path = args.key or Path("keys/iel_signing.pem")
    signature = _compute_signature(Path(args.sign), Path(key_path))
    payload = {
        "artifact": str(args.sign),
        "signature": signature,
        "algorithm": "SHA256",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Stored signature at {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    raise SystemExit(main())
