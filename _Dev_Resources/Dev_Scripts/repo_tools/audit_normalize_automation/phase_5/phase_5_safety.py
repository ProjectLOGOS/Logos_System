"""
LOGOS / Audit-Normalize Automation
Phase 5 — Safety Layer (Backup + Rollback)

Append-only JSONL backup log:
- Captures PRE state before any mutation
- Captures POST state after mutation
- Stores enough metadata to restore the original file atomically

Fail-closed:
- If backup cannot be written, mutation is forbidden
- If any invariant is violated, halt immediately
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


@dataclass(frozen=True)
class BackupRecord:
    ts: str
    task_id: str
    action: str
    src_path: str
    dest_path: Optional[str]
    pre_exists: bool
    pre_sha256: Optional[str]
    pre_bytes_b64: Optional[str]
    post_exists: Optional[bool] = None
    post_sha256: Optional[str] = None
    note: Optional[str] = None


def _b64(b: bytes) -> str:
    import base64

    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    import base64

    return base64.b64decode(s.encode("ascii"))


def append_backup_record(jsonl_path: Path, rec: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")


def capture_pre_state(task_id: str, action: str, src: Path, dest: Optional[Path], backup_jsonl: Path) -> Dict[str, Any]:
    pre_exists = src.exists()
    pre_hash = None
    pre_b64 = None
    if pre_exists and src.is_file():
        data = read_bytes(src)
        pre_hash = sha256_bytes(data)
        pre_b64 = _b64(data)

    rec = BackupRecord(
        ts=utc_now(),
        task_id=task_id,
        action=action,
        src_path=str(src.as_posix()),
        dest_path=str(dest.as_posix()) if dest else None,
        pre_exists=pre_exists,
        pre_sha256=pre_hash,
        pre_bytes_b64=pre_b64,
        note="PRE",
    )
    d = rec.__dict__
    append_backup_record(backup_jsonl, d)
    return d


def capture_post_state(task_id: str, action: str, src: Path, dest: Optional[Path], backup_jsonl: Path, note: str = "POST") -> Dict[str, Any]:
    post_exists = None
    post_hash = None
    target = dest if (dest and dest.exists()) else src
    post_exists = target.exists()
    if post_exists and target.is_file():
        data = read_bytes(target)
        post_hash = sha256_bytes(data)

    rec = {
        "ts": utc_now(),
        "task_id": task_id,
        "action": action,
        "src_path": str(src.as_posix()),
        "dest_path": str(dest.as_posix()) if dest else None,
        "post_exists": post_exists,
        "post_sha256": post_hash,
        "note": note,
    }
    append_backup_record(backup_jsonl, rec)
    return rec


def rollback_from_pre_record(pre: Dict[str, Any], repo_root: Path) -> None:
    src = repo_root / Path(pre["src_path"])
    pre_exists = bool(pre.get("pre_exists"))
    pre_b64 = pre.get("pre_bytes_b64")

    if pre_exists:
        if pre_b64 is None:
            raise RuntimeError("Rollback fail-closed: PRE exists but bytes missing.")
        data = _unb64(pre_b64)
        write_bytes_atomic(src, data)
    else:
        if src.exists():
            if src.is_file():
                src.unlink()
            else:
                raise RuntimeError("Rollback fail-closed: src exists and is not a file.")
