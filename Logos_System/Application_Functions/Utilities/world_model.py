# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Unified World Model (UWM) snapshot utilities."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional, Tuple

SNAPSHOT_VERSION = 1
DEFAULT_IDENTITY_PATH = Path("state/agent_identity.json")
DEFAULT_SNAPSHOT_PATH = Path("state/world_model_snapshot.json")
DEFAULT_CATALOG_PATH = Path("training_data/index/catalog.jsonl")
DEFAULT_PLANNER_DIGEST_POINTER = Path("state/latest_planner_digest_archive.txt")
DEFAULT_MISSION_PROFILE_PATH = Path("state/mission_profile.json")
DEFAULT_BELIEFS_FEED_PATH = Path("state/beliefs_feed.jsonl")
MAX_TIER_ENTRIES = 200
CATALOG_TAIL_LINES = 200
NOTES_MAX_CHARS = 500


def canonical_json(obj: Any) -> bytes:
    """Return canonical JSON bytes for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return a sha256 digest string with prefix."""
    import hashlib

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON payload to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_latest_pointer(pointer_path: Path) -> Optional[Path]:
    if not pointer_path.exists():
        return None
    content = pointer_path.read_text(encoding="utf-8").strip()
    if not content:
        return None
    return Path(content)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def _read_catalog_entries(path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(path)


def _read_last_lines(path: Path, max_lines: int) -> List[str]:
    lines: List[str] = []
    if not path.exists():
        return lines
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.rstrip("\n")
            lines.append(raw)
            if len(lines) > max_lines:
                lines.pop(0)
    return lines


def _hash_directory_listing(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_dir():
        return None
    entries: List[str] = []
    for root, _dirs, files in os.walk(path):
        files.sort()
        rel_root = Path(root).relative_to(path)
        for filename in files:
            entries.append(str(rel_root / filename))
    payload = "\n".join(entries).encode("utf-8")
    return sha256_bytes(payload)


def _hash_directory(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_dir():
        return None
    payload: List[str] = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        rel_root = Path(root).relative_to(path)
        for directory in dirs:
            payload.append(str(rel_root / directory) + "/")
        for filename in files:
            payload.append(str(rel_root / filename))
    data = "\n".join(payload).encode("utf-8")
    return sha256_bytes(data)


def _compute_catalog_tail_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) > CATALOG_TAIL_LINES:
        lines = lines[-CATALOG_TAIL_LINES:]
    tail_content = "".join(lines)
    return _sha256_hex(tail_content.encode("utf-8"))


def _gather_beliefs(feed_path: Path) -> List[Dict[str, Any]]:
    entries = _read_jsonl(feed_path)
    entries.sort(key=lambda item: item.get("timestamp_utc", ""), reverse=True)
    return entries[:MAX_TIER_ENTRIES]


def _gather_provisionals(catalog_entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    provisionals: List[Dict[str, Any]] = []
    for entry in catalog_entries:
        policy_class = entry.get("policy_class")
        if policy_class in {"repair", "enhancement"}:
            provisionals.append(
                {
                    "entry_id": entry.get("entry_id"),
                    "label": entry.get("improvement_type") or entry.get("label"),
                    "status": "deployed" if entry.get("deployed") else "staged",
                    "provenance": {
                        "policy_class": policy_class,
                        "artifact_path": entry.get("artifact_path"),
                        "deployment_path": entry.get("deployment_path"),
                    },
                }
            )
    provisionals.sort(key=lambda item: item["entry_id"] or "")
    if len(provisionals) > MAX_TIER_ENTRIES:
        provisionals = provisionals[-MAX_TIER_ENTRIES:]
    return provisionals


def _corpus_reference(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        return {"source": "absent", "ref": None, "hash": None, "note": "interpretive"}
    return {
        "source": label,
        "ref": str(path),
        "hash": _hash_directory(path),
        "note": "interpretive",
    }


def load_snapshot(snapshot_path: Path = DEFAULT_SNAPSHOT_PATH) -> Optional[Dict[str, Any]]:
    """Load snapshot JSON if present."""
    return _load_json(snapshot_path)


def validate_snapshot(snapshot: Dict[str, Any], identity: Dict[str, Any], repo_root: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """Validate snapshot consistency against identity."""
    reasons: List[str] = []
    proof_gate = identity.get("proof_gate", {})
    context = snapshot.get("context", {})
    integrity = snapshot.get("integrity", {})

    if snapshot.get("truth_core", {}).get("theory_hash") != proof_gate.get("theory_hash"):
        reasons.append("theory_hash mismatch")

    identity_mission = identity.get("mission", {})
    if identity_mission.get("mission_profile_hash") and context.get("mission_profile_hash"):
        if identity_mission["mission_profile_hash"] != context["mission_profile_hash"]:
            reasons.append("mission_profile_hash mismatch")

    identity_cap = identity.get("capabilities", {})
    if identity_cap.get("catalog_tail_hash") and context.get("catalog_tail_hash"):
        if identity_cap["catalog_tail_hash"] != context["catalog_tail_hash"]:
            reasons.append("catalog_tail_hash mismatch")

    planner_digest = context.get("last_planner_digest_archive")
    if planner_digest:
        digest_path = Path(planner_digest)
        if repo_root and not digest_path.is_absolute():
            digest_path = (repo_root / digest_path).resolve()
        if not digest_path.exists():
            reasons.append(f"planner digest missing: {planner_digest}")

    calculated_hash = integrity.get("snapshot_hash")
    sanitized = dict(snapshot)
    sanitized_integrity = dict(integrity)
    sanitized_integrity["snapshot_hash"] = None
    sanitized["integrity"] = sanitized_integrity
    recomputed = sha256_bytes(canonical_json(sanitized))
    if calculated_hash and calculated_hash != recomputed:
        reasons.append("snapshot hash mismatch")

    return not reasons, reasons


def build_snapshot(
    identity: Dict[str, Any],
    catalog_path: Path,
    planner_digest_path: Optional[Path],
    mission_profile_path: Optional[Path],
    beliefs_feed_path: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    created_utc = datetime.now(timezone.utc).isoformat()
    proof_gate = identity.get("proof_gate", {})

    tier1_refs: List[Dict[str, Any]] = [
        {
            "kind": "theory_hash",
            "ref": proof_gate.get("theory_hash"),
            "hash": proof_gate.get("theory_hash"),
        }
    ]

    pxl_core_dir = repo_root / "external" / "Logos_AGI" / "PXL_core"
    if pxl_core_dir.exists():
        tier1_refs.append(
            {
                "kind": "module",
                "ref": str(pxl_core_dir),
                "hash": _hash_directory(pxl_core_dir),
            }
        )

    catalog_entries = _read_catalog_entries(catalog_path)
    provisionals = _gather_provisionals(catalog_entries)
    beliefs = _gather_beliefs(beliefs_feed_path)

    data_root = repo_root / "data"
    scripture_ref = _corpus_reference(data_root / "scripture", "scripture")
    patristics_ref = _corpus_reference(data_root / "patristics", "patristics")

    catalog_tail_hash = _compute_catalog_tail_hash(catalog_path)
    mission_profile_hash: Optional[str] = None
    if mission_profile_path and mission_profile_path.exists():
        mission_profile_hash = _sha256_hex(mission_profile_path.read_bytes())

    state_dir = repo_root / "state"
    protopraxis_dir = repo_root / "Protopraxis"

    notes = f"Snapshot generated {created_utc}"
    if len(notes) > NOTES_MAX_CHARS:
        notes = notes[:NOTES_MAX_CHARS]

    snapshot: Dict[str, Any] = {
        "world_model_version": SNAPSHOT_VERSION,
        "created_utc": created_utc,
        "truth_core": {
            "theory_hash": proof_gate.get("theory_hash"),
            "axiom_policy_profile": proof_gate.get("axiom_policy_profile"),
            "tier1_refs": tier1_refs,
        },
        "beliefs_tier_2": beliefs,
        "provisionals_tier_3": provisionals,
        "corpora_tier_3": {
            "scripture": scripture_ref,
            "patristics": patristics_ref,
        },
        "context": {
            "mission_label": identity.get("mission", {}).get("mission_label"),
            "mission_profile_hash": mission_profile_hash
            or identity.get("mission", {}).get("mission_profile_hash"),
            "last_planner_digest_archive": str(planner_digest_path) if planner_digest_path else None,
            "catalog_tail_hash": catalog_tail_hash
            or identity.get("capabilities", {}).get("catalog_tail_hash"),
            "notes": notes,
            "state_reference": {
                "root": str(state_dir),
                "hash": _hash_directory_listing(state_dir),
            },
            "protopraxis_reference": {
                "root": str(protopraxis_dir),
                "hash": _hash_directory_listing(protopraxis_dir),
            },
        },
        "integrity": {"snapshot_hash": None},
    }
    tooling_refs = _tooling_references(repo_root)
    if tooling_refs:
        snapshot["tooling"] = tooling_refs
    return snapshot

def _tooling_references(repo_root: Path) -> Optional[Dict[str, str]]:
    tooling_dir = repo_root / "state" / "tool_optimizer"
    files = {
        "tool_registry_ref": tooling_dir / "tool_registry.json",
        "tool_chain_profiles_ref": tooling_dir / "tool_chain_profiles.json",
        "tool_optimizer_report_ref": tooling_dir / "tool_optimizer_report.json",
    }
    refs: Dict[str, str] = {}
    for key, path in files.items():
        if path.exists() and path.is_file():
            try:
                digest = _sha256_hex(path.read_bytes())
            except OSError:
                continue
            refs[key] = f"{str(path)}#{digest}"
    return refs or None


def write_snapshot(snapshot_path: Path, snapshot: Dict[str, Any]) -> str:
    sanitized = dict(snapshot)
    sanitized_integrity = dict(snapshot.get("integrity", {}))
    sanitized_integrity["snapshot_hash"] = None
    sanitized["integrity"] = sanitized_integrity
    digest = sha256_bytes(canonical_json(sanitized))
    snapshot["integrity"]["snapshot_hash"] = digest
    atomic_write_json(snapshot_path, snapshot)
    return digest


def update_world_model(
    identity_path: Path = DEFAULT_IDENTITY_PATH,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    catalog_path: Optional[Path] = None,
    planner_digest_path: Optional[Path] = None,
    mission_profile_path: Optional[Path] = None,
    beliefs_feed_path: Path = DEFAULT_BELIEFS_FEED_PATH,
    planner_digest_pointer: Path = DEFAULT_PLANNER_DIGEST_POINTER,
) -> Dict[str, Any]:
    identity = _load_json(identity_path)
    if not identity:
        raise FileNotFoundError(f"Identity record not found at {identity_path}")

    repo_root = Path(identity.get("repo", {}).get("root", ".")).resolve()

    if catalog_path:
        resolved_catalog = catalog_path if catalog_path.is_absolute() else (repo_root / catalog_path)
    else:
        identity_catalog = identity.get("capabilities", {}).get("catalog_path")
        if identity_catalog:
            resolved_catalog = repo_root / identity_catalog
        else:
            resolved_catalog = repo_root / DEFAULT_CATALOG_PATH
    resolved_catalog = resolved_catalog.resolve()
    if not resolved_catalog.exists():
        resolved_catalog = (repo_root / DEFAULT_CATALOG_PATH).resolve()

    resolved_mission_profile = (mission_profile_path or repo_root / DEFAULT_MISSION_PROFILE_PATH)
    if not resolved_mission_profile.exists():
        resolved_mission_profile = None

    resolved_planner_digest = planner_digest_path
    if not resolved_planner_digest or not resolved_planner_digest.exists():
        pointer_path = planner_digest_pointer if planner_digest_pointer.is_absolute() else repo_root / planner_digest_pointer
        pointer_target = _read_latest_pointer(pointer_path)
        if pointer_target:
            resolved_planner_digest = (pointer_target if pointer_target.is_absolute() else (repo_root / pointer_target)).resolve()
    if resolved_planner_digest and not resolved_planner_digest.exists():
        resolved_planner_digest = None

    resolved_beliefs_feed = beliefs_feed_path if beliefs_feed_path.is_absolute() else repo_root / beliefs_feed_path

    snapshot = build_snapshot(
        identity=identity,
        catalog_path=resolved_catalog,
        planner_digest_path=resolved_planner_digest,
        mission_profile_path=resolved_mission_profile,
        beliefs_feed_path=resolved_beliefs_feed,
        repo_root=repo_root,
    )
    snapshot_file = snapshot_path if snapshot_path.is_absolute() else repo_root / snapshot_path
    digest = write_snapshot(snapshot_file, snapshot)
    return {
        "snapshot_hash": digest,
        "snapshot_path": str(snapshot_file),
        "world_model_version": snapshot.get("world_model_version", SNAPSHOT_VERSION),
    }
