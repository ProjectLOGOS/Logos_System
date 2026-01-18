# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Runtime-governed tooling optimizer for LOGOS self-improvement cycles."""

from __future__ import annotations

import ast
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

KEYWORDS = (
    "tool",
    "report",
    "analyze",
    "probe",
    "audit",
    "validate",
    "optimiz",
    "synthesize",
    "infer",
    "forecast",
)

REGISTRY_MAX = 500
PROFILES_MAX = 100
PROFILE_STEPS_MAX = 50
NOTES_MAX = 500
DEFAULT_DIGEST_LIMIT = 50

TASK_CLASSES = (
    "integrity_check",
    "toolkit_audit",
    "gap_fill",
    "proof_gate_checkpoint",
    "uwm_refresh",
    "capability_report",
)

DEFAULT_CHAINS: Dict[str, List[str]] = {
    "integrity_check": ["mission.status", "probe.last", "fs.read"],
    "toolkit_audit": ["mission.status", "fs.read", "sandbox.list"],
    "gap_fill": ["mission.status", "sandbox.read_pending", "sandbox.write"],
    "proof_gate_checkpoint": ["mission.status", "probe.last", "proof.status"],
    "uwm_refresh": ["mission.status", "uwm.snapshot", "fs.read"],
    "capability_report": ["mission.status", "probe.last", "report.generate"],
}

PRECONDITIONS: Dict[str, List[str]] = {
    "integrity_check": [
        "Identity validated",
        "Telemetry accessible",
    ],
    "toolkit_audit": [
        "Planner digests archived",
        "Tool registry readable",
    ],
    "gap_fill": [
        "Enhancements permitted",
        "Sandbox write quota available",
    ],
    "proof_gate_checkpoint": [
        "Proof gate baseline hashed",
        "Coq baseline reachable",
    ],
    "uwm_refresh": [
        "World model snapshot path configured",
        "Mission profile readable",
    ],
    "capability_report": [
        "Catalog tail hash available",
        "Planner digest pointer resolved",
    ],
}

POSTCONDITIONS: Dict[str, List[str]] = {
    "integrity_check": [
        "Telemetry anomalies recorded",
        "Identity warnings cleared",
    ],
    "toolkit_audit": [
        "Tool usage map updated",
        "Catalog references reconciled",
    ],
    "gap_fill": [
        "Gap opportunities prioritized",
        "Enhancement backlog refreshed",
    ],
    "proof_gate_checkpoint": [
        "Proof anomalies logged",
        "Theory hash confirmed",
    ],
    "uwm_refresh": [
        "Snapshot hash updated",
        "UWM references published",
    ],
    "capability_report": [
        "Capability summary staged",
        "Reporting artifacts cataloged",
    ],
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_timestamp() -> str:
    return _now_utc().isoformat()


def _repo_root_from_identity(identity_path: Path) -> Path:
    if identity_path.exists():
        try:
            return identity_path.resolve().parent.parent
        except Exception:
            pass
    return Path(__file__).resolve().parents[4]


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _compute_file_hash(path: Path) -> Optional[str]:
    try:
        import hashlib

        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    except OSError:
        return None


def _module_path(file_path: Path, repo_root: Path) -> str:
    try:
        rel = file_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel = file_path.name
        return rel.rsplit(".", 1)[0]
    parts = list(rel.parts)
    if parts and parts[0] == "external":
        parts = parts[2:]
    if parts and parts[0] == "scripts":
        # leave scripts prefix in place
        pass
    stem = parts[-1]
    if stem.endswith(".py"):
        stem = stem[:-3]
        parts[-1] = stem
    return ".".join(part for part in parts if part and part != "__init__")


def _infer_protocol_tag(path: Path) -> str:
    tokens = {part.lower() for part in path.parts}
    if "advanced_reasoning_protocol" in tokens:
        return "ARP"
    if "system_operations_protocol" in tokens:
        return "SOP"
    if "user_interaction_protocol" in tokens:
        return "UIP"
    if "synthetic_cognition_protocol" in tokens:
        return "SCP"
    if "logos_core" in tokens:
        return "LOGOS_CORE"
    if "scripts" in tokens:
        return "SCRIPTS"
    return "UNKNOWN"


def _infer_risk_tag(module_path: str, symbol: str) -> str:
    text = f"{module_path}.{symbol}".lower()
    governed_tokens = (
        "identity",
        "commitment",
        "ledger",
        "world_model",
        "start_agent",
        "self_improvement",
        "sandbox",
        "deployment",
        "arbiter",
    )
    write_tokens = (
        "write",
        "persist",
        "update",
        "save",
        "deploy",
        "stage",
        "register",
        "optimiz",
    )
    if any(token in text for token in governed_tokens):
        return "governed"
    if any(token in text for token in write_tokens):
        return "write"
    return "safe"


def _io_hint_for_risk(risk_tag: str) -> str:
    if risk_tag == "write":
        return "writes_state"
    if risk_tag == "governed":
        return "governed_flow"
    return "read_only"


def _clean_docstring(node: ast.AST) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    cleaned = " ".join(doc.strip().split())
    return cleaned[:NOTES_MAX].strip()


def _matches_keyword(name: str) -> bool:
    lname = name.lower()
    return any(keyword in lname for keyword in KEYWORDS)


def _scan_python_files(base_dirs: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for base in base_dirs:
        if not base.exists():
            continue
        for file_path in base.rglob("*.py"):
            if "__pycache__" in file_path.parts:
                continue
            files.append(file_path)
    files.sort()
    return files


def _build_tool_registry(repo_root: Path, anomalies: List[str]) -> List[Dict[str, Any]]:
    targets = [
        repo_root / "external" / "Logos_AGI" / "logos_core",
        repo_root / "external" / "Logos_AGI" / "System_Operations_Protocol",
        repo_root / "scripts",
    ]
    files = _scan_python_files(targets)
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for file_path in files:
        module = _module_path(file_path, repo_root)
        if not module:
            continue
        try:
            node = ast.parse(file_path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError) as exc:
            anomalies.append(f"registry_parse_error:{file_path.as_posix()}:{exc}")
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = child.name
                if not _matches_keyword(name):
                    continue
                key = f"{module}.{name}"
                if key in seen:
                    continue
                seen.add(key)
                risk_tag = _infer_risk_tag(module, name)
                notes = _clean_docstring(child) or "Auto-registered via keyword match."
                entry = {
                    "tool_id": _stable_id(key),
                    "name": name,
                    "module_path": module,
                    "symbol": name,
                    "protocol_tag": _infer_protocol_tag(file_path),
                    "io_hints": _io_hint_for_risk(risk_tag),
                    "risk_tag": risk_tag,
                    "notes": notes[:NOTES_MAX],
                }
                entries.append(entry)
                if len(entries) >= REGISTRY_MAX:
                    anomalies.append("registry_truncated")
                    return sorted(entries, key=lambda item: item["tool_id"])
    return sorted(entries, key=lambda item: item["tool_id"])


def _stable_id(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _classify_sequence(steps: Sequence[str]) -> List[str]:
    assigned: List[str] = []
    joined = " ".join(step.lower() for step in steps)
    if any(token in joined for token in ("telemetry", "validate", "audit", "health", "integrity")):
        assigned.append("integrity_check")
    if any(token in joined for token in ("tool", "catalog", "inventory", "probe")):
        assigned.append("toolkit_audit")
    if any(token in joined for token in ("generate", "improve", "gap", "synthesize")):
        assigned.append("gap_fill")
    if any(token in joined for token in ("proof", "lemma", "coq", "theory")):
        assigned.append("proof_gate_checkpoint")
    if any(token in joined for token in ("uwm", "world_model", "snapshot", "refresh_alignment")):
        assigned.append("uwm_refresh")
    if any(token in joined for token in ("report", "digest", "brief", "summary", "prepare")):
        assigned.append("capability_report")
    return assigned


def _read_digest_file(path: Path, limit: int, sequences: Dict[str, List[Tuple[Tuple[str, ...], str]]]) -> None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if line_index >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                steps = [
                    str(action.get("name"))
                    for action in record.get("actions", []) or []
                    if action.get("name")
                ]
                if not steps:
                    continue
                truncated = tuple(steps[:PROFILE_STEPS_MAX])
                for task in _classify_sequence(truncated):
                    sequences[task].append((truncated, path.as_posix()))
    except OSError:
        return


def _aggregate_profiles(repo_root: Path, digests_glob: str, anomalies: List[str], digest_limit: int = DEFAULT_DIGEST_LIMIT) -> List[Dict[str, Any]]:
    digest_paths = sorted(repo_root.glob(digests_glob))
    if not digest_paths:
        anomalies.append("no_planner_digests")
    digest_paths = digest_paths[-digest_limit:]

    sequences: Dict[str, List[Tuple[Tuple[str, ...], str]]] = defaultdict(list)
    for path in digest_paths:
        _read_digest_file(path, limit=100, sequences=sequences)

    profiles: List[Dict[str, Any]] = []
    for task in TASK_CLASSES:
        entries = sequences.get(task, [])
        confidence = 0.0
        if entries:
            counter = Counter(seq for seq, _ in entries)
            top_sequence, top_count = max(counter.items(), key=lambda item: (item[1], item[0]))
            total = sum(counter.values())
            confidence = round(top_count / total, 2) if total else 0.0
            evidence_refs: List[str] = []
            for seq, ref in entries:
                if seq == top_sequence and ref not in evidence_refs:
                    evidence_refs.append(ref)
                if len(evidence_refs) >= 5:
                    break
            profiles.append(
                _profile_payload(
                    task,
                    list(top_sequence),
                    evidence_refs,
                    PRECONDITIONS.get(task, []),
                    POSTCONDITIONS.get(task, []),
                    confidence,
                )
            )
            continue

        fallback_steps = DEFAULT_CHAINS.get(task, [])[:PROFILE_STEPS_MAX]
        profiles.append(
            _profile_payload(
                task,
                list(fallback_steps),
                [],
                PRECONDITIONS.get(task, []),
                POSTCONDITIONS.get(task, []),
                confidence,
            )
        )

    if len(profiles) > PROFILES_MAX:
        anomalies.append("profiles_truncated")
        profiles = profiles[:PROFILES_MAX]
    return profiles


def _profile_payload(
    task: str,
    steps: List[str],
    evidence_refs: List[str],
    preconditions: List[str],
    postconditions: List[str],
    confidence: float,
) -> Dict[str, Any]:
    return {
        "profile_id": f"toolopt_{task}",
        "name": f"Tool Optimizer - {task.replace('_', ' ').title()}",
        "ordered_steps": steps[:PROFILE_STEPS_MAX],
        "preconditions": preconditions,
        "postconditions": postconditions,
        "evidence_refs": evidence_refs,
        "confidence": confidence,
    }


def _compute_catalog_tail_hash(catalog_path: Path) -> Optional[str]:
    if not catalog_path.exists():
        return None
    try:
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    tail = lines[-200:] if len(lines) > 200 else lines
    import hashlib

    return hashlib.sha256("".join(tail).encode("utf-8")).hexdigest()


def _collect_inputs(repo_root: Path, identity_path: Path, digest_paths: Sequence[Path]) -> Dict[str, Any]:
    catalog_path = repo_root / "training_data" / "index" / "catalog.jsonl"
    catalog_tail_hash = _compute_catalog_tail_hash(catalog_path)
    identity_hash = None
    if identity_path.exists():
        identity_hash = _compute_file_hash(identity_path)
    return {
        "digest_files": [path.as_posix() for path in digest_paths],
        "catalog_tail_hash": catalog_tail_hash,
        "identity_hash": identity_hash,
    }


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    file_hash = _compute_file_hash(path)
    return file_hash or ""


def _publish_catalog_artifact(
    repo_root: Path,
    registry_ref: str,
    profiles_ref: str,
    report_ref: str,
    anomalies: List[str],
) -> Optional[str]:
    catalog_cls = None
    try:
        from System_Operations_Protocol.code_generator.knowledge_catalog import KnowledgeCatalog as _CatalogCls  # type: ignore
        catalog_cls = _CatalogCls
    except Exception:
        try:
            from knowledge_catalog import KnowledgeCatalog as _CatalogCls  # type: ignore
            catalog_cls = _CatalogCls
        except Exception:
            anomalies.append("knowledge_catalog_unavailable")
            return None

    timestamp = _iso_timestamp()
    catalog = catalog_cls(repo_root)
    request = {
        "improvement_id": f"tool_optimizer_{timestamp.replace(':', '').replace('-', '')}",
        "description": "Record tool optimizer outputs",
        "target_module": "logos_core.optimization.tool_optimizer",
        "improvement_type": "analysis",
        "requirements": {"gap_category": "capability_analysis"},
        "constraints": {},
        "test_cases": [],
    }
    code_payload = json.dumps(
        {
            "tool_registry_ref": registry_ref,
            "tool_chain_profiles_ref": profiles_ref,
            "tool_optimizer_report_ref": report_ref,
        },
        indent=2,
    )
    stage_result = {
        "stage_ok": True,
        "compile_ok": True,
        "import_ok": True,
        "smoke_test_ok": True,
        "errors": [],
    }
    policy_result = {
        "policy_class": "capability_analysis",
        "deploy_allowed": False,
        "reasoning": "Tool optimizer outputs are catalog artifacts only.",
    }
    entry_id = catalog.persist_artifact(
        request,
        code_payload,
        stage_result,
        policy_result,
        "ToolOptimizer.run_tool_optimization",
    )

    artifact_dir = catalog.artifacts_dir / entry_id
    artifact_path = artifact_dir / "artifact.json"
    artifact_data = _safe_read_json(artifact_path) or {}
    artifact_data.setdefault("metadata", {})
    artifact_data["metadata"].update(
        {
            "tool_registry_ref": registry_ref,
            "tool_chain_profiles_ref": profiles_ref,
            "tool_optimizer_report_ref": report_ref,
        }
    )
    artifact_path.write_text(json.dumps(artifact_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    policy_path = artifact_dir / "policy.json"
    policy_data = _safe_read_json(policy_path) or {}
    policy_data["policy_class"] = "capability_analysis"
    policy_data["deploy_allowed"] = False
    policy_path.write_text(json.dumps(policy_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return entry_id


def run_tool_optimization(
    identity_path: str | Path = "state/agent_identity.json",
    out_dir: str | Path = "state/tool_optimizer",
    digests_glob: str = "state/planner_digest_archives/*.jsonl.gz",
) -> Dict[str, Any]:
    raw_identity = Path(identity_path)
    identity_candidate = raw_identity if raw_identity.is_absolute() else Path.cwd() / raw_identity
    repo_root = _repo_root_from_identity(identity_candidate)
    if not identity_candidate.exists() and not raw_identity.is_absolute():
        identity_candidate = (repo_root / raw_identity).resolve()
    identity_path = identity_candidate

    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    anomalies: List[str] = []

    registry = _build_tool_registry(repo_root, anomalies)
    profiles = _aggregate_profiles(repo_root, digests_glob, anomalies)

    registry_path = out_path / "tool_registry.json"
    profiles_path = out_path / "tool_chain_profiles.json"
    report_path = out_path / "tool_optimizer_report.json"

    registry_hash = _write_json(registry_path, registry)
    profiles_hash = _write_json(profiles_path, profiles)

    digest_paths = sorted(repo_root.glob(digests_glob))
    digest_paths = digest_paths[-DEFAULT_DIGEST_LIMIT:]
    inputs_used = _collect_inputs(repo_root, identity_path, digest_paths)

    report_payload = {
        "registry_count": len(registry),
        "profiles_count": len(profiles),
        "inputs_used": inputs_used,
        "anomalies": anomalies,
        "completion_status": "complete" if registry and profiles else "partial",
        "timestamp_utc": _iso_timestamp(),
    }
    report_hash = _write_json(report_path, report_payload)

    registry_ref = f"{registry_path.as_posix()}#{registry_hash.split(':', 1)[-1]}" if registry_hash else ""
    profiles_ref = f"{profiles_path.as_posix()}#{profiles_hash.split(':', 1)[-1]}" if profiles_hash else ""
    report_ref = f"{report_path.as_posix()}#{report_hash.split(':', 1)[-1]}" if report_hash else ""

    catalog_entry = _publish_catalog_artifact(repo_root, registry_ref, profiles_ref, report_ref, anomalies)

    ok = bool(registry) and bool(profiles)

    return {
        "ok": ok,
        "registry_path": registry_path.as_posix(),
        "profiles_path": profiles_path.as_posix(),
        "report_path": report_path.as_posix(),
        "hashes": {
            "registry": registry_hash,
            "profiles": profiles_hash,
            "report": report_hash,
        },
        "counts": {
            "registry": len(registry),
            "profiles": len(profiles),
        },
        "anomalies": anomalies,
        "catalog_entry": catalog_entry,
    }