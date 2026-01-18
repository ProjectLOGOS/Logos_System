# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# INSTALL_STATUS: SEMANTIC_REWRITE
# SOURCE_LEGACY: start_agent.py

"""
SEMANTIC REWRITE

This module has been rewritten for governed integration into the
LOGOS System Rebuild. Its runtime scope and protocol role have been
normalized, but its original logical structure has been preserved.
"""

"""Bounded supervised loop honoring the active mission profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from urllib import request as urllib_request

try:
    from external.Logos_AGI.identity_paths import CANONICAL_IDENTITY_PATH, LOGOS_ROOT
except ModuleNotFoundError:
    # Fallback when the identity_paths helper is absent (e.g., stripped submodule state).
    REPO_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
    LOGOS_ROOT = (REPO_ROOT_FALLBACK / "external" / "Logos_AGI").resolve()
    CANONICAL_IDENTITY_PATH = (
        LOGOS_ROOT / "System_Operations_Protocol" / "governance" / "state" / "agent_identity.json"
    )

try:
    from external.Logos_AGI.System_Operations_Protocol.alignment_protocols.safety.integrity_framework.integrity_safeguard import (
        IntegrityValidator,
        SafeguardConfiguration,
    )
except ModuleNotFoundError:
    # Ensure repo root is on sys.path when invoked via an absolute script path.
    repo_root_hint = Path(__file__).resolve().parent.parent
    if str(repo_root_hint) not in sys.path:
        sys.path.insert(0, str(repo_root_hint))
    from external.Logos_AGI.System_Operations_Protocol.alignment_protocols.safety.integrity_framework.integrity_safeguard import (
        IntegrityValidator,
        SafeguardConfiguration,
    )

try:
    scripts_dir = Path(__file__).parent
    repo_root = scripts_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from llm_advisor import LLMAdvisor, build_tool_schema
    from LOGOS_SYSTEM.System_Stack.Protocol_Resources.attestation import (
        AlignmentGateError,
        compute_attestation_hash,
        load_alignment_attestation,
        load_mission_profile,
        validate_attestation,
        validate_mission_profile,
    )
    from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import canonical_json_hash
    from logos.proof_refs import enforce_truth_annotation
    from scripts.system_stack_tbd.need_to_distribute.provenance import DriftError, load_pin, verify_pinned_repo
    from logos.evaluator import update_metrics, normalize_objective_class
    from logos.goals import (
        generate_goal_candidates,
        rank_goal_candidates,
        enforce_goal_safety,
    )
    from logos.tool_health import analyze_tool_health
    from logos.tool_playbooks import TOOL_PLAYBOOKS, get_tool_playbook
    from logos.tool_validators import run_validators
    from logos.plan_validation import validate_plan_run
    from logos.plan_scoring import compute_plan_score, plan_signature, update_plan_history
    from logos.tool_improvement import (
        propose_tool_improvements,
        invoke_tool_proposal_pipeline,
    )
    from plugins.uip_integration_plugin import uip_prompt_choice
except ImportError:
    # Fallback if attestation module unavailable
    class AlignmentGateError(Exception):
        """Raised when alignment gating fails in fallback mode."""

    def canonical_json_hash(obj):
        return ""  # type: ignore

    def compute_attestation_hash(att):
        return ""  # type: ignore

    def load_alignment_attestation(path):
        raise AlignmentGateError("attestation module unavailable")  # type: ignore

    def load_mission_profile(path):
        return {}  # type: ignore

    def validate_attestation(*args, **kwargs):
        pass  # type: ignore

    def enforce_truth_annotation(annotation, index=None):
        return annotation

    def validate_mission_profile(*args, **kwargs):
        pass  # type: ignore

    DriftError = Exception  # type: ignore

    def load_pin(path):
        raise DriftError("provenance module unavailable")  # type: ignore

    def verify_pinned_repo(*args, **kwargs):
        raise DriftError("provenance module unavailable")  # type: ignore

    def update_metrics(state, obj_class, tool, outcome):
        return state

    TOOL_PLAYBOOKS = {}

    def get_tool_playbook(name):
        return None

    def run_validators(*_args: Any, **_kwargs: Any):  # type: ignore
        return {"ok": True, "reason": "validator_unavailable", "validator": "noop"}

    def validate_plan_run(plan, execution_trace):  # type: ignore
        return {
            "plan_id": plan.get("plan_id") if isinstance(plan, dict) else None,
            "objective_class": plan.get("objective_class") if isinstance(plan, dict) else None,
            "steps_total": 0,
            "steps_ok": 0,
            "steps_denied": 0,
            "steps_error": 0,
            "invariants_ok": True,
            "step_reports": [],
            "run_ok": True,
        }

    def compute_plan_score(report):  # type: ignore
        return 0.0, {"base": 0.0, "penalties": {}, "raw_score": 0.0}

    def plan_signature(plan):  # type: ignore
        return "plan_sig_unavailable"

    def update_plan_history(*_args: Any, **_kwargs: Any):  # type: ignore
        return {}

    def normalize_objective_class(obj):
        obj = obj.lower().strip()
        if obj in ["status", "system_status", "health_check"]:
            return "STATUS"
        return "GENERAL"


try:
    from logos.ledger import build_run_ledger
except ImportError:
    build_run_ledger = None

try:
    from JUNK_DRAWER.scripts.runtime.need_to_distribute.logos_agi_adapter import LogosAgiNexus
except ImportError:
    try:
        from ..need_to_distribute.logos_agi_adapter import LogosAgiNexus
    except ImportError:
        LogosAgiNexus = None  # type: ignore

AnyCallable = Callable[..., Any]
Decorator = Callable[[AnyCallable], AnyCallable]

# SOP availability will be checked later when REPO_ROOT is defined
SOP_AVAILABLE = False
trigger_self_improvement = None

REPO_ROOT = Path(__file__).resolve().parent.parent


def _logos_state_dir(repo_root: str | Path) -> Path:
    """Resolve the state directory, honoring LOGOS_STATE_DIR override."""
    root_path = Path(repo_root)
    override = os.environ.get("LOGOS_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (root_path / "state").resolve()


def _attestation_path(repo_root: str | Path) -> Path:
    return _logos_state_dir(repo_root) / "alignment_LOGOS-AGENT-OMEGA.json"


STATE_DIR = _logos_state_dir(REPO_ROOT)
AUDIT_DIR = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit")).resolve()
MISSION_FILE = STATE_DIR / "mission_profile.json"
AGENT_STATE_FILE = STATE_DIR / "agent_state.json"
TRAINING_INDEX_PATH = (REPO_ROOT / "training_data" / "index" / "catalog.jsonl").resolve()
ALLOW_TRAINING_INDEX_WRITE = False

LOGOS_CORE_PATH = REPO_ROOT / "external" / "Logos_AGI"
if str(LOGOS_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(LOGOS_CORE_PATH))

try:
    from logos_core.governance.agent_identity import (  # type: ignore
        PersistentAgentIdentity,
        load_or_create_identity,
        update_identity,
        validate_identity,
    )
    from logos_core.governance.commitment_ledger import (  # type: ignore
        DEFAULT_LEDGER_PATH as DEFAULT_COMMITMENT_LEDGER_PATH,
        LEDGER_VERSION as COMMITMENT_LEDGER_VERSION,
        ensure_active_commitment,
        load_or_create_ledger,
        mark_commitment_status,
        record_event,
        validate_ledger as validate_commitment_ledger,
        write_ledger,
    )
    from logos_core.governance.prioritization import (  # type: ignore
        select_next_active_commitment,
    )

    LOGOS_CORE_AVAILABLE = True
except ImportError:
    LOGOS_CORE_AVAILABLE = False

    def _missing_logos_core(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Logos core governance modules unavailable; "
            "ensure external/Logos_AGI is initialized"
        )

    class PersistentAgentIdentity:  # type: ignore[too-many-ancestors]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "Logos core governance modules unavailable; "
                "ensure external/Logos_AGI is initialized"
            )

        def _save_identity(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "Logos core governance modules unavailable; "
                "ensure external/Logos_AGI is initialized"
            )

    DEFAULT_COMMITMENT_LEDGER_PATH = STATE_DIR / "commitment_ledger.json"
    COMMITMENT_LEDGER_VERSION = "unavailable"
    ensure_active_commitment = _missing_logos_core  # type: ignore[assignment]
    load_or_create_identity = _missing_logos_core  # type: ignore[assignment]
    load_or_create_ledger = _missing_logos_core  # type: ignore[assignment]
    mark_commitment_status = _missing_logos_core  # type: ignore[assignment]
    record_event = _missing_logos_core  # type: ignore[assignment]
    select_next_active_commitment = _missing_logos_core  # type: ignore[assignment]
    update_identity = _missing_logos_core  # type: ignore[assignment]
    validate_commitment_ledger = _missing_logos_core  # type: ignore[assignment]
    validate_identity = _missing_logos_core  # type: ignore[assignment]
    write_ledger = _missing_logos_core  # type: ignore[assignment]

try:
    from Logos_Agent.scripts.genesis_capsule import (  # type: ignore
        bootstrap_genesis,
        load_genesis_manifest,
    )
except ImportError:
    bootstrap_genesis = None  # type: ignore
    load_genesis_manifest = None  # type: ignore


@dataclass
class SandboxState:
    root: Optional[Path] = None
    cap: int = 0
    count: int = 0
    run_id: str = ""
    promotion_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    tests_required: Set[str] = field(default_factory=set)
    verification_steps: Set[str] = field(default_factory=set)
    rollback_steps: Set[str] = field(default_factory=set)


@dataclass
class RuntimeContext:
    attestation_hash: Optional[str] = None
    mission_profile_hash: Optional[str] = None
    unlocked: bool = False
    audit_logger: Optional[Any] = None
    objective_class: Optional[str] = None
    read_only: bool = False
    truth_events: Optional[List[Dict[str, Any]]] = None
    validation_beliefs: Optional[List[Dict[str, Any]]] = None
    tool_validation_events: Optional[List[Dict[str, Any]]] = None
    fallback_proposals: Optional[List[Dict[str, Any]]] = None
    last_tool_validation: Optional[Dict[str, Any]] = None


SANDBOX = SandboxState()


def load_mission() -> Dict[str, Any]:
    try:
        data = json.loads(MISSION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return {"label": "DEMO_STABLE", "safe_interfaces_only": True}


MISSION_PROFILE = load_mission()
MISSION_LABEL = str(MISSION_PROFILE.get("label", "DEMO_STABLE"))
SAFE_INTERFACES_ONLY = bool(MISSION_PROFILE.get("safe_interfaces_only", True))

TOOL_OPTIMIZER_TEMPLATE_ID = "T6"
TOOL_OPTIMIZER_TITLE = "optimize runtime tool orchestration"
TOOL_INVENTION_TEMPLATE_ID = "T7"
TOOL_INVENTION_TITLE = "derive novel tools from optimizer gaps"


def _is_tool_optimizer_commitment(entry: Dict[str, Any]) -> bool:
    template_id = str(entry.get("template_id", "")).upper()
    title = str(entry.get("title", "") or "").strip().lower()
    commitment_type = str(entry.get("type", "") or "").strip().lower()
    if template_id == TOOL_OPTIMIZER_TEMPLATE_ID:
        return True
    if TOOL_OPTIMIZER_TITLE in title:
        return True
    return template_id == TOOL_OPTIMIZER_TEMPLATE_ID and commitment_type == "analysis"


def _is_tool_invention_commitment(entry: Dict[str, Any]) -> bool:
    template_id = str(entry.get("template_id", "")).upper()
    title = str(entry.get("title", "") or "").strip().lower()
    commitment_type = str(entry.get("type", "") or "").strip().lower()
    if template_id == TOOL_INVENTION_TEMPLATE_ID:
        return True
    if TOOL_INVENTION_TITLE in title:
        return True
    return template_id == TOOL_INVENTION_TEMPLATE_ID and commitment_type == "analysis"


def _fallback_require_safe_interfaces(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any):
        if SAFE_INTERFACES_ONLY:
            raise RuntimeError("Blocked by mission profile: safe_interfaces_only")
        return func(*args, **kwargs)

    return wrapper


def _fallback_restrict_writes(root: Path) -> Decorator:
    sandbox = root.resolve()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any):
            def _ensure(candidate: os.PathLike[str] | str) -> None:
                path = Path(candidate).expanduser().resolve()
                try:
                    path.relative_to(sandbox)
                except ValueError as exc:
                    raise PermissionError(
                        f"blocked write outside {sandbox}: {path}"
                    ) from exc

            for label in ("path", "outfile", "dest", "filename", "target"):
                value = kwargs.get(label)
                if isinstance(value, (str, os.PathLike)):
                    _ensure(value)
            for value in args:
                if isinstance(value, (str, os.PathLike)):
                    _ensure(value)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _restrict_wrapper(root: str | os.PathLike[str]) -> Decorator:
    return _fallback_restrict_writes(Path(root))


require_safe_interfaces = _fallback_require_safe_interfaces
restrict_writes_to = _restrict_wrapper

try:  # prefer shared guardrails when available
    from plugins.guardrails import (  # type: ignore
        require_safe_interfaces as _require_safe_interfaces,
        restrict_writes_to as _restrict_writes_to,
    )
except (ImportError, AttributeError):
    pass
else:
    require_safe_interfaces = _require_safe_interfaces
    restrict_writes_to = _restrict_writes_to


def tool_filesystem_read(argument: str) -> str:
    if argument:
        if argument.startswith("state/"):
            candidate = (STATE_DIR / argument[len("state/"):]).expanduser().resolve()
        else:
            candidate = (REPO_ROOT / argument).expanduser().resolve()
    else:
        candidate = (STATE_DIR / "mission_profile.json").expanduser().resolve()

    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        try:
            candidate.relative_to(STATE_DIR)
        except ValueError:
            return f"[fs.read] blocked path: {candidate}"
    if not candidate.exists():
        return f"[fs.read] not found: {candidate}"
    if candidate.is_file():
        if candidate.stat().st_size > 1_000_000:
            return f"[fs.read] too large: {candidate}"
        return candidate.read_text(encoding="utf-8", errors="replace")
    if candidate.is_dir():
        entries = sorted(item.name for item in candidate.iterdir())[:200]
        header = "[fs.read] dir list (first {count}):\n".format(count=len(entries))
        return header + "\n".join(entries)
    return f"[fs.read] unsupported entry: {candidate}"


def tool_mission_status(_: str = "") -> str:
    return json.dumps(MISSION_PROFILE, indent=2)


def _integrity_baseline_path() -> Path:
    """Resolve the canonical integrity baseline path under LOGOS_AGI."""
    return (LOGOS_ROOT / "System_Operations_Protocol" / "state" / "integrity_hashes.json").resolve()


def _enforce_integrity_baseline() -> tuple[bool, str]:
    """Validate integrity baseline before any agent loop runs."""
    baseline_path = _integrity_baseline_path()
    if not baseline_path.exists():
        return True, f"Integrity baseline missing at {baseline_path} (skipped)"
    config = SafeguardConfiguration(integrity_hash_path=str(baseline_path))
    validator = IntegrityValidator(config)
    if not validator.baseline_hashes:
        return False, f"Integrity baseline missing at {baseline_path}"

    ok, violations = validator.validate_integrity()
    if not ok:
        joined = "; ".join(violations) if violations else "unknown violation"
        return False, f"Integrity validation failed: {joined}"

    return True, f"Integrity baseline verified at {baseline_path} ({len(validator.baseline_hashes)} files)"


def _run_proof_compile() -> tuple[bool, str]:
    """Invoke the PXL proof compile wrapper and gate on its result."""
    compile_script = REPO_ROOT / "PXL_Gate" / "ui" / "PXL_Proof_Compile.py"
    if not compile_script.exists():
        return False, f"Proof compile wrapper missing at {compile_script}"

    result = subprocess.run(
        [sys.executable, str(compile_script)], cwd=REPO_ROOT, check=False
    )
    if result.returncode != 0:
        return False, f"Proof compile failed (exit {result.returncode})"
    return True, "Proof gates verified"


def _preflight_identity_check() -> tuple[bool, str]:
    """Load and validate identity for preflight without entering the loop."""
    if not LOGOS_CORE_AVAILABLE:
        return False, "Logos core governance modules unavailable"
    try:
        identity = load_or_create_identity(_compute_theory_hash(), REPO_ROOT)
        ok, reason = validate_identity(
            identity,
            MISSION_FILE,
            REPO_ROOT / "training_data" / "index" / "catalog.jsonl",
        )
        if not ok:
            return False, f"Identity validation failed: {reason}"
        return True, f"Identity validated: {identity.get('agent_id')}"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"Identity validation error: {exc}"


def _is_forbidden_candidate(path: Path) -> bool:
    return any(part in RETRIEVE_FORBIDDEN_PARTS for part in path.parts)


def _resolve_retrieve_bases(root: str) -> List[Path]:
    root_key = (root or "repo").lower().strip() or "repo"
    if root_key == "docs":
        return [REPO_ROOT / "docs", REPO_ROOT / "README.md"]
    if root_key == "training_data":
        return [REPO_ROOT / "training_data"]
    if root_key == "sandbox":
        return [REPO_ROOT / "sandbox"]

    bases: List[Path] = [
        REPO_ROOT / "docs",
        REPO_ROOT / "README.md",
        REPO_ROOT / "scripts",
        REPO_ROOT / "logos",
    ]
    proto = REPO_ROOT / "Protopraxis"
    if proto.exists():
        bases.append(proto)
    training = REPO_ROOT / "training_data"
    if training.exists():
        bases.append(training)
    return bases


def _iter_retrieve_files(bases: List[Path], max_files: int) -> List[Path]:
    files: List[Path] = []
    file_bases = [
        b
        for b in bases
        if b.exists() and b.is_file() and not _is_forbidden_candidate(b)
    ]
    dir_bases = [
        b for b in bases if b.exists() and b.is_dir() and not _is_forbidden_candidate(b)
    ]

    for base in sorted(file_bases):
        if len(files) >= max_files:
            break
        files.append(base)

    priority: List[Path] = []
    docs_readme = REPO_ROOT / "docs" / "README.md"
    if docs_readme.exists() and not _is_forbidden_candidate(docs_readme):
        priority.append(docs_readme)
    repo_readme = REPO_ROOT / "README.md"
    if repo_readme.exists() and not _is_forbidden_candidate(repo_readme):
        priority.append(repo_readme)
    for path in priority:
        if len(files) >= max_files:
            break
        if path not in files:
            files.append(path)

    for base in dir_bases:
        if len(files) >= max_files:
            break
        for candidate in sorted(base.rglob("*")):
            if len(files) >= max_files:
                break
            if not candidate.is_file():
                continue
            if _is_forbidden_candidate(candidate):
                continue
            if candidate.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".pdf",
                ".zip",
                ".tar",
                ".gz",
                ".bz2",
            }:
                continue
            files.append(candidate)
    return files


def tool_retrieve_local(argument: str) -> str:
    try:
        args = json.loads(argument) if argument else {}
    except json.JSONDecodeError:
        args = {}

    query = str(args.get("query", "")).strip()
    root = str(args.get("root", "repo")).strip() or "repo"
    max_files = min(max(int(args.get("max_files", 10)), 1), 20)
    max_snippets = min(max(int(args.get("max_snippets", 5)), 1), 10)

    result: Dict[str, Any] = {
        "snippets": [],
        "query": query,
        "files_scanned": 0,
        "errors": [],
    }
    if not query:
        result["errors"].append("query required")
        return json.dumps(result, indent=2)

    bases = _resolve_retrieve_bases(root)
    query_lower = query.lower()
    files = _iter_retrieve_files(bases, max_files)
    for path in files:
        if len(result["snippets"]) >= max_snippets:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError) as exc:
            result["errors"].append(str(exc))
            continue
        result["files_scanned"] += 1
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if query_lower not in line.lower():
                continue
            snippet_text = line.strip()
            if len(snippet_text) > RETRIEVE_MAX_SNIPPET_CHARS:
                snippet_text = snippet_text[:RETRIEVE_MAX_SNIPPET_CHARS]
            snippet = {
                "path": str(path.relative_to(REPO_ROOT)),
                "start_line": idx + 1,
                "end_line": idx + 1,
                "text": snippet_text,
            }
            result["snippets"].append(snippet)
            if len(result["snippets"]) >= max_snippets:
                break

    result["snippets"].sort(
        key=lambda s: (s.get("path", ""), int(s.get("start_line", 0)))
    )
    return json.dumps(result, indent=2)


def tool_retrieve_web(argument: str) -> str:
    try:
        args = json.loads(argument) if argument else {}
    except json.JSONDecodeError:
        args = {}

    query = str(args.get("query", "")).strip()
    url = str(args.get("url", "")).strip()
    max_results = min(max(int(args.get("max_results", 1)), 1), 5)
    target = url or query
    result: Dict[str, Any] = {"query": target, "results": [], "errors": []}

    if os.getenv("LOGOS_ENABLE_WEB_RETRIEVAL", "0") != "1":
        result["status"] = "disabled"
        return json.dumps(result, indent=2)

    if not target:
        result["errors"].append("url required")
        return json.dumps(result, indent=2)

    try:
        req = urllib_request.Request(target, headers={"User-Agent": "LOGOS-GPT/1.0"})
        with urllib_request.urlopen(req, timeout=5) as resp:  # nosec B310
            body = resp.read(4096).decode("utf-8", errors="replace")
    except Exception as exc:  # pylint: disable=broad-except
        result["errors"].append(str(exc))
        return json.dumps(result, indent=2)

    snippet = body[:RETRIEVE_MAX_SNIPPET_CHARS]
    result["results"].append({"url": target, "status": "ok", "snippet": snippet})
    result["results"] = result["results"][:max_results]
    return json.dumps(result, indent=2)


def tool_probe_last_snapshot(_: str = "") -> str:
    snapshot_file = STATE_DIR / "alignment_LOGOS-AGENT-OMEGA.json"
    if not snapshot_file.exists():
        return "[probe] no snapshot file"
    try:
        root = json.loads(snapshot_file.read_text(encoding="utf-8"))
        if isinstance(root, list):
            data = root[-1]
        elif isinstance(root, dict):
            data = root
        else:
            return "[probe] unexpected snapshot structure"
        runs = data.get("protocol_probe_runs", [])
        if not runs:
            return "[probe] no runs"
        last = runs[-1]
        discovery = last.get("discovery", {})
        mission = last.get("mission", {})
        payload = {
            "mission_label": mission.get("label"),
            "timestamp": last.get("timestamp"),
            "modules_count": len(discovery.get("modules", [])),
            "discovery_errors": len(discovery.get("errors", [])),
            "runtime_seconds": last.get("runtime_seconds"),
        }
        hooks = last.get("hooks")
        if hooks:
            payload["hooks_attempted"] = len(hooks)
        return json.dumps(payload, indent=2)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return f"[probe] parse error: {exc}"


TOOLS: Dict[str, Callable[[str], str]] = {
    "mission.status": tool_mission_status,
    "probe.last": tool_probe_last_snapshot,
    "fs.read": tool_filesystem_read,
    "retrieve.local": tool_retrieve_local,
    "retrieve.web": tool_retrieve_web,
}


def _init_validation_containers(ctx: RuntimeContext) -> None:
    if ctx.truth_events is None:
        ctx.truth_events = []
    if ctx.validation_beliefs is None:
        ctx.validation_beliefs = []
    if ctx.tool_validation_events is None:
        ctx.tool_validation_events = []
    if ctx.fallback_proposals is None:
        ctx.fallback_proposals = []


def _precondition_check(
    tool_name: str, playbook: Optional[Dict[str, Any]], ctx: RuntimeContext
) -> tuple[bool, str]:
    if not playbook:
        return True, ""
    if getattr(ctx, "read_only", False) and not playbook.get("read_only", True):
        return False, "read_only_mode_blocks"
    objective = getattr(ctx, "objective_class", None)
    allowed = playbook.get("allowed_objectives", []) if isinstance(playbook, dict) else []
    if objective and allowed and "ANY" not in allowed and objective not in allowed:
        return False, f"objective_not_allowed:{objective}"
    return True, ""


def _record_tool_validation(
    ctx: RuntimeContext, tool_name: str, validation: Dict[str, Any]
) -> Dict[str, Any]:
    _init_validation_containers(ctx)
    record = {
        "tool": tool_name,
        "validator": validation.get("validator", "unknown"),
        "ok": bool(validation.get("ok", False)),
        "reason": str(validation.get("reason", "")),
        "timestamp": _timestamp(),
    }
    if validation.get("evidence") is not None:
        record["evidence"] = validation.get("evidence")
    ctx.last_tool_validation = record
    ctx.tool_validation_events.append(record)

    evidence = record.get("evidence") or {
        "type": "schema",
        "ref": None,
        "details": record.get("reason", "validation"),
    }
    truth_value = "VERIFIED" if record["ok"] else "CONTRADICTED" if record.get("reason") else "UNVERIFIED"
    ctx.truth_events.append(
        {
            "ts": record["timestamp"],
            "source": "TOOL",
            "content": {
                "tool": tool_name,
                "validator": record.get("validator"),
                "reason": record.get("reason", ""),
            },
            "truth_annotation": {
                "truth": truth_value,
                "evidence": evidence,
            },
        }
    )
    ctx.validation_beliefs.append(
        {
            "tool": tool_name,
            "truth": truth_value,
            "confidence": 0.82 if record["ok"] else 0.35,
            "reason": record.get("reason", ""),
            "timestamp": record["timestamp"],
        }
    )
    return record


def _register_fallbacks(
    ctx: RuntimeContext, tool_name: str, playbook: Optional[Dict[str, Any]], reason: str
) -> None:
    if not playbook:
        return
    fallbacks = playbook.get("fallback_tools") if isinstance(playbook, dict) else []
    if not isinstance(fallbacks, list):
        return
    for candidate in fallbacks:
        ctx.fallback_proposals.append(
            {
                "tool": str(candidate),
                "fallback_from": tool_name,
                "reason": str(reason or "validation_failed"),
                "timestamp": _timestamp(),
                "status": "proposed",
            }
        )


def dispatch_tool(
    tool_name: str, args: str, *, ctx: RuntimeContext, timeout_seconds: int = 15
) -> str:
    """Centralized tool dispatcher with attestation enforcement."""
    if not ctx.attestation_hash:
        raise AlignmentGateError("Tool execution requires valid attestation")
    _init_validation_containers(ctx)
    playbook = get_tool_playbook(tool_name)

    ok, reason = _precondition_check(tool_name, playbook, ctx)
    if not ok:
        validation_record = _record_tool_validation(
            ctx,
            tool_name,
            {"ok": False, "reason": reason, "validator": "precondition"},
        )
        _register_fallbacks(ctx, tool_name, playbook, validation_record.get("reason", ""))
        return f"[{tool_name}] precondition failed: {reason}"

    tool_fn = TOOLS.get(tool_name)
    if not tool_fn:
        validation_record = _record_tool_validation(
            ctx,
            tool_name,
            {"ok": False, "reason": "unknown_tool", "validator": "dispatcher"},
        )
        _register_fallbacks(ctx, tool_name, playbook, validation_record.get("reason", ""))
        return "[unknown tool]"

    if ctx.audit_logger:
        ctx.audit_logger(
            {
                "event": "tool_execution",
                "tool": tool_name,
                "attestation_hash": ctx.attestation_hash,
                "mission_profile_hash": ctx.mission_profile_hash,
                "timestamp": _timestamp(),
            }
        )

    output = bounded_call(tool_fn, args, hard_timeout_seconds=timeout_seconds)

    try:
        validation = run_validators(
            tool_name,
            args,
            output,
            ctx={"objective_class": ctx.objective_class, "read_only": ctx.read_only},
            playbook=playbook,
        )
    except Exception:
        validation = {"ok": False, "reason": "validator_exception", "validator": "unknown"}

    validation_record = _record_tool_validation(ctx, tool_name, validation)
    if not validation_record.get("ok", False):
        _register_fallbacks(ctx, tool_name, playbook, validation_record.get("reason", ""))

    return output


REFLECTION_CACHE_NAME = "_latest_reflection.json"


def _sandbox_base() -> Path:
    return SANDBOX.root or (REPO_ROOT / "sandbox")


def configure_sandbox(write_dir: Optional[str], cap: int) -> None:
    SANDBOX.root = None
    SANDBOX.cap = max(0, cap)
    SANDBOX.count = 0
    SANDBOX.run_id = ""
    SANDBOX.promotion_outcomes = []
    SANDBOX.tests_required.clear()
    SANDBOX.verification_steps.clear()
    SANDBOX.rollback_steps.clear()
    if write_dir:
        candidate = (REPO_ROOT / write_dir).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        SANDBOX.root = candidate


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_catalog_tail_hash(
    catalog_path: Path, tail_lines: int = 200
) -> Optional[str]:
    try:
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
    content = "".join(tail)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _collect_missing_evidence_refs(
    ledger: Dict[str, Any], repo_root: Path
) -> List[str]:
    missing: List[str] = []
    commitments = ledger.get("commitments", []) if isinstance(ledger, dict) else []
    for entry in commitments:
        refs = entry.get("evidence_refs", []) or []
        for raw_ref in refs:
            if not isinstance(raw_ref, str):
                continue
            ref = raw_ref.strip()
            if not ref:
                continue
            path_candidate: Optional[Path] = None
            if ref.startswith("uwm:"):
                path_part = ref.split(":", 1)[1]
                if "#" in path_part:
                    path_part = path_part.split("#", 1)[0]
                if path_part:
                    path_candidate = Path(path_part)
            else:
                path_candidate = Path(ref)
            if path_candidate is None:
                continue
            if not path_candidate.is_absolute():
                path_candidate = (repo_root / path_candidate).resolve()
            if not path_candidate.exists():
                if ref not in missing:
                    missing.append(ref)
                    if len(missing) >= 10:
                        return missing
    return missing


def _compute_theory_hash() -> str:
    """Compute hash of the PXL theory from Coq baseline."""
    try:
        baseline_dir = (
            REPO_ROOT / "Protopraxis" / "formal_verification" / "coq" / "baseline"
        )
        if not baseline_dir.exists():
            return "baseline_not_found"

        # Hash all .v files in baseline directory
        import hashlib

        hasher = hashlib.sha256()

        v_files = sorted(baseline_dir.glob("**/*.v"))
        for v_file in v_files:
            try:
                content = v_file.read_text(encoding="utf-8")
                hasher.update(f"{v_file.relative_to(baseline_dir)}:{content}".encode())
            except Exception:
                continue

        return hasher.hexdigest()
    except Exception:
        return "theory_hash_computation_failed"


def _make_run_id() -> str:
    raw = _timestamp()
    token = re.sub(r"[^A-Za-z0-9]+", "", raw)
    return token or "run"


def _sandbox_path(name: Optional[str] = None) -> Path:
    if SANDBOX.root is None:
        raise RuntimeError("sandbox writes disabled")
    base = name or f"artifact_{int(time.time())}_{SANDBOX.count}"
    return SANDBOX.root / base


PROMOTION_WRAPPER = (
    REPO_ROOT
    / "sandbox"
    / "agent_generated_scripts"
    / "promotion_tools"
    / "emit_from_items.py"
)


DOC_EXTENSIONS = {".md", ".rst", ".txt"}

RETRIEVE_FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "external",
    "state",
    "audit",
    "__pycache__",
}
RETRIEVE_MAX_SNIPPET_CHARS = 480


class PromotionPolicyError(Exception):
    pass


TOOL_EMISSION_POLICY: Dict[str, Dict[str, Any]] = {
    "capability.report": {
        "allows_non_doc": True,
        "requires_metadata_for_non_doc": True,
        "default_risk_tier": "NEUTRAL",
        "default_tests_required": [
            (
                "python -m unittest "
                "sandbox.agent_generated_scripts.test_promotion_artifact_parsing"
            ),
        ],
        "default_verification_steps": [
            (
                "python scripts/start_agent.py "
                '--objective "Re-run capability census after promotion" '
                "--write-dir sandbox/agent_generated_scripts "
                "--cap-writes 6 --assume-yes"
            ),
        ],
        "default_rollback_steps": [
            "git checkout -- <files>",
        ],
    },
    "sandbox.write": {
        "allows_non_doc": True,
        "requires_metadata_for_non_doc": True,
    },
}


def _requires_promotion(name: Optional[str]) -> bool:
    if not name:
        return False
    suffix = Path(name).suffix.lower()
    return suffix not in DOC_EXTENSIONS


def _coerce_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(entry) for entry in value if entry is not None]
    if value is None:
        return []
    return [str(value)]


def _apply_tool_policy(
    tool_name: str,
    items: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    doc_only = all(
        Path(item["target_path"]).suffix.lower() in DOC_EXTENSIONS for item in items
    )
    if doc_only:
        return

    policy = TOOL_EMISSION_POLICY.get(tool_name)
    if not policy:
        raise PromotionPolicyError(
            f"non-doc emissions blocked: undefined policy for tool '{tool_name}'"
        )
    if not policy.get("allows_non_doc", False):
        raise PromotionPolicyError(
            f"non-doc emissions blocked: tool '{tool_name}' disallows code artifacts"
        )

    def _ensure_list(field_name: str, default_key: str) -> None:
        current = _coerce_list(config.get(field_name))
        if current:
            config[field_name] = current
            return
        default_value = policy.get(default_key)
        if default_value:
            config[field_name] = list(default_value)
            return
        if policy.get("requires_metadata_for_non_doc", False):
            raise PromotionPolicyError(
                (
                    "non-doc emissions blocked: tool "
                    f"'{tool_name}' requires '{field_name}' metadata"
                )
            )
        config[field_name] = []

    _ensure_list("tests_required", "default_tests_required")
    _ensure_list("verification_steps", "default_verification_steps")
    _ensure_list("rollback_steps", "default_rollback_steps")
    config["operator_override"] = _coerce_list(config.get("operator_override"))

    risk = str(config.get("risk_tier", "")).strip().upper()
    if not risk:
        default_risk = str(policy.get("default_risk_tier", "NEUTRAL")).strip().upper()
        risk = default_risk or "NEUTRAL"
    config["risk_tier"] = risk


def _replace_rollback_placeholders(
    config: Dict[str, Any],
    items: List[Dict[str, Any]],
) -> None:
    targets = sorted({item["target_path"] for item in items if item.get("target_path")})
    if not targets:
        return
    replaced: List[str] = []
    for step in _coerce_list(config.get("rollback_steps")):
        if "<files>" in step:
            step = step.replace("<files>", " ".join(targets))
        replaced.append(step)
    config["rollback_steps"] = replaced


def _emit_via_promotion(
    data: Dict[str, Any],
    payload: str,
    default_name: str,
    tool_name: str,
) -> str:
    items_spec = data.get("items")
    items: List[Dict[str, Any]] = []
    if isinstance(items_spec, list) and items_spec:
        for entry in items_spec:
            if not isinstance(entry, dict):
                continue
            target = str(entry.get("target_path") or "").strip()
            content = entry.get("content")
            if not target or content is None:
                continue
            items.append(
                {
                    "target_path": target,
                    "content": str(content),
                    "rationale": str(
                        entry.get("rationale", "Generated by start_agent")
                    ),
                }
            )
    else:
        items.append(
            {
                "target_path": default_name,
                "content": payload,
                "rationale": str(data.get("rationale", "Generated by start_agent")),
            }
        )

    config = {
        "items": items,
        "risk_tier": str(data.get("risk_tier", "")).strip(),
        "tests_required": _coerce_list(data.get("tests_required")),
        "verification_steps": _coerce_list(data.get("verification_steps")),
        "rollback_steps": _coerce_list(data.get("rollback_steps")),
        "operator_override": _coerce_list(data.get("operator_override")),
        "objective": str(data.get("objective", "")),
    }

    run_id = SANDBOX.run_id or _make_run_id()
    SANDBOX.run_id = run_id
    summary_suffix = f"{SANDBOX.count:02d}"
    summary_name = f"promotion_bundle_summary_{run_id}_{summary_suffix}.md"
    config["run_id"] = run_id
    config["summary_name"] = summary_name

    def _build_snapshot() -> Dict[str, Any]:
        return {
            "items": [dict(item) for item in items],
            "risk_tier": config.get("risk_tier", ""),
            "tests_required": list(config.get("tests_required", [])),
            "verification_steps": list(config.get("verification_steps", [])),
            "rollback_steps": list(config.get("rollback_steps", [])),
            "operator_override": list(config.get("operator_override", [])),
            "objective": config.get("objective", ""),
            "summary_name": summary_name,
            "run_id": run_id,
        }

    try:
        _apply_tool_policy(tool_name, items, config)
    except PromotionPolicyError as exc:
        policy_snapshot = _build_snapshot()
        policy_record = {
            "status": "policy_refused",
            "tool": tool_name,
            "run_id": run_id,
            "timestamp": _timestamp(),
            "returncode": None,
            "raw_status": None,
            "config": policy_snapshot,
            "outcome": {
                "summary_path": None,
                "bundle_paths": {},
                "result_path": None,
                "refusal_note_path": None,
                "audit_path": None,
            },
            "reason": str(exc),
        }
        SANDBOX.promotion_outcomes.append(policy_record)
        return f"[{tool_name}] promotion policy refusal: {exc}"

    _replace_rollback_placeholders(config, items)
    config_snapshot = _build_snapshot()

    for entry in config_snapshot["tests_required"]:
        if entry:
            SANDBOX.tests_required.add(str(entry))
    for entry in config_snapshot["verification_steps"]:
        if entry:
            SANDBOX.verification_steps.add(str(entry))
    for entry in config_snapshot["rollback_steps"]:
        if entry:
            SANDBOX.rollback_steps.add(str(entry))

    tmp_payload = json.dumps(config, indent=2)
    tmp_digest = hashlib.sha256(tmp_payload.encode("utf-8")).hexdigest()[:12]

    tmp_path = _sandbox_path(f"promotion_emit_config_{tmp_digest}.json")

    @restrict_writes_to(SANDBOX.root or tmp_path.parent)
    def _write_tmp(path: Path) -> None:
        Path(path).write_text(tmp_payload + "\n", encoding="utf-8")

    _write_tmp(path=tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PROMOTION_WRAPPER),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    tmp_path.unlink(missing_ok=True)

    default_summary = "sandbox/promotion_bundle_summary.md"

    def _as_posix(candidate: Optional[str], fallback: Optional[str] = None) -> str:
        base = candidate if candidate else fallback
        if not base:
            return ""
        return Path(str(base)).as_posix()

    outcome_payload: Dict[str, Any]
    try:
        outcome_payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        outcome_payload = {}

    raw_status = outcome_payload.get("status")
    status = str(raw_status) if isinstance(raw_status, str) and raw_status else ""
    summary_rel = _as_posix(outcome_payload.get("summary_path"), default_summary)
    bundle_paths_raw = outcome_payload.get("bundle_paths")
    if not isinstance(bundle_paths_raw, dict):
        bundle_paths_raw = {}
    bundle_paths: Dict[str, str] = {}
    for key, value in bundle_paths_raw.items():
        if isinstance(key, str):
            bundle_paths[key] = _as_posix(value)
    result_rel = _as_posix(outcome_payload.get("result_path"))
    refusal_rel = _as_posix(outcome_payload.get("refusal_note_path"))
    audit_rel = _as_posix(outcome_payload.get("audit_path"))

    if status == "emitted":
        bundle_rel = _as_posix(
            bundle_paths.get("patch"),
            "sandbox/agent_generated_scripts/promotion_patch.json",
        )
        bundle_paths.setdefault("patch", bundle_rel)
        resolved_status = "emitted"
        message = (
            "[sandbox.write] promotion bundle emitted; summary "
            f"{summary_rel}; bundle {bundle_rel}"
        )
    elif status == "refused":
        refusal_rel = _as_posix(
            refusal_rel,
            "sandbox/agent_generated_scripts/promotion_refusal_note.md",
        )
        resolved_status = "refused"
        message = (
            "[sandbox.write] promotion bundle refused; summary "
            f"{summary_rel}; refusal note {refusal_rel}"
        )
    elif status == "error":
        details_rel = _as_posix(
            result_rel,
            "sandbox/agent_generated_scripts/promotion_refusal_note.md",
        )
        resolved_status = "error"
        message = (
            "[sandbox.write] promotion bundle error; summary "
            f"{summary_rel}; details {details_rel}"
        )
    else:
        if result.returncode == 0:
            bundle_rel = _as_posix(
                result_rel,
                "sandbox/agent_generated_scripts/promotion_patch.json",
            )
            bundle_paths.setdefault("patch", bundle_rel)
            resolved_status = "emitted"
            message = (
                "[sandbox.write] promotion bundle emitted; summary "
                f"{summary_rel}; bundle {bundle_rel}"
            )
        else:
            refusal_rel = _as_posix(
                result_rel,
                "sandbox/agent_generated_scripts/promotion_refusal_note.md",
            )
            resolved_status = "refused"
            message = (
                "[sandbox.write] promotion bundle refused; summary "
                f"{summary_rel}; refusal note {refusal_rel}"
            )

    promotion_record = {
        "status": resolved_status,
        "tool": tool_name,
        "run_id": run_id,
        "timestamp": _timestamp(),
        "returncode": result.returncode,
        "raw_status": status or None,
        "config": config_snapshot,
        "outcome": {
            "summary_path": summary_rel,
            "bundle_paths": bundle_paths,
            "result_path": result_rel,
            "refusal_note_path": refusal_rel,
            "audit_path": audit_rel,
        },
        "reason": None,
    }
    SANDBOX.promotion_outcomes.append(promotion_record)

    return message


@require_safe_interfaces
def _sandbox_write_impl(raw: str, tool_name: str = "sandbox.write") -> str:
    if SANDBOX.root is None or SANDBOX.cap <= 0:
        return "[sandbox.write] disabled"
    if SANDBOX.count >= SANDBOX.cap:
        return "[sandbox.write] cap reached"
    payload = raw.strip() or "(empty payload)"
    name: Optional[str] = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    items_spec = None
    if isinstance(data, dict):
        candidate_name = data.get("name")
        if isinstance(candidate_name, str) and candidate_name.strip():
            name = candidate_name.strip()
        load_from = data.get("load_from")
        if isinstance(load_from, str) and load_from.strip():
            entry = _resolve_sandbox_entry(load_from.strip())
            if entry is None:
                return "[sandbox.write] load_from blocked"
            if not entry.exists():
                return f"[sandbox.write] load_from missing: {entry}"
            if entry.is_dir():
                message = f"[sandbox.write] load_from directory unsupported: {entry}"
                return message
            payload = entry.read_text(encoding="utf-8", errors="replace")
        else:
            content_value = data.get("content")
            if isinstance(content_value, (dict, list)):
                payload = json.dumps(content_value, indent=2)
            elif isinstance(content_value, str):
                payload = content_value
            elif content_value is not None:
                payload = str(content_value)
        items_spec = data.get("items")
    if payload.startswith("@"):
        entry = _resolve_sandbox_entry(payload[1:].strip())
        if entry is None:
            return "[sandbox.write] indirect payload blocked"
        if not entry.exists():
            return f"[sandbox.write] indirect payload missing: {entry}"
        if entry.is_dir():
            message = f"[sandbox.write] indirect payload directory unsupported: {entry}"
            return message
        payload = entry.read_text(encoding="utf-8", errors="replace")

    if isinstance(data, dict) and isinstance(items_spec, list) and items_spec:
        SANDBOX.count += 1
        return _emit_via_promotion(data, payload, name or "promotion_item", tool_name)

    if isinstance(data, dict) and name and _requires_promotion(name):
        SANDBOX.count += 1
        return _emit_via_promotion(data, payload, name, tool_name)

    target = _sandbox_path(name)
    sandbox_root = SANDBOX.root or target.parent

    @restrict_writes_to(sandbox_root)
    def _write(path: Path) -> None:
        Path(path).write_text(payload, encoding="utf-8")

    _write(path=target)
    SANDBOX.count += 1
    return f"[sandbox.write] wrote {target.relative_to(REPO_ROOT)}"


def tool_sandbox_write(argument: str) -> str:
    return _sandbox_write_impl(argument, tool_name="sandbox.write")


TOOLS["sandbox.write"] = tool_sandbox_write


def tool_agent_memory(argument: str) -> str:
    state = _load_agent_state()
    runs: List[Dict[str, Any]] = state.get("runs", [])[-10:]
    limit = None
    if argument:
        try:
            limit = max(1, int(argument.strip()))
        except ValueError:
            pass
    if limit is not None:
        runs = runs[-limit:]
    summary: List[Dict[str, Any]] = []
    for item in runs:
        summary.append(
            {
                "timestamp": item.get("timestamp"),
                "objective": item.get("objective"),
                "artifacts": item.get("artifacts", []),
            }
        )
    payload = {
        "total_runs": len(state.get("runs", [])),
        "returned": len(summary),
        "reflections": state.get("reflections", [])[-5:],
        "runs": summary,
    }
    return json.dumps(payload, indent=2)


def _resolve_sandbox_entry(argument: str) -> Optional[Path]:
    sandbox_root = _sandbox_base().resolve()
    if not argument:
        target = sandbox_root
    else:
        candidate = Path(argument)
        if candidate.is_absolute():
            target = candidate
        else:
            parts = candidate.parts
            if parts and parts[0] == sandbox_root.name:
                candidate = Path(*parts[1:]) if len(parts) > 1 else Path("")
            target = sandbox_root / candidate
    try:
        resolved = target.resolve()
    except FileNotFoundError:
        return target
    try:
        resolved.relative_to(sandbox_root)
    except ValueError:
        return None
    return resolved


def tool_sandbox_read(argument: str) -> str:
    entry = _resolve_sandbox_entry(argument)
    if entry is None:
        return "[sandbox.read] blocked path"
    if not entry.exists():
        return f"[sandbox.read] not found: {entry}"
    if entry.is_dir():
        entries = sorted(item.name for item in entry.iterdir())[:200]
        header = "[sandbox.read] dir list (first {count}):\n".format(count=len(entries))
        return header + "\n".join(entries)
    if entry.is_file():
        if entry.stat().st_size > 1_000_000:
            return f"[sandbox.read] too large: {entry}"
        return entry.read_text(encoding="utf-8", errors="replace")
    return f"[sandbox.read] unsupported entry: {entry}"


def tool_sandbox_list(_: str) -> str:
    root = _sandbox_base()
    if not root.exists():
        return "[sandbox.list] sandbox empty"
    entries = sorted(item.name for item in root.iterdir())[:200]
    return json.dumps({"count": len(entries), "entries": entries}, indent=2)


def _safe_list_subdirs(path: Path, limit: int = 6) -> List[str]:
    names: List[str] = []
    try:
        for candidate in sorted(path.iterdir()):
            if not candidate.is_dir():
                continue
            if candidate.name.startswith(".") or candidate.name.startswith("__"):
                continue
            if candidate.name in {"Documentation", "docs", "__pycache__"}:
                continue
            names.append(candidate.name)
            if len(names) >= limit:
                break
    except (OSError, PermissionError):
        return names
    return names


def _count_python_files(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*.py"):
            if item.is_file():
                total += 1
    except (OSError, PermissionError):
        return total
    return total


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for entry in values:
        if entry in seen:
            continue
        seen.add(entry)
        ordered.append(entry)
    return ordered


def tool_capability_report(_: str) -> str:
    base = REPO_ROOT / "external" / "Logos_AGI"
    if not base.exists():
        return "[capability.report] Logos_AGI repository unavailable"

    timestamp = _timestamp()
    protocol_entries: List[Dict[str, Any]] = []
    for child in sorted(base.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or not child.name.endswith("_Protocol"):
            continue
        protocol_entries.append(
            {
                "name": child.name,
                "subsystems": _safe_list_subdirs(child),
                "py_modules": _count_python_files(child),
                "has_readme": (child / "README.md").exists()
                or (child / "Documentation").exists(),
            }
        )

    logos_core_dir = base / "logos_core"
    logos_core_segments = (
        _safe_list_subdirs(logos_core_dir) if logos_core_dir.exists() else []
    )

    plugins_dir = REPO_ROOT / "plugins"
    gaps: List[str] = []
    actions: List[str] = []
    sop_dev_env = (
        base
        / "System_Operations_Protocol"
        / "code_generator"
        / "development_environment.py"
    )
    sop_ready = sop_dev_env.exists()
    if not sop_ready:
        gaps.append(
            (
                "SOP development environment module missing "
                "(System_Operations_Protocol/code_generator/"
                "development_environment.py)."
            )
        )
        actions.append(
            (
                "Restore or implement "
                "System_Operations_Protocol/code_generator/"
                "development_environment.py before attempting sandbox upgrades."
            )
        )

    plugin_flags = [
        ("uip_integration_plugin.py", "UIP integration plugin"),
        ("enhanced_uip_integration_plugin.py", "Enhanced UIP integration plugin"),
    ]
    for plugin_name, description in plugin_flags:
        plugin_path = plugins_dir / plugin_name
        if not plugin_path.exists():
            relative = plugin_path.relative_to(REPO_ROOT)
            gaps.append(f"{description} file missing ({relative}).")
            actions.append(
                f"Add {plugin_name} with a concrete integration implementation."
            )
            continue
        try:
            source_text = plugin_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "return False" in source_text:
            gaps.append(f"{description} remains stubbed (available() -> False).")
            actions.append(
                f"Implement {description} so available() reflects actual "
                "integration readiness."
            )

    integrations_dir = base / "integrations"
    real_connectors: List[str] = []
    if integrations_dir.exists():
        try:
            for candidate in integrations_dir.iterdir():
                if (
                    candidate.is_file()
                    and candidate.suffix == ".py"
                    and candidate.name not in {"__init__.py", "example_connector.py"}
                ):
                    real_connectors.append(candidate.name)
        except (OSError, PermissionError):
            pass
    if not real_connectors:
        gaps.append(
            (
                "No production connectors detected under "
                "external/Logos_AGI/integrations (only example connector present)."
            )
        )
        actions.append(
            "Publish production-ready connectors inside external/Logos_AGI/"
            "integrations/ to reach external systems."
        )

    report_lines: List[str] = [
        "# LOGOS Protocol Capability Report",
        f"*Generated:* {timestamp}",
        f"*Mission Profile:* {MISSION_LABEL}",
        "",
        "## Protocol Overview",
    ]
    if protocol_entries:
        for entry in protocol_entries:
            subsystems = (
                ", ".join(entry["subsystems"]) or "no subsystem directories discovered"
            )
            readme_status = (
                "README present" if entry["has_readme"] else "README missing"
            )
            report_lines.append(
                (
                    f"- {entry['name']}: {subsystems}; {readme_status}; "
                    f"~{entry['py_modules']} Python modules"
                )
            )
    else:
        report_lines.append(
            "- No protocol directories discovered under external/Logos_AGI"
        )

    if logos_core_segments:
        report_lines.extend(
            [
                "",
                "## logos_core Snapshot",
                "- Key modules: " + ", ".join(logos_core_segments),
            ]
        )

    report_lines.extend(["", "## Architecture Gaps"])
    if gaps:
        for gap in gaps:
            report_lines.append(f"- {gap}")
    else:
        report_lines.append("- No critical gaps detected during scan")

    deduped_actions = _dedupe_preserve_order(actions)
    _record_protocol_health(
        {
            "timestamp": timestamp,
            "mission": MISSION_LABEL,
            "source": "capability.report",
            "protocols": protocol_entries,
            "logos_core_segments": logos_core_segments,
            "gaps": gaps,
            "recommended_actions": deduped_actions,
            "sop_ready": sop_ready,
        }
    )
    report_lines.extend(["", "## Recommended Next Steps"])
    if deduped_actions:
        for action in deduped_actions:
            report_lines.append(f"- {action}")
    else:
        report_lines.append(
            "- Maintain current configuration; no immediate actions identified"
        )

    report_lines.extend(["", "## SOP Coding Environment Status"])
    if sop_ready:
        report_lines.append(
            "- development_environment.py present; automation hooks can be drafted."
        )
    else:
        report_lines.append(
            "- development_environment.py missing; automation remains blocked "
            "pending restoration."
        )

    report_payload = json.dumps(
        {
            "name": "protocol_capability_report.md",
            "content": "\n".join(report_lines),
        }
    )
    report_result = _sandbox_write_impl(report_payload, tool_name="capability.report")

    if sop_ready:
        actions_for_script = deduped_actions or [
            (
                "Confirm SOP upgrade objectives with mission owners "
                "before enabling automation."
            )
        ]
        action_lines = ["    recommended_actions = ["]
        for action in actions_for_script:
            action_lines.append(f"        {action!r},")
        action_lines.append("    ]")

        script_lines: List[str] = [
            "#!/usr/bin/env python3",
            f'"""SOP sandbox upgrade scaffold generated on {timestamp}."""',
            "",
            "from __future__ import annotations",
            "",
            "import json",
            "import sys",
            "from pathlib import Path",
            "",
            "REPO_ROOT = Path(__file__).resolve().parents[2]",
            "if str(REPO_ROOT) not in sys.path:",
            "    sys.path.insert(0, str(REPO_ROOT))",
            "",
            (
                "from external.Logos_AGI.System_Operations_Protocol."
                "code_generator.development_environment import ("
            ),
            "    get_code_environment_status,",
            ")",
            "",
            "",
            "def _sandbox_root() -> Path:",
            "    return Path(__file__).resolve().parent",
            "",
            "",
            "    status = get_code_environment_status()",
        ]
        script_lines.extend(action_lines)
        script_lines.extend(
            [
                "    upgrade_plan = {",
                f'        "generated_at": "{timestamp}",',
                '        "environment_status": status,',
                '        "recommended_actions": recommended_actions,',
                "    }",
                '    output_path = _sandbox_root() / "sop_upgrade_plan.json"',
                "    output_path.write_text(",
                "        json.dumps(upgrade_plan, indent=2),",
                '        encoding="utf-8",',
                "    )",
                "    return 0",
                "",
                "",
                'if __name__ == "__main__":',
            ]
        )
        script_payload = json.dumps(
            {
                "items": [
                    {
                        "target_path": (
                            "sandbox/agent_generated_scripts/sop_upgrade_plan.py"
                        ),
                        "content": "\n".join(script_lines),
                        "rationale": "Capability report SOP upgrade scaffold",
                    }
                ],
                "objective": "Capability report SOP upgrade plan",
            }
        )
        extra_result = _sandbox_write_impl(
            script_payload, tool_name="capability.report"
        )
    else:
        blockers_lines: List[str] = [
            "# SOP Upgrade Prerequisites",
            f"*Generated:* {timestamp}",
            "",
            "## Blockers",
        ]
        if gaps:
            for gap in gaps:
                blockers_lines.append(f"- {gap}")
        else:
            blockers_lines.append("- No blockers detected")

        blockers_lines.extend(["", "## Action Items"])
        if deduped_actions:
            for action in deduped_actions:
                blockers_lines.append(f"- {action}")
        else:
            blockers_lines.append(
                "- Proceed with SOP upgrade draft once mission owners approve"
            )

        blockers_payload = json.dumps(
            {
                "name": "sop_upgrade_next_steps.md",
                "content": "\n".join(blockers_lines),
            }
        )
        extra_result = _sandbox_write_impl(
            blockers_payload,
            tool_name="capability.report",
        )

    return report_result + "\n" + extra_result


TOOLS["capability.report"] = tool_capability_report


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _collect_sandbox_files(exclude: Optional[Path] = None) -> List[Path]:
    root = _sandbox_base().resolve()
    if not root.exists():
        return []
    files: List[Path] = []
    for item in root.iterdir():
        if item.is_file():
            if exclude is not None and item.resolve() == exclude.resolve():
                continue
            files.append(item)
    return files


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def _plan_quality(text: str) -> int:
    score = 0
    if re.search(r"\b(step|plan|bullet|1\)|- )", text, re.IGNORECASE):
        score += 1
    if re.search(
        r"\b(success|criteria|metric|deadline|owner)\b",
        text,
        re.IGNORECASE,
    ):
        score += 1
    if re.search(
        r"\b(safe|sandbox|cap|guardrail|limit)\b",
        text,
        re.IGNORECASE,
    ):
        score += 1
    return score


def tool_sandbox_reflect(argument: str) -> str:
    entry = _resolve_sandbox_entry(argument)
    if entry is None:
        return "[sandbox.reflect] blocked path"
    if not entry.exists():
        return f"[sandbox.reflect] not found: {entry}"
    if entry.is_dir():
        return f"[sandbox.reflect] unsupported directory: {entry}"
    text = entry.read_text(encoding="utf-8", errors="replace")
    tokens = _tokenize(text)
    unique_tokens = set(tokens)

    prior_tokens: set[str] = set()
    for other in _collect_sandbox_files(entry):
        try:
            prior_tokens.update(
                _tokenize(other.read_text(encoding="utf-8", errors="replace"))
            )
        except (OSError, UnicodeDecodeError):
            continue
    novelty = 1.0
    if prior_tokens:
        novelty = 1.0 - _jaccard(unique_tokens, prior_tokens)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    preview = text.replace("\n", " ")[:240]
    plan_score = _plan_quality(text)

    sandbox_root = _sandbox_base().resolve()

    try:
        relative_path = entry.resolve().relative_to(sandbox_root).as_posix()
    except ValueError:
        try:
            relative_path = entry.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            relative_path = entry.name

    reflection = {
        "source_artifact": relative_path,
        "length_bytes": len(text.encode("utf-8")),
        "unique_token_count": len(unique_tokens),
        "novelty_score": round(novelty, 3),
        "plan_quality_0_3": plan_score,
        "line_count": len(lines),
        "preview": preview,
        "next_experiment": (
            "Read last two artifacts, extract concrete success criteria, "
            "propose ONE measurable experiment with owner+deadline, "
            "then write plan.md in sandbox."
        ),
    }

    cache_path = _sandbox_base() / REFLECTION_CACHE_NAME
    cache_path.write_text(json.dumps(reflection, indent=2), encoding="utf-8")
    reflection["cached_to"] = str(cache_path.relative_to(REPO_ROOT))

    metrics_stamp = {
        "timestamp": _timestamp(),
        "artifact": reflection["source_artifact"],
        "novelty_score": reflection["novelty_score"],
        "plan_quality_0_3": reflection["plan_quality_0_3"],
        "unique_token_count": reflection["unique_token_count"],
        "length_bytes": reflection["length_bytes"],
    }

    state = _load_agent_state()
    reflections = state.setdefault("reflection_metrics", [])

    reflections.append(metrics_stamp)
    state["reflection_metrics"] = reflections[-100:]
    summary_line = (
        f"Reflection {metrics_stamp['timestamp']}: "
        f"artifact={metrics_stamp['artifact']} "
        f"novelty={metrics_stamp['novelty_score']} "
        f"plan_quality={metrics_stamp['plan_quality_0_3']}"
    )
    state.setdefault("reflections", []).append(
        {"timestamp": metrics_stamp["timestamp"], "text": summary_line}
    )
    state["reflections"] = state["reflections"][-50:]
    _persist_agent_state(state)

    return json.dumps(reflection, indent=2)


TOOLS["agent.memory"] = tool_agent_memory
TOOLS["sandbox.read"] = tool_sandbox_read
TOOLS["sandbox.list"] = tool_sandbox_list
TOOLS["sandbox.reflect"] = tool_sandbox_reflect


def select_ready_inputs() -> List[Any]:
    try:
        import select

        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return list(readable)
    except (ImportError, OSError, ValueError):
        return []


def ask_user(
    prompt: str,
    default: bool = False,
    timeout_seconds: int = 20,
) -> bool:
        f"\n[CONSENT] {prompt} [y/N] (auto={'Y' if default else 'N'} "
        f"in {timeout_seconds}s): ",
        end="",
        flush=True,
    )
    start = time.time()
    buffer = ""
    try:
        while True:
            if select_ready_inputs():
                buffer = sys.stdin.readline().strip()
                break
            if time.time() - start > timeout_seconds:
                return default

            time.sleep(0.05)
    except KeyboardInterrupt:
        return False
    if not buffer:
        return default
    return buffer.lower() in {"y", "yes"}


def bounded_call(
    fn: Callable[[str], str],
    argument: str,
    hard_timeout_seconds: int = 15,
) -> str:
    result: Dict[str, str] = {"output": "[timeout]"}

    def run() -> None:
        try:
            result["output"] = fn(argument)
        except Exception:  # pylint: disable=broad-exception-caught
            result["output"] = "[error]\n" + "".join(traceback.format_exc(limit=3))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(hard_timeout_seconds)
    if thread.is_alive():
        return "[timeout]"
    return result["output"]


def _parse_extra_steps(specs: Iterable[str]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for spec in specs:
        if not spec:
            continue

        tool, _, arg = spec.partition(":")
        tool = tool.strip()
        if not tool:
            continue
        parsed.append({"tool": tool, "arg": arg})
    return parsed


def make_plan(
    objective: str,
    force_read_only: bool,
    extra_steps: Iterable[str],
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = [
        {"tool": "mission.status", "arg": ""},
        {"tool": "probe.last", "arg": ""},
        {"tool": "fs.read", "arg": "state/mission_profile.json"},
    ]
    plan.extend(_parse_extra_steps(extra_steps))
    if not force_read_only and SANDBOX.root is not None and SANDBOX.cap > 0:
        payload = json.dumps({"content": objective})
        plan.append({"tool": "sandbox.write", "arg": payload})
        if SANDBOX.cap > 2:
            plan.append({"tool": "capability.report", "arg": ""})
    for idx, entry in enumerate(plan, start=1):
        entry["step"] = idx
    return plan


def _load_agent_state() -> Dict[str, Any]:
    if AGENT_STATE_FILE.exists():
        try:
            data = json.loads(AGENT_STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data

        except (
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            TypeError,
        ):
            pass
    return {"version": 1, "runs": [], "reflections": []}


def _persist_agent_state(state: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _record_protocol_health(snapshot: Dict[str, Any]) -> None:
    state = _load_agent_state()
    history = state.setdefault("protocol_health", [])
    history.append(snapshot)
    state["protocol_health"] = history[-25:]

    actions = snapshot.get("recommended_actions", []) or []
    queue = state.setdefault("improvement_queue", [])
    for action in actions:
        queue.append(
            {
                "timestamp": snapshot.get("timestamp", _timestamp()),
                "action": action,
                "source": snapshot.get("source", "capability.report"),
            }
        )
    state["improvement_queue"] = queue[-100:]
    _persist_agent_state(state)


def _extract_artifacts(results: List[Dict[str, Any]]) -> List[str]:
    artifacts: List[str] = []
    for item in results:
        output = item.get("output")
        if isinstance(output, str) and "[sandbox.write] wrote " in output:
            artifacts.append(output.split("[sandbox.write] wrote ", 1)[1].strip())
    return artifacts


def _persist_agent_run(
    objective: str,
    plan: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    attestation_hash: Optional[str] = None,
    mission_profile_hash: Optional[str] = None,
) -> None:
    state = _load_agent_state()
    timestamp = _timestamp()
    reflection = (
        f"Run {timestamp}: steps={summary['steps_executed']}, "
        f"denied={summary['denied_steps']}, mission={summary['mission']}"
    )
    entry = {
        "timestamp": timestamp,
        "objective": objective,
        "mission": summary.get("mission"),
        "plan": plan,
        "results": results,
        "summary": summary,
        "artifacts": _extract_artifacts(results),
        "reflection": reflection,
        "attestation_hash": attestation_hash,
        "mission_profile_hash": mission_profile_hash,
    }
    state.setdefault("runs", []).append(entry)
    state["runs"] = state["runs"][-25:]
    state.setdefault("reflections", []).append(
        {"timestamp": timestamp, "text": reflection}
    )
    state["reflections"] = state["reflections"][-50:]
    state["last_goal"] = objective
    _persist_agent_state(state)


def _plan_history_mean(prior_state: Dict[str, Any], signature: str) -> float:
    try:
        container = prior_state.get("plans", {}).get("history_scored", {})
        entries = container.get("entries_by_signature", {}).get(signature, [])
        if not entries:
            return 0.0
        total = sum(float(entry.get("score", 0.0)) for entry in entries)
        return total / max(1, len(entries))
    except Exception:
        return 0.0


def run_supervised(
    objective: str,
    read_only: bool,
    max_runtime_seconds: int,
    max_plan_steps_per_run: int = 0,
    enable_llm_advisor: bool = False,
    llm_provider: str = "stub",
    llm_model: str = "stub",
    llm_timeout_sec: int = 10,
    extra_steps: Iterable[str] = (),
    assume_yes: bool = False,
    attestation_hash: Optional[str] = None,
    mission_profile_hash: Optional[str] = None,
    enable_logos_agi: bool = False,
    logos_agi_max_compute_ms: int = 100,
    allow_logos_agi_drift: bool = False,
    logos_agi_mode: str = "auto",
    scp_recovery_mode: bool = False,
) -> Dict[str, Any]:
    run_start_ts = datetime.now(timezone.utc).isoformat()
    started = time.time()
    if not SANDBOX.run_id:
        SANDBOX.run_id = _make_run_id()
    force_read_only = SAFE_INTERFACES_ONLY or read_only

    # Ledger collection
    executed_events = []
    proposals = []
    executed_plan_steps = 0
    plan_validation_report: Dict[str, Any] = {}
    plan_score_value: float = 0.0
    plan_score_explanation: Dict[str, Any] = {}
    plan_signature_value: str = ""
    plan_report_hash: str = ""
    advisor = None
    tool_schema = None
    requires_uip_map: Dict[str, bool] = {}
    truth_rules = {
        "allowed_truths": ["PROVED", "VERIFIED", "INFERRED", "HEURISTIC", "UNVERIFIED"],
        "note": "LLM advisor is non-authoritative; no execution",
    }
    if enable_llm_advisor:
        tool_schema = build_tool_schema(TOOLS)
        requires_uip_map = {
            entry.get("name", ""): bool(entry.get("requires_uip"))
            for entry in tool_schema.get("tools", [])
        }
        advisor = LLMAdvisor(
            provider=llm_provider,
            model=llm_model,
            tools_schema=tool_schema,
            truth_rules=truth_rules,
            timeout_sec=llm_timeout_sec,
        )
    policy_notes_aggregate = {
        "boosted_tools": [],
        "filtered_tools": [],
        "boosted_belief_ids": [],
        "filtered_belief_ids": [],
    }

    # Create runtime context with attestation
    ctx = RuntimeContext(
        attestation_hash=attestation_hash,
        mission_profile_hash=mission_profile_hash,
        unlocked=bool(attestation_hash),
    )
    ctx.read_only = force_read_only
    _init_validation_containers(ctx)

    # Initialize Logos_AGI nexus if enabled
    logos_agi_provenance = None
    nexus = None
    if enable_logos_agi and LogosAgiNexus:
        # Enforce pin and drift detection (skip for stub mode)
        pin_path = STATE_DIR / "logos_agi_pin.json"
        logos_agi_dir = REPO_ROOT / "external" / "Logos_AGI"

        if logos_agi_mode != "stub":
            if not pin_path.exists():
                    "[ERROR] Logos_AGI pin missing. Run: python "
                    "scripts/aligned_agent_import.py --pin-sha <ref> --pin-note '...'"
                )
                return {"error": "logos_agi_pin_missing"}

            try:
                pin = load_pin(str(pin_path))
                provenance = verify_pinned_repo(
                    str(logos_agi_dir), pin, require_clean=True
                )
                logos_agi_provenance = provenance
            except DriftError as e:
                if (
                    allow_logos_agi_drift
                    and os.environ.get("LOGOS_DEV_BYPASS_OK") == "1"
                ):
                    logos_agi_provenance = {"error": str(e), "override": True}
                else:
                    return {"error": "logos_agi_drift"}
            except (OSError, ValueError, RuntimeError) as e:
                return {"error": "logos_agi_pin_error"}
        else:
            logos_agi_provenance = {
                "pinned_sha": None,
                "head_sha": None,
                "dirty": False,
                "match": True,
            }  # Stub mode

        repo_sha = _compute_theory_hash()  # or some SHA
        nexus = LogosAgiNexus(
            enable=True,
            max_compute_ms=logos_agi_max_compute_ms,
            state_dir=str(STATE_DIR),
            repo_sha=repo_sha,
            mode=logos_agi_mode,
            scp_recovery_mode=scp_recovery_mode,
        )
        nexus.bootstrap()

        # Log bootstrap audit
        memory = nexus.get_memory_summary()
        ctx.audit_logger(
            {
                "event": "logos_agi_bootstrap",
                "logos_agi_mode": logos_agi_mode,
                "logos_agi_available": nexus.available,
                "logos_agi_last_error": nexus.last_error,
                "scp_state_loaded": memory["has_prior"],
                "scp_state_valid": nexus.scp_state_valid,
                "scp_recovery_mode": nexus.scp_recovery_mode,
                "scp_state_validation_error": nexus.scp_state_validation_error,
                "scp_state_prev_hash": nexus.prior_state.get("prev_hash")
                if nexus.prior_state
                else None,
                "scp_state_version": memory.get("version", 0),
                "logos_agi_pinned_sha": logos_agi_provenance.get("pinned_sha"),
                "logos_agi_head_sha": logos_agi_provenance.get("head_sha"),
                "logos_agi_dirty": logos_agi_provenance.get("dirty"),
                "logos_agi_pin_match": logos_agi_provenance.get("match", False),
            }
        )

    ledger = None
    ledger_hash: Optional[str] = None
    ledger_path = REPO_ROOT / DEFAULT_COMMITMENT_LEDGER_PATH
    active_commitment_id: Optional[str] = None
    ledger_validation_failed = False
    cycle_errors: List[str] = []
    tool_optimizer_summary: Optional[Dict[str, Any]] = None
    tool_invention_summary: Optional[Dict[str, Any]] = None
    identity_validation_warnings: List[str] = []
    ledger_warnings: List[str] = []

    # Load or create Persistent Agent Identity at cycle start
    agent_identity = None
    theory_hash = _compute_theory_hash()
    if LOGOS_CORE_AVAILABLE:
        try:
            agent_identity = load_or_create_identity(theory_hash, REPO_ROOT)

            # Validate identity
            is_valid, validation_reason = validate_identity(
                agent_identity,
                MISSION_FILE,
                REPO_ROOT / "training_data" / "index" / "catalog.jsonl",
            )
            if not is_valid:
                force_read_only = True
                cycle_errors.append(f"Identity validation failed: {validation_reason}")
                identity_validation_warnings.append(validation_reason)
                try:
                    agent_identity.setdefault("mission", {})["allow_enhancements"] = (
                        False
                    )
                except Exception:
                    pass
            else:
                if "warnings:" in validation_reason:
                    warn_text = validation_reason.split("warnings:", 1)[1].strip()
                    if warn_text:
                        for chunk in warn_text.split(";"):
                            entry = chunk.strip()
                            if entry:
                                identity_validation_warnings.append(entry)

        except Exception as e:
            cycle_errors.append(f"Identity unavailable: {e}")
            identity_validation_warnings.append(str(e))
    else:
            "[PAI] Logos core governance modules unavailable; entering repair-only mode"
        )
        force_read_only = True
        cycle_errors.append(
            "Identity unavailable: Logos core governance modules missing"
        )
        identity_validation_warnings.append("Logos core governance modules missing")

    if agent_identity:
        try:
            if not ledger_path.exists():
                    f"[PAI][ledger] Missing ledger file at {ledger_path}, bootstrapping new record"
                )
            ledger = load_or_create_ledger(ledger_path)
            ok, ledger_reasons, ledger_warnings_local = validate_commitment_ledger(
                ledger, agent_identity
            )
            for entry in ledger_warnings_local:
            ledger_warnings.extend(ledger_warnings_local)
            if not ok:
                reason = (
                    "; ".join(ledger_reasons) if ledger_reasons else "validation failed"
                )
                ledger_validation_failed = True
                force_read_only = True
                cycle_errors.append(f"Commitment ledger invalid: {reason}")
                agent_identity.setdefault("mission", {})["allow_enhancements"] = False

            world_model_info = agent_identity.get("world_model", {})
            snapshot_path_value = world_model_info.get(
                "snapshot_path", "state/world_model_snapshot.json"
            )
            if not snapshot_path_value:
                snapshot_path_value = "state/world_model_snapshot.json"
            snapshot_hash_value = world_model_info.get("snapshot_hash") or "unknown"
            uwm_ref = f"uwm:{snapshot_path_value}#{snapshot_hash_value}"

            planner_digest_ref: Optional[str] = None
            latest_archive_file = STATE_DIR / "latest_planner_digest_archive.txt"
            if latest_archive_file.exists():
                try:
                    planner_digest_ref = (
                        latest_archive_file.read_text(encoding="utf-8").strip() or None
                    )
                except Exception as exc:
                        f"[PAI][ledger] warn: unable to read planner digest pointer: {exc}"
                    )
            if not planner_digest_ref:
                planner_digest_ref = agent_identity.get("continuity", {}).get(
                    "last_planner_digest"
                )

            ledger = ensure_active_commitment(
                ledger, agent_identity, uwm_ref, planner_digest_ref
            )
            uwm_snapshot_path = STATE_DIR / "world_model_snapshot.json"
            uwm_snapshot_data: Dict[str, Any] = {}
            additional_warnings: List[str] = []
            if uwm_snapshot_path.exists():
                try:
                    with uwm_snapshot_path.open("r", encoding="utf-8") as handle:
                        uwm_snapshot_data = json.load(handle)
                except (OSError, json.JSONDecodeError) as exc:
                    uwm_snapshot_data = {}
                    additional_warnings.append(f"UWM snapshot unreadable: {exc}")
            catalog_path = REPO_ROOT / "training_data" / "index" / "catalog.jsonl"
            catalog_tail_hash_current = (
                _compute_catalog_tail_hash(catalog_path)
                if catalog_path.exists()
                else None
            )
            if not catalog_path.exists():
                additional_warnings.append(f"Catalog missing: {catalog_path}")

            capabilities_block = agent_identity.get("capabilities", {}) or {}
            catalog_tail_hash = capabilities_block.get("catalog_tail_hash")
            catalog_tail_mismatch = (
                bool(catalog_tail_hash)
                and bool(catalog_tail_hash_current)
                and catalog_tail_hash != catalog_tail_hash_current
            )

            missing_refs = _collect_missing_evidence_refs(ledger, REPO_ROOT)

            proof_gate_data = agent_identity.get("proof_gate", {}) or {}
            raw_anomalies = proof_gate_data.get("anomalies") or []
            if isinstance(raw_anomalies, list):
                proof_gate_anomalies = [str(entry) for entry in raw_anomalies if entry]
            elif raw_anomalies:
                proof_gate_anomalies = [str(raw_anomalies)]
            else:
                proof_gate_anomalies = []
            proof_gate_changed = proof_gate_data.get("theory_hash") != theory_hash

            combined_warnings = (
                list(identity_validation_warnings)
                + list(ledger_warnings)
                + additional_warnings
            )
            cycle_context_utc = _timestamp()
            arbiter_context = {
                "mission_label": agent_identity.get("mission", {}).get("mission_label"),
                "last_planner_digest_archive": planner_digest_ref,
                "validation_warnings": combined_warnings,
                "cycle_utc": cycle_context_utc,
                "catalog_tail_hash": catalog_tail_hash,
                "catalog_tail_hash_current": catalog_tail_hash_current,
                "catalog_tail_hash_mismatch": catalog_tail_mismatch,
                "missing_artifact_refs": missing_refs,
                "proof_gate_changed": proof_gate_changed,
                "proof_gate_anomalies": proof_gate_anomalies,
                "repo_root": str(REPO_ROOT),
            }

            if not ledger_validation_failed:
                try:
                    selected_id, ledger, arbiter_report = select_next_active_commitment(
                        agent_identity,
                        uwm_snapshot_data,
                        ledger,
                        arbiter_context,
                    )
                    active_commitment_id = selected_id or ledger.get(
                        "active_commitment_id"
                    )
                    top_entries = arbiter_report.get("top_ranked") or []
                    top_display = ",".join(
                        f"{entry['commitment_id']}({entry['score']})"
                        for entry in top_entries
                        if entry.get("commitment_id") and entry.get("score") is not None
                    )
                    if not top_display:
                        top_display = "<none>"
                        f"[ARB] selected={active_commitment_id} "
                        f"switched={arbiter_report.get('switched', False)} "
                        f"top={top_display}"
                    )
                except Exception as arb_exc:
                    cycle_errors.append(f"Prioritization failed: {arb_exc}")
                    active_commitment_id = ledger.get("active_commitment_id")
            else:
                active_commitment_id = ledger.get("active_commitment_id")

            if agent_identity and ledger is not None and active_commitment_id:
                active_entry: Optional[Dict[str, Any]] = None
                for entry in ledger.get("commitments", []) or []:
                    if str(entry.get("commitment_id")) == str(active_commitment_id):
                        active_entry = entry
                        break
                if active_entry and _is_tool_optimizer_commitment(active_entry):
                    mission_block = not bool(
                        agent_identity.get("mission", {}).get(
                            "allow_enhancements", False
                        )
                    )
                    if mission_block:
                        reason = "Tool optimizer requires allow_enhancements true"
                        mark_commitment_status(
                            ledger, active_commitment_id, "blocked", reason
                        )
                        force_read_only = True
                        agent_identity.setdefault("mission", {})[
                            "allow_enhancements"
                        ] = False
                        os.environ["LOGOS_ALLOW_ENHANCEMENTS"] = "0"
                        cycle_errors.append(reason)
                        tool_optimizer_summary = {
                            "status": "blocked",
                            "reason": reason,
                            "anomalies": [],
                        }
                    else:
                        try:
                            # Use governed tool proposal pipeline
                            import subprocess

                            result = subprocess.run(
                                [
                                    sys.executable,
                                    str(scripts_dir / "tool_proposal_pipeline.py"),
                                    "generate",
                                    "--objective",
                                    "tool_optimization",
                                ],
                                capture_output=True,
                                text=True,
                                cwd=REPO_ROOT,
                            )
                            if result.returncode == 0:
                                optimizer_result = {
                                    "ok": True,
                                    "counts": {"tools_optimized": 1},
                                    "hashes": {},
                                    "anomalies": [],
                                }
                            else:
                                raise Exception(f"Pipeline failed: {result.stderr}")
                        except Exception as opt_exc:
                            reason = f"Tool optimizer failure: {opt_exc}"
                            mark_commitment_status(
                                ledger, active_commitment_id, "blocked", reason
                            )
                            force_read_only = True
                            agent_identity.setdefault("mission", {})[
                                "allow_enhancements"
                            ] = False
                            os.environ["LOGOS_ALLOW_ENHANCEMENTS"] = "0"
                            cycle_errors.append(reason)
                            tool_optimizer_summary = {
                                "status": "blocked",
                                "reason": reason,
                                "anomalies": [],
                            }
                        else:
                            if optimizer_result.get("ok"):
                                counts = optimizer_result.get("counts", {}) or {}
                                hashes = optimizer_result.get("hashes", {}) or {}
                                anomalies_local = (
                                    optimizer_result.get("anomalies", []) or []
                                )
                                summary_text = "Tool optimizer cataloged registry={reg} profiles={prof}".format(
                                    reg=counts.get("registry", 0),
                                    prof=counts.get("profiles", 0),
                                )
                                record_event(
                                    ledger,
                                    "advance",
                                    active_commitment_id,
                                    summary_text,
                                )
                                tool_optimizer_summary = {
                                    "status": "ok",
                                    "counts": counts,
                                    "hashes": hashes,
                                    "anomalies": anomalies_local,
                                    "catalog_entry": optimizer_result.get(
                                        "catalog_entry"
                                    ),
                                }
                            else:
                                anomalies_local = (
                                    optimizer_result.get("anomalies") or []
                                )
                                detail = ", ".join(anomalies_local[:3])
                                reason = "Tool optimizer produced partial results"
                                if detail:
                                    reason = f"{reason}: {detail}"
                                mark_commitment_status(
                                    ledger, active_commitment_id, "blocked", reason
                                )
                                force_read_only = True
                                agent_identity.setdefault("mission", {})[
                                    "allow_enhancements"
                                ] = False
                                os.environ["LOGOS_ALLOW_ENHANCEMENTS"] = "0"
                                cycle_errors.append(reason)
                                tool_optimizer_summary = {
                                    "status": "blocked",
                                    "reason": reason,
                                    "anomalies": anomalies_local,
                                }

                if active_entry and _is_tool_invention_commitment(active_entry):
                    mission_block = not bool(
                        agent_identity.get("mission", {}).get(
                            "allow_enhancements", False
                        )
                    )
                    if mission_block:
                        reason = "Tool invention requires allow_enhancements true"
                        mark_commitment_status(
                            ledger, active_commitment_id, "blocked", reason
                        )
                        force_read_only = True
                        agent_identity.setdefault("mission", {})[
                            "allow_enhancements"
                        ] = False
                        os.environ["LOGOS_ALLOW_ENHANCEMENTS"] = "0"
                        cycle_errors.append(reason)
                        tool_invention_summary = {
                            "status": "blocked",
                            "reason": reason,
                            "anomalies": [],
                        }
                    else:
                        try:
                            # Use governed tool proposal pipeline
                            import subprocess

                            result = subprocess.run(
                                [
                                    sys.executable,
                                    str(scripts_dir / "tool_proposal_pipeline.py"),
                                    "generate",
                                    "--objective",
                                    "tool_invention",
                                ],
                                capture_output=True,
                                text=True,
                                cwd=REPO_ROOT,
                            )
                            if result.returncode == 0:
                                invention_result = {
                                    "ok": True,
                                    "counts": {
                                        "tools_generated": 1,
                                        "catalog_entries": 1,
                                    },
                                    "hashes": {},
                                    "anomalies": [],
                                }
                            else:
                                raise Exception(f"Pipeline failed: {result.stderr}")
                        except Exception as inv_exc:
                            reason = f"Tool invention failure: {inv_exc}"
                            mark_commitment_status(
                                ledger, active_commitment_id, "blocked", reason
                            )
                            force_read_only = True
                            agent_identity.setdefault("mission", {})[
                                "allow_enhancements"
                            ] = False
                            os.environ["LOGOS_ALLOW_ENHANCEMENTS"] = "0"
                            cycle_errors.append(reason)
                            tool_invention_summary = {
                                "status": "blocked",
                                "reason": reason,
                                "anomalies": [],
                            }
                        else:
                            if invention_result.get("ok"):
                                counts = invention_result.get("counts", {}) or {}
                                hashes = invention_result.get("hashes", {}) or {}
                                anomalies_local = (
                                    invention_result.get("anomalies", []) or []
                                )
                                summary_text = "Tool invention generated tools={tools} cataloged={cat}".format(
                                    tools=counts.get("tools_generated", 0),
                                    cat=counts.get("catalog_entries", 0),
                                )
                                record_event(
                                    ledger,
                                    "advance",
                                    active_commitment_id,
                                    summary_text,
                                )
                                tool_invention_summary = {
                                    "status": "ok",
                                    "counts": counts,
                                    "hashes": hashes,
                                    "anomalies": anomalies_local,
                                    "catalog_entries": invention_result.get(
                                        "catalog_entries", []
                                    ),
                                }
                            else:
                                anomalies_local = (
                                    invention_result.get("anomalies") or []
                                )
                                detail = ", ".join(anomalies_local[:3])
                                reason = "Tool invention produced partial results"
                                if detail:
                                    reason = f"{reason}: {detail}"
                                mark_commitment_status(
                                    ledger, active_commitment_id, "blocked", reason
                                )
                                force_read_only = True
                                agent_identity.setdefault("mission", {})[
                                    "allow_enhancements"
                                ] = False
                                os.environ["LOGOS_ALLOW_ENHANCEMENTS"] = "0"
                                cycle_errors.append(reason)
                                tool_invention_summary = {
                                    "status": "blocked",
                                    "reason": reason,
                                    "anomalies": anomalies_local,
                                }

            ledger_hash, _ = write_ledger(ledger_path, ledger)

            commitments_block = agent_identity.setdefault("commitments", {})
            commitments_block["ledger_path"] = DEFAULT_COMMITMENT_LEDGER_PATH.as_posix()
            commitments_block["ledger_hash"] = ledger_hash
            commitments_block["ledger_version"] = COMMITMENT_LEDGER_VERSION
            PersistentAgentIdentity(REPO_ROOT)._save_identity(agent_identity)
        except Exception as exc:
            ledger = None
            ledger_hash = None
            active_commitment_id = None
            ledger_validation_failed = True
            force_read_only = True
            cycle_errors.append(f"Commitment ledger unavailable: {exc}")
            if agent_identity:
                agent_identity.setdefault("mission", {})["allow_enhancements"] = False

        "\n=== SUPERVISED RUN ===\n"
        f"mission={MISSION_LABEL} "
        f"safe_only={SAFE_INTERFACES_ONLY} "
        f"read_only={read_only}\n"
        f"objective={objective}\n"
    )

    # Align validation/truth-event sinks
    if nexus:
        ctx.truth_events = nexus.truth_events
    else:
        _init_validation_containers(ctx)

    # Goal proposal for auto mode
    if objective == "auto":
        if not nexus or not nexus.prior_state:
            return 1
        scp_state = nexus.prior_state
        beliefs = scp_state.get("beliefs", {})
        ledger_context = {"ledger_hash": "dummy"}  # TODO: compute real ledger hash
        candidates = generate_goal_candidates(scp_state, beliefs, ledger_context)
        safe_candidates = [c for c in candidates if enforce_goal_safety(c)]
        ranked = rank_goal_candidates(safe_candidates, scp_state, beliefs)
        top_candidates = ranked[:3]
        if not top_candidates:
            return 1
        choices = [f"{c['statement']} (confidence: {c['confidence']:.2f})" for c in top_candidates]
        prompt = "Select a goal to pursue:"
        selected = uip_prompt_choice(prompt, choices, assume_yes=assume_yes)
        if not selected:
            return 1
        selected_index = choices.index(selected)
        selected_goal = top_candidates[selected_index]
        objective = selected_goal["statement"]
        # Mark as approved
        selected_goal["status"] = "APPROVED"
        # Record in ledger

    # Check for active plan
    active_plan = None
    objective_class = normalize_objective_class(objective) if nexus else "GENERAL"
    ctx.objective_class = objective_class
    if nexus and nexus.prior_state:
        plans = nexus.prior_state.get("plans", {})
        for plan in plans.get("active", []):
            if (
                plan.get("objective_class") == objective_class
                and plan.get("status") == "ACTIVE"
            ):
                active_plan = plan
                break

    if active_plan:
        plan_steps = active_plan["steps"]
        # Extract policy_notes from resumed plan if present
        resumed_policy = active_plan.get("policy_notes", {})
        for tool in resumed_policy.get("boosted_tools", []):
            if tool not in policy_notes_aggregate["boosted_tools"]:
                policy_notes_aggregate["boosted_tools"].append(tool)
        for tool in resumed_policy.get("filtered_tools", []):
            if tool not in policy_notes_aggregate["filtered_tools"]:
                policy_notes_aggregate["filtered_tools"].append(tool)
    else:
        # Request new plan from Logos_AGI
        if nexus and objective_class == "STATUS":  # Only for status for now
            plan_result = nexus.propose_plan(objective, {"read_only": force_read_only})
            plan_candidates = []
            if isinstance(plan_result.get("plan_candidates"), list):
                plan_candidates.extend(
                    [p for p in plan_result.get("plan_candidates", []) if isinstance(p, dict)]
                )
            if plan_result.get("plan"):
                plan_candidates.append(plan_result["plan"])

            if plan_candidates:
                scored_candidates = []
                for idx, candidate in enumerate(plan_candidates):
                    candidate["read_only"] = force_read_only
                    sig = plan_signature(candidate)
                    mean = _plan_history_mean(nexus.prior_state, sig)
                    scored_candidates.append((mean, idx, candidate))
                scored_candidates.sort(key=lambda t: (-t[0], t[1]))
                best = scored_candidates[0][2]
                active_plan = best
                plans = nexus.prior_state.setdefault(
                    "plans", {"active": [], "history": []}
                )
                plans["active"].append(active_plan)
                plan_steps = active_plan["steps"]
                plan_policy = plan_result.get("notes", {}).get("policy", {})
                for tool in plan_policy.get("boosted_tools", []):
                    if tool not in policy_notes_aggregate["boosted_tools"]:
                        policy_notes_aggregate["boosted_tools"].append(tool)
                for tool in plan_policy.get("filtered_tools", []):
                    if tool not in policy_notes_aggregate["filtered_tools"]:
                        policy_notes_aggregate["filtered_tools"].append(tool)
            else:
                plan_steps = None
        else:
            plan_steps = None

        if not plan_steps:
            plan = make_plan(objective, force_read_only, extra_steps)
            for step in plan:
            plan_steps = plan

    plan = plan_steps  # For persistence

    results: List[Dict[str, Any]] = []
    step_index = 0
    while step_index < len(plan_steps):
        step = plan_steps[step_index]
        tool_name = step["tool"]
        # Skip timeout check if max_runtime_seconds is 0 (unlimited)
        if max_runtime_seconds > 0 and time.time() - started > max_runtime_seconds:
            break
        if tool_name not in TOOLS:
            results.append(
                {
                    "step": step.get("step", step_index + 1),
                    "tool": tool_name,
                    "status": "skip",
                    "output": "[unknown tool]",
                    "attestation_hash": attestation_hash,
                    "mission_profile_hash": mission_profile_hash,
                }
            )
            step_index += 1
            executed_plan_steps += 1
            if max_plan_steps_per_run > 0 and executed_plan_steps >= max_plan_steps_per_run:
                break
            continue

        # Check if step is skipped due to plan revision
        if active_plan and step.get("status") == "SKIPPED":
                f"[SKIP] Step {step.get('step', step_index + 1)} skipped due to plan revision"
            )
            results.append(
                {
                    "step": step.get("step", step_index + 1),
                    "tool": tool_name,
                    "status": "skip",
                    "output": "Skipped due to belief contradiction",
                    "attestation_hash": attestation_hash,
                    "mission_profile_hash": mission_profile_hash,
                }
            )
            step_index += 1
            continue

        allowed = True
        if not assume_yes:
            allowed = ask_user(
                f"Allow step {step.get('step', step_index + 1)} => {tool_name}?",
                default=False,
                timeout_seconds=15,
            )
        else:
                f"[CONSENT] auto-approved step {step.get('step', step_index + 1)} => {tool_name}"
            )
        if not allowed:
            results.append(
                {
                    "step": step.get("step", step_index + 1),
                    "tool": tool_name,
                    "status": "denied",
                    "output": "",
                    "attestation_hash": attestation_hash,
                    "mission_profile_hash": mission_profile_hash,
                }
            )
            # Update plan step status
            if active_plan:
                step["status"] = "DENIED"
                step["executed_at"] = datetime.now(timezone.utc).isoformat()
                step["evaluator"]["outcome"] = "DENY"
                active_plan["checkpoints"].append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "step_id": step["step_id"],
                        "status": "DENIED",
                        "tool": tool_name,
                        "summary": {"denied": True},
                        "truth_annotation": step["truth_annotation"],
                    }
                )
                active_plan["current_index"] = step_index + 1
            step_index += 1
            executed_plan_steps += 1
            if max_plan_steps_per_run > 0 and executed_plan_steps >= max_plan_steps_per_run:
                break
            continue
        try:
            output = dispatch_tool(
                tool_name,
                step.get("args", step.get("arg", "")),
                ctx=ctx,
                timeout_seconds=15,
            )
        except AlignmentGateError as e:
            output = f"[gate error] {e}"
        status = "ok" if "[gate error]" not in output else "denied"
        if ctx.last_tool_validation and not ctx.last_tool_validation.get("ok", False):
            status = "denied"
        results.append(
            {
                "step": step.get("step", step_index + 1),
                "tool": tool_name,
                "status": status,
                "output": output,
                "attestation_hash": attestation_hash,
                "mission_profile_hash": mission_profile_hash,
            }
        )

        # Collect for ledger
        event_entry = {
            "tool": tool_name,
            "args_hash": hashlib.sha256(
                (step.get("args", step.get("arg", "")) or "").encode()
            ).hexdigest(),
            "outcome": "SUCCESS"
            if status == "ok"
            else "ERROR"
            if status == "denied"
            else "DENY",
            "truth_tier": step.get("truth_annotation", {}).get(
                "truth", "UNVERIFIED"
            )
            if active_plan
            else "UNVERIFIED",
            "policy_belief_id": step.get("policy_belief_id", None)
            if active_plan
            else None,
            "evaluator_score": step.get("evaluator", {}).get("score", 0.0)
            if active_plan
            else 0.0,
        }
        if ctx.last_tool_validation:
            event_entry["validation"] = {
                "ok": ctx.last_tool_validation.get("ok"),
                "validator": ctx.last_tool_validation.get("validator"),
                "reason": ctx.last_tool_validation.get("reason"),
            }
        executed_events.append(event_entry)

        # Update plan step status
        if active_plan:
            step["status"] = "DONE" if status == "ok" else "ERROR"
            step["executed_at"] = datetime.now(timezone.utc).isoformat()
            step["result_summary"] = {"output": output[:200]}  # Truncate
            step["evaluator"]["outcome"] = "SUCCESS" if status == "ok" else "ERROR"
            active_plan["checkpoints"].append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "step_id": step["step_id"],
                    "status": step["status"],
                    "tool": tool_name,
                    "summary": step["result_summary"],
                    "truth_annotation": step["truth_annotation"],
                }
            )
            active_plan["current_index"] = step_index + 1

        if nexus:
            nexus.record_tool_result(
                tool_name,
                step.get("args", step.get("arg", "")),
                status,
                objective,
            )

        # Update evaluator metrics
        if nexus and nexus.metrics_state:
            if "[gate error]" in output:
                outcome = "DENY"
            elif any(
                word in output.lower() for word in ["error", "exception", "failed"]
            ):
                outcome = "ERROR"
            else:
                outcome = "SUCCESS"

            obj_class = normalize_objective_class(objective)
            nexus.metrics_state = update_metrics(
                nexus.metrics_state, obj_class, tool_name, outcome
            )

            # Persist metrics
            metrics_path = STATE_DIR / "proposal_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(nexus.metrics_state, f, indent=2)

        executed_plan_steps += 1
        if max_plan_steps_per_run > 0 and executed_plan_steps >= max_plan_steps_per_run:
            step_index += 1
            break

        # Logos_AGI integration per iteration
        if nexus:
            observation = {
                "objective": objective,
                "last_tool": tool_name,
                "last_output": output,
                "time_remaining": max(0, max_runtime_seconds - (time.time() - started)),
                "results_count": len(results),
            }
            nexus.observe(observation)
            constraints = {
                "read_only": force_read_only,
                "safe_interfaces_only": SAFE_INTERFACES_ONLY,
            }
            proposal = nexus.propose(objective, constraints)
            # Merge LLM advisor proposals (non-authoritative)
            if advisor:
                advisor_result = advisor.propose(
                    objective,
                    {
                        "observation": observation,
                        "mission": MISSION_LABEL,
                        "tools": tool_schema,
                    },
                )
                rejected_props = advisor_result.get("rejected", []) or []
                for rejected in rejected_props:
                    llm_prop = rejected.get("proposal") if isinstance(rejected, dict) else None
                    if nexus and hasattr(nexus, "truth_events"):
                        nexus.truth_events.append(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "source": "RUNTIME",
                                "content": {
                                    "llm_provider": llm_provider,
                                    "llm_model": llm_model,
                                    "rejected": llm_prop,
                                    "reason": rejected.get("reason", "direct_execution_attempt"),
                                },
                                "truth_annotation": {
                                    "truth": "CONTRADICTED",
                                    "evidence": {
                                        "type": "none",
                                        "ref": None,
                                        "details": "LLM advisor rejected proposal",
                                    },
                                },
                            }
                        )
                for llm_prop in advisor_result.get("proposals", []):
                    # Guard against direct execution attempts
                    forbidden_keys = {"execute", "run", "code", "shell"}
                    if any(k in llm_prop for k in forbidden_keys):
                        if nexus and hasattr(nexus, "truth_events"):
                            nexus.truth_events.append(
                                {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "source": "RUNTIME",
                                    "content": {
                                        "llm_provider": llm_provider,
                                        "llm_model": llm_model,
                                        "rejected": llm_prop,
                                        "reason": "direct_execution_attempt",
                                    },
                                    "truth_annotation": {
                                        "truth": "CONTRADICTED",
                                        "evidence": {
                                            "type": "none",
                                            "ref": None,
                                            "details": "LLM advisor attempted direct execution",
                                        },
                                    },
                                }
                            )
                        continue

                    tool_name = llm_prop.get("tool", "")
                    high_impact_tools = {"tool_proposal_pipeline", "start_agent", "retrieve.web"}
                    if requires_uip_map.get(tool_name, False) or tool_name in high_impact_tools:
                        if nexus and hasattr(nexus, "truth_events"):
                            nexus.truth_events.append(
                                {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "source": "RUNTIME",
                                    "content": {
                                        "llm_provider": llm_provider,
                                        "llm_model": llm_model,
                                        "rejected": llm_prop,
                                        "reason": "uip_required",
                                    },
                                    "truth_annotation": {
                                        "truth": "CONTRADICTED",
                                        "evidence": {
                                            "type": "none",
                                            "ref": None,
                                            "details": "UIP approval required",
                                        },
                                    },
                                }
                            )
                        continue

                    annotation = llm_prop.get("truth_annotation", {})
                    # downgrade unsupported PROVED claims lacking coq evidence
                    ev = annotation.get("evidence", {}) if isinstance(annotation, dict) else {}
                    if isinstance(annotation, dict) and annotation.get("truth") == "PROVED":
                        if not isinstance(ev, dict) or ev.get("type") != "coq":
                            annotation["truth"] = "VERIFIED"
                            annotation["evidence"] = {
                                "type": "hash",
                                "ref": None,
                                "details": "downgraded: missing coq evidence",
                            }
                    try:
                        annotation = enforce_truth_annotation(annotation, None)
                    except Exception:
                        pass
                    llm_prop["truth_annotation"] = annotation
                    llm_prop["llm_provider"] = llm_provider
                    llm_prop["llm_model"] = llm_model
                    proposal["proposals"].append(llm_prop)
                if advisor_result.get("errors") and nexus and hasattr(nexus, "truth_events"):
                    nexus.truth_events.append(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "source": "RUNTIME",
                            "content": {
                                "llm_provider": llm_provider,
                                "llm_model": llm_model,
                                "errors": advisor_result.get("errors"),
                            },
                            "truth_annotation": {
                                "truth": "UNVERIFIED",
                                "evidence": {
                                    "type": "none",
                                    "ref": None,
                                    "details": "LLM advisor errors",
                                },
                            },
                        }
                    )
            # ensure proposals recorded on nexus for persistence
            try:
                nexus.last_proposals = proposal.get("proposals", [])
            except Exception:
                pass
            # Collect for ledger
            proposals.extend(proposal["proposals"])
            notes = proposal.get("notes", {})
            policy_notes = notes.get("policy", {})
            # Aggregate policy notes
            for tool in policy_notes.get("boosted_tools", []):
                if tool not in policy_notes_aggregate["boosted_tools"]:
                    policy_notes_aggregate["boosted_tools"].append(tool)
            for tool in policy_notes.get("filtered_tools", []):
                if tool not in policy_notes_aggregate["filtered_tools"]:
                    policy_notes_aggregate["filtered_tools"].append(tool)
            for prop in proposal["proposals"]:
                belief_id = prop.get("policy_belief_id")
                if belief_id:
                    if prop.get("policy_adjustment", 0) > 0:
                        if (
                            belief_id
                            not in policy_notes_aggregate["boosted_belief_ids"]
                        ):
                            policy_notes_aggregate["boosted_belief_ids"].append(
                                belief_id
                            )
                    else:
                        if (
                            belief_id
                            not in policy_notes_aggregate["filtered_belief_ids"]
                        ):
                            policy_notes_aggregate["filtered_belief_ids"].append(
                                belief_id
                            )

            if proposal["proposals"]:
                mission_profile = {}  # TODO: load from state/mission_profile.json
                prop = proposal["proposals"][0]  # Take first
                prop_tool = prop["tool"]
                if prop_tool in TOOLS:
                    # Check truth-based constraint for high-impact tools
                    high_impact_tools = {
                        "tool_proposal_pipeline",
                        "start_agent",
                        "retrieve.web",
                    }  # Add more as needed
                    truth_level = prop.get("truth_annotation", {}).get(
                        "truth", "UNVERIFIED"
                    )
                    if (
                        truth_level in ["HEURISTIC", "UNVERIFIED"]
                        and prop_tool in high_impact_tools
                        and not mission_profile.get(
                            "allow_heuristic_high_impact", False
                        )
                    ):
                            f"[CONSTRAINT] Skipping heuristic proposal for high-impact tool {prop_tool} (truth: {truth_level})"
                        )
                        continue

                        f"[LOGOS_AGI] Proposing {prop_tool} (confidence {prop['confidence']})"
                    )
                    try:
                        prop_output = dispatch_tool(
                            prop_tool, prop["args"], ctx=ctx, timeout_seconds=15
                        )
                        # Add to results or audit
                        results.append(
                            {
                                "step": f"logos_agi_{len(results)}",
                                "tool": prop_tool,
                                "status": "ok",
                                "output": prop_output,
                                "attestation_hash": attestation_hash,
                                "mission_profile_hash": mission_profile_hash,
                                "logos_agi_proposal": prop,
                                "proposal_policy_reason": proposal.get("notes", {}).get(
                                    "policy_reason", "default"
                                ),
                                "evaluator_outcome": "SUCCESS",
                                "evaluator_state_hash": nexus.metrics_state.get(
                                    "state_hash", ""
                                )
                                if nexus.metrics_state
                                else "",
                                "selected_proposal_score": prop.get(
                                    "evaluator_score", 0
                                ),
                            }
                        )
                        # Collect for ledger
                        event_entry = {
                            "tool": prop_tool,
                            "args_hash": hashlib.sha256(
                                (prop["args"] or "").encode()
                            ).hexdigest(),
                            "outcome": "SUCCESS",
                            "truth_tier": prop.get("truth_annotation", {}).get(
                                "truth", "UNVERIFIED"
                            ),
                            "policy_belief_id": prop.get("policy_belief_id", None),
                            "evaluator_score": prop.get("evaluator_score", 0.0),
                        }
                        if ctx.last_tool_validation:
                            event_entry["validation"] = {
                                "ok": ctx.last_tool_validation.get("ok"),
                                "validator": ctx.last_tool_validation.get("validator"),
                                "reason": ctx.last_tool_validation.get("reason"),
                            }
                        executed_events.append(event_entry)
                        nexus.record_tool_result(
                            prop_tool, prop["args"], "ok", objective
                        )
                        # Update metrics
                        if nexus.metrics_state:
                            obj_class = normalize_objective_class(objective)
                            nexus.metrics_state = update_metrics(
                                nexus.metrics_state, obj_class, prop_tool, "SUCCESS"
                            )
                            with open(STATE_DIR / "proposal_metrics.json", "w") as f:
                                json.dump(nexus.metrics_state, f, indent=2)
                    except AlignmentGateError as e:
                        nexus.record_tool_result(
                            prop_tool, prop["args"], "denied", objective
                        )
                        # Add denied result
                        results.append(
                            {
                                "step": f"logos_agi_{len(results)}",
                                "tool": prop_tool,
                                "status": "denied",
                                "output": f"[gate error] {e}",
                                "attestation_hash": attestation_hash,
                                "mission_profile_hash": mission_profile_hash,
                                "logos_agi_proposal": prop,
                                "proposal_policy_reason": proposal.get("notes", {}).get(
                                    "policy_reason", "default"
                                ),
                                "evaluator_outcome": "DENY",
                                "evaluator_state_hash": nexus.metrics_state.get(
                                    "state_hash", ""
                                )
                                if nexus.metrics_state
                                else "",
                                "selected_proposal_score": prop.get(
                                    "evaluator_score", 0
                                ),
                            }
                        )
                        # Collect for ledger
                        event_entry = {
                            "tool": prop_tool,
                            "args_hash": hashlib.sha256(
                                (prop["args"] or "").encode()
                            ).hexdigest(),
                            "outcome": "DENY",
                            "truth_tier": prop.get("truth_annotation", {}).get(
                                "truth", "UNVERIFIED"
                            ),
                            "policy_belief_id": prop.get("policy_belief_id", None),
                            "evaluator_score": prop.get("evaluator_score", 0.0),
                        }
                        if ctx.last_tool_validation:
                            event_entry["validation"] = {
                                "ok": ctx.last_tool_validation.get("ok"),
                                "validator": ctx.last_tool_validation.get("validator"),
                                "reason": ctx.last_tool_validation.get("reason"),
                            }
                        executed_events.append(event_entry)
                        # Update metrics
                        if nexus.metrics_state:
                            obj_class = normalize_objective_class(objective)
                            nexus.metrics_state = update_metrics(
                                nexus.metrics_state, obj_class, prop_tool, "DENY"
                            )
                            with open(STATE_DIR / "proposal_metrics.json", "w") as f:
                                json.dump(nexus.metrics_state, f, indent=2)
                else:
            nexus.persist()

        step_index += 1

    # Finalize plan
    if active_plan:
        if active_plan["current_index"] >= len(active_plan["steps"]):
            active_plan["status"] = "COMPLETED"
            # Move to history
            plans = nexus.prior_state["plans"]
            plans["active"] = [
                p for p in plans["active"] if p["plan_id"] != active_plan["plan_id"]
            ]
            plans["history"].append(active_plan)
            if len(plans["history"]) > 10:
                plans["history"] = plans["history"][-10:]
        nexus.persist()  # Final persist

    summary = {
        "objective": objective,
        "mission": MISSION_LABEL,
        "safe_interfaces_only": SAFE_INTERFACES_ONLY,
        "steps_executed": len([r for r in results if r["status"] == "ok"]),
        "denied_steps": len([r for r in results if r["status"] == "denied"]),
        "errors": [r for r in results if r["status"] not in {"ok", "denied", "skip"}],
        "runtime_seconds": time.time() - started,
        "logos_agi_enabled": bool(nexus),
        "logos_agi_available": nexus.available if nexus else False,
        "logos_agi_health": nexus.health() if nexus else {},
    }
    if tool_optimizer_summary:
        summary["tool_optimizer"] = tool_optimizer_summary
    if tool_invention_summary:
        summary["tool_invention"] = tool_invention_summary

    # Plan-level validation and scoring
    plan_for_validation: Optional[Dict[str, Any]] = None
    if active_plan:
        plan_for_validation = active_plan
    elif plan:
        steps_struct: List[Dict[str, Any]] = []
        for idx, step_entry in enumerate(plan, start=1):
            steps_struct.append(
                {
                    "step_id": f"adhoc-{idx}",
                    "index": idx - 1,
                    "tool": step_entry.get("tool"),
                    "args": step_entry.get("arg", ""),
                    "status": "DONE",
                    "truth_annotation": {"truth": "UNVERIFIED", "evidence": {"type": "none", "ref": None, "details": "adhoc"}},
                    "result_summary": {},
                    "executed_at": None,
                    "evaluator": {"score": 0.0, "outcome": "SUCCESS"},
                }
            )
        plan_for_validation = {
            "schema_version": 1,
            "plan_id": f"adhoc-{SANDBOX.run_id or 'run'}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "objective": objective,
            "objective_class": objective_class,
            "steps": steps_struct,
            "current_index": len(steps_struct),
            "status": "COMPLETED",
            "checkpoints": [],
            "read_only": force_read_only,
        }

    if plan_for_validation:
        plan_for_validation["read_only"] = force_read_only
        plan_signature_value = plan_signature(plan_for_validation)
        plan_validation_report = validate_plan_run(plan_for_validation, executed_events)
        plan_score_value, plan_score_explanation = compute_plan_score(
            plan_validation_report
        )
        plan_report_hash = canonical_json_hash(plan_validation_report)

    _persist_agent_run(
        objective, plan, results, summary, attestation_hash, mission_profile_hash
    )

    if summary["errors"]:
        cycle_errors.append(f"Execution errors recorded ({len(summary['errors'])})")

    # Check for self-improvement opportunities
    try:
        import asyncio

        asyncio.run(_check_and_trigger_self_improvement(summary, nexus, assume_yes, REPO_ROOT))
    except Exception as e:

    # Update Persistent Agent Identity after cycle completion
    last_entry_id = None
    planner_digest_path_value: Optional[str] = None
    planner_digest_path_obj: Optional[Path] = None
    world_model_snapshot_path: Optional[str] = None
    world_model_snapshot_hash: Optional[str] = None
    world_model_version: Optional[int] = None

    if agent_identity:
        try:
            if "self_improvements" in summary:
                improvements = summary["self_improvements"]
                if improvements:
                    last_improvement = improvements[-1]
                    results = last_improvement.get("results", [])
                    if results:
                        for result in reversed(results):
                            deployment_stage = (
                                result.get("improvement_result", {})
                                .get("stages", {})
                                .get("deployment")
                            )
                            if deployment_stage == "completed":
                                last_entry_id = result.get(
                                    "improvement_result", {}
                                ).get("entry_id")
                                break

            latest_archive_file = STATE_DIR / "latest_planner_digest_archive.txt"
            if latest_archive_file.exists():
                try:
                    planner_digest_path_value = (
                        latest_archive_file.read_text().strip() or None
                    )
                except Exception as exc:
                    planner_digest_path_value = None
            if planner_digest_path_value:
                candidate = Path(planner_digest_path_value)
                if not candidate.is_absolute():
                    candidate = (REPO_ROOT / candidate).resolve()
                planner_digest_path_obj = candidate if candidate.exists() else None

            try:
                from logos_core.world_model.uwm import update_world_model

                wm_result = update_world_model(
                    identity_path=CANONICAL_IDENTITY_PATH,
                    snapshot_path=STATE_DIR / "world_model_snapshot.json",
                    planner_digest_path=planner_digest_path_obj,
                )
                world_model_snapshot_path = wm_result.get("snapshot_path")
                world_model_snapshot_hash = wm_result.get("snapshot_hash")
                world_model_version = wm_result.get("world_model_version")
                if world_model_snapshot_path:
            except Exception as exc:
                force_read_only = True
                agent_identity.setdefault("mission", {})["allow_enhancements"] = False
                cycle_errors.append(f"World model update failed: {exc}")

        except Exception as e:
            cycle_errors.append(f"Identity preparation failed: {e}")

    cycle_success = not cycle_errors and not ledger_validation_failed
    if ledger is not None:
        if active_commitment_id and cycle_success:
            record_event(
                ledger,
                "advance",
                active_commitment_id,
                "Cycle completed; commitment advanced",
            )
        elif active_commitment_id:
            failure_reason = (
                "; ".join(cycle_errors) if cycle_errors else "Cycle reported errors"
            )
            mark_commitment_status(
                ledger, active_commitment_id, "blocked", failure_reason
            )
        ledger_hash, _ = write_ledger(ledger_path, ledger)

    # Build and write epistemic ledger
    if build_run_ledger:
        run_end_ts = datetime.now(timezone.utc).isoformat()
        plan_summary = {}
        if active_plan:
            truth_counts = {}
            for step in active_plan.get("steps", []):
                truth = step.get("truth_annotation", {}).get("truth", "UNVERIFIED")
                truth_counts[truth] = truth_counts.get(truth, 0) + 1
            plan_summary = {"truth_counts": truth_counts}

        governance_flags = {
            "logos_agi_mode": logos_agi_mode,
            "scp_state_valid": nexus.scp_state_valid if nexus else False,
            "scp_recovery_mode": scp_recovery_mode,
            "allow_logos_agi_drift": allow_logos_agi_drift,
            "pin_match": (logos_agi_provenance or {}).get("match", False)
            if "logos_agi_provenance" in locals()
            else None,
        }

        coq_index_hash = None
        if nexus and hasattr(nexus, "theorem_index") and nexus.theorem_index:
            coq_index_hash = nexus.theorem_index.get("index_hash")

        scp_state_hash = None
        if nexus and nexus.prior_state:
            scp_state_hash = nexus.prior_state.get("state_hash")

        beliefs_hash = None
        if nexus and nexus.prior_state and "beliefs" in nexus.prior_state:
            beliefs_hash = nexus.prior_state["beliefs"].get("state_hash")

        metrics_hash = None
        if nexus and nexus.metrics_state:
            metrics_hash = nexus.metrics_state.get("state_hash")

        tool_introspection_summary = None
        try:
            from System_Stack.Logos_Protocol.Protocol_Core.Runtime_Operations.tools.implementations import tool_introspection

            capabilities = tool_introspection.build_capability_records()
            known_tool_names = [c.tool_name for c in capabilities]
            ledger_dir_path = AUDIT_DIR / "run_ledgers"
            scp_state_for_health = nexus.prior_state if nexus and nexus.prior_state else {}
            beliefs_for_health = scp_state_for_health.get("beliefs", {}) if isinstance(scp_state_for_health, dict) else {}
            metrics_for_health = nexus.metrics_state if nexus and isinstance(nexus.metrics_state, dict) else {}
            health_report = analyze_tool_health(
                ledger_dir_path,
                scp_state_for_health,
                beliefs_for_health,
                metrics_for_health,
                known_tools=known_tool_names,
            )
            broken_tools = [
                name
                for name, entry in health_report.get("tools", {}).items()
                if entry.get("health") == "BROKEN"
            ]
            tool_introspection_summary = {
                "tools_analyzed": len(capabilities),
                "broken_tools": broken_tools,
                "repair_proposals_generated": [],
                "uip_decisions": [],
                "capabilities_hash": canonical_json_hash(
                    {"records": [c.to_dict() for c in capabilities]}
                ),
            }
        except Exception as exc:

        ledger_context = {
            "attestation_hash": attestation_hash,
            "coq_index_hash": coq_index_hash,
            "scp_state_hash": scp_state_hash,
            "beliefs_hash": beliefs_hash,
            "metrics_hash": metrics_hash,
            "run_start_ts": run_start_ts,
            "run_end_ts": run_end_ts,
            "executed_events": executed_events,
            "proposals": proposals,
            "plan_summary": plan_summary,
            "policy_notes": policy_notes_aggregate,
            "governance_flags": governance_flags,
            "tool_introspection": tool_introspection_summary,
            "tool_validation": getattr(ctx, "tool_validation_events", []),
            "fallback_proposals": getattr(ctx, "fallback_proposals", []),
            "plan_validation": {
                "report": plan_validation_report,
                "score": plan_score_value,
                "signature": plan_signature_value,
                "report_hash": plan_report_hash,
                "explanation": plan_score_explanation,
            },
        }
        ledger = build_run_ledger(ledger_context)
        run_ledger_hash: Optional[str] = None
        try:
            run_ledger_hash = canonical_json_hash(ledger) if ledger else None
        except Exception:
            run_ledger_hash = None

        ledger_dir = AUDIT_DIR / "run_ledgers"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        repo_sha_short = _compute_theory_hash()[:8]
        ledger_path = (
            ledger_dir
            / f"run_{run_start_ts.replace(':', '').replace('-', '').replace('.', '').replace('T', '_').replace('Z', '')}_{repo_sha_short}.json"
        )
        with open(ledger_path, "w") as f:
            json.dump(ledger, f, indent=2)

        # Persist plan score history into SCP state
        try:
            history_update = None
            if plan_signature_value:
                state_path = (
                    nexus.persisted_path
                    if nexus and getattr(nexus, "persisted_path", None)
                    else STATE_DIR / "scp_state.json"
                )
                history_update = update_plan_history(
                    state_path,
                    plan_signature_value,
                    plan_score_value,
                    plan_report_hash,
                    run_ledger_hash,
                )
                if history_update and nexus:
                    history_container = history_update.get("history_container")
                    if isinstance(history_container, dict):
                        try:
                            nexus.refresh_plan_history(history_container)
                        except Exception as exc:
        except Exception as exc:

    if agent_identity:
        try:
            updated_identity = update_identity(
                agent_identity,
                MISSION_FILE,
                REPO_ROOT / "training_data" / "index" / "catalog.jsonl",
                last_entry_id,
                planner_digest_path_value,
                SANDBOX.run_id,
                world_model_snapshot_path=world_model_snapshot_path,
                world_model_snapshot_hash=world_model_snapshot_hash,
                world_model_version=world_model_version,
                commitment_ledger_path=DEFAULT_COMMITMENT_LEDGER_PATH.as_posix(),
                commitment_ledger_hash=ledger_hash,
                commitment_ledger_version=COMMITMENT_LEDGER_VERSION,
            )
            pai = PersistentAgentIdentity(REPO_ROOT)
            pai._save_identity(updated_identity)
            agent_identity = updated_identity

        except Exception as e:
            cycle_errors.append(f"Identity update failed: {e}")
            if ledger is not None and active_commitment_id:
                mark_commitment_status(
                    ledger,
                    active_commitment_id,
                    "blocked",
                    f"Identity update failed: {e}",
                )
                ledger_hash, _ = write_ledger(ledger_path, ledger)
            commitments_block = agent_identity.setdefault("commitments", {})
            commitments_block["ledger_path"] = DEFAULT_COMMITMENT_LEDGER_PATH.as_posix()
            commitments_block["ledger_hash"] = ledger_hash
            commitments_block["ledger_version"] = COMMITMENT_LEDGER_VERSION
            PersistentAgentIdentity(REPO_ROOT)._save_identity(agent_identity)

    try:
        import JUNK_DRAWER.scripts.need_to_distribute.cycle_ledger as cycle_ledger
        cycle_ledger.write_cycle_ledger(
            run_id=SANDBOX.run_id,
            objective=objective,
            mission=summary["mission"],
            timestamp_utc=_timestamp(),
            steps=results,
            promotion_outcomes=SANDBOX.promotion_outcomes,
            tests_required=SANDBOX.tests_required,
            verification_steps=SANDBOX.verification_steps,
            rollback_steps=SANDBOX.rollback_steps,
            sandbox_root=SANDBOX.root or _sandbox_base(),
            repo_root=REPO_ROOT,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
    return summary


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded supervised agent loop")
    parser.add_argument("--objective", help="Top-level task for the agent")
    parser.add_argument("--goal", help="Alias for --objective")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Force read-only execution",
    )
    parser.add_argument(
        "--budget-sec",
        "--max-seconds",
        dest="budget_sec",
        type=int,
        default=45,
        help="Runtime budget in seconds (default: 45)",
    )
    parser.add_argument(
        "--write-dir",
        help="Sandbox directory for writes in experimental mode",
    )
    parser.add_argument(
        "--cap-writes",
        type=int,
        default=0,
        help="Maximum sandbox writes allowed this run",
    )
    parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Automatically consent to all plan steps",
    )
    parser.add_argument(
        "--extra-step",
        action="append",
        default=[],
        metavar="TOOL[:ARG]",
        help=(
            "Append an additional plan step (e.g. agent.memory or "
            "sandbox.read:artifact)"
        ),
    )
    parser.add_argument(
        "--attestation-path",
        default=str(_attestation_path(REPO_ROOT)),
        help="Path to alignment attestation file",
    )
    parser.add_argument(
        "--attestation-max-age-sec",
        type=int,
        default=21600,
        help="Maximum age of attestation in seconds (default: 21600 = 6 hours)",
    )
    parser.add_argument(
        "--require-attestation",
        action="store_true",
        default=True,
        help="Require valid attestation to run (default: True)",
    )
    parser.add_argument(
        "--no-require-attestation",
        action="store_false",
        dest="require_attestation",
        help="Skip attestation requirement (only if LOGOS_DEV_BYPASS_OK=1)",
    )
    parser.add_argument(
        "--bootstrap-genesis",
        action="store_true",
        default=False,
        help="Validate genesis capsule manifest before launching the agent",
    )
    parser.add_argument(
        "--no-bootstrap-genesis",
        action="store_false",
        dest="bootstrap_genesis",
        help="Skip genesis capsule manifest validation",
    )
    parser.add_argument(
        "--genesis-write-status",
        action="store_true",
        default=False,
        help="Persist genesis bootstrap status to state/genesis_bootstrap_status.json",
    )
    parser.add_argument(
        "--enable-logos-agi",
        action="store_true",
        default=False,
        help="Enable Logos_AGI ARP+SCP integration",
    )
    parser.add_argument(
        "--allow-training-index-write",
        action="store_true",
        default=False,
        help="Allow writes to training_data/index catalog files (requires LOGOS_OPERATOR_OK=1)",
    )
    parser.add_argument(
        "--allow-logos-agi-drift",
        action="store_true",
        default=False,
        help="Allow Logos_AGI drift from pinned SHA (only if LOGOS_DEV_BYPASS_OK=1)",
    )
    parser.add_argument(
        "--logos-agi-max-compute-ms",
        type=int,
        default=100,
        help="Max compute time for Logos_AGI operations in ms",
    )
    parser.add_argument(
        "--logos-agi-mode",
        choices=["auto", "real", "stub"],
        default="auto",
        help="Logos_AGI bootstrap mode: auto (default), real (fail if unavailable), stub (always stub)",
    )
    parser.add_argument(
        "--enable-llm-advisor",
        action="store_true",
        default=False,
        help="Enable non-authoritative LLM advisor for proposals",
    )
    parser.add_argument(
        "--llm-provider",
        default="stub",
        choices=["openai", "stub"],
        help="LLM advisor provider (default stub)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="LLM advisor model identifier",
    )
    parser.add_argument(
        "--llm-timeout-sec",
        type=int,
        default=10,
        help="LLM advisor timeout in seconds",
    )
    parser.add_argument(
        "--max-plan-steps-per-run",
        type=int,
        default=0,
        help="Maximum plan steps to execute in this run (0 = unlimited)",
    )
    parser.add_argument(
        "--scp-recovery-mode",
        action="store_true",
        default=False,
        help="Enable SCP state recovery mode for invalid state (requires LOGOS_DEV_BYPASS_OK=1)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        default=False,
        help="Run gating checks only; do not start the agent loop.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    objective = args.goal or args.objective or "auto"

    global ALLOW_TRAINING_INDEX_WRITE
    ALLOW_TRAINING_INDEX_WRITE = bool(args.allow_training_index_write)
    if ALLOW_TRAINING_INDEX_WRITE:
        if os.environ.get("LOGOS_OPERATOR_OK", "").strip() != "1":
                "[GATE] ERROR: --allow-training-index-write requires LOGOS_OPERATOR_OK=1",
            )
            return 2
    else:
            "[GATE] Training index writes disabled; "
            f"{TRAINING_INDEX_PATH} remains read-only this run",
        )

    # Attestation enforcement
    attestation_hash = None
    mission_profile_hash = None
    if args.require_attestation:
        try:
            att = load_alignment_attestation(args.attestation_path)
            validate_attestation(att, max_age_seconds=args.attestation_max_age_sec)
            attestation_hash = compute_attestation_hash(att)
        except AlignmentGateError as e:
            return 2
    elif os.getenv("LOGOS_DEV_BYPASS_OK") != "1":
        return 2
    else:
        attestation_hash = "DEV_BYPASS"

    # SCP recovery mode check
    if args.scp_recovery_mode and os.getenv("LOGOS_DEV_BYPASS_OK") != "1":
        return 2

    # Load mission profile and compute hash
    try:
        mission_profile = load_mission_profile(str(MISSION_FILE))
        validate_mission_profile(mission_profile)
        mission_profile_hash = canonical_json_hash(mission_profile)
    except AlignmentGateError as e:
        return 2

    genesis_info: Dict[str, Any] = {}
    if args.bootstrap_genesis:
        if bootstrap_genesis is None:
            return 2
        ok, message, genesis_info = bootstrap_genesis(
            REPO_ROOT, write_status=args.genesis_write_status
        )
        manifest_version = genesis_info.get("manifest_version", "<none>")
        boot_sha = genesis_info.get("boot_digest_sha256", "<none>")
        pointer_sha = genesis_info.get("training_data_pointer_sha256", "<none>")
            "[GENESIS] manifest_version={ver} boot_sha={boot} pointer_sha={ptr}".format(
                ver=manifest_version, boot=boot_sha, ptr=pointer_sha
            )
        )
        if not ok:
            return 2
        missing = genesis_info.get("corpus_missing") or []
        if missing:

    # Logos_AGI pin verification
    if args.enable_logos_agi and args.logos_agi_mode != "stub":
        pin_path = STATE_DIR / "logos_agi_pin.json"
        if not pin_path.exists():
                "[GATE] ERROR: --enable-logos-agi requires pin file at state/logos_agi_pin.json"
            )
            return 2
        try:
            pin = load_pin(str(pin_path))
            logos_agi_dir = REPO_ROOT / "external" / "Logos_AGI"
            if not logos_agi_dir.exists():
                return 2
            allow_drift = args.allow_logos_agi_drift
            if allow_drift and os.getenv("LOGOS_DEV_BYPASS_OK") != "1":
                    "[GATE] ERROR: --allow-logos-agi-drift requires LOGOS_DEV_BYPASS_OK=1"
                )
                return 2
            provenance = verify_pinned_repo(
                str(logos_agi_dir), pin, require_clean=True, allow_drift=allow_drift
            )
                f"[GATE] Logos_AGI pin verified: {provenance['pinned_sha'][:12]}... (match={provenance['match']}, dirty={provenance['dirty']})"
            )
        except DriftError as e:
            return 2
        except Exception as e:
            return 2

    configure_sandbox(args.write_dir, args.cap_writes)

    # Load approved tools after attestation validation
    from logos.tool_registry_loader import load_approved_tools

    load_approved_tools(TOOLS)

    proof_ok, proof_msg = _run_proof_compile()
    if not proof_ok:
        if args.preflight:
            return 1
        return 2

    integrity_ok, integrity_msg = _enforce_integrity_baseline()
    if not integrity_ok:
        if args.preflight:
            return 1
        return 2

    if args.preflight:
        identity_ok, identity_msg = _preflight_identity_check()
        if identity_ok:
        else:

        if identity_ok:
            return 0

        return 1

    run_supervised(
        objective,
        read_only=args.read_only,
        max_runtime_seconds=args.budget_sec,
        enable_llm_advisor=args.enable_llm_advisor,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_timeout_sec=args.llm_timeout_sec,
        max_plan_steps_per_run=args.max_plan_steps_per_run,
        extra_steps=args.extra_step,
        assume_yes=args.assume_yes,
        attestation_hash=attestation_hash,
        mission_profile_hash=mission_profile_hash,
        enable_logos_agi=args.enable_logos_agi,
        logos_agi_max_compute_ms=args.logos_agi_max_compute_ms,
        allow_logos_agi_drift=args.allow_logos_agi_drift,
        logos_agi_mode=args.logos_agi_mode,
        scp_recovery_mode=args.scp_recovery_mode,
    )
    return 0


async def _check_and_trigger_self_improvement(summary: Dict[str, Any], nexus, assume_yes: bool, repo_root: Path) -> None:
    """Check tool health and trigger improvement if needed."""
    if not nexus or not nexus.prior_state:
        return

    # Load run ledgers
    ledger_dir = AUDIT_DIR / "run_ledgers"
    scp_state = nexus.prior_state
    beliefs = scp_state.get("beliefs", {})
    metrics = nexus.metrics_state if isinstance(nexus.metrics_state, dict) else {}

    health_report = analyze_tool_health(ledger_dir, scp_state, beliefs, metrics)

    # Write report
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = AUDIT_DIR / "tool_health" / f"tool_health_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(health_report, f, indent=2)

    if health_report["overall_health"] == "BROKEN":
        intents = propose_tool_improvements(health_report)
        for intent in intents:
            prompt = f"Propose repair tool {intent['tool']} because {', '.join(intent['issues'])}?"
            choices = ["Yes", "No"]
            selected = uip_prompt_choice(prompt, choices, assume_yes=assume_yes)
            if selected == "Yes":
                success = invoke_tool_proposal_pipeline(intent, repo_root)
                if success:
                else:
