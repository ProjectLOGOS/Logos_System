# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""FastAPI surface for LOGOS-GPT (advisor-only, UIP gated)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
scripts_dir = REPO_ROOT / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

AUDIT_ROOT = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit"))

import scripts.llm_interface_suite.logos_gpt_chat as chat_loop  # noqa: E402
# pylint: disable=wrong-import-position
from logos.ledger import build_run_ledger  # noqa: E402  # type: ignore
from logos.proof_refs import \
    enforce_truth_annotation  # noqa: E402  # type: ignore
from logos.uwm import (add_memory_item,  # noqa: E402  # type: ignore
                       calculate_initial_salience, init_working_memory, recall,
                       stable_item_id)
from scripts.llm_interface_suite.llm_advisor import LLMAdvisor, build_tool_schema  # noqa: E402
from scripts.evidence import (  # noqa: E402
    evidence_to_citation_string,
    normalize_evidence_refs,
)
from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import (canonical_json_hash,  # noqa: E402
                             validate_grounded_reply, validate_scp_state)
from scripts.start_agent import (STATE_DIR, TOOLS,  # noqa: E402
                                 RuntimeContext, dispatch_tool, load_mission)
from logos.tool_registry_loader import load_approved_tools  # noqa: E402
from scripts.nexus_manager import NEXUS_MANAGER

try:
    from attestation import (  # noqa: E402  # pylint: disable=wrong-import-position
        AlignmentGateError, compute_attestation_hash,
        load_alignment_attestation, validate_attestation,
        validate_mission_profile)
except ImportError:
    # pylint: disable=wrong-import-position
    from LOGOS_SYSTEM.System_Stack.Protocol_Resources.attestation import (AlignmentGateError,  # noqa: E402
                                     compute_attestation_hash,
                                     load_alignment_attestation,
                                     validate_attestation,
                                     validate_mission_profile)

MAX_PROPOSALS = 3
LOW_IMPACT_TOOLS = {"mission.status", "probe.last"}
HIGH_IMPACT_TOOLS = {"tool_proposal_pipeline", "start_agent", "retrieve.web"}
STATE_PATH = STATE_DIR / "scp_state.json"
AUDIT_DIR = AUDIT_ROOT / "run_ledgers"
APPROVAL_TTL_SECONDS = int(os.getenv("LOGOS_APPROVAL_TTL_SEC", "300"))

load_approved_tools(TOOLS)

_tool_schema = build_tool_schema(TOOLS)
_requires_uip_map = {
    entry["name"]: entry.get("requires_uip", False)
    for entry in _tool_schema.get("tools", [])
}

METRICS = {
    "chats": 0,
    "proposals": 0,
    "approvals": 0,
    "executions": 0,
    "denials": 0,
    "errors": 0,
}

LAST_PROVIDER_STATUS: Dict[str, Any] = {
    "provider": "stub",
    "mode": "stub",
    "reason": "",
    "errors": [],
    "timestamp": "",
}


@dataclass
class SessionData:
    pending: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_ledger: Optional[str] = None
    consumed: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    used_hashes: Set[str] = field(default_factory=set)


SESSIONS: Dict[str, SessionData] = {}
app = FastAPI(title="LOGOS-GPT")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_session_id(request: Request, provided: Optional[str]) -> str:
    header_sid = request.headers.get("x-session-id") if request else None
    cookie_sid = request.cookies.get("session_id") if request else None
    return header_sid or cookie_sid or provided or uuid.uuid4().hex


def _resolve_ws_session_id(websocket: WebSocket, provided: Optional[str]) -> str:
    header_sid = websocket.headers.get("x-session-id") if websocket else None
    cookie_sid = websocket.cookies.get("session_id") if websocket else None
    return header_sid or cookie_sid or provided or uuid.uuid4().hex


def _ensure_session(session_id: str) -> SessionData:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionData()
    return SESSIONS[session_id]


def _bump_metric(name: str, delta: int = 1) -> None:
    if name not in METRICS:
        return
    METRICS[name] = max(0, METRICS.get(name, 0) + delta)


def _sanitize_claims_for_runtime(
    claims: List[Dict[str, Any]], default_text: str
) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    verified_types = {"file", "url", "schema", "hash"}
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("text", "")).strip() or default_text
        truth = str(claim.get("truth", "HEURISTIC"))
        notes = claim.get("notes", "")
        refs_raw = (
            claim.get("evidence_refs")
            if isinstance(claim.get("evidence_refs"), list)
            else []
        )
        refs = normalize_evidence_refs(refs_raw)
        if truth == "PROVED":
            truth = (
                "VERIFIED"
                if any(r.get("type") in verified_types for r in refs)
                else "HEURISTIC"
            )
        if truth == "VERIFIED":
            if not refs or not any(r.get("type") in verified_types for r in refs):
                truth = "HEURISTIC"
        cleaned.append(
            {
                "text": text,
                "truth": truth,
                "evidence_refs": refs,
                "notes": str(notes) if notes is not None else "",
            }
        )
    if not cleaned:
        cleaned.append(
            {
                "text": default_text,
                "truth": "HEURISTIC",
                "evidence_refs": [],
                "notes": "grounding_default",
            }
        )
    try:
        validate_grounded_reply(
            {"reply": default_text, "claims": cleaned, "proposals": []}
        )
    except (ValueError, TypeError):
        pass
    return cleaned


def _citation_markers(claims: List[Dict[str, Any]]) -> List[str]:
    refs: List[Dict[str, Any]] = []
    for claim in claims or []:
        refs.extend(claim.get("evidence_refs", []))
    merged = normalize_evidence_refs(refs)
    citations: List[str] = []
    for ref in merged:
        citations.append(evidence_to_citation_string(ref))
        if len(citations) >= 8:
            break
    return citations


def _claim_truth_counts(claims: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for claim in claims or []:
        truth = str(claim.get("truth", "HEURISTIC"))
        counts[truth] = counts.get(truth, 0) + 1
    return counts


def _perform_local_retrieval(
    message: str,
    ctx: RuntimeContext,
    max_files: int = 10,
    max_snippets: int = 5,
) -> Dict[str, Any]:
    args = {
        "query": message,
        "max_files": max_files,
        "max_snippets": max_snippets,
        "root": "repo",
    }
    meta: Dict[str, Any] = {
        "attempted": False,
        "snippets": [],
        "files_scanned": 0,
        "errors": [],
        "query": message,
    }
    try:
        raw = dispatch_tool("retrieve.local", json.dumps(args), ctx=ctx)
        meta["attempted"] = True
        data = json.loads(raw) if raw else {}
        if isinstance(data, dict):
            meta["snippets"] = data.get("snippets", [])
            meta["files_scanned"] = int(data.get("files_scanned", 0))
            meta["query"] = data.get("query", message)
    except Exception as exc:  # pylint: disable=broad-except
        meta["errors"].append(str(exc))
    return meta


def _proposal_hash(proposal: Dict[str, Any]) -> str:
    try:
        return canonical_json_hash(proposal)
    except Exception:
        return canonical_json_hash({"proposal": str(proposal)})


def _expires_at(created_at: str) -> str:
    try:
        ts = datetime.fromisoformat(created_at)
    except ValueError:
        ts = datetime.now(timezone.utc)
    return (ts + timedelta(seconds=APPROVAL_TTL_SECONDS)).isoformat()


def _approval_expired(item: Dict[str, Any]) -> bool:
    created_at = item.get("created_at")
    if not created_at:
        return False
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - created).total_seconds() > APPROVAL_TTL_SECONDS


def _record_provider_status(notes: Dict[str, Any]) -> None:
    errors = (
        notes.get("errors", []) if isinstance(notes.get("errors", []), list) else []
    )
    LAST_PROVIDER_STATUS.update(
        {
            "provider": notes.get("provider", LAST_PROVIDER_STATUS.get("provider")),
            "mode": notes.get("mode", LAST_PROVIDER_STATUS.get("mode")),
            "reason": notes.get("reason", ""),
            "errors": errors,
            "timestamp": _now(),
        }
    )


def _check_ledger_writable() -> Dict[str, Any]:
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        probe = AUDIT_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"status": "ok"}
    except Exception as exc:  # pylint: disable=broad-except
        return {"status": "error", "reason": str(exc)}


def _attestation(require: bool = True) -> RuntimeContext:
    mission_profile = load_mission()
    validate_mission_profile(mission_profile)
    mission_hash = canonical_json_hash(mission_profile)
    att_hash: Optional[str] = None
    if require:
        att = load_alignment_attestation(STATE_DIR / "alignment_LOGOS-AGENT-OMEGA.json")
        validate_attestation(att, max_age_seconds=21600)
        att_hash = compute_attestation_hash(att)
    else:
        if os.getenv("LOGOS_DEV_BYPASS_OK") == "1":
            att_hash = "DEV_BYPASS"
    if not att_hash:
        raise AlignmentGateError("attestation missing or expired")
    return RuntimeContext(attestation_hash=att_hash, mission_profile_hash=mission_hash)


def _load_state() -> Dict[str, Any]:
    return chat_loop._load_state(STATE_PATH)  # pylint: disable=protected-access


def _persist_state(state: Dict[str, Any]) -> None:
    state["observations"] = state.get("observations", [])[-50:]
    state["last_proposals"] = state.get("last_proposals", [])[-50:]
    state["last_tool_results"] = state.get("last_tool_results", [])[-50:]
    state["truth_events"] = state.get("truth_events", [])[-50:]
    state["updated_at"] = _now()
    state["version"] = state.get("version", 0) + 1
    state["prev_hash"] = state.get("state_hash")
    state["state_hash"] = canonical_json_hash(
        {k: v for k, v in state.items() if k != "state_hash"}
    )
    try:
        validate_scp_state(state)
    except (ValueError, TypeError):
        # ignore schema validation issues when persisting interim state
        pass
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _make_wm_item(
    role: str,
    message: str,
    objective_class: str,
    session_tag: str,
    evidence_refs: Optional[List[Dict[str, Any]]] = None,
    truth: str = "HEURISTIC",
) -> Dict[str, Any]:
    content = {"role": role, "message": message}
    evidence = {"type": "none", "ref": None, "details": "uip_web"}
    if evidence_refs:
        content["evidence_refs"] = evidence_refs[:5]
        evidence = {
            "type": "schema",
            "ref": None,
            "details": f"grounded_refs:{len(evidence_refs)}",
        }
    item_id = stable_item_id("SCP", content, [objective_class, session_tag])
    salience = calculate_initial_salience(truth, "SCP", {"status": "SUCCESS"})
    ts = _now()
    return {
        "id": item_id,
        "created_at": ts,
        "last_accessed_at": ts,
        "objective_tags": list(
            dict.fromkeys([objective_class, "CHAT", session_tag])
        ),
        "truth": truth,
        "evidence": evidence,
        "content": content,
        "salience": salience,
        "decay_rate": 0.15,
        "access_count": 1,
        "source": "SCP",
    }


def _downgrade_proved(annotation: Dict[str, Any]) -> Dict[str, Any]:
    ev = annotation.get("evidence", {}) if isinstance(annotation, dict) else {}
    if isinstance(annotation, dict) and annotation.get("truth") == "PROVED":
        if not isinstance(ev, dict) or ev.get("type") != "coq":
            annotation["truth"] = "VERIFIED"
            annotation["evidence"] = {
                "type": "hash",
                "ref": None,
                "details": "downgraded: missing coq evidence",
            }
    return annotation


def _advisor(provider: str, model: str, timeout_sec: int = 10) -> LLMAdvisor:
    return LLMAdvisor(
        provider=provider,
        model=model,
        tools_schema=_tool_schema,
        truth_rules={},
        timeout_sec=timeout_sec,
    )


def _pending_or_execute(
    session: SessionData,
    proposals: List[Dict[str, Any]],
    ctx: RuntimeContext,
    state: Dict[str, Any],
    objective_class: str,
    read_only: bool,
) -> Dict[str, Any]:
    pending: List[Dict[str, Any]] = []
    executed_results: List[Dict[str, Any]] = []
    executed_events: List[Dict[str, Any]] = []
    ctx.objective_class = objective_class
    ctx.read_only = read_only
    if ctx.tool_validation_events is None:
        ctx.tool_validation_events = []
    if ctx.fallback_proposals is None:
        ctx.fallback_proposals = []
    if ctx.truth_events is None:
        ctx.truth_events = state.setdefault("truth_events", [])
    else:
        state.setdefault("truth_events", ctx.truth_events)
    for prop in proposals:
        prop_copy = dict(prop)
        prop_hash = prop_copy.get("proposal_hash") or _proposal_hash(prop_copy)
        prop_copy["proposal_hash"] = prop_hash
        tool = prop_copy.get("tool", "")
        needs_approval = (
            _requires_uip_map.get(tool, False)
            or tool in HIGH_IMPACT_TOOLS
            or tool not in LOW_IMPACT_TOOLS
        )
        if read_only:
            needs_approval = True
        if needs_approval:
            approval_id = str(uuid.uuid4())
            created_at = _now()
            expires_at = _expires_at(created_at)
            session.pending[approval_id] = {
                "proposal": prop_copy,
                "objective_class": objective_class,
                "proposal_hash": prop_hash,
                "created_at": created_at,
                "expires_at": expires_at,
            }
            pending.append(
                {
                    "approval_id": approval_id,
                    "proposal": prop_copy,
                    "proposal_hash": prop_hash,
                    "expires_at": expires_at,
                    "reason": "approval_required",
                }
            )
            continue
        try:
            output = dispatch_tool(tool, str(prop_copy.get("args", "")), ctx=ctx)
            outcome = "SUCCESS"
            if ctx.last_tool_validation and not ctx.last_tool_validation.get(
                "ok", False
            ):
                outcome = "ERROR"
        except Exception as exc:  # pylint: disable=broad-except
            output = f"[error] {exc}"
            outcome = "ERROR"
            _bump_metric("errors")
        _bump_metric("executions")
        state.setdefault("last_tool_results", []).append(
            {
                "tool": tool,
                "args": prop_copy.get("args", ""),
                "status": outcome,
                "objective": objective_class,
            }
        )
        executed_results.append(
            {
                "tool": tool,
                "args": prop_copy.get("args", ""),
                "output": output,
                "outcome": outcome,
            }
        )
        event_entry = {
            "tool": tool,
            "args_hash": canonical_json_hash({"args": prop_copy.get("args", "")}),
            "outcome": outcome,
            "proposal_hash": prop_hash,
            "truth_tier": (
                (prop_copy.get("truth_annotation") or {}).get("truth", "UNVERIFIED")
                if isinstance(prop_copy, dict)
                else "UNVERIFIED"
            ),
        }
        if ctx.last_tool_validation:
            event_entry["validation"] = {
                "ok": ctx.last_tool_validation.get("ok"),
                "validator": ctx.last_tool_validation.get("validator"),
                "reason": ctx.last_tool_validation.get("reason"),
            }
        executed_events.append(event_entry)
    return {
        "pending": pending,
        "executed_results": executed_results,
        "executed_events": executed_events,
    }


def _session_tag(session_id: str) -> str:
    return f"SESSION:{session_id[:8]}" if session_id else "SESSION:unknown"


def _get_nexus(session_id: str, mode: str, max_compute_ms: int):
    return NEXUS_MANAGER.get(
        session_id=session_id,
        mode=mode,
        max_compute_ms=max_compute_ms,
        audit_logger=lambda evt: None,
    )


def _build_ledger(
    executed_events: List[Dict[str, Any]],
    proposals: List[Dict[str, Any]],
    advisor_meta: Dict[str, Any],
    ctx: RuntimeContext,
    run_start_ts: str,
    read_only: bool,
    failure_notes: Optional[Dict[str, Any]] = None,
    retrieval_meta: Optional[Dict[str, Any]] = None,
    claim_counts: Optional[Dict[str, int]] = None,
    evidence_ref_count: int = 0,
) -> Dict[str, Any]:
    ledger_ctx = {
        "attestation_hash": ctx.attestation_hash or "",
        "coq_index_hash": None,
        "scp_state_hash": "",
        "beliefs_hash": "",
        "metrics_hash": "",
        "run_start_ts": run_start_ts,
        "run_end_ts": _now(),
        "executed_events": executed_events,
        "proposals": proposals,
        "plan_summary": {},
        "policy_notes": failure_notes or {},
        "governance_flags": {"read_only": read_only, "assume_yes": False},
        "advisor": advisor_meta,
        "tool_validation": getattr(ctx, "tool_validation_events", []),
        "fallback_proposals": getattr(ctx, "fallback_proposals", []),
    }
    ledger = build_run_ledger(ledger_ctx)
    if retrieval_meta or claim_counts is not None:
        local_meta = retrieval_meta or {}
        ledger["grounding"] = {
            "retrieval": {
                "local_attempted": bool(local_meta.get("attempted")),
                "local_snippets": len(
                    local_meta.get("snippets", [])
                    if isinstance(local_meta.get("snippets"), list)
                    else []
                ),
                "local_files_scanned": int(local_meta.get("files_scanned", 0) or 0),
                "local_errors": local_meta.get("errors", []),
            },
            "claims": claim_counts or {},
            "evidence_ref_count": int(evidence_ref_count),
        }
    return ledger


def _finalize_turn(
    user_msg: str,
    reply: str,
    result: Dict[str, Any],
    session: SessionData,
    ctx: RuntimeContext,
    objective_class: str,
    advisor_meta: Dict[str, Any],
    claims: List[Dict[str, Any]],
    retrieval_meta: Dict[str, Any],
    read_only: bool,
    safety_hold: Optional[Dict[str, str]] = None,
    session_tag: str = "SESSION:unknown",
) -> Dict[str, Any]:
    state = _load_state()
    init_working_memory(state)
    sanitized_claims = _sanitize_claims_for_runtime(claims, reply or user_msg)
    claim_counts = _claim_truth_counts(sanitized_claims)
    merged_refs = normalize_evidence_refs(
        [ref for claim in sanitized_claims for ref in claim.get("evidence_refs", [])]
    )
    citations = _citation_markers(sanitized_claims)
    reply_text = reply.strip()
    if citations:
        reply_text = (reply_text + " " + " ".join(citations)).strip()

    raw_proposals = (result.get("proposals", []) or [])[:MAX_PROPOSALS]
    proposals: List[Dict[str, Any]] = []
    for prop in raw_proposals:
        prop_copy = dict(prop)
        prop_copy["proposal_hash"] = _proposal_hash(prop_copy)
        proposals.append(prop_copy)
    _bump_metric("proposals", len(proposals))
    rejected = result.get("rejected", []) or []
    advisor_errors = result.get("errors", []) or []
    if advisor_errors and not safety_hold:
        safety_hold = {
            "reason": "provider_error",
            "details": ";".join(str(err) for err in advisor_errors if err),
        }
        _bump_metric("errors")

    for rej in rejected:
        content = {
            "rejected": rej.get("proposal") if isinstance(rej, dict) else rej,
            "reason": (
                rej.get("reason", "rejected")
                if isinstance(rej, dict)
                else "rejected"
            ),
        }
        state.setdefault("truth_events", []).append(
            {
                "ts": _now(),
                "source": "UIP",
                "content": content,
                "truth_annotation": {
                    "truth": "CONTRADICTED",
                    "evidence": {
                        "type": "none",
                        "ref": None,
                        "details": "advisor rejected",
                    },
                },
            }
        )

    reply_truth = "HEURISTIC"
    if claim_counts.get("VERIFIED"):
        reply_truth = "VERIFIED"
    reply_annotation = {
        "truth": reply_truth,
        "evidence": {
            "type": "schema",
            "ref": citations[0] if citations else None,
            "details": "grounded_reply" if citations else "advisor_reply",
        },
    }
    reply_annotation = _downgrade_proved(reply_annotation)
    try:
        reply_annotation = enforce_truth_annotation(reply_annotation, None)
    except (ValueError, TypeError):
        pass

    add_memory_item(
        state, _make_wm_item("user", user_msg, objective_class, session_tag)
    )
    add_memory_item(
        state,
        _make_wm_item(
            "assistant",
            reply_text,
            objective_class,
            session_tag,
            evidence_refs=merged_refs[:5],
            truth=reply_truth,
        ),
    )

    for claim in sanitized_claims:
        refs = normalize_evidence_refs(claim.get("evidence_refs", []))
        citation = evidence_to_citation_string(refs[0]) if refs else None
        evidence = {
            "type": "schema",
            "ref": citation,
            "details": "grounded_claim" if citation else "claim_no_evidence",
        }
        state.setdefault("truth_events", []).append(
            {
                "ts": _now(),
                "source": "SCP",
                "content": {
                    "claim": claim.get("text", ""),
                    "objective_class": objective_class,
                },
                "truth_annotation": {
                    "truth": claim.get("truth", "HEURISTIC"),
                    "evidence": evidence,
                },
            }
        )

    run_start_ts = _now()
    gating = {"pending": [], "executed_results": [], "executed_events": []}
    failure_notes: Optional[Dict[str, Any]] = None
    if safety_hold:
        _bump_metric("denials")
        failure_notes = {"safety_hold": safety_hold}
    else:
        gating = _pending_or_execute(
            session, proposals, ctx, state, objective_class, read_only
        )

    for prop in proposals:
        prop_copy = dict(prop)
        prop_copy["truth_annotation"] = prop_copy.get("truth_annotation") or {
            "truth": "HEURISTIC",
            "evidence": {
                "type": "none",
                "ref": None,
                "details": "advisor_default",
            },
        }
        state.setdefault("last_proposals", []).append(prop_copy)

    state.setdefault("truth_events", []).append(
        {
            "ts": _now(),
            "source": "UIP",
            "content": {"reply": reply_text, "objective_class": objective_class},
            "truth_annotation": reply_annotation,
        }
    )

    _persist_state(state)

    ledger_ctx = _build_ledger(
        gating["executed_events"],
        proposals,
        advisor_meta,
        ctx,
        run_start_ts,
        read_only,
        failure_notes,
        retrieval_meta,
        claim_counts,
        len(merged_refs),
    )
    ledger_ctx["scp_state_hash"] = state.get("state_hash", "")
    ledger_ctx["beliefs_hash"] = state.get("beliefs", {}).get("state_hash", "")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = AUDIT_DIR / f"logos_gpt_web_{uuid.uuid4().hex}.json"
    ledger_path.write_text(json.dumps(ledger_ctx, indent=2), encoding="utf-8")
    session.last_ledger = str(ledger_path)

    response = {
        "reply": reply_text,
        "claims": sanitized_claims,
        "proposals": proposals,
        "pending_approvals": gating["pending"],
        "executed_results": gating["executed_results"],
        "ledger_path": str(ledger_path),
        "retrieval": retrieval_meta,
    }
    if safety_hold:
        response["safety_hold"] = safety_hold
    return response


@app.get("/health")
async def health() -> JSONResponse:
    try:
        ctx = _attestation(require=False)
        status = "ok" if ctx.attestation_hash else "bypass"
    except Exception as exc:  # pylint: disable=broad-except
        return JSONResponse({"status": "error", "reason": str(exc)}, status_code=503)


@app.post("/api/chat")
async def chat(body: Dict[str, Any], request: Request) -> JSONResponse:
    message = str(body.get("message", "")).strip()
    if not message:
        raise HTTPException(status_code=400, detail="message required")
    session_id = _resolve_session_id(request, body.get("session_id"))
    session = _ensure_session(session_id)
    provider = body.get("provider") or os.getenv("LOGOS_LLM_PROVIDER", "stub")
    model = body.get("model") or os.getenv("LOGOS_LLM_MODEL", "gpt-4.1-mini")
    logos_mode = os.getenv("LOGOS_AGI_MODE", "stub")
    max_compute_ms = int(os.getenv("LOGOS_AGI_MAX_COMPUTE_MS", "100") or 100)
    read_only = bool(body.get("read_only", False))
    objective_class = body.get("objective_class", "CHAT")
    session_tag = _session_tag(session_id)

    _bump_metric("chats")
    safety_hold: Optional[Dict[str, str]] = None

    require_attestation = (
        body.get("require_attestation", False)
        if body.get("require_attestation") is not None
        else False
    )
    try:
        ctx = _attestation(require=require_attestation)
    except AlignmentGateError as exc:
        dev_bypass = os.getenv("LOGOS_DEV_BYPASS_OK") == "1"
        if not require_attestation and dev_bypass:
            ctx = RuntimeContext(attestation_hash="DEV_BYPASS", mission_profile_hash=None)
        else:
            _bump_metric("errors")
            _bump_metric("denials")
            return JSONResponse({"error": str(exc)}, status_code=403)

    state = _load_state()
    nexus = _get_nexus(session_id, logos_mode, max_compute_ms)
    nexus.prior_state = state
    if state.get("plans", {}).get("history_scored"):
        try:
            nexus.refresh_plan_history(state["plans"]["history_scored"])
        except Exception:
            pass
    init_working_memory(state)
    retrieval_meta = _perform_local_retrieval(
        message, ctx, max_files=10, max_snippets=5
    )
    advisor = _advisor(provider, model)
    context = {
        "conversation_recall": recall(state, objective_class, k=5),
        "belief_summary": state.get("beliefs", {}),
        "tool_summary": {
            "tools": sorted(TOOLS.keys()),
            "high_impact": sorted(HIGH_IMPACT_TOOLS),
        },
        "retrieval": {"local": retrieval_meta.get("snippets", [])},
    }
    try:
        result = advisor.propose(message, context)
    except Exception as exc:  # pylint: disable=broad-except
        _bump_metric("errors")
        safety_hold = safety_hold or {
            "reason": "advisor_exception",
            "details": str(exc),
        }
        result = {
            "proposals": [],
            "rejected": [],
            "claims": [],
            "reply": "Advisory-only: advisor unavailable.",
            "errors": [str(exc)],
            "notes": {
                "provider": provider,
                "model": model,
                "mode": "stub",
                "reason": str(exc),
            },
        }
    reply = result.get("reply") or "Acknowledged."
    notes = result.get("notes") or {}
    if notes:
        _record_provider_status(notes)
    if notes.get("mode") == "stub" and provider != "stub" and not safety_hold:
        safety_hold = {
            "reason": "provider_unavailable",
            "details": notes.get("reason", "provider unavailable"),
        }
        _bump_metric("denials")

    advisor_meta = {
        "enabled": True,
        "provider": provider,
        "model": model,
        "stream": False,
        "notes": notes,
        "errors": result.get("errors", []),
    }
    finalized = _finalize_turn(
        message,
        reply,
        result,
        session,
        ctx,
        objective_class,
        advisor_meta,
        result.get("claims", []),
        retrieval_meta,
        read_only,
        safety_hold,
        session_tag,
    )
    try:
        nexus.prior_state = _load_state()
    except Exception:
        pass
    finalized["session_id"] = session_id
    response = JSONResponse(finalized)
    response.set_cookie("session_id", session_id)
    return response


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        initial_text = await websocket.receive_text()
        try:
            payload = json.loads(initial_text)
        except json.JSONDecodeError:
            await websocket.close(code=1003)
            return
        message = str(payload.get("message", "")).strip()
        if not message:
            await websocket.close(code=1003)
            return
        session_id = _resolve_ws_session_id(websocket, payload.get("session_id"))
        session = _ensure_session(session_id)
        provider = payload.get("provider") or os.getenv("LOGOS_LLM_PROVIDER", "stub")
        model = payload.get("model") or os.getenv("LOGOS_LLM_MODEL", "gpt-4.1-mini")
        logos_mode = os.getenv("LOGOS_AGI_MODE", "stub")
        max_compute_ms = int(os.getenv("LOGOS_AGI_MAX_COMPUTE_MS", "100") or 100)
        read_only = bool(payload.get("read_only", False))
        objective_class = payload.get("objective_class", "CHAT")
        session_tag = _session_tag(session_id)
        require_attestation = (
            payload.get("require_attestation", False)
            if payload.get("require_attestation") is not None
            else False
        )
        try:
            ctx = _attestation(require=require_attestation)
        except AlignmentGateError as exc:
            dev_bypass = os.getenv("LOGOS_DEV_BYPASS_OK") == "1"
            if not require_attestation and dev_bypass:
                ctx = RuntimeContext(attestation_hash="DEV_BYPASS", mission_profile_hash=None)
            else:
                await websocket.send_json({"error": str(exc)})
                await websocket.close()
                return

        state = _load_state()
        nexus = _get_nexus(session_id, logos_mode, max_compute_ms)
        nexus.prior_state = state
        if state.get("plans", {}).get("history_scored"):
            try:
                nexus.refresh_plan_history(state["plans"]["history_scored"])
            except Exception:
                pass
        init_working_memory(state)
        retrieval_meta = _perform_local_retrieval(
            message, ctx, max_files=10, max_snippets=5
        )
        advisor = _advisor(provider, model)
        context = {
            "conversation_recall": recall(state, objective_class, k=5),
            "belief_summary": state.get("beliefs", {}),
            "tool_summary": {
                "tools": sorted(TOOLS.keys()),
                "high_impact": sorted(HIGH_IMPACT_TOOLS),
            },
            "retrieval": {"local": retrieval_meta.get("snippets", [])},
        }

        stream = advisor.propose_stream(message, context)
        reply_chunks: List[str] = []
        result: Dict[str, Any] = {}
        while True:
            try:
                chunk = next(stream)
            except StopIteration as stop:
                result = stop.value or {}
                break
            if chunk:
                reply_chunks.append(str(chunk))
                await websocket.send_text(str(chunk))
        reply = result.get("reply") or "".join(reply_chunks) or "Acknowledged."
        advisor_meta = {
            "enabled": True,
            "provider": provider,
            "model": model,
            "stream": True,
        }
        finalized = _finalize_turn(
            message,
            reply,
            result,
            session,
            ctx,
            objective_class,
            advisor_meta,
            result.get("claims", []),
            retrieval_meta,
            read_only,
            None,
            session_tag,
        )
        try:
            nexus.prior_state = _load_state()
        except Exception:
            pass
        finalized["session_id"] = session_id
        await websocket.send_json({"type": "final", **finalized})
    except WebSocketDisconnect:
        return


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    session = _ensure_session(session_id)
    state = _load_state()
    summary = {
        "session_id": session_id,
        "last_ledger": session.last_ledger,
        "observations": state.get("observations", [])[-5:],
        "last_proposals": state.get("last_proposals", [])[-5:],
    }
    return JSONResponse(summary)


if os.getenv("TEST_MODE") == "1":

    @app.get("/api/debug/nexus")
    async def debug_nexus() -> JSONResponse:
        snapshot = NEXUS_MANAGER.debug_snapshot()
        return JSONResponse(snapshot)


@app.post("/api/approve")
async def approve(body: Dict[str, Any]) -> JSONResponse:
    session_id = body.get("session_id")
    approval_id = body.get("approval_id")
    decision = str(body.get("decision", "")).lower()
    if not session_id or not approval_id:
        raise HTTPException(
            status_code=400, detail="session_id and approval_id required"
        )
    provided_hash = str(body.get("proposal_hash", ""))
    if not provided_hash:
        raise HTTPException(status_code=400, detail="proposal_hash required")
    session = _ensure_session(session_id)
    _bump_metric("approvals")

    if approval_id in session.consumed:
        consumed = session.consumed.get(approval_id, {})
        _bump_metric("denials")
        return JSONResponse(
            {
                "error": "approval already processed",
                "reason": consumed.get("reason", "replay"),
                "proposal_hash": consumed.get("proposal_hash"),
            },
            status_code=409,
        )

    item = session.pending.get(approval_id)
    if not item:
        _bump_metric("denials")
        raise HTTPException(status_code=404, detail="pending approval not found")

    stored_hash = item.get("proposal_hash") or _proposal_hash(item.get("proposal", {}))
    if provided_hash != stored_hash:
        session.pending.pop(approval_id, None)
        session.consumed[approval_id] = {
            "reason": "hash_mismatch",
            "proposal_hash": stored_hash,
        }
        _bump_metric("denials")
        return JSONResponse(
            {"error": "approval hash mismatch", "reason": "hash_mismatch"},
            status_code=409,
        )

    if stored_hash in session.used_hashes:
        session.pending.pop(approval_id, None)
        session.consumed[approval_id] = {
            "reason": "replay",
            "proposal_hash": stored_hash,
        }
        _bump_metric("denials")
        return JSONResponse(
            {"error": "approval already applied", "reason": "replay"},
            status_code=409,
        )

    if _approval_expired(item):
        session.pending.pop(approval_id, None)
        session.consumed[approval_id] = {
            "reason": "expired",
            "proposal_hash": stored_hash,
        }
        _bump_metric("denials")
        return JSONResponse(
            {"error": "approval expired", "reason": "expired"},
            status_code=410,
        )

    item = session.pending.pop(approval_id, None) or item
    proposal = item.get("proposal", {})
    objective_class = item.get("objective_class", "CHAT")
    tool = proposal.get("tool", "")
    read_only = bool(body.get("read_only", False))

    try:
        ctx = _attestation(
            require=body.get("require_attestation", False)
            if body.get("require_attestation") is not None
            else False
        )
    except AlignmentGateError as exc:
        _bump_metric("errors")
        _bump_metric("denials")
        return JSONResponse({"error": str(exc)}, status_code=403)

    state = _load_state()
    init_working_memory(state)

    outcome = "REJECTED" if decision != "approve" else "APPROVED"
    output = ""
    executed_events: List[Dict[str, Any]] = []
    executed_results: List[Dict[str, Any]] = []

    if decision == "approve" and not read_only:
        try:
            output = dispatch_tool(tool, str(proposal.get("args", "")), ctx=ctx)
            outcome = "SUCCESS"
            _bump_metric("executions")
        except Exception as exc:  # pylint: disable=broad-except
            output = f"[error] {exc}"
            outcome = "ERROR"
            _bump_metric("errors")
        executed_results.append(
            {
                "tool": tool,
                "args": proposal.get("args", ""),
                "output": output,
                "outcome": outcome,
            }
        )
        executed_events.append(
            {
                "tool": tool,
                "args_hash": canonical_json_hash(
                    {"args": proposal.get("args", "")}
                ),
                "outcome": outcome,
                "proposal_hash": stored_hash,
                "truth_tier": (
                    (proposal.get("truth_annotation") or {}).get(
                        "truth", "UNVERIFIED"
                    )
                    if isinstance(proposal, dict)
                    else "UNVERIFIED"
                ),
            }
        )
        state.setdefault("last_tool_results", []).append(
            {
                "tool": tool,
                "args": proposal.get("args", ""),
                "status": outcome,
                "objective": objective_class,
            }
        )
        session.used_hashes.add(stored_hash)
    else:
        outcome = "APPROVED_NO_EXEC" if decision == "approve" else "REJECTED"
        _bump_metric("denials")
        session.used_hashes.add(stored_hash)

    state.setdefault("truth_events", []).append({
        "ts": _now(),
        "source": "UIP",
        "content": {"approval_id": approval_id, "tool": tool, "decision": decision},
        "truth_annotation": {
            "truth": "UNVERIFIED",
            "evidence": {"type": "none", "ref": None, "details": "uip_decision"},
        },
    })

    session.consumed[approval_id] = {
        "reason": outcome.lower(),
        "proposal_hash": stored_hash,
    }

    ledger_ctx = {
        "attestation_hash": ctx.attestation_hash or "",
        "coq_index_hash": None,
        "scp_state_hash": state.get("state_hash", ""),
        "beliefs_hash": state.get("beliefs", {}).get("state_hash", ""),
        "metrics_hash": "",
        "run_start_ts": _now(),
        "run_end_ts": _now(),
        "executed_events": executed_events,
        "proposals": [],
        "plan_summary": {},
        "policy_notes": {
            "uip_decision": decision,
            "tool": tool,
            "proposal_hash": stored_hash,
            "outcome": outcome,
            "reason": "read_only" if read_only else decision,
        },
        "governance_flags": {"read_only": read_only, "assume_yes": False},
        "advisor": {
            "enabled": False,
            "provider": "stub",
            "model": "stub",
            "stream": False,
        },
    }
    ledger_error = None
    try:
        ledger = build_run_ledger(ledger_ctx)
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        ledger_path = AUDIT_DIR / f"logos_gpt_web_approval_{approval_id}.json"
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        session.last_ledger = str(ledger_path)
    except Exception as exc:  # pylint: disable=broad-except
        ledger_error = str(exc)
        _bump_metric("errors")
        ledger_path = AUDIT_DIR / f"logos_gpt_web_approval_{approval_id}_FAILED.json"

    _persist_state(state)

    response_body = {
        "decision": decision,
        "outcome": outcome,
        "output": output,
        "executed_results": executed_results,
        "ledger_path": str(ledger_path),
        "proposal_hash": stored_hash,
    }
    if ledger_error:
        response_body["ledger_error"] = ledger_error
    return JSONResponse(response_body)


if __name__ == "__main__":
    import uvicorn

    if os.environ.get("LOGOS_OPERATOR_OK", "").strip() != "1":
        print(
            "ERROR: operator ack required. Set LOGOS_OPERATOR_OK=1 to run this server.",
            file=sys.stderr,
        )
        sys.exit(2)

    parser = argparse.ArgumentParser(
        description="FastAPI surface for LOGOS-GPT (advisor-only, UIP gated)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Acknowledge this script will start a local server (uvicorn/FastAPI).",
    )
    cli_args = parser.parse_args()

    if not cli_args.serve:
        print("ERROR: --serve is required to start the server.", file=sys.stderr)
        sys.exit(2)

    uvicorn.run(
        "scripts.llm_interface_suite.logos_gpt_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
