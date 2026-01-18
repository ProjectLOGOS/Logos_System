# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Shared schema validators and canonical hashing helpers for LOGOS scripts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterable, Mapping, MutableSequence, Sequence

# Truth tiers observed across the runtime stack. The validator remains permissive
# but this set is helpful for normalizing casing in error messages.
_ALLOWED_TRUTH_VALUES = {
    "PROVED",
    "PROVEN",
    "VERIFIED",
    "HEURISTIC",
    "ANALOGICAL",
    "INFERRED",
    "UNVERIFIED",
    "CONTRADICTED",
    "REFUTED",
    "ALIGNED",
    "AUDITED",
}


class SchemaError(ValueError):
    """Raised when a document fails validation."""


def canonical_json_hash(payload: Any) -> str:
    """Return the SHA-256 of the canonical JSON form of ``payload``.

    The function mirrors the historical helper that lived under
    ``System_Stack.is_this_needed_question`` so downstream modules keep their
    deterministic hashing guarantees.
    """

    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise SchemaError(f"Object is not JSON serializable: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _is_iso8601(value: str) -> bool:
    try:
        # ``fromisoformat`` cannot parse ``Z`` directly, so normalize first.
        sanitized = value.replace("Z", "+00:00")
        datetime.fromisoformat(sanitized)
        return True
    except Exception:  # pragma: no cover - best-effort validator
        return False


def _ensure_hex(value: str, length: int | None = None) -> None:
    _require(isinstance(value, str) and value, "expected non-empty hex string")
    if length is not None:
        _require(len(value) == length, f"expected hex string of length {length}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SchemaError("value must be hexadecimal") from exc


def validate_truth_annotation(annotation: Mapping[str, Any]) -> None:
    """Validate a truth annotation block used throughout the planner stack."""

    _require(isinstance(annotation, Mapping), "truth_annotation must be a mapping")
    truth = annotation.get("truth")
    _require(isinstance(truth, str) and truth.strip(), "truth_annotation.truth must be a non-empty string")
    truth_upper = truth.upper()
    if truth_upper not in _ALLOWED_TRUTH_VALUES:
        # Remain permissive but ensure consistent casing for downstream logic.
        annotation["truth"] = truth_upper  # type: ignore[index]

    confidence = annotation.get("confidence")
    if confidence is not None:
        _require(isinstance(confidence, (int, float)), "truth_annotation.confidence must be numeric")
        _require(0.0 <= float(confidence) <= 1.0, "truth_annotation.confidence must be between 0 and 1")

    evidence_refs = annotation.get("evidence_refs")
    if evidence_refs is not None:
        _require(
            isinstance(evidence_refs, Sequence),
            "truth_annotation.evidence_refs must be a sequence",
        )

    notes = annotation.get("notes")
    if notes is not None:
        _require(isinstance(notes, (str, Mapping)), "truth_annotation.notes must be str or mapping")


def validate_goal_candidate(candidate: Mapping[str, Any]) -> None:
    """Validate the deterministic goal candidate envelope."""

    _require(isinstance(candidate, Mapping), "goal candidate must be a mapping")
    required_str = [
        "goal_id",
        "created_at",
        "objective_class",
        "statement",
        "rationale",
        "risk_tier",
        "required_approval",
        "status",
    ]
    for key in required_str:
        _require(isinstance(candidate.get(key), str) and candidate[key], f"missing goal candidate field: {key}")

    _require(
        isinstance(candidate.get("confidence"), (int, float)),
        "goal candidate confidence must be numeric",
    )
    confidence = float(candidate["confidence"])
    _require(0.0 <= confidence <= 1.0, "goal candidate confidence must be between 0 and 1")

    for list_field in ("supporting_refs", "contradicting_refs"):
        refs = candidate.get(list_field)
        _require(isinstance(refs, Sequence), f"goal candidate {list_field} must be a sequence")

    truth_annotation = candidate.get("truth_annotation")
    _require(isinstance(truth_annotation, Mapping), "goal candidate missing truth_annotation")
    validate_truth_annotation(truth_annotation)

    timestamp = candidate.get("created_at")
    _require(isinstance(timestamp, str) and _is_iso8601(timestamp), "goal candidate timestamp must be ISO-8601")


def validate_tool_proposal(proposal: Mapping[str, Any]) -> None:
    """Validate tool proposal metadata before ingestion."""

    _require(isinstance(proposal, Mapping), "tool proposal must be a mapping")
    _require(isinstance(proposal.get("schema_version"), int), "proposal.schema_version must be int")
    str_fields = [
        "proposal_id",
        "created_at",
        "created_by",
        "objective_class",
        "tool_name",
        "description",
    ]
    for field in str_fields:
        _require(isinstance(proposal.get(field), str) and proposal[field], f"proposal missing {field}")

    _require(isinstance(proposal.get("inputs_schema"), Mapping), "proposal.inputs_schema must be a mapping")
    _require(isinstance(proposal.get("outputs_schema"), Mapping), "proposal.outputs_schema must be a mapping")
    _require(isinstance(proposal.get("code"), str), "proposal.code must be a string")
    _require(isinstance(proposal.get("provenance"), Mapping), "proposal.provenance must be a mapping")

    safety = proposal.get("safety")
    _require(isinstance(safety, Mapping), "proposal.safety must be a mapping")
    _require(isinstance(safety.get("no_network"), bool), "safety.no_network must be bool")
    _require(isinstance(safety.get("sandbox_only"), bool), "safety.sandbox_only must be bool")
    max_runtime = safety.get("max_runtime_ms")
    _require(isinstance(max_runtime, (int, float)), "safety.max_runtime_ms must be numeric")
    _require(int(max_runtime) > 0, "safety.max_runtime_ms must be positive")

    truth_annotation = proposal.get("truth_annotation")
    if isinstance(truth_annotation, Mapping):
        validate_truth_annotation(truth_annotation)


def validate_tool_validation_report(report: Mapping[str, Any]) -> None:
    """Validate the structured report emitted by the validation stage."""

    _require(isinstance(report, Mapping), "validation report must be a mapping")
    _require(isinstance(report.get("proposal_id"), str), "validation report missing proposal_id")
    _require(isinstance(report.get("exec_ok"), bool), "validation report exec_ok must be bool")
    runtime = report.get("runtime_ms")
    _require(isinstance(runtime, (int, float)), "validation report runtime_ms must be numeric")
    _require(int(runtime) >= 0, "validation report runtime_ms must be non-negative")
    _require(isinstance(report.get("output"), str), "validation report output must be string")
    errors = report.get("errors")
    _require(isinstance(errors, Sequence), "validation report errors must be sequence")
    for entry in errors:
        _require(isinstance(entry, str), "validation report errors must contain strings")


def validate_plan_history_scored(container: Mapping[str, Any]) -> None:
    """Validate the bounded plan history ledger."""

    _require(isinstance(container, Mapping), "history_scored must be a mapping")
    _require(isinstance(container.get("schema_version"), int), "history_scored.schema_version must be int")
    _require(isinstance(container.get("updated_at"), str), "history_scored.updated_at must be set")
    state_hash = container.get("state_hash")
    if isinstance(state_hash, str) and state_hash:
        _ensure_hex(state_hash, 64)

    entries = container.get("entries_by_signature")
    _require(isinstance(entries, Mapping), "history_scored.entries_by_signature must be mapping")
    for signature, history in entries.items():
        _require(isinstance(signature, str) and signature, "plan signature must be non-empty string")
        _require(isinstance(history, Sequence), "plan history must be a list")
        for entry in history:
            _require(isinstance(entry, Mapping), "plan history entry must be mapping")
            _require(isinstance(entry.get("timestamp"), str), "plan history entry missing timestamp")
            _require(isinstance(entry.get("score"), (int, float)), "plan history entry score must be numeric")
            _require(isinstance(entry.get("report_hash"), str), "plan history entry missing report_hash")
            ledger_hash = entry.get("ledger_hash")
            if ledger_hash is not None:
                _require(isinstance(ledger_hash, str), "plan history ledger_hash must be str when present")
            prev_hash = entry.get("prev_hash")
            if prev_hash is not None:
                _require(isinstance(prev_hash, str), "plan history prev_hash must be str when present")
            entry_hash = entry.get("entry_hash")
            if isinstance(entry_hash, str) and entry_hash:
                _ensure_hex(entry_hash, 64)


def validate_scp_state(state: Mapping[str, Any]) -> None:
    """Validate the persisted SCP state document loaded by orchestration scripts."""

    _require(isinstance(state, Mapping), "scp_state must be a mapping")
    _require(isinstance(state.get("schema_version"), int), "scp_state schema_version must be int")
    state_hash = state.get("state_hash")
    if isinstance(state_hash, str) and state_hash:
        _ensure_hex(state_hash, 64)
    _require(isinstance(state.get("updated_at"), str), "scp_state updated_at must be set")
    _require(isinstance(state.get("version"), (int, float)), "scp_state version must be numeric")

    plans = state.get("plans")
    _require(isinstance(plans, Mapping), "scp_state.plans must be mapping")
    history = plans.get("history_scored")
    if isinstance(history, Mapping):
        validate_plan_history_scored(history)

    beliefs = state.get("beliefs")
    _require(isinstance(beliefs, Mapping), "scp_state.beliefs must be mapping")

    working_memory = state.get("working_memory")
    _require(isinstance(working_memory, Mapping), "scp_state.working_memory must be mapping")


def validate_grounded_reply(reply: Mapping[str, Any]) -> None:
    """Validate grounded replies returned by advisor or server endpoints."""

    _require(isinstance(reply, Mapping), "grounded reply must be a mapping")
    _require(isinstance(reply.get("reply"), str), "grounded reply missing text")
    claims = reply.get("claims")
    _require(isinstance(claims, Sequence), "grounded reply claims must be a sequence")
    for claim in claims:
        _require(isinstance(claim, Mapping), "claim must be a mapping")
        _require(isinstance(claim.get("text"), str), "claim missing text field")
        truth = claim.get("truth", "HEURISTIC")
        _require(isinstance(truth, str), "claim truth must be string")
        refs = claim.get("evidence_refs", [])
        _require(isinstance(refs, Sequence), "claim evidence_refs must be a sequence")
    proposals = reply.get("proposals")
    if proposals is not None:
        _require(isinstance(proposals, Sequence), "grounded reply proposals must be a sequence when present")


__all__ = [
    "SchemaError",
    "canonical_json_hash",
    "validate_goal_candidate",
    "validate_grounded_reply",
    "validate_plan_history_scored",
    "validate_scp_state",
    "validate_tool_proposal",
    "validate_tool_validation_report",
    "validate_truth_annotation",
]
