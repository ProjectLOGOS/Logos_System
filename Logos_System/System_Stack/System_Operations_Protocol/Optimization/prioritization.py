# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Deterministic commitment prioritization for LOGOS AGI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...Logos_Protocol.Unified_Working_Memory.World_Modeling.commitment_ledger import (
    JUSTIFICATION_MAX,
    SUMMARY_MAX,
    mark_commitment_status,
    record_event,
    set_active_commitment,
    upsert_commitment,
)

SELECTABLE_STATUSES = {"active", "deferred", "blocked"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _inherit_constraints(identity: Dict[str, Any]) -> Dict[str, bool]:
    mission = identity.get("mission", {})
    allow_enhancements = bool(mission.get("allow_enhancements", False))
    safe_only = bool(mission.get("safe_interfaces_only", False))
    read_only = bool(mission.get("safe_interfaces_only", False))
    return {
        "allow_enhancements": allow_enhancements,
        "safe_only": safe_only,
        "read_only": read_only,
    }


def _dedupe_key(commitment: Dict[str, Any]) -> Tuple[str, str]:
    title = str(commitment.get("title", "")).strip().lower()
    ctype = str(commitment.get("type", "other")).strip().lower()
    return (title, ctype)


def _build_uwm_ref(identity: Dict[str, Any], uwm_snapshot: Dict[str, Any]) -> Optional[str]:
    world_model = identity.get("world_model", {})
    snapshot_path = world_model.get("snapshot_path")
    snapshot_hash = world_model.get("snapshot_hash")
    if not snapshot_hash:
        snapshot_hash = (
            uwm_snapshot.get("integrity", {}).get("snapshot_hash")
            if isinstance(uwm_snapshot, dict)
            else None
        )
    if snapshot_path:
        ref = f"uwm:{snapshot_path}"
        if snapshot_hash:
            ref = f"{ref}#{snapshot_hash}"
        return ref
    return None


def _clamp(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

def _evidence_refs(base_refs: Iterable[str]) -> List[str]:
    refs: List[str] = []
    for candidate in base_refs:
        if not candidate:
            continue
        if candidate not in refs:
            refs.append(candidate)
    return refs


def _default_success_criteria(template_id: str) -> List[str]:
    if template_id == "T1":
        return [
            "Identity validation executes without warnings",
            "World model binds with expected hash",
            "Ledger integrity verified",
        ]
    if template_id == "T2":
        return [
            "All referenced artifacts reachable",
            "Catalog tail hash reconciled with identity",
            "Orphan artifact count reduced to zero",
        ]
    if template_id == "T3":
        return [
            "Governance README updated with latest changes",
            "Presentation brief includes commitment policy",
            "Stakeholder notes synchronized",
        ]
    if template_id == "T4":
        return [
            "Proof gate health report generated",
            "Theory hash cross-checked with baseline",
            "All proof gate anomalies resolved",
        ]
    if template_id == "T5":
        return [
            "Planner digests reviewed for recurring blockers",
            "Mitigation plan documented",
            "Next cycle backlog updated",
        ]
    if template_id == "T6":
        return [
            "Tool registry generated and hashed",
            "Tool chain profiles generated and hashed",
            "UWM snapshot updated with tooling refs",
        ]
    if template_id == "T7":
        return [
            "Tool invention gaps analyzed and scored",
            "New tools generated and validated",
            "Tool catalog updated with invention artifacts",
        ]
    return ["Commitment executed successfully"]


def propose_candidate_commitments(
    identity: Dict[str, Any],
    uwm_snapshot: Dict[str, Any],
    ledger: Dict[str, Any],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mission_label = str(identity.get("mission", {}).get("mission_label", "") or "")
    validation_warnings = list(context.get("validation_warnings", []))
    missing_refs = list(context.get("missing_artifact_refs", []))
    catalog_tail_hash = context.get("catalog_tail_hash")
    catalog_tail_hash_current = context.get("catalog_tail_hash_current")
    catalog_tail_changed = bool(context.get("catalog_tail_hash_mismatch"))
    last_planner_digest = context.get("last_planner_digest_archive")
    proof_gate_changed = bool(context.get("proof_gate_changed"))
    proof_gate_anomalies = list(context.get("proof_gate_anomalies", []))

    uwm_ref = _build_uwm_ref(identity, uwm_snapshot)
    evidence: List[str] = []
    if uwm_ref:
        evidence.append(uwm_ref)
    if last_planner_digest:
        evidence.append(str(last_planner_digest))

    constraints = _inherit_constraints(identity)
    cycle_utc = context.get("cycle_utc") or _now_iso()
    proposals: List[Dict[str, Any]] = []

    repo_root = Path(context.get("repo_root", "."))
    tool_optimizer_dir = repo_root / "state" / "tool_optimizer"
    tool_optimizer_report_path = tool_optimizer_dir / "tool_optimizer_report.json"
    tool_report_data: Optional[Dict[str, Any]] = None
    tool_optimizer_missing = not tool_optimizer_report_path.exists()
    tool_optimizer_registry_zero = False
    tool_optimizer_stale = False
    tool_optimizer_report_timestamp: Optional[str] = None
    if not tool_optimizer_missing:
        try:
            tool_report_data = json.loads(tool_optimizer_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            tool_optimizer_missing = True
        else:
            registry_count_value = int(tool_report_data.get("registry_count", 0) or 0)
            tool_optimizer_registry_zero = registry_count_value == 0
            tool_optimizer_report_timestamp = str(tool_report_data.get("timestamp_utc") or "") or None
            timestamp_raw = tool_report_data.get("timestamp_utc")
            if timestamp_raw:
                try:
                    parsed = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00"))
                except ValueError:
                    tool_optimizer_stale = True
                else:
                    parsed_utc = parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                    age = datetime.now(timezone.utc) - parsed_utc
                    tool_optimizer_stale = age > timedelta(hours=24)
            else:
                tool_optimizer_stale = True


    def _make_candidate(template_id: str, title: str, ctype: str, priority: int, justification: str) -> Dict[str, Any]:
        payload = {
            "commitment_id": None,
            "template_id": template_id,
            "title": title,
            "type": ctype,
            "status": "deferred",
            "priority": priority,
            "created_utc": cycle_utc,
            "updated_utc": cycle_utc,
            "justification": _clamp(justification, JUSTIFICATION_MAX),
            "evidence_refs": _evidence_refs(evidence),
            "constraints": dict(constraints),
            "success_criteria": _default_success_criteria(template_id),
            "blockers": [],
            "provenance": {
                "arbiter": "commitment_prioritization",
                "template": template_id,
                "cycle_utc": cycle_utc,
            },
        }
        return payload

    if validation_warnings and len(proposals) < 5:
        justification = "Resolve outstanding validation warnings from identity/world model checks."
        proposals.append(
            _make_candidate(
                "T1",
                "Resolve identity/world-model validation warnings",
                "repair",
                1,
                justification,
            )
        )

    catalog_issue = catalog_tail_changed or bool(missing_refs)
    if catalog_issue and len(proposals) < 5:
        justification = "Investigate catalog or artifact discrepancies to restore alignment evidence."
        proposals.append(
            _make_candidate(
                "T2",
                "Investigate missing or orphan artifacts",
                "repair",
                2,
                justification,
            )
        )
        proposals[-1]["provenance"].update(
            {
                "catalog_tail_hash": str(catalog_tail_hash),
                "catalog_tail_hash_current": str(catalog_tail_hash_current),
                "missing_refs": missing_refs[:5],
            }
        )

    if ("demo" in mission_label.lower() or "presentation" in mission_label.lower()) and len(proposals) < 5:
        justification = "Document recent governance updates for demo/presentation readiness."
        proposals.append(
            _make_candidate(
                "T3",
                "Update documentation for latest governance changes",
                "doc",
                4,
                justification,
            )
        )

    if (proof_gate_changed or proof_gate_anomalies) and len(proposals) < 5:
        justification = "Audit proof-gate health after detected theory change or anomalies."
        proposals.append(
            _make_candidate(
                "T4",
                "Run proof-gate health check",
                "proof_task",
                3,
                justification,
            )
        )
        proposals[-1]["provenance"].update({"proof_gate_anomalies": proof_gate_anomalies[:5]})

    if last_planner_digest and len(proposals) < 5:
        justification = "Review latest planner digests for recurring blockers impacting commitments."
        proposals.append(
            _make_candidate(
                "T5",
                "Analyze recent planner digests for recurring blockers",
                "analysis",
                4,
                justification,
            )
        )

    mission_allow = constraints.get("allow_enhancements", False)
    tool_optimizer_anomalies = tool_report_data.get("anomalies", []) if tool_report_data else []
    tool_optimizer_has_anomalies = bool(tool_optimizer_anomalies)

    tool_optimizer_triggers: List[str] = []
    if tool_optimizer_missing:
        tool_optimizer_triggers.append("report missing")
    if tool_optimizer_stale:
        tool_optimizer_triggers.append("report older than 24h")
    if tool_optimizer_registry_zero:
        tool_optimizer_triggers.append("registry empty")
    if tool_optimizer_has_anomalies:
        tool_optimizer_triggers.append(f"anomalies present ({len(tool_optimizer_anomalies)})")
    if mission_allow and tool_optimizer_triggers and len(proposals) < 5:
        justification = "Refresh governed tool orchestration catalog to maintain runtime optimization integrity."
        if tool_optimizer_triggers:
            justification += " Triggers: " + ", ".join(tool_optimizer_triggers) + "."
        proposals.append(
            _make_candidate(
                "T6",
                "Optimize runtime tool orchestration",
                "analysis",
                2,
                justification,
            )
        )
        proposals[-1]["provenance"].update(
            {
                "tool_optimizer_report_path": str(tool_optimizer_report_path),
                "tool_optimizer_triggers": tool_optimizer_triggers,
                "tool_optimizer_report_timestamp": tool_optimizer_report_timestamp,
                "tool_optimizer_registry_count": tool_report_data.get("registry_count") if tool_report_data else None,
                "tool_optimizer_anomalies": tool_optimizer_anomalies[:5] if tool_optimizer_anomalies else [],
            }
        )

    # T7: Tool invention - propose when tool optimizer has fresh reports and enhancements allowed
    tool_invention_dir = repo_root / "state" / "tool_invention"
    tool_invention_report_path = tool_invention_dir / "tool_invention_report.json"
    tool_invention_missing = not tool_invention_report_path.exists()
    tool_invention_stale = False
    tool_invention_report_timestamp: Optional[str] = None
    if not tool_invention_missing:
        try:
            invention_report_data = json.loads(tool_invention_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            tool_invention_missing = True
        else:
            invention_timestamp_raw = invention_report_data.get("timestamp_utc")
            if invention_timestamp_raw:
                try:
                    parsed = datetime.fromisoformat(str(invention_timestamp_raw).replace("Z", "+00:00"))
                except ValueError:
                    tool_invention_stale = True
                else:
                    parsed_utc = parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                    age = datetime.now(timezone.utc) - parsed_utc
                    tool_invention_stale = age > timedelta(hours=24)
                tool_invention_report_timestamp = str(invention_timestamp_raw)
            else:
                tool_invention_stale = True

    tool_invention_triggers: List[str] = []
    if tool_invention_missing:
        tool_invention_triggers.append("report missing")
    if tool_invention_stale:
        tool_invention_triggers.append("report older than 24h")
    if mission_allow and not tool_optimizer_missing and tool_invention_triggers and len(proposals) < 5:
        justification = "Derive novel tools from optimizer gap analysis to enhance toolkit capabilities."
        if tool_invention_triggers:
            justification += " Triggers: " + ", ".join(tool_invention_triggers) + "."
        proposals.append(
            _make_candidate(
                "T7",
                "Derive novel tools from optimizer gaps",
                "analysis",
                3,
                justification,
            )
        )
        proposals[-1]["provenance"].update(
            {
                "tool_invention_report_path": str(tool_invention_report_path),
                "tool_invention_triggers": tool_invention_triggers,
                "tool_invention_report_timestamp": tool_invention_report_timestamp,
                "tool_optimizer_report_available": not tool_optimizer_missing,
            }
        )

    return proposals[:5]


TYPE_SCORES = {
    "repair": 120,
    "proof_task": 80,
    "analysis": 40,
    "data_ingest": 30,
    "doc": 10,
}


def _parse_priority(value: Any) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, num))


def _iso_to_timestamp(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return 0.0


def _find_commitment(ledger: Dict[str, Any], commitment_id: str) -> Optional[Dict[str, Any]]:
    if not ledger or not commitment_id:
        return None
    for item in ledger.get("commitments", []) or []:
        if str(item.get("commitment_id")) == str(commitment_id):
            return item
    return None


def score_commitment(
    commitment: Dict[str, Any],
    identity: Dict[str, Any],
    uwm_snapshot: Dict[str, Any],
    ledger: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[int, List[str]]:
    reasons: List[str] = []
    score = 0

    mission_allow = bool(identity.get("mission", {}).get("allow_enhancements", False))
    constraints = commitment.get("constraints", {})
    if bool(constraints.get("allow_enhancements", False)) and not mission_allow:
        reasons.append("Disqualified: exceeds mission allow_enhancements policy")
        return -10000, reasons

    status = str(commitment.get("status", "deferred"))
    if status not in SELECTABLE_STATUSES:
        reasons.append(f"Status {status} not selectable")
        return -10000, reasons

    if status == "active":
        score += 50
        reasons.append("Current active commitment (+50)")
    elif status == "blocked":
        score -= 200
        reasons.append("Blocked status (-200)")
    elif status == "deferred":
        score -= 50
        reasons.append("Deferred status (-50)")

    ctype = str(commitment.get("type", "other")).lower()
    type_score = TYPE_SCORES.get(ctype, 0)
    score += type_score
    reasons.append(f"Type {ctype} (+{type_score})")

    priority = _parse_priority(commitment.get("priority"))
    priority_bonus = 60 - 10 * priority
    score += priority_bonus
    reasons.append(f"Priority {priority} (+{priority_bonus})")

    latest_digest = context.get("last_planner_digest_archive")
    repo_root = Path(context.get("repo_root", "."))
    evidence_refs = commitment.get("evidence_refs", []) or []
    digest_bonus_applied = False
    if latest_digest:
        digest_path = Path(latest_digest)
        if not digest_path.is_absolute():
            digest_path = repo_root / digest_path
        if digest_path.exists() and str(latest_digest) in evidence_refs:
            score += 20
            digest_bonus_applied = True
            reasons.append("Evidence includes latest planner digest (+20)")

    uwm_hash = identity.get("world_model", {}).get("snapshot_hash")
    if not uwm_hash and isinstance(uwm_snapshot, dict):
        uwm_hash = uwm_snapshot.get("integrity", {}).get("snapshot_hash")
    if uwm_hash:
        for ref in evidence_refs:
            if isinstance(ref, str) and ref.startswith("uwm:") and f"#{uwm_hash}" in ref:
                score += 10
                reasons.append("UWM reference matches identity hash (+10)")
                break

    success_criteria = commitment.get("success_criteria", []) or []
    if len(success_criteria) >= 3:
        score += 5
        reasons.append("Robust success criteria (+5)")

    blockers = commitment.get("blockers", []) or []
    blocker_penalty = -25 * min(len(blockers), 4)
    if blocker_penalty:
        score += blocker_penalty
        reasons.append(f"Blockers present ({len(blockers)}) ({blocker_penalty})")

    if not digest_bonus_applied and latest_digest and str(latest_digest) in evidence_refs:
        # latest digest referenced but missing on disk
        reasons.append("Latest planner digest ref missing on disk (no bonus)")

    return score, reasons


def select_next_active_commitment(
    identity: Dict[str, Any],
    uwm_snapshot: Dict[str, Any],
    ledger: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    if ledger is None:
        return None, ledger, {"selected_id": None, "switched": False, "top_ranked": []}

    proposals = propose_candidate_commitments(identity, uwm_snapshot, ledger, context)
    existing_keys = { _dedupe_key(item) for item in ledger.get("commitments", []) }
    for candidate in proposals:
        key = _dedupe_key(candidate)
        if key in existing_keys:
            continue
        candidate.setdefault("status", "deferred")
        candidate.setdefault("priority", 3)
        candidate.setdefault("created_utc", context.get("cycle_utc") or _now_iso())
        candidate.setdefault("updated_utc", candidate["created_utc"])
        upsert_commitment(ledger, candidate)
        cid = candidate.get("commitment_id")
        if cid:
            record_event(
                ledger,
                "propose",
                cid,
                f"Candidate introduced via template {candidate.get('template_id', 'unknown')}"
            )
        existing_keys.add(key)

    scored: List[Tuple[int, Dict[str, Any], List[str]]] = []
    for commitment in ledger.get("commitments", []):
        status = str(commitment.get("status", "deferred"))
        if status not in SELECTABLE_STATUSES:
            continue
        score, reasons = score_commitment(commitment, identity, uwm_snapshot, ledger, context)
        if score <= -10000:
            continue
        scored.append((score, commitment, reasons))

    if not scored:
        return ledger.get("active_commitment_id"), ledger, {
            "selected_id": ledger.get("active_commitment_id"),
            "switched": False,
            "top_ranked": [],
        }

    scored.sort(
        key=lambda entry: (
            -entry[0],
            _parse_priority(entry[1].get("priority")),
            -_iso_to_timestamp(entry[1].get("updated_utc") or entry[1].get("created_utc")),
            str(entry[1].get("commitment_id") or ""),
        )
    )

    top_ranked = [
        {
            "commitment_id": item[1].get("commitment_id"),
            "score": item[0],
            "reasons": item[2],
        }
        for item in scored[:3]
    ]

    selected_commitment = scored[0][1]
    selected_id = selected_commitment.get("commitment_id")
    current_active_id = ledger.get("active_commitment_id")
    switched = selected_id != current_active_id

    if switched and current_active_id:
        mark_commitment_status(
            ledger,
            current_active_id,
            "deferred",
            "Preempted by higher priority commitment",
        )

    if selected_id:
        set_active_commitment(ledger, selected_id)
        rationale = scored[0][2][:3]
        summary = "; ".join(rationale)
        summary = _clamp(summary, SUMMARY_MAX)
        record_event(ledger, "arbiter_select", selected_id, summary or "Arbiter selection applied")

    decision_report = {
        "selected_id": selected_id,
        "switched": switched,
        "top_ranked": top_ranked,
    }
    return selected_id, ledger, decision_report


__all__ = [
    "propose_candidate_commitments",
    "score_commitment",
    "select_next_active_commitment",
]
