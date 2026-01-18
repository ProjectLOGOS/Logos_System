# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""Minimal UIP chat loop for LOGOS-GPT (advisor-only, gated execution)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_ROOT = Path(os.getenv("LOGOS_AUDIT_DIR", REPO_ROOT / "audit"))

from logos.uwm import (  # noqa: E402  # pylint: disable=wrong-import-position
    add_memory_item,
    calculate_initial_salience,
    init_working_memory,
    recall,
    stable_item_id,
)
from logos.ledger import (  # noqa: E402  # pylint: disable=wrong-import-position
    build_run_ledger,
)
from logos.proof_refs import (  # noqa: E402  # pylint: disable=wrong-import-position
    enforce_truth_annotation,
)
from scripts.llm_interface_suite.llm_advisor import (  # noqa: E402  # pylint: disable=wrong-import-position
    LLMAdvisor,
    build_tool_schema,
)
from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import (  # noqa: E402  # pylint: disable=wrong-import-position
    canonical_json_hash,
    validate_scp_state,
)

# Reuse dispatcher and tool registry
from scripts.start_agent import (  # noqa: E402  # pylint: disable=wrong-import-position
    STATE_DIR,
    TOOLS,
    RuntimeContext,
    dispatch_tool,
    load_mission,
)
from logos.tool_registry_loader import load_approved_tools  # noqa: E402
from LOGOS_SYSTEM.System_Stack.Protocol_Resources.attestation import (  # noqa: E402  # pylint: disable=wrong-import-position
    AlignmentGateError,
    compute_attestation_hash,
    load_alignment_attestation,
    validate_attestation,
    validate_mission_profile,
)

DEFAULT_MAX_TURNS = 20
MAX_PROPOSALS = 3
HIGH_IMPACT_TOOLS = {"tool_proposal_pipeline", "start_agent"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOGOS-GPT UIP chat loop (advisor-only)"
    )
    parser.add_argument("--enable-llm-advisor", action="store_true", default=False)
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "stub"],
        default="stub",
    )
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-timeout-sec", type=int, default=10)
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--read-only", action="store_true", default=False)
    parser.add_argument("--state-dir", default=str(STATE_DIR))
    parser.add_argument(
        "--audit-dir", default=os.getenv("LOGOS_AUDIT_DIR", str(REPO_ROOT / "audit"))
    )
    parser.add_argument("--assume-yes", action="store_true", default=False)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--objective-class", default="CHAT")
    parser.add_argument("--require-attestation", action="store_true", default=True)
    parser.add_argument(
        "--no-require-attestation",
        dest="require_attestation",
        action="store_false",
    )
    parser.add_argument(
        "--attestation-path",
        default=str(STATE_DIR / "alignment_LOGOS-AGENT-OMEGA.json"),
    )
    parser.add_argument("--attestation-max-age-sec", type=int, default=21600)
    return parser.parse_args(argv)


def _default_state() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": _now(),
        "version": 0,
        "prev_hash": None,
        "state_hash": "",
        "observations": [],
        "last_proposals": [],
        "last_tool_results": [],
        "truth_events": [],
        "arp_status": {},
        "scp_status": {},
        "working_memory": init_working_memory({}),
        "plans": {"active": [], "history": []},
        "beliefs": {
            "schema_version": 1,
            "updated_at": _now(),
            "beliefs_version": 0,
            "prev_hash": "",
            "state_hash": "",
            "items": [],
        },
    }


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_state()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return _default_state()


def _persist_state(state: Dict[str, Any], path: Path) -> None:
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
        # keep stub-friendly behavior; validation may fail in stub mode
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _make_wm_item(role: str, message: str, objective_class: str) -> Dict[str, Any]:
    content = {"role": role, "message": message}
    truth = "HEURISTIC"
    evidence = {"type": "none", "ref": None, "details": "uip_chat"}
    item_id = stable_item_id("SCP", content, [objective_class])
    salience = calculate_initial_salience(truth, "SCP", {"status": "SUCCESS"})
    ts = _now()
    return {
        "id": item_id,
        "created_at": ts,
        "last_accessed_at": ts,
        "objective_tags": [objective_class],
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


def _tool_summary() -> Dict[str, Any]:
    return {"tools": sorted(TOOLS.keys()), "high_impact": sorted(HIGH_IMPACT_TOOLS)}


def _approve_tool(
    proposal: Dict[str, Any],
    requires_uip: bool,
    assume_yes: bool,
    read_only: bool,
) -> bool:
    # High-impact or UIP-required tools must be approved
    if requires_uip or proposal.get("tool") in HIGH_IMPACT_TOOLS:
        return assume_yes
    if read_only:
        # Allow only low-impact proposals in read-only mode
        return True
    return assume_yes


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    state_dir = Path(args.state_dir)
    audit_dir = Path(args.audit_dir)
    state_path = state_dir / "scp_state.json"

    if state_dir.resolve() != STATE_DIR.resolve():
        print(
            "WARN: --state-dir affects chat state/ledgers only; mission/attestation come from LOGOS_STATE_DIR"  # noqa: E501
            f" ({STATE_DIR})",
            file=sys.stderr,
        )

    load_approved_tools(TOOLS)

    # Attestation
    attestation_hash = None
    mission_profile_hash = None
    mission_profile = load_mission()
    validate_mission_profile(mission_profile)
    mission_profile_hash = canonical_json_hash(mission_profile)
    if args.require_attestation:
        try:
            att = load_alignment_attestation(args.attestation_path)
            validate_attestation(att, max_age_seconds=args.attestation_max_age_sec)
            attestation_hash = compute_attestation_hash(att)
        except AlignmentGateError as exc:
            print(f"[GATE] ERROR: {exc}")
            return 2
    elif os.getenv("LOGOS_DEV_BYPASS_OK") != "1":
        print("[GATE] ERROR: --no-require-attestation requires LOGOS_DEV_BYPASS_OK=1")
        return 2
    else:
        attestation_hash = "DEV_BYPASS"

    ctx = RuntimeContext(
        attestation_hash=attestation_hash,
        mission_profile_hash=mission_profile_hash,
    )

    # Load state and init UWM
    state = _load_state(state_path)
    init_working_memory(state)

    # Advisor setup
    tool_schema = build_tool_schema(TOOLS)
    requires_uip_map = {
        entry["name"]: entry.get("requires_uip", False)
        for entry in tool_schema.get("tools", [])
    }
    advisor = None
    if args.enable_llm_advisor:
        advisor = LLMAdvisor(
            provider=args.llm_provider,
            model=args.llm_model,
            tools_schema=tool_schema,
            truth_rules={},
            timeout_sec=args.llm_timeout_sec,
        )

    executed_events: List[Dict[str, Any]] = []
    run_start_ts = _now()
    objective_class = args.objective_class

    for turn in range(args.max_turns):
        try:
            user_msg = input().strip()
        except EOFError:
            break
        if not user_msg:
            break

        # Recall context
        recalled = recall(state, objective_class, k=5)
        context = {
            "conversation_recall": recalled,
            "belief_summary": state.get("beliefs", {}),
            "tool_summary": _tool_summary(),
        }

        advisor_reply = "Acknowledged."
        proposals: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        result: Dict[str, Any] = {}
        if advisor:
            if args.stream:
                reply_chunks: List[str] = []
                stream_gen = advisor.propose_stream(user_msg, context)
                print("Assistant: ", end="", flush=True)
                while True:
                    try:
                        chunk = next(stream_gen)
                    except StopIteration as stop:
                        result = stop.value or {}
                        break
                    if chunk:
                        reply_chunks.append(str(chunk))
                        print(str(chunk), end="", flush=True)
                print("")
                advisor_reply = (
                    result.get("reply") or "".join(reply_chunks) or advisor_reply
                )
            else:
                result = advisor.propose(user_msg, context)
                advisor_reply = result.get("reply", advisor_reply)

            proposals = (result.get("proposals", []) or [])[:MAX_PROPOSALS]
            rejected = result.get("rejected", []) or []

        # Record rejected proposals as truth events
        for rej in rejected:
            content = {
                "rejected": rej.get("proposal") if isinstance(rej, dict) else rej,
                "reason": rej.get("reason", "rejected"),
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

        # Truth-annotate reply
        reply_annotation = {
            "truth": "HEURISTIC",
            "evidence": {
                "type": "inference",
                "ref": None,
                "details": "advisor_reply",
            },
        }
        reply_annotation = _downgrade_proved(reply_annotation)
        try:
            reply_annotation = enforce_truth_annotation(reply_annotation, None)
        except Exception:
            pass

        # Store user + assistant messages in UWM
        add_memory_item(state, _make_wm_item("user", user_msg, objective_class))
        add_memory_item(
            state,
            _make_wm_item("assistant", advisor_reply, objective_class),
        )

        # Tool proposals with gating
        run_results: List[Dict[str, Any]] = []
        for prop in proposals:
            tool = prop.get("tool", "")
            ann = _downgrade_proved(prop.get("truth_annotation", {}) or {})
            try:
                ann = enforce_truth_annotation(ann, None)
            except Exception:
                pass
            requires_uip = requires_uip_map.get(tool, False)
            approved = _approve_tool(
                prop,
                requires_uip,
                args.assume_yes,
                args.read_only,
            )
            if not approved:
                state.setdefault("truth_events", []).append(
                    {
                        "ts": _now(),
                        "source": "UIP",
                        "content": {"rejected": prop, "reason": "uip_denied"},
                        "truth_annotation": {
                            "truth": "UNVERIFIED",
                            "evidence": {
                                "type": "none",
                                "ref": None,
                                "details": "uip denied",
                            },
                        },
                    }
                )
                continue
            if requires_uip or tool in HIGH_IMPACT_TOOLS:
                state.setdefault("truth_events", []).append(
                    {
                        "ts": _now(),
                        "source": "UIP",
                        "content": {"approved": prop, "reason": "uip_approved"},
                        "truth_annotation": {
                            "truth": "UNVERIFIED",
                            "evidence": {
                                "type": "none",
                                "ref": None,
                                "details": "uip approval",
                            },
                        },
                    }
                )
            try:
                output = dispatch_tool(tool, str(prop.get("args", "")), ctx=ctx)
                outcome = "SUCCESS"
            except Exception as exc:  # pylint: disable=broad-except
                output = f"[error] {exc}"
                outcome = "ERROR"
            state.setdefault("last_tool_results", []).append(
                {
                    "tool": tool,
                    "args": prop.get("args", ""),
                    "status": outcome,
                    "objective": objective_class,
                }
            )
            run_results.append(
                {
                    "tool": tool,
                    "args": prop.get("args", ""),
                    "output": output,
                    "outcome": outcome,
                    "truth_annotation": ann,
                }
            )
            executed_events.append(
                {
                    "tool": tool,
                    "args_hash": canonical_json_hash({"args": prop.get("args", "")}),
                    "outcome": outcome,
                    "truth_tier": ann.get("truth", "UNVERIFIED")
                    if isinstance(ann, dict)
                    else "UNVERIFIED",
                }
            )
            # Memory of tool results
            tool_content = {"tool": tool, "output": output, "status": outcome}
            tool_item_id = stable_item_id("TOOL", tool_content, [objective_class])
            tool_item = {
                "id": tool_item_id,
                "created_at": _now(),
                "last_accessed_at": _now(),
                "objective_tags": [objective_class],
                "truth": ann.get("truth", "UNVERIFIED")
                if isinstance(ann, dict)
                else "UNVERIFIED",
                "evidence": ann.get(
                    "evidence", {"type": "none", "ref": None, "details": "tool run"}
                )
                if isinstance(ann, dict)
                else {"type": "none", "ref": None, "details": "tool run"},
                "content": tool_content,
                "salience": calculate_initial_salience(
                    "VERIFIED" if outcome == "SUCCESS" else "UNVERIFIED",
                    "TOOL",
                    tool_content,
                ),
                "decay_rate": 0.15,
                "access_count": 1,
                "source": "TOOL",
            }
            add_memory_item(state, tool_item)

        # Update last proposals
        for prop in proposals:
            prop_copy = dict(prop)
            prop_copy["truth_annotation"] = prop_copy.get("truth_annotation") or {
                "truth": "HEURISTIC",
                "evidence": {"type": "none", "ref": None, "details": "advisor_default"},
            }
            state.setdefault("last_proposals", []).append(prop_copy)

        # Truth event for reply
        state.setdefault("truth_events", []).append(
            {
                "ts": _now(),
                "source": "UIP",
                "content": {"reply": advisor_reply, "objective_class": objective_class},
                "truth_annotation": reply_annotation,
            }
        )

        # Emit ledger per turn
        ledger_ctx = {
            "attestation_hash": attestation_hash or "",
            "coq_index_hash": None,
            "scp_state_hash": state.get("state_hash", ""),
            "beliefs_hash": state.get("beliefs", {}).get("state_hash", ""),
            "metrics_hash": "",
            "run_start_ts": run_start_ts,
            "run_end_ts": _now(),
            "executed_events": executed_events,
            "proposals": proposals,
            "plan_summary": {},
            "policy_notes": {},
            "governance_flags": {
                "read_only": args.read_only,
                "assume_yes": args.assume_yes,
            },
            "advisor": {
                "enabled": bool(advisor),
                "provider": args.llm_provider if advisor else "stub",
                "model": args.llm_model if advisor else "stub",
                "stream": bool(advisor and args.stream),
            },
        }
        ledger = build_run_ledger(ledger_ctx)
        ledger_filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        ledger_path = (
            audit_dir / "run_ledgers" / f"logos_gpt_chat_{turn}_{ledger_filename}.json"
        )
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

        if not (advisor and args.stream):
            print(f"Assistant: {advisor_reply}")
        for res in run_results:
            print(f"Executed {res['tool']}: {res['outcome']}")

    _persist_state(state, state_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
