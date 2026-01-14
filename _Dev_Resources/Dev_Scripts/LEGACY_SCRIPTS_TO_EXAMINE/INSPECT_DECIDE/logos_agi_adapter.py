"""Adapter for Logos_AGI ARP+SCP integration into Entry_Point supervised loop."""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import canonical_json_hash, validate_scp_state

try:
    from logos.evaluator import load_metrics, choose_best, normalize_objective_class
except ImportError:
    # Fallback if evaluator not available
    def load_metrics(_path):
        return {"metrics": {}}

    def choose_best(proposals, _obj_class, _metrics):
        return proposals

    def normalize_objective_class(obj):
        return "GENERAL"


try:
    from logos.uwm import (
        init_working_memory,
        add_memory_item,
        decay_and_promote,
        recall,
        stable_item_id,
        calculate_initial_salience,
    )
except ImportError:
    # Fallback if UWM not available
    def init_working_memory(state):
        return state.get("working_memory", {})

    def add_memory_item(state, item):
        pass

    def decay_and_promote(state):
        pass

    def recall(state, obj_class, k=5, include_contradicted=False):
        return []

    def stable_item_id(source, content, objective_tags):
        return str(hash((source, tuple(sorted(objective_tags)), str(content))))

    def calculate_initial_salience(truth, source, content):
        return 0.5


try:
    from logos.proof_refs import load_theorem_index, enforce_truth_annotation
    from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import validate_truth_annotation
except ImportError:
    # Fallback
    def load_theorem_index(path):
        return None

    def enforce_truth_annotation(annotation, index):
        return annotation

    def validate_truth_annotation(annotation):
        pass


try:
    from logos.policy import apply_belief_policy
except ImportError:

    def apply_belief_policy(proposals, obj_class, beliefs, **kwargs):
        return proposals, {}


def init_beliefs_container(state):
    if "beliefs" not in state:
        state["beliefs"] = {"items": []}


def consolidate_beliefs(state, **kwargs):
    run_id = kwargs.get("run_id")
    mode = kwargs.get("mode", "auto")
    stub_mode = bool(kwargs.get("stub_mode", False)) or mode == "stub"
    prev = state.get("beliefs", {}) if isinstance(state, dict) else {}
    prev_hash = prev.get("state_hash") if isinstance(prev, dict) else None
    items = list(prev.get("items", [])) if isinstance(prev, dict) else []

    # In real mode, drop stub/synthesized artifacts unless explicitly stub_mode
    if not stub_mode:
        items = [
            b
            for b in items
            if not (b.get("source") == "STUB" or b.get("synthesized") is True)
        ]

    # Synthesize a bounded stub belief only when in stub mode
    if stub_mode and not items:
        last_result = None
        if isinstance(state, dict):
            tool_results = state.get("last_tool_results", [])
            if isinstance(tool_results, list) and tool_results:
                last_result = tool_results[-1]

        now = datetime.now(timezone.utc).isoformat()
        belief_content = last_result or {
            "note": "synthetic belief for stub SCP",
        }
        items.append(
            {
                "belief_id": str(uuid.uuid4()),
                "id": None,  # filled below for compatibility
                "created_at": now,
                "updated_at": now,
                "objective_tags": ["STATUS"],
                "content": {
                    "observation": belief_content,
                    "evidence": {
                        "type": "inference",
                        "ref": None,
                        "details": "stub synthesis",
                    },
                },
                "truth": "HEURISTIC",
                "confidence": 0.6,
                "supporting_refs": [],
                "contradicting_refs": [],
                "status": "ACTIVE",
                "notes": {},
                "source": "STUB",
                "mode": "stub",
                "synthesized": True,
            }
        )
        items[-1]["id"] = items[-1]["belief_id"]

    container = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "beliefs_version": (prev.get("beliefs_version", 0) + 1)
        if isinstance(prev, dict)
        else 1,
        "prev_hash": prev_hash,
        "items": items,
    }
    container["state_hash"] = canonical_json_hash(container)
    return container


def apply_plan_revision(state, beliefs):
    # Stub implementation: mark steps as SKIPPED if beliefs have contradictions
    quarantined = [
        b
        for b in beliefs.get("items", [])
        if b.get("status") == "QUARANTINED"
        and b.get("source") != "STUB"
        and not b.get("synthesized", False)
    ]
    if quarantined:
        if "plans" in state and "active" in state["plans"]:
            for plan in state["plans"]["active"]:
                if "steps" in plan:
                    for step in plan["steps"]:
                        if step.get("status") in ["PENDING", "DENIED"]:
                            step["status"] = "SKIPPED"
                            # Add checkpoint
                            if "checkpoints" not in plan:
                                plan["checkpoints"] = []
                            plan["checkpoints"].append(
                                {
                                    "step_index": step["index"],
                                    "reason": "belief contradiction detected",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            )


try:
    # from logos.beliefs import consolidate_beliefs, apply_plan_revision
    pass
except ImportError:
    # Fallback if beliefs not available
    pass


def get_belief_summary(beliefs, obj_class):
    return {"count": 0, "top_beliefs": [], "high_confidence_count": 0}


class StubSCPNexus:
    """Stub SCP nexus for when real imports fail."""

    def __init__(self):
        self.initialized = True

    async def initialize(self):
        return True

    async def process_agent_request(self, request):
        # Stub response
        return type(
            "Response",
            (),
            {"success": True, "error": None, "data": {"message": "stub response"}},
        )()


class LogosAgiNexus:
    """Adapter wrapping ARP bootstrap + SCP nexus for safe integration."""

    def __init__(
        self,
        *,
        enable: bool,
        audit_logger,
        max_compute_ms: int,
        state_dir: str,
        repo_sha: str,
        mode: str = "auto",
        scp_recovery_mode: bool = False,
    ):
        self.enable = enable
        self.audit_logger = audit_logger
        self.max_compute_ms = max_compute_ms
        self.state_dir = Path(state_dir)
        self.repo_sha = repo_sha
        self.mode = mode
        self.scp_recovery_mode = scp_recovery_mode
        self.available = False
        self.arp_reasoner = None
        self.scp_nexus = None
        self.last_error = None
        self.observations = []
        self.prior_state = None
        self.last_proposals = []
        self.last_tool_results = []
        self.truth_events = []
        self.persisted_path = self.state_dir / "logos_agi_scp_state.json"
        self.compat_scp_state_path = self.state_dir / "scp_state.json"
        self.metrics_path = self.state_dir / "proposal_metrics.json"
        self.metrics_state = None
        self.working_memory = None
        self.scp_state_valid = True
        self.scp_state_validation_error = None
        self.theorem_index = None

    def bootstrap(self) -> None:
        """Import and initialize ARP/SCP components."""
        if not self.enable:
            return

        if self.mode == "stub":
            # Always use stub implementations
            self.scp_nexus = StubSCPNexus()
            self.arp_reasoner = None
            self.available = True
            self.last_error = None
        elif self.mode == "real":
            # Require real implementations, fail if unavailable
            try:
                self._bootstrap_real()
                self.available = True
                self.last_error = None
            except Exception as e:
                raise RuntimeError(f"Real Logos_AGI bootstrap failed: {e}") from e
        else:  # auto
            # Try real, fallback to stub
            try:
                self._bootstrap_real()
                self.available = True
                self.last_error = None
            except Exception as e:
                self.last_error = f"Using stub due to bootstrap error: {e}"
                self.scp_nexus = StubSCPNexus()
                self.arp_reasoner = None
                self.available = True

        # Load persisted state and metrics regardless of mode
        self._load_persisted_state()
        self.metrics_state = load_metrics(self.metrics_path)

        # Load theorem index
        try:
            self.theorem_index = load_theorem_index(
                str(self.state_dir / "coq_theorem_index.json")
            )
        except Exception:
            self.theorem_index = None

        # Ensure working_memory is initialized
        if self.prior_state:
            self.working_memory = init_working_memory(self.prior_state)

    def _bootstrap_real(self) -> None:
        """Attempt real bootstrap."""
        # Add external to path
        import sys
        from pathlib import Path

        external_path = Path(__file__).parent.parent / "external"
        if str(external_path) not in sys.path:
            sys.path.insert(0, str(external_path))

        # Import ARP bootstrap
        from Logos_AGI.Advanced_Reasoning_Protocol.arp_bootstrap import AdvancedReasoner

        self.arp_reasoner = AdvancedReasoner(agent_identity=self.repo_sha)
        self.arp_reasoner.start()

        # Import SCP nexus
        from Logos_AGI.Synthetic_Cognition_Protocol.system_utilities.nexus.scp_nexus import (
            SCPNexus,
        )

        try:
            from Logos_AGI.System_Operations_Protocol.infrastructure.agent_system.base_nexus import (
                AgentRequest,
            )
        except ImportError:
            # Define minimal AgentRequest if import fails
            class AgentRequest:
                def __init__(self, request_id, operation, payload):
                    self.request_id = request_id
                    self.operation = operation
                    self.payload = payload

        self.scp_nexus = SCPNexus()
        # Run async init
        success = asyncio.run(self.scp_nexus.initialize())
        if not success:
            raise RuntimeError("SCP initialization failed")

    def _load_persisted_state(self) -> None:
        """Load and validate previously persisted state."""
        load_path = (
            self.persisted_path
            if self.persisted_path.exists()
            else self.compat_scp_state_path
        )
        if load_path.exists():
            try:
                with open(load_path) as f:
                    state = json.load(f)
                # Try to validate (skip in stub mode)
                if self.mode != "stub":
                    validate_scp_state(state)
                validation_error = None
                stored_hash = state.get("state_hash")
                if stored_hash:
                    temp_state = dict(state)
                    temp_state.pop("state_hash", None)
                    computed_hash = canonical_json_hash(temp_state)
                    if computed_hash != stored_hash:
                        validation_error = "state_hash mismatch"
                # If validation fails, allow recovery when explicitly enabled
                if validation_error:
                    self.scp_state_validation_error = validation_error
                    self.scp_state_valid = False
                    recovery_allowed = (
                        self.scp_recovery_mode and os.getenv("LOGOS_DEV_BYPASS_OK") == "1"
                    )
                    if not recovery_allowed:
                        return
                # If validation succeeds or skipped
                self.prior_state = state
                self.observations = state.get("observations", [])
                self.last_proposals = state.get("last_proposals", [])
                self.last_tool_results = state.get("last_tool_results", [])
                self.working_memory = state.get("working_memory", {})
                # Initialize UWM if missing
                init_working_memory(state)
                # Initialize beliefs if missing
                init_beliefs_container(state)
                # Ensure plans exist
                if "plans" not in state:
                    state["plans"] = {"active": [], "history": []}
                if validation_error:
                    self.scp_state_valid = False
                    self.scp_state_validation_error = validation_error
                else:
                    self.scp_state_valid = True
                    self.scp_state_validation_error = None
            except (json.JSONDecodeError, ValueError, TypeError, OSError) as e:
                error_msg = f"Failed to validate persisted state: {e}"
                self.scp_state_validation_error = error_msg[:200]  # Bound length
                self.scp_state_valid = False
                # Check if recovery mode is enabled
                recovery_allowed = (
                    self.scp_recovery_mode and os.getenv("LOGOS_DEV_BYPASS_OK") == "1"
                )
                if recovery_allowed:
                    # Load the state anyway for recovery
                    self.prior_state = state
                    self.observations = state.get("observations", [])
                    self.last_proposals = state.get("last_proposals", [])
                    self.last_tool_results = state.get("last_tool_results", [])
                    self.working_memory = state.get("working_memory", {})
                    # Initialize UWM if missing
                    init_working_memory(state)
                    # Initialize beliefs if missing
                    init_beliefs_container(state)
                    # Ensure plans exist
                    if "plans" not in state:
                        state["plans"] = {"active": [], "history": []}
                    # Record recovery in truth events
                    ts = datetime.now(timezone.utc).isoformat()
                    event = {
                        "ts": ts,
                        "source": "RUNTIME",
                        "content": {
                            "recovery_mode": True,
                            "validation_error": error_msg,
                        },
                        "truth_annotation": {
                            "truth": "UNVERIFIED",
                            "evidence": {
                                "type": "none",
                                "ref": None,
                                "details": "SCP state recovery mode activated",
                            },
                        },
                    }
                    self.truth_events.append(event)
                    self.last_error = (
                        f"Loaded invalid state in recovery mode: {error_msg}"
                    )
                else:
                    self.last_error = error_msg
                    self.prior_state = None

    def refresh_plan_history(self, history_scored: Dict[str, Any]) -> None:
        """Refresh in-memory plan history from validated source-of-truth."""
        if not isinstance(history_scored, dict):
            return

        if self.prior_state is None:
            self.prior_state = {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
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
                "beliefs": {},
            }

        plans_block = self.prior_state.setdefault(
            "plans", {"active": [], "history": []}
        )
        plans_block["history_scored"] = history_scored

        if not isinstance(self.truth_events, list):
            self.truth_events = []

        ts = datetime.now(timezone.utc).isoformat()
        evidence_ref = (
            self.prior_state.get("state_hash")
            if isinstance(self.prior_state, dict)
            else None
        )
        self.truth_events.append(
            {
                "ts": ts,
                "source": "RUNTIME",
                "content": {
                    "event": "plan_history_refreshed",
                    "entries_by_signature": list(
                        history_scored.get("entries_by_signature", {}).keys()
                    ),
                },
                "truth_annotation": {
                    "truth": "VERIFIED",
                    "evidence": {
                        "type": "hash",
                        "ref": evidence_ref,
                        "details": "In-process refresh from persisted plan history",
                    },
                },
            }
        )

    def record_tool_result(
        self, tool: str, args: str, status: str, objective: str
    ) -> None:
        """Record the result of a tool execution for replay logic."""
        self.last_tool_results.append(
            {
                "tool": tool,
                "args": args,
                "status": status,
                "objective": objective,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Add to UWM
        if self.prior_state:
            objective_class = normalize_objective_class(objective)
            content = {
                "tool": tool,
                "args": args,
                "status": status,
                "outcome": "SUCCESS" if status in ["ok", "success"] else "FAILURE",
            }
            truth = "VERIFIED" if status in ["ok", "success"] else "HEURISTIC"
            item_id = stable_item_id("TOOL", content, [objective_class])
            item = {
                "id": item_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed_at": datetime.now(timezone.utc).isoformat(),
                "objective_tags": [objective_class],
                "truth": truth,
                "evidence": {
                    "type": "hash",
                    "ref": self.prior_state.get("state_hash"),
                    "details": f"Tool execution result: {tool}",
                },
                "content": content,
                "salience": calculate_initial_salience(truth, "TOOL", content),
                "decay_rate": 0.15,
                "access_count": 0,
                "source": "TOOL",
            }
            add_memory_item(self.prior_state, item)

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get short summary of replayed state for proposal logic."""
        if not self.prior_state:
            return {"has_prior": False}
        last_result = self.last_tool_results[-1] if self.last_tool_results else None
        last_proposal = self.last_proposals[-1] if self.last_proposals else None
        return {
            "has_prior": True,
            "version": self.prior_state.get("version", 0),
            "last_objective": last_result.get("objective") if last_result else None,
            "last_tool": last_result.get("tool") if last_result else None,
            "last_status": last_result.get("status") if last_result else None,
            "last_proposal_tool": last_proposal.get("tool") if last_proposal else None,
        }

    def observe(self, observation: Dict[str, Any]) -> None:
        """Pass observation to SCP for cognitive processing."""
        self.observations.append(observation)
        if not self.available or not self.scp_nexus:
            return

        try:
            from Logos_AGI.Synthetic_Cognition_Protocol.system_utilities.nexus.scp_nexus import (
                AgentRequest,
                AgentType,
            )

            # Use enhance_cognition to process the observation
            request = AgentRequest(
                agent_id=str(uuid.uuid4()),
                operation="enhance_cognition",
                payload={
                    "observation": observation,
                    "context": self.observations[-10:],
                },  # Last 10 for context
                agent_type=AgentType.SYSTEM_AGENT,
            )
            # Schedule async processing deterministically
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._process_observation_async(request))
            except RuntimeError:
                # No running event loop, record diagnostic
                self.last_error = "No running event loop; observation async skipped"
                # Optionally add a truth event
                ts = datetime.now(timezone.utc).isoformat()
                event = {
                    "ts": ts,
                    "source": "RUNTIME",
                    "content": {
                        "diagnostic": "async observation skipped",
                        "reason": "no event loop",
                    },
                    "truth_annotation": {
                        "truth": "UNVERIFIED",
                        "evidence": {
                            "type": "none",
                            "ref": None,
                            "details": "Runtime diagnostic",
                        },
                    },
                }
                self.truth_events.append(event)
        except (ImportError, AttributeError) as e:
            # Record error instead of silent pass
            self.last_error = f"Observation setup error: {e}"
            ts = datetime.now(timezone.utc).isoformat()
            event = {
                "ts": ts,
                "source": "RUNTIME",
                "content": {"diagnostic": "observation error", "error": str(e)},
                "truth_annotation": {
                    "truth": "UNVERIFIED",
                    "evidence": {
                        "type": "none",
                        "ref": None,
                        "details": "Error diagnostic",
                    },
                },
            }
            self.truth_events.append(event)

    async def _process_observation_async(self, request: Any) -> None:
        """Async helper to process observation without blocking."""
        try:
            response = await self.scp_nexus.process_agent_request(request)
            if not response.success:
                self.last_error = f"Observation processing failed: {response.error}"
        except Exception as e:
            self.last_error = f"Async observation error: {e}"

    def propose(self, objective: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate proposal using ARP/SCP with replayed state influence."""
        constraints = constraints or {}
        if not self.available:
            return {
                "proposals": [],
                "notes": {"reason": "Logos_AGI unavailable"},
                "errors": [self.last_error or "Not bootstrapped"],
            }

        try:
            # Get memory summary for decision logic
            memory = self.get_memory_summary()
            proposal_policy_reason = "default"

            # Recall relevant UWM items
            objective_class = normalize_objective_class(objective)
            recalled_items = (
                recall(self.prior_state, objective_class, k=5)
                if self.prior_state
                else []
            )
            recalled_summaries = [
                f"{item['source']}: {item['content'].get('tool', 'unknown')} "
                f"({item['truth']})"
                for item in recalled_items
            ]

            # Get belief summary
            belief_summary = (
                get_belief_summary(self.prior_state.get("beliefs", {}), objective_class)
                if self.prior_state
                else {"count": 0, "top_beliefs": [], "high_confidence_count": 0}
            )
            belief_summaries = [
                (
                    f"belief: {b['content'].get('tool', 'unknown')} "
                    f"({b['truth']}, conf={b['confidence']:.2f})"
                )
                for b in belief_summary["top_beliefs"][:2]
            ]
            recalled_summaries.extend(belief_summaries)

            # Apply plan revision if active plans exist
            if self.prior_state and self.prior_state.get("plans", {}).get("active"):
                apply_plan_revision(
                    self.prior_state, self.prior_state.get("beliefs", {})
                )

            # Apply replay rules
            avoid_tools = set()
            if memory["has_prior"]:
                # Rule 1: Avoid tools that failed last time for same objective
                if (
                    memory["last_objective"] == objective
                    and memory["last_status"] in ["denied", "error"]
                    and memory["last_tool"]
                ):
                    avoid_tools.add(memory["last_tool"])
                    proposal_policy_reason = (
                        f"avoiding previously failed tool {memory['last_tool']}"
                    )

                # Rule 2: If last objective was "status" and mission.status succeeded,
                # then try probe.last
                if (
                    memory["last_objective"] == "status"
                    and memory["last_tool"] == "mission.status"
                    and memory["last_status"] == "ok"
                    and objective == "status"
                    and self.mode != "stub"
                ):
                    # Propose probe.last instead
                    proposals = [
                        {
                            "tool": "probe.last",
                            "args": "",
                            "rationale": (
                                "Replaying: last status check succeeded, now probing "
                                "recent activity. Recalled: "
                                f"{', '.join(recalled_summaries[:3])}"
                            ),
                            "confidence": 0.85,
                            "truth_annotation": {
                                "truth": "VERIFIED",
                                "evidence": {
                                    "type": "hash",
                                    "ref": self.prior_state.get("state_hash")
                                    if self.prior_state
                                    else None,
                                    "details": (
                                        "Deterministic replay from validated SCP "
                                        "state"
                                    ),
                                },
                            },
                        }
                    ]

                    # Apply belief policy even to replayed proposals
                    objective_class = normalize_objective_class(objective)
                    proposals, policy_notes = apply_belief_policy(
                        proposals, objective_class, self.prior_state.get("beliefs", {})
                    )

                    self.last_proposals.extend(proposals)

                    # Append truth event
                    ts = datetime.now(timezone.utc).isoformat()
                    for proposal in proposals:
                        event = {
                            "ts": ts,
                            "source": "ARP",
                            "content": proposal,
                            "truth_annotation": proposal["truth_annotation"],
                        }
                        self.truth_events.append(event)

                    return {
                        "proposals": proposals,
                        "notes": {
                            "method": "replayed_state",
                            "policy": policy_notes,
                            "policy_reason": proposal_policy_reason,
                        },
                        "errors": [],
                    }

            # Default proposal logic (similar to before)
            proposals = []
            if "status" in objective.lower() and "mission.status" not in avoid_tools:
                proposals.append(
                    {
                        "tool": "mission.status",
                        "args": "",
                        "rationale": (
                            "SCP meta-reasoning analysis suggests status check. "
                            f"Recalled: {', '.join(recalled_summaries[:3])}"
                        ),
                        "confidence": 0.9,
                        "truth_annotation": {
                            "truth": "HEURISTIC",
                            "evidence": {
                                "type": "inference",
                                "ref": None,
                                "details": (
                                    "Pattern-based reasoning from objective keywords"
                                ),
                            },
                        },
                    }
                )
            elif "probe" in objective.lower():
                proposals.append(
                    {
                        "tool": "probe.last",
                        "args": "",
                        "rationale": (
                            "SCP cognitive enhancement suggests probing recent "
                            f"activity. Recalled: {', '.join(recalled_summaries[:3])}"
                        ),
                        "confidence": 0.8,
                        "truth_annotation": {
                            "truth": "HEURISTIC",
                            "evidence": {
                                "type": "inference",
                                "ref": None,
                                "details": (
                                    "Pattern-based reasoning from objective keywords"
                                ),
                            },
                        },
                    }
                )
            else:
                proposals.append(
                    {
                        "tool": "mission.status",
                        "args": "",
                        "rationale": (
                            "Conservative proposal from SCP meta-reasoning. "
                            f"Recalled: {', '.join(recalled_summaries[:3])}"
                        ),
                        "confidence": 0.7,
                        "truth_annotation": {
                            "truth": "UNVERIFIED",
                            "evidence": {
                                "type": "none",
                                "ref": None,
                                "details": "Fallback default proposal",
                            },
                        },
                    }
                )

            # Apply belief policy
            objective_class = normalize_objective_class(objective)
            proposals, policy_notes = apply_belief_policy(
                proposals, objective_class, self.prior_state.get("beliefs", {})
            )

            # Apply evaluator ranking
            proposals = choose_best(proposals, objective_class, self.metrics_state)

            self.last_proposals.extend(proposals)

            # Add proposals to UWM
            if self.prior_state:
                for proposal in proposals:
                    content = {
                        "tool": proposal["tool"],
                        "args": proposal.get("args", ""),
                        "rationale": proposal.get("rationale", ""),
                        "confidence": proposal.get("confidence", 0.5),
                    }
                    truth = proposal.get("truth_annotation", {}).get(
                        "truth", "HEURISTIC"
                    )
                    item_id = stable_item_id("ARP", content, [objective_class])
                    item = {
                        "id": item_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "last_accessed_at": datetime.now(timezone.utc).isoformat(),
                        "objective_tags": [objective_class],
                        "truth": truth,
                        "evidence": proposal.get("truth_annotation", {}).get(
                            "evidence", {}
                        ),
                        "content": content,
                        "salience": calculate_initial_salience(truth, "ARP", content),
                        "decay_rate": 0.15,
                        "access_count": 0,
                        "source": "ARP",
                    }
                    add_memory_item(self.prior_state, item)

            # Append truth events for proposals
            ts = datetime.now(timezone.utc).isoformat()
            for proposal in proposals:
                event = {
                    "ts": ts,
                    "source": "ARP",
                    "content": proposal,
                    "truth_annotation": proposal["truth_annotation"],
                }
                self.truth_events.append(event)

            return {
                "proposals": proposals,
                "notes": {
                    "method": "default_with_replay",
                    "policy": policy_notes,
                    "evaluator_applied": True,
                },
                "errors": [],
            }
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            return {
                "proposals": [],
                "notes": {"reason": "Proposal error"},
                "errors": [str(e)],
            }

    def propose_plan(
        self, objective: str, constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a deterministic plan for multi-step objectives."""
        constraints = constraints or {}
        if not self.available:
            return {
                "plan": None,
                "notes": {"reason": "Logos_AGI unavailable"},
                "errors": [self.last_error or "Not bootstrapped"],
            }

        # Initialize prior_state if None
        if self.prior_state is None:
            self.prior_state = {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
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
            }

        try:
            objective_class = normalize_objective_class(objective)
            recalled_items = (
                recall(self.prior_state, objective_class, k=5)
                if self.prior_state
                else []
            )

            # Conservative plan for "status" objective
            if objective_class == "STATUS":
                # Create candidate proposals for tools
                candidate_proposals = [
                    {
                        "tool": "mission.status",
                        "args": "",
                        "rationale": (
                            f"Initial status check. Recalled: "
                            f"{len(recalled_items)} items."
                        ),
                        "confidence": 0.9,
                        "truth_annotation": {
                            "truth": "VERIFIED"
                            if any(
                                item["truth"] == "VERIFIED" and item["source"] == "TOOL"
                                for item in recalled_items
                            )
                            else "HEURISTIC",
                            "evidence": {
                                "type": "inference",
                                "ref": None,
                                "details": "Deterministic plan generation",
                            },
                        },
                    },
                    {
                        "tool": "probe.last",
                        "args": "",
                        "rationale": "Probe recent activity for completeness.",
                        "confidence": 0.8,
                        "truth_annotation": {
                            "truth": "HEURISTIC",
                            "evidence": {
                                "type": "inference",
                                "ref": None,
                                "details": "Deterministic plan generation",
                            },
                        },
                    },
                ]

                # Apply belief policy to filter and boost candidate tools
                filtered_proposals, policy_notes = apply_belief_policy(
                    candidate_proposals,
                    objective_class,
                    self.prior_state.get("beliefs", {}),
                )

                # Build steps from filtered proposals
                steps = []
                for idx, proposal in enumerate(filtered_proposals):
                    steps.append(
                        {
                            "step_id": str(uuid.uuid4()),
                            "index": idx,
                            "tool": proposal["tool"],
                            "args": proposal.get("args", ""),
                            "rationale": proposal.get("rationale", ""),
                            "truth_annotation": proposal.get("truth_annotation", {}),
                            "status": "PENDING",
                            "result_summary": {},
                            "executed_at": None,
                            "evaluator": {
                                "score": proposal.get("confidence", 0.5),
                                "outcome": "SUCCESS",
                            },
                            "policy_adjustment": proposal.get("policy_adjustment", 0.0),
                            "policy_reason": proposal.get("policy_reason", ""),
                            "policy_belief_id": proposal.get("policy_belief_id", None),
                        }
                    )

                plan = {
                    "schema_version": 1,
                    "plan_id": str(uuid.uuid4()),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "objective": objective,
                    "objective_class": objective_class,
                    "steps": steps,
                    "current_index": 0,
                    "status": "ACTIVE",
                    "checkpoints": [],
                    "policy_notes": policy_notes,
                }

                # Apply belief-based plan revision before execution
                plan_state = {"plans": {"active": [plan], "history": []}}
                apply_plan_revision(plan_state, self.prior_state.get("beliefs", {}))

                # Enforce PROVED claims
                if self.theorem_index:
                    for step in plan["steps"]:
                        if "truth_annotation" in step:
                            step["truth_annotation"] = enforce_truth_annotation(
                                step["truth_annotation"], self.theorem_index
                            )

                # Save plan to state
                if self.prior_state is None:
                    self.prior_state = {"plans": {"active": [], "history": []}}
                self.prior_state["plans"] = plan_state["plans"]

                return {
                    "plan": plan_state["plans"]["active"][0],
                    "notes": {
                        "method": "conservative_status_plan",
                        "policy": policy_notes,
                        "theorem_index_hash": self.theorem_index["index_hash"]
                        if self.theorem_index
                        else None,
                    },
                    "errors": [],
                }
            else:
                # No plan for other objectives yet
                return {
                    "plan": None,
                    "notes": {"reason": "Objective not supported for planning"},
                    "errors": [],
                }
        except Exception as e:
            return {
                "plan": None,
                "notes": {"reason": "Plan generation error"},
                "errors": [str(e)],
            }

    def persist(self) -> None:
        """Persist SCP cognitive state with schema validation."""
        if not self.available:
            return
        try:
            # Initialize prior_state if None
            if self.prior_state is None:
                self.prior_state = {
                    "schema_version": 1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "version": 0,
                    "prev_hash": "",
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
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "beliefs_version": 0,
                        "prev_hash": "",
                        "state_hash": "",
                        "items": [],
                    },
                }

            # Enforce max sizes
            self.observations = self.observations[-50:]  # Keep last 50
            self.last_proposals = self.last_proposals[-50:]
            self.last_tool_results = self.last_tool_results[-50:]
            self.truth_events = self.truth_events[-50:]

            # Apply UWM decay and promotion
            decay_and_promote(self.prior_state)

            # Consolidate beliefs
            run_id = str(uuid.uuid4())
            beliefs_container = consolidate_beliefs(
                self.prior_state,
                run_id=run_id,
                mode=self.mode,
                stub_mode=isinstance(self.scp_nexus, StubSCPNexus),
            )
            self.prior_state["beliefs"] = beliefs_container

            # Apply plan revision based on beliefs
            apply_plan_revision(self.prior_state, beliefs_container)

            cognitive_status = {"message": "default status"}

            # Handle stub vs real SCP
            if isinstance(self.scp_nexus, StubSCPNexus):
                cognitive_status = {
                    "message": "stub cognitive status",
                    "stub": True,
                    "mode": self.mode,
                    "last_error": self.last_error,
                    "scp_state_valid": self.scp_state_valid,
                    "scp_recovery_mode": self.scp_recovery_mode,
                    "scp_state_validation_error": self.scp_state_validation_error,
                }
            else:
                try:
                    from Logos_AGI.System_Operations_Protocol.infrastructure.agent_system.base_nexus import (
                        AgentRequest,
                    )
                except ImportError:

                    class AgentRequest:
                        def __init__(self, request_id, operation, payload):
                            self.request_id = request_id
                            self.operation = operation
                            self.payload = payload

                try:
                    request = AgentRequest(
                        str(uuid.uuid4()), "get_cognitive_status", {}
                    )
                    response = asyncio.run(
                        self.scp_nexus.process_agent_request(request)
                    )
                    cognitive_status = (
                        response.data if response.success else {"error": response.error}
                    )
                    cognitive_status["mode"] = self.mode
                    cognitive_status["last_error"] = self.last_error
                    cognitive_status["scp_state_valid"] = self.scp_state_valid
                    cognitive_status["scp_recovery_mode"] = self.scp_recovery_mode
                    cognitive_status["scp_state_validation_error"] = (
                        self.scp_state_validation_error
                    )
                except (RuntimeError, TypeError, ValueError, OSError) as e:
                    cognitive_status = {
                        "error": str(e),
                        "mode": self.mode,
                        "last_error": self.last_error,
                        "scp_state_valid": self.scp_state_valid,
                        "scp_recovery_mode": self.scp_recovery_mode,
                        "scp_state_validation_error": self.scp_state_validation_error,
                    }

            # Build state dict
            temp_state = {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "version": (self.prior_state.get("version", 0) + 1)
                if self.prior_state
                else 1,
                "prev_hash": self.prior_state.get("state_hash")
                if self.prior_state
                else None,
                "observations": self.observations,
                "last_proposals": self.last_proposals,
                "last_tool_results": self.last_tool_results,
                "truth_events": self.truth_events,
                "arp_status": self.arp_reasoner.status() if self.arp_reasoner else {},
                "scp_status": cognitive_status,
                "working_memory": self.prior_state.get(
                    "working_memory", init_working_memory({})
                ),
                "plans": self.prior_state.get("plans", {"active": [], "history": []}),
                "beliefs": self.prior_state.get("beliefs", {"items": []}),
            }

            # Compute state_hash
            temp_state["state_hash"] = canonical_json_hash(temp_state)

            # Validate before writing
            if not isinstance(self.scp_nexus, StubSCPNexus):
                validate_scp_state(temp_state)

            with open(self.persisted_path, "w") as f:
                json.dump(temp_state, f, indent=2)
            if self.persisted_path != self.compat_scp_state_path:
                with open(self.compat_scp_state_path, "w") as f:
                    json.dump(temp_state, f, indent=2)

            # Update prior_state for next run
            self.prior_state = temp_state
        except Exception as e:
            self.last_error = f"Persist error: {e}"

    def health(self) -> Dict[str, Any]:
        """Return health status."""
        wm = self.working_memory or {}
        return {
            "available": self.available,
            "last_error": self.last_error,
            "arp_online": self.arp_reasoner.online if self.arp_reasoner else False,
            "observations_count": len(self.observations),
            "persisted_path": str(self.persisted_path),
            "uwm_short_term_count": len(wm.get("short_term", [])),
            "uwm_long_term_count": len(wm.get("long_term", [])),
        }
