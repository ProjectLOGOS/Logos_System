from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Import the shared governance dataclasses created with ETGC.
# NOTE: Use the fully-qualified package path to avoid import ambiguity.
from Logos_System.System_Stack.Logos_Protocol.Runtime_Operations.Governance.Constraint_Result import ConstraintResult
from Logos_System.System_Stack.Logos_Protocol.Runtime_Operations.Governance.Constraint_Context import ConstraintContext


@dataclass(frozen=True)
class Triune_Principal_Report:
    """
    Structured report emitted by the Triune principal validator.

    - sign_ok/mind_ok/bridge_ok are independent checks.
    - mesh_ok is the commutation/holism check over the three.
    - tags carries per-check diagnostics and future audit hooks.
    """

    sign_ok: bool
    mind_ok: bool
    bridge_ok: bool
    mesh_ok: bool
    tags: Dict[str, Any]


class Triune_Principal_Validator:
    """
    Application-neutral constraint layer for:
      SIGN / MIND / BRIDGE independent checks
    followed by:
      MESH commutation/holism check.

    Intended usage:
      - Validate any payload (internal or external) before it is allowed to:
        * enter memory
        * influence planning
        * affect agent outputs
        * propagate across protocols

    Grounding sources (for later stub expansion):
      - Arguments/ION_Argument_Complete.txt
      - Arguments/MESH_Argument.txt
      - Arguments/Three Pillars Formalization Axiomatic and Computational Framework.md
      - Arguments/Three Pillars of Divine Necessity.md
    """

    @staticmethod
    def validate(payload: Dict[str, Any], *, context: ConstraintContext) -> ConstraintResult:
        """
        Returns ConstraintResult(ok=...) with tags describing the triune evaluation.
        FAIL-CLOSED by default if any principal fails or Mesh commutation fails.
        """
        tags: Dict[str, Any] = {
            "validator": "Triune_Principal_Validator",
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "source": context.source,
        }

        sign_ok, sign_msg, sign_tags = Triune_Principal_Validator._check_sign(payload, context)
        mind_ok, mind_msg, mind_tags = Triune_Principal_Validator._check_mind(payload, context)
        bridge_ok, bridge_msg, bridge_tags = Triune_Principal_Validator._check_bridge(payload, context)

        tags["sign"] = {"ok": sign_ok, "msg": sign_msg, "tags": sign_tags}
        tags["mind"] = {"ok": mind_ok, "msg": mind_msg, "tags": mind_tags}
        tags["bridge"] = {"ok": bridge_ok, "msg": bridge_msg, "tags": bridge_tags}

        # Mesh commutation must evaluate *after* independent checks.
        mesh_ok, mesh_msg, mesh_tags = Triune_Principal_Validator._check_mesh_commutation(
            payload=payload,
            context=context,
            sign=(sign_ok, sign_tags),
            mind=(mind_ok, mind_tags),
            bridge=(bridge_ok, bridge_tags),
        )
        tags["mesh"] = {"ok": mesh_ok, "msg": mesh_msg, "tags": mesh_tags}

        report = Triune_Principal_Report(
            sign_ok=sign_ok,
            mind_ok=mind_ok,
            bridge_ok=bridge_ok,
            mesh_ok=mesh_ok,
            tags=tags,
        )
        tags["triune_report"] = {
            "sign_ok": report.sign_ok,
            "mind_ok": report.mind_ok,
            "bridge_ok": report.bridge_ok,
            "mesh_ok": report.mesh_ok,
        }

        failed = []
        if not sign_ok:
            failed.append("SIGN")
        if not mind_ok:
            failed.append("MIND")
        if not bridge_ok:
            failed.append("BRIDGE")
        if sign_ok and mind_ok and bridge_ok and not mesh_ok:
            failed.append("MESH_COMMUTATION")

        if failed:
            return ConstraintResult(
                ok=False,
                reason=f"Triune principal failure: {', '.join(failed)}",
                tags=tags,
            )

        return ConstraintResult(
            ok=True,
            reason="Triune principal checks passed (SIGN/MIND/BRIDGE + MESH commutation).",
            tags=tags,
        )

    # ----------------------------
    # Independent principal checks
    # ----------------------------

    @staticmethod
    def _check_sign(payload: Dict[str, Any], context: ConstraintContext) -> Tuple[bool, str, Dict[str, Any]]:
        """
        SIGN (Simultaneous Interconnected Governing Nexus) check stub.

        Intended meaning (to implement after audit):
          - semantic/structural integrity
          - non-degenerate symbol grounding
          - stable schema (no malformed/ambiguous signal)
          - “nexus fit”: payload can be embedded into the LOGOS lattice without tearing
        """
        if payload is None or not isinstance(payload, dict):
            return False, "SIGN: payload must be a dict", {"error": "type"}
        # Minimal scaffolding: require any result field to exist (ETGC does this too; redundancy is ok).
        if "result" not in payload:
            return False, "SIGN: missing 'result'", {"missing": ["result"]}
        return True, "SIGN: stub-pass", {"stub": True}

    @staticmethod
    def _check_mind(payload: Dict[str, Any], context: ConstraintContext) -> Tuple[bool, str, Dict[str, Any]]:
        """
        MIND (Metaphysical Instantiative Necessity Driver) check stub.

        Intended meaning (to implement after audit):
          - intentionality / goal alignment constraints (not goodness—this is instantiation stability)
          - non-arbitrary generation: outputs must be derivable/traceable in principle
          - bounded creativity: avoids ungrounded leaps
        """
        if payload is None or not isinstance(payload, dict):
            return False, "MIND: payload must be a dict", {"error": "type"}
        return True, "MIND: stub-pass", {"stub": True}

    @staticmethod
    def _check_bridge(payload: Dict[str, Any], context: ConstraintContext) -> Tuple[bool, str, Dict[str, Any]]:
        """
        BRIDGE check stub.

        Intended meaning (to implement after audit):
          - cross-domain mapping integrity (E↔O bijection/commutation constraints)
          - preserves critical invariants across transforms
          - enforces “gap barrier” rules (e.g., privation barrier patterns)
        """
        if payload is None or not isinstance(payload, dict):
            return False, "BRIDGE: payload must be a dict", {"error": "type"}
        return True, "BRIDGE: stub-pass", {"stub": True}

    # ----------------------------
    # Mesh commutation / holism
    # ----------------------------

    @staticmethod
    def _check_mesh_commutation(
        *,
        payload: Dict[str, Any],
        context: ConstraintContext,
        sign: Tuple[bool, Dict[str, Any]],
        mind: Tuple[bool, Dict[str, Any]],
        bridge: Tuple[bool, Dict[str, Any]],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        MESH holism / commutation check.

        Policy:
          - If any principal failed, Mesh is not evaluated as "pass" (it is effectively failed).
          - If all principals passed, Mesh checks joint satisfiability:
              * tags/claims must not contradict
              * combined constraints must be simultaneously satisfiable
              * optionally: verify closure invariants (to be wired after audit)

        This is the "forced commutation layer":
          passed individually != passed collectively.
        """
        sign_ok, sign_tags = sign
        mind_ok, mind_tags = mind
        bridge_ok, bridge_tags = bridge

        if not (sign_ok and mind_ok and bridge_ok):
            return False, "MESH: prerequisite principal failure", {
                "prereq": {"sign_ok": sign_ok, "mind_ok": mind_ok, "bridge_ok": bridge_ok},
                "policy": "fail-closed",
            }

        # Stub joint checks: detect explicit contradiction markers if present.
        contradictions = []
        # Example: if any sub-check emits a tag indicating conflict.
        for name, tags in [("SIGN", sign_tags), ("MIND", mind_tags), ("BRIDGE", bridge_tags)]:
            if isinstance(tags, dict) and tags.get("contradiction") is True:
                contradictions.append(name)

        if contradictions:
            return False, "MESH: contradiction among principals", {"contradictions": contradictions}

        # Future: add commutation invariants derived from MESH argument files.
        return True, "MESH: stub-pass (joint satisfiable)", {"stub": True}
