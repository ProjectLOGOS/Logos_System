# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Governed tool invention pipeline driven by tool optimizer outputs."""

from __future__ import annotations

import json
import hashlib
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from System_Operations_Protocol.code_generator.development_environment import (
    CodeGenerationRequest,
    SOPCodeEnvironment,
)

MAX_TOOLS_PER_CYCLE = 3
GAP_DEFINITIONS: Sequence[Dict[str, Any]] = (
    {
        "gap_id": "tool_chain_executor",
        "title": "Deterministic Tool Chain Executor",
        "recommended_tool_type": "orchestration",
        "target_module": "logos_core.tools.tool_chain_executor",
        "risk_tag": "governed",
        "acceptance_tests": [
            "Execute sequence of three callables and confirm deterministic ordering",
            "Record telemetry for each step outcome and surface aggregated status",
        ],
    },
    {
        "gap_id": "io_normalizer",
        "title": "Standardized Tool IO Normalizer",
        "recommended_tool_type": "analysis",
        "target_module": "logos_core.tools.io_normalizer",
        "risk_tag": "safe",
        "acceptance_tests": [
            "Normalize heterogeneous tool payloads into canonical dict form",
            "Strip unsafe keys while preserving audit metadata",
        ],
    },
    {
        "gap_id": "uwm_packager",
        "title": "Cross-Domain UWM Packager",
        "recommended_tool_type": "synthesis",
        "target_module": "logos_core.tools.uwm_packager",
        "risk_tag": "safe",
        "acceptance_tests": [
            "Package tool outputs into UWM-compatible artifact records",
            "Attach provenance hashes for downstream verification",
        ],
    },
    {
        "gap_id": "regression_checker",
        "title": "Cycle Regression Checker",
        "recommended_tool_type": "analysis",
        "target_module": "logos_core.tools.regression_checker",
        "risk_tag": "safe",
        "acceptance_tests": [
            "Compare baseline and candidate tool outputs and flag deltas",
            "Emit structured report with pass/fail summary",
        ],
    },
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _file_hash(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _relative_ref(repo_root: Path, path: Path, digest: Optional[str]) -> Optional[str]:
    if not digest:
        return None
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        base = rel.as_posix()
    except ValueError:
        base = path.as_posix()
    return f"{base}#{digest}"


class ToolInventionManager:
    """Derives new instrumental tools based on optimizer gaps."""

    def __init__(
        self,
        identity_path: Path,
        optimizer_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        proposal_only: Optional[bool] = None,
    ) -> None:
        self.identity_path = identity_path
        self.repo_root = self._repo_root_from_identity(identity_path)
        self.optimizer_dir = optimizer_dir or (self.repo_root / "state" / "tool_optimizer")
        self.output_dir = output_dir or (self.repo_root / "state" / "tool_invention")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        env_toggle = os.getenv("LOGOS_TOOL_INVENTION_DEPLOY", "0")
        default_proposal_only = env_toggle != "1"
        self.proposal_only = default_proposal_only if proposal_only is None else bool(proposal_only)
        self.proposals_dir = self.output_dir / "proposals"
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        self.code_env = SOPCodeEnvironment()

    def run(self) -> Dict[str, Any]:
        identity = _read_json(self.identity_path)
        if not identity:
            raise FileNotFoundError("Agent identity not found; cannot derive tools")
        if not bool(identity.get("mission", {}).get("allow_enhancements", False)):
            raise RuntimeError("Tool invention requires allow_enhancements==true")

        registry_path = self.optimizer_dir / "tool_registry.json"
        profiles_path = self.optimizer_dir / "tool_chain_profiles.json"
        report_path = self.optimizer_dir / "tool_optimizer_report.json"

        registry = _read_json(registry_path) or []
        profiles = _read_json(profiles_path) or []
        optimizer_report = _read_json(report_path) or {}
        if optimizer_report.get("completion_status") != "complete":
            raise RuntimeError("Tool optimizer must complete before tool invention")

        registry_hash = _file_hash(registry_path)
        profiles_hash = _file_hash(profiles_path)

        selected_gaps = self._select_gaps(registry, profiles, optimizer_report)
        gap_report_path = self.output_dir / "tool_gap_report.json"
        gap_payload = {
            "generated_utc": _now_iso(),
            "inputs": {
                "registry_hash": registry_hash,
                "profiles_hash": profiles_hash,
            },
            "gaps": selected_gaps,
            "proposal_only": self.proposal_only,
        }
        gap_report_path.write_text(json.dumps(gap_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        gap_report_hash = _file_hash(gap_report_path)

        generation_results = self._generate_tools(selected_gaps)
        if self.proposal_only:
            deployed_count = 0
        else:
            deployed_count = sum(1 for item in generation_results if item.get("deployed"))
        invention_report_path = self.output_dir / "tool_invention_report.json"
        invention_payload = {
            "timestamp_utc": _now_iso(),
            "proposal_only": self.proposal_only,
            "proposals_dir": self._relative_path(self.proposals_dir),
            "proposed_gaps": selected_gaps,
            "generated": generation_results,
            "deployed_count": deployed_count,
            "new_tool_entry_ids": [item["entry_id"] for item in generation_results if item.get("entry_id")],
        }
        invention_report_path.write_text(
            json.dumps(invention_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        invention_report_hash = _file_hash(invention_report_path)

        if not selected_gaps:
            ok = True
        else:
            ok = all(item.get("success") for item in generation_results)

        return {
            "ok": ok,
            "proposal_only": self.proposal_only,
            "gap_report_path": _relative_ref(self.repo_root, gap_report_path, gap_report_hash),
            "invention_report_path": _relative_ref(self.repo_root, invention_report_path, invention_report_hash),
            "generated": generation_results,
            "deployed_count": deployed_count,
        }

    def _select_gaps(
        self,
        registry: Sequence[Dict[str, Any]],
        profiles: Sequence[Dict[str, Any]],
        optimizer_report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        registry_names = {
            f"{item.get('module_path', '')}.{item.get('symbol', '')}".lower()
            for item in registry
            if isinstance(item, dict)
        }
        registry_symbols = {str(item.get("name", "")).lower() for item in registry if isinstance(item, dict)}
        step_counter = Counter()
        profile_names = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            profile_names.append(str(profile.get("name", "")).lower())
            for step in profile.get("ordered_steps", []) or []:
                step_counter[str(step).lower()] += 1
        max_steps = 0
        for profile in profiles:
            if isinstance(profile, dict):
                max_steps = max(max_steps, len(profile.get("ordered_steps", []) or []))
        anomalies = optimizer_report.get("anomalies", []) or []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in GAP_DEFINITIONS:
            gap_id = candidate["gap_id"]
            score = self._score_gap(
                gap_id,
                registry_names,
                registry_symbols,
                step_counter,
                profile_names,
                max_steps,
                anomalies,
            )
            if score <= 0:
                continue
            risk_priority = 0 if candidate["risk_tag"] == "safe" else 1
            rationale = self._gap_rationale(
                gap_id,
                score,
                max_steps,
                step_counter,
                anomalies,
            )
            entry = dict(candidate)
            entry.update(
                {
                    "rationale": rationale,
                    "score": score,
                }
            )
            scored.append(((score, -risk_priority, gap_id), entry))

        scored.sort(key=lambda item: (-item[0][0], item[0][1], item[0][2]))
        selections = [entry for _key, entry in scored[:MAX_TOOLS_PER_CYCLE]]
        for entry in selections:
            entry.pop("score", None)
        return selections

    def _score_gap(
        self,
        gap_id: str,
        registry_names: Sequence[str],
        registry_symbols: Sequence[str],
        step_counter: Counter,
        profile_names: Sequence[str],
        max_steps: int,
        anomalies: Sequence[str],
    ) -> float:
        registry_text = " ".join(registry_names) + " " + " ".join(registry_symbols)
        if gap_id == "tool_chain_executor":
            if "executor" in registry_text:
                return 0.0
            if max_steps < 3:
                return 0.0
            return float(max_steps)
        if gap_id == "io_normalizer":
            if "normalize" in registry_text or "io_normalizer" in registry_text:
                return 0.0
            io_hits = step_counter.get("fs.read", 0) + step_counter.get("sandbox.write", 0)
            return float(io_hits)
        if gap_id == "uwm_packager":
            if "packager" in registry_text or "uwm" in registry_text:
                return 0.0
            count = sum(1 for name in profile_names if "report" in name or "capability" in name)
            return float(count)
        if gap_id == "regression_checker":
            if "regression" in registry_text or "delta" in registry_text or "compare" in registry_text:
                return 0.0
            anomaly_bonus = sum(1 for item in anomalies if "hash" in str(item) or "mismatch" in str(item))
            return float(anomaly_bonus + step_counter.get("probe.last", 0))
        return 0.0

    def _gap_rationale(
        self,
        gap_id: str,
        score: float,
        max_steps: int,
        step_counter: Counter,
        anomalies: Sequence[str],
    ) -> str:
        if gap_id == "tool_chain_executor":
            return (
                f"Observed chain profiles up to {max_steps} steps without executor tooling; score={int(score)}"
            )
        if gap_id == "io_normalizer":
            io_hits = step_counter.get("fs.read", 0) + step_counter.get("sandbox.write", 0)
            return f"High IO interaction frequency ({io_hits}) without normalizer; score={int(score)}"
        if gap_id == "uwm_packager":
            return "Capability reporting profiles lack packager to emit UWM-ready artifacts"
        if gap_id == "regression_checker":
            anomaly_excerpt = anomalies[0] if anomalies else "no anomaly details"
            return f"Optimizer anomalies ({anomaly_excerpt}) suggest regression tracking gap"
        return ""

    def _generate_tools(self, gaps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for index, gap in enumerate(gaps[:MAX_TOOLS_PER_CYCLE]):
            request = self._build_request(gap, index)
            try:
                if self.proposal_only:
                    generation = self.code_env.generate_code_draft(
                        request,
                        allow_enhancements=True,
                        proposal_reason="tool_invention_proposal_only",
                    )
                else:
                    generation = self.code_env.generate_code(request, allow_enhancements=True)
            except Exception as exc:
                results.append(
                    {
                        "gap_id": gap["gap_id"],
                        "success": False,
                        "error": str(exc),
                        "entry_id": None,
                        "deployed": False,
                        "deployment_path": None,
                        "stage_ok": False,
                        "proposal_only": self.proposal_only,
                    }
                )
                continue
            if self.proposal_only:
                draft_path = self._write_proposal(gap, request, generation)
                draft_hash = _file_hash(draft_path) if draft_path else None
                results.append(
                    {
                        "gap_id": gap["gap_id"],
                        "entry_id": generation.get("entry_id"),
                        "success": bool(generation.get("staged")),
                        "deployed": False,
                        "deployment_path": None,
                        "stage_ok": bool(generation.get("staged")),
                        "policy_class": generation.get("policy_class"),
                        "policy_reasoning": generation.get("policy_reasoning"),
                        "proposal_only": True,
                        "proposal_path": _relative_ref(self.repo_root, draft_path, draft_hash) if draft_path else None,
                        "proposal_hash": draft_hash,
                        "stage_errors": generation.get("stage_errors", []),
                    }
                )
            else:
                results.append(
                    {
                        "gap_id": gap["gap_id"],
                        "entry_id": generation.get("entry_id"),
                        "success": bool(generation.get("success")),
                        "deployed": bool(generation.get("deployed")),
                        "deployment_path": generation.get("deployment_path"),
                        "stage_ok": bool(generation.get("staged")),
                        "policy_class": generation.get("policy_class"),
                        "proposal_only": False,
                    }
                )
        return results

    def _write_proposal(
        self,
        gap: Dict[str, Any],
        request: CodeGenerationRequest,
        generation: Dict[str, Any],
    ) -> Optional[Path]:
        code = generation.get("code")
        if not code:
            return None

        meta_lines = [
            "# Tool invention draft (proposal-only)",
            f"# improvement_id: {request.improvement_id}",
            f"# gap_id: {gap.get('gap_id')}",
            f"# target_module: {request.target_module}",
            "# origin: tool_optimizer",
            f"# policy_class: {generation.get('policy_class')}",
            f"# stage_ok: {generation.get('staged')}",
            f"# policy_reasoning: {generation.get('policy_reasoning')}",
            f"# entry_id: {generation.get('entry_id')}",
            f"# timestamp_utc: {_now_iso()}",
        ]

        filename = f"{request.improvement_id}.py"
        path = self.proposals_dir / filename
        path.write_text("\n".join(meta_lines) + "\n\n" + str(code), encoding="utf-8")
        return path

    def _build_request(self, gap: Dict[str, Any], index: int) -> CodeGenerationRequest:
        timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        improvement_id = f"tool_invention_{gap['gap_id']}_{timestamp_token}_{index}"
        description = (
            f"Derive {gap['title']} to satisfy tool optimizer gap {gap['gap_id']}"
        )
        requirements = {
            "gap_category": gap["gap_id"],
            "gap_id": gap["gap_id"],
            "recommended_tool_type": gap["recommended_tool_type"],
            "risk_tag": gap["risk_tag"],
            "acceptance_tests": gap["acceptance_tests"],
            "origin": "tool_optimizer",
        }
        constraints = {
            "allow_enhancements": True,
            "max_tools_per_cycle": MAX_TOOLS_PER_CYCLE,
        }
        test_cases = [
            {
                "name": "import_check",
                "description": "Generated module imports without error",
                "expected": "success",
            },
            {
                "name": "api_contract",
                "description": "Primary class exposes callable interface",
                "expected": "callable",
            },
        ]
        return CodeGenerationRequest(
            improvement_id=improvement_id,
            description=description,
            target_module=gap["target_module"],
            improvement_type="module",
            requirements=requirements,
            constraints=constraints,
            test_cases=test_cases,
        )

    @staticmethod
    def _repo_root_from_identity(identity_path: Path) -> Path:
        candidate = identity_path.resolve()
        if candidate.exists():
            return candidate.parent.parent
        return Path(__file__).resolve().parents[4]

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def run_tool_invention(
    identity_path: str | Path = "state/agent_identity.json",
    optimizer_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    proposal_only: bool | None = None,
) -> Dict[str, Any]:
    manager = ToolInventionManager(
        Path(identity_path),
        Path(optimizer_dir) if optimizer_dir else None,
        Path(output_dir) if output_dir else None,
        proposal_only,
    )
    return manager.run()
