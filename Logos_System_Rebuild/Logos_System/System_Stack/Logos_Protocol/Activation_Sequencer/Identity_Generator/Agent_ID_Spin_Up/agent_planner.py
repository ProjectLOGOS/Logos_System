# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Prototype planner producing safe action suggestions post-alignment."""

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # Import Observation lazily to avoid heavy startup cost during proofs.
    from plugins.perception_ingestors import Observation
except (ImportError, AttributeError):  # pragma: no cover - generated docs

    @dataclass
    class Observation:  # type: ignore
        identifier: str
        modality: str
        payload: Dict[str, object]


BASE_SAFE_ACTIONS: Dict[str, str] = {
    "generate_symbolic_digest": (
        "Summarize symbolic observations into audit-ready prose."
    ),
    "curate_training_brief": (
        "Incorporate operator-provided corpora with provenance hashes maintained."
    ),
    "review_runtime_telemetry": (
        "Inspect recent SOP telemetry for resource anomalies before acting."
    ),
    "prepare_agenda_snapshot": (
        "Capture current goal stack and mission objectives for operator review."
    ),
    "synthesize_transfer_digest": (
        "Highlight cross-domain motifs supporting abstraction and transfer tests."
    ),
    "refresh_alignment_snapshot": (
        "Re-run proof-gated alignment audit to confirm safety status."
    ),
    "prepare_planning_brief": (
        "Draft a planning brief linking observations to mission objectives."
    ),
}

MISSION_OBJECTIVES: Dict[str, str] = {
    "symbolic": "Maintain metaphysical audit coherence",
    "text": "Integrate operator-provided corpora into mission brief",
    "telemetry": "Stabilize resource footprint before action",
}


@dataclass
class PlannedAction:
    """Structured action recommendation produced by the planner."""

    name: str
    rationale: str
    sources: Sequence[str] = field(default_factory=tuple)


@dataclass
class Goal:
    """Agenda item representing a mission objective to satisfy."""

    name: str
    description: str
    priority: int
    sources: Sequence[str] = field(default_factory=tuple)


class AlignmentRequiredError(RuntimeError):
    """Raised when planning is attempted before alignment verification."""


class AlignmentAwarePlanner:
    """Minimal planner that only activates once alignment is confirmed."""

    def __init__(self, safe_action_catalog: Dict[str, str] | None = None) -> None:
        self.safe_action_catalog = dict(safe_action_catalog or BASE_SAFE_ACTIONS)
        self._alignment_verified = False
        self._agenda: List[Goal] = []
        self._last_digest: Optional[Dict[str, Any]] = None

    def mark_alignment_verified(self, verified: bool = True) -> None:
        """Record whether the proof-gated alignment checks have succeeded."""

        self._alignment_verified = bool(verified)

    def available_actions(self) -> List[str]:
        """Return the catalog of safe action identifiers."""

        return sorted(self.safe_action_catalog.keys())

    def agenda(self) -> List[Goal]:
        """Return a copy of the current agenda stack."""

        return list(self._agenda)

    def reset_agenda(self) -> None:
        """Clear the tracked agenda."""

        self._agenda.clear()

    def derive_goals(self, observations: Iterable[Observation]) -> List[Goal]:
        """Populate mission goals from categorized observations."""

        modality_buckets: Dict[str, List[str]] = {}
        for obs in observations:
            modality_buckets.setdefault(obs.modality, []).append(obs.identifier)

        derived: List[Goal] = []
        for modality, sources in modality_buckets.items():
            description = MISSION_OBJECTIVES.get(modality)
            if not description:
                description = "Document unclassified observations for operator triage"
            if modality == "telemetry":
                priority = 1
            elif modality == "symbolic":
                priority = 2
            else:
                priority = 3
            goal = Goal(
                name=f"mission:{modality}",
                description=description,
                priority=priority,
                sources=tuple(sources),
            )
            derived.append(goal)

        derived.sort(key=lambda goal: goal.priority)
        self._agenda = derived
        return self.agenda()

    def plan(self, observations: Iterable[Observation]) -> List[PlannedAction]:
        """Infer high-level actions from the provided observations."""

        if not self._alignment_verified:
            raise AlignmentRequiredError(
                "planner requires constructive LEM verification before activation"
            )

        observation_list = list(observations)
        goals = self.derive_goals(observation_list)
        planned: List[PlannedAction] = []
        collected_sources: List[str] = []
        training_sources: List[str] = []
        telemetry_sources: List[str] = []
        telemetry_payloads: List[Dict[str, object]] = []
        for obs in observation_list:
            collected_sources.append(obs.identifier)
            if obs.modality == "symbolic":
                rationale = (
                    "Connect symbolic concept '{title}' to the current mission audit."
                ).format(title=obs.payload.get("title", obs.identifier))
                planned.append(
                    PlannedAction(
                        name="generate_symbolic_digest",
                        rationale=rationale,
                        sources=[obs.identifier],
                    )
                )
            elif obs.modality == "text":
                training_sources.append(obs.identifier)
            elif obs.modality == "telemetry":
                telemetry_sources.append(obs.identifier)
                if isinstance(obs.payload, dict):
                    telemetry_payloads.append(obs.payload)

        if training_sources:
            planned.append(
                PlannedAction(
                    name="curate_training_brief",
                    rationale=(
                        "Synthesize operator-provided corpus entries into the "
                        "mission alignment narrative while preserving provenance."
                    ),
                    sources=tuple(training_sources),
                )
            )

        if telemetry_sources:
            peak_cpu = 0
            last_event = None
            sampled_events = None
            cpu_trend: Optional[str] = None
            memory_trend: Optional[str] = None
            cpu_anomaly = False
            memory_anomaly = False
            if telemetry_payloads:
                latest = telemetry_payloads[-1]
                peak_cpu = int(latest.get("peak_cpu_slots", 0))
                last_event = latest.get("last_event")
                sampled_events = latest.get("sampled_events")
                cpu_trend_value = latest.get("cpu_trend")
                cpu_trend = str(cpu_trend_value) if cpu_trend_value else None
                memory_trend_value = latest.get("memory_trend")
                memory_trend = str(memory_trend_value) if memory_trend_value else None
                cpu_anomaly = bool(latest.get("cpu_anomaly"))
                memory_anomaly = bool(latest.get("memory_anomaly"))
            rationale_parts = [
                "Review SOP runtime telemetry before executing new plans.",
            ]
            if peak_cpu:
                rationale_parts.append(f"Peak CPU slots observed: {peak_cpu}.")
            if last_event:
                rationale_parts.append(f"Last event timestamp: {last_event}.")
            if sampled_events:
                rationale_parts.append(f"Evaluated {sampled_events} recent samples.")
            if cpu_trend and cpu_trend != "steady":
                rationale_parts.append(
                    f"CPU trend indicates {cpu_trend} usage trajectory."
                )
            if memory_trend and memory_trend != "steady":
                rationale_parts.append(
                    f"Memory trend indicates {memory_trend} usage trajectory."
                )
            if cpu_anomaly:
                rationale_parts.append(
                    "CPU anomaly threshold exceeded; escalate resource review."
                )
            if memory_anomaly:
                rationale_parts.append(
                    "Memory anomaly threshold exceeded; validate sandbox constraints."
                )
            planned.append(
                PlannedAction(
                    name="review_runtime_telemetry",
                    rationale=" ".join(rationale_parts),
                    sources=tuple(telemetry_sources),
                )
            )

        if collected_sources:
            planned.append(
                PlannedAction(
                    name="refresh_alignment_snapshot",
                    rationale=(
                        "Ensure audit logs capture insights from new observations."
                    ),
                    sources=tuple(collected_sources),
                )
            )

        if goals:
            agenda_sources: List[str] = []
            for goal in goals:
                agenda_sources.extend(goal.sources)
            planned.append(
                PlannedAction(
                    name="prepare_agenda_snapshot",
                    rationale=(
                        "Summarize agenda stack so operators can sequence mission "
                        "objectives post-alignment."
                    ),
                    sources=tuple(agenda_sources),
                )
            )

        if training_sources or telemetry_sources:
            planned.append(
                PlannedAction(
                    name="synthesize_transfer_digest",
                    rationale=(
                        "Outline abstraction or transfer opportunities across "
                        "ingested modalities for follow-on testing."
                    ),
                    sources=tuple(collected_sources),
                )
            )

        if planned:
            planned.append(
                PlannedAction(
                    name="prepare_planning_brief",
                    rationale=(
                        "Summarize observation-driven actions for operator review."
                    ),
                    sources=tuple(collected_sources),
                )
            )

        self._last_digest = self._build_digest(
            observation_list,
            goals,
            planned,
            telemetry_payloads,
        )
        return planned

    def latest_digest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent planning digest, if available."""

        if self._last_digest is None:
            return None
        return deepcopy(self._last_digest)

    def _build_digest(
        self,
        observations: Sequence[Observation],
        goals: Sequence[Goal],
        actions: Sequence[PlannedAction],
        telemetry_payloads: Sequence[Dict[str, object]],
    ) -> Dict[str, Any]:
        """Construct a structured explanation payload for operators."""

        modality_counts: Dict[str, int] = {}
        for obs in observations:
            modality_counts[obs.modality] = modality_counts.get(obs.modality, 0) + 1

        telemetry_snapshot: Dict[str, Any] = {}
        if telemetry_payloads:
            telemetry_snapshot = {
                key: value
                for key, value in telemetry_payloads[-1].items()
                if key
                in {
                    "sampled_events",
                    "peak_cpu_slots",
                    "average_cpu_slots",
                    "latest_cpu_slots",
                    "cpu_trend",
                    "cpu_anomaly",
                    "peak_memory_mb",
                    "average_memory_mb",
                    "latest_memory_mb",
                    "memory_trend",
                    "memory_anomaly",
                    "peak_disk_mb",
                    "last_event",
                }
            }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "observation_counts": modality_counts,
            "agenda": [
                {
                    "name": goal.name,
                    "description": goal.description,
                    "priority": goal.priority,
                    "sources": list(goal.sources),
                }
                for goal in goals
            ],
            "actions": [
                {
                    "name": action.name,
                    "rationale": action.rationale,
                    "sources": list(action.sources),
                }
                for action in actions
            ],
            "telemetry": telemetry_snapshot,
        }


def append_digest_to_log(digest: Dict[str, Any], log_path: Path) -> None:
    """Append the provided digest to an append-only JSONL log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps(digest, separators=(",", ":"))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry + "\n")


def snapshot_digest_log(log_path: Path, archive_dir: Path) -> Optional[Path]:
    """Compress the current planner log into a timestamped archive."""

    if not log_path.exists() or log_path.stat().st_size == 0:
        return None

    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_dir / f"planner_digests_{timestamp}.jsonl.gz"

    with log_path.open("rb") as source, gzip.open(archive_path, "wb") as target:
        target.write(source.read())

    return archive_path


def latest_digest_archive(archive_dir: Path) -> Optional[Dict[str, Any]]:
    """Return metadata for the most recent planner digest archive."""

    if not archive_dir.exists():
        return None

    archives = [
        path
        for path in archive_dir.glob("planner_digests_*.jsonl.gz")
        if path.is_file()
    ]
    if not archives:
        return None

    archives.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    latest = archives[0]
    stats = latest.stat()
    return {
        "path": latest,
        "size_bytes": stats.st_size,
        "modified_at": datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat(),
    }
