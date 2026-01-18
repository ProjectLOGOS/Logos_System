# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Tests for observation ingestor health reporting and simulation ingestion."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

from Logos_AGI.Logos_Agent.creator_packet.reflection_builder.perception_ingestors import (
    Observation,
    ObservationBroker,
    RuntimeTelemetryIngestor,
    SIMULATION_LOG_DIR,
    SimulationEventIngestor,
)


class _DummyIngestor:
    """Simple ingestor stub for health report verification."""

    def __init__(self, name: str, available: bool, detail: str) -> None:
        self.dataset_path = Path(name)
        self._available = available
        self._detail = detail

    def available(self) -> bool:
        return self._available

    def collect(self) -> List[Observation]:
        return []

    def status_detail(self) -> str:
        return self._detail

    def trace_digest(self) -> Dict[str, Any]:
        return {
            "name": self.dataset_path.name,
            "path": str(self.dataset_path),
            "available": self._available,
            "status": self._detail,
            "updated_at": None,
            "extra": {},
        }


def _write_events(path: Path, count: int = 3) -> None:
    events: List[Dict[str, object]] = []
    for tick in range(count):
        events.append(
            {
                "tick": tick,
                "agent_state": {"x": tick, "y": 0},
                "environment_state": {"boundary": 4},
                "reward": 1.0 if tick == count - 1 else 0.0,
                "done": tick == count - 1,
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


class SimulationEventIngestorTests(unittest.TestCase):
    def test_collects_events_into_observation(self) -> None:
        log_path = SIMULATION_LOG_DIR / "test_events_ingestor.jsonl"
        try:
            _write_events(log_path, count=5)

            ingestor = SimulationEventIngestor(dataset_path=log_path, sample_size=10)
            observations = ingestor.collect()

            self.assertEqual(len(observations), 1)
            observation = observations[0]
            self.assertEqual(observation.modality, "simulation")
            self.assertEqual(observation.payload["sampled_events"], 5)
            self.assertTrue(observation.payload["terminated"])
            self.assertEqual(observation.payload["last_tick"], 4)

            detail = ingestor.status_detail()
            self.assertIn("bytes logged", detail)
        finally:
            if log_path.exists():
                log_path.unlink()


class ObservationBrokerHealthTests(unittest.TestCase):
    def test_health_report_returns_status_objects(self) -> None:
        ready = _DummyIngestor("ready.json", True, "dataset ready")
        waiting = _DummyIngestor("waiting.json", False, "awaiting operator upload")

        broker = ObservationBroker([ready, waiting])
        report = broker.health_report()

        self.assertEqual(len(report), 2)
        self.assertEqual(report[0].name, "ready.json")
        self.assertTrue(report[0].available)
        self.assertEqual(report[0].detail, "dataset ready")

        self.assertEqual(report[1].name, "waiting.json")
        self.assertFalse(report[1].available)
        self.assertEqual(report[1].detail, "awaiting operator upload")

    def test_trace_digest_returns_entries(self) -> None:
        ready = _DummyIngestor("ready.json", True, "dataset ready")
        waiting = _DummyIngestor("waiting.json", False, "awaiting operator upload")

        broker = ObservationBroker([ready, waiting])
        digest = broker.trace_digest()

        self.assertEqual(len(digest), 2)
        self.assertEqual(digest[0]["name"], "ready.json")
        self.assertTrue(digest[0]["available"])
        self.assertEqual(digest[1]["status"], "awaiting operator upload")


class RuntimeTelemetryIngestorTests(unittest.TestCase):
    def setUp(self) -> None:
        base_dir = Path(__file__).resolve().parent
        self.fixture_path = base_dir / "fixtures" / "resource_events_sample.jsonl"

    def test_collect_summarizes_resource_statistics(self) -> None:
        ingestor = RuntimeTelemetryIngestor(
            dataset_path=self.fixture_path,
            sample_size=10,
        )
        observations = ingestor.collect()

        self.assertEqual(len(observations), 1)
        observation = observations[0]
        payload = observation.payload

        self.assertEqual(observation.identifier, "telemetry:runtime-resource")
        self.assertEqual(observation.modality, "telemetry")
        self.assertEqual(payload["sampled_events"], 2)
        self.assertEqual(payload["peak_cpu_slots"], 4)
        self.assertAlmostEqual(payload["average_cpu_slots"], 2.5)
        self.assertEqual(payload["peak_memory_mb"], 768)
        self.assertEqual(payload["peak_disk_mb"], 1200)
        self.assertEqual(payload["cpu_trend"], "rising")
        self.assertEqual(payload["memory_trend"], "rising")
        self.assertTrue(payload["cpu_anomaly"])
        self.assertTrue(payload["memory_anomaly"])

        last_timestamp = datetime.fromtimestamp(1700003600.0, timezone.utc)
        expected_last = last_timestamp.isoformat().replace("+00:00", "Z")
        self.assertEqual(payload["last_event"], expected_last)

    def test_trace_digest_exposes_summary_metadata(self) -> None:
        ingestor = RuntimeTelemetryIngestor(
            dataset_path=self.fixture_path,
            sample_size=10,
        )
        digest = ingestor.trace_digest()

        self.assertEqual(digest["name"], self.fixture_path.name)
        self.assertTrue(digest["available"])
        self.assertIn("sampled_events", digest["extra"])
        self.assertEqual(digest["extra"]["cpu_trend"], "rising")

    def test_status_detail_reports_last_update(self) -> None:
        ingestor = RuntimeTelemetryIngestor(
            dataset_path=self.fixture_path,
            sample_size=10,
        )
        detail = ingestor.status_detail()
        self.assertIn("log updated", detail)


if __name__ == "__main__":
    unittest.main()
