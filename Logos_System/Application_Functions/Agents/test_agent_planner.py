# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Lightweight tests for AlignmentAwarePlanner agenda and planning behavior."""

from __future__ import annotations

import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Protopraxis.agent_planner import (
    AlignmentAwarePlanner,
    AlignmentRequiredError,
    Observation,
    append_digest_to_log,
    snapshot_digest_log,
    latest_digest_archive,
)


class AlignmentAwarePlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = AlignmentAwarePlanner()
        self.symbolic_obs = Observation(
            identifier="concept:triune-unity",
            modality="symbolic",
            payload={"title": "Triune Unity"},
        )
        self.training_obs = Observation(
            identifier="training:data/training/PXL/Protopraxic_Logic_PXL.md",
            modality="text",
            payload={},
        )
        self.telemetry_obs = Observation(
            identifier="telemetry:runtime-resource",
            modality="telemetry",
            payload={
                "peak_cpu_slots": 2,
                "average_cpu_slots": 1.5,
                "latest_cpu_slots": 2,
                "cpu_trend": "steady",
                "cpu_anomaly": False,
                "peak_memory_mb": 512,
                "average_memory_mb": 256,
                "latest_memory_mb": 512,
                "memory_trend": "steady",
                "memory_anomaly": False,
                "peak_disk_mb": 1024,
                "sampled_events": 3,
                "last_event": "2025-12-23T00:00:00Z",
            },
        )

    def test_plan_requires_alignment(self) -> None:
        with self.assertRaises(AlignmentRequiredError):
            _ = self.planner.plan([self.symbolic_obs])

    def test_agenda_and_actions_after_alignment(self) -> None:
        self.planner.mark_alignment_verified(True)
        actions = self.planner.plan(
            [self.symbolic_obs, self.training_obs, self.telemetry_obs]
        )
        agenda = self.planner.agenda()
        self.assertTrue(actions)
        self.assertEqual(len(agenda), 3)
        agenda_modalities = {goal.name for goal in agenda}
        self.assertIn("mission:symbolic", agenda_modalities)
        self.assertIn("mission:text", agenda_modalities)
        self.assertIn("mission:telemetry", agenda_modalities)
        action_names = {action.name for action in actions}
        self.assertIn("prepare_agenda_snapshot", action_names)
        self.assertIn("synthesize_transfer_digest", action_names)

    def test_latest_digest_contains_structured_payload(self) -> None:
        self.planner.mark_alignment_verified(True)
        self.planner.plan(
            [
                self.symbolic_obs,
                self.training_obs,
                self.telemetry_obs,
            ]
        )
        digest = self.planner.latest_digest()
        self.assertIsNotNone(digest)
        digest_data = digest or {}
        self.assertIn("generated_at", digest_data)
        self.assertIn("actions", digest_data)
        self.assertGreater(len(digest_data["actions"]), 0)
        self.assertEqual(digest_data["observation_counts"].get("symbolic"), 1)
        telemetry = digest_data.get("telemetry", {})
        self.assertEqual(telemetry.get("sampled_events"), 3)
        digest_data["actions"][0]["name"] = "mutated"
        fresh = self.planner.latest_digest()
        self.assertIsNotNone(fresh)
        fresh_data = fresh or {}
        self.assertNotEqual(fresh_data["actions"][0]["name"], "mutated")

    def test_append_and_snapshot_preserves_log(self) -> None:
        self.planner.mark_alignment_verified(True)
        self.planner.plan(
            [
                self.symbolic_obs,
                self.training_obs,
                self.telemetry_obs,
            ]
        )
        digest = self.planner.latest_digest()
        self.assertIsNotNone(digest)

        with TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            log_path = tmp_root / "planner_digests.jsonl"
            archive_dir = tmp_root / "archives"
            append_digest_to_log(digest or {}, log_path)
            original = log_path.read_text(encoding="utf-8")

            archive_path = snapshot_digest_log(log_path, archive_dir)
            self.assertIsNotNone(archive_path)
            self.assertTrue(archive_path)
            self.assertTrue(log_path.exists())
            self.assertTrue(archive_path and archive_path.exists())

            with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
                archived = handle.read()
            self.assertEqual(archived, original)
            self.assertEqual(log_path.read_text(encoding="utf-8"), original)

            info = latest_digest_archive(archive_dir)
            self.assertIsNotNone(info)
            self.assertEqual(info and info.get("path"), archive_path)
            self.assertEqual(
                info and info.get("size_bytes"),
                archive_path.stat().st_size,
            )


if __name__ == "__main__":
    unittest.main()
