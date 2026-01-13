#!/usr/bin/env python3
"""Smoke tests for plan history persistence and chaining."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import scripts.system_stack_tbd.need_to_distribute._bootstrap as _bootstrap  # noqa: F401

from logos.plan_scoring import update_plan_history
from LOGOS_SYSTEM.System_Stack.Protocol_Resources.schemas import validate_plan_history_scored


class PlanHistoryUpdateSmokeTests(unittest.TestCase):
    def test_history_bounded_and_chained(self) -> None:
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "scp_state.json"
            for idx in range(25):
                update_plan_history(
                    state_path,
                    "sig-A",
                    float(idx) / 10.0,
                    f"report-hash-{idx}",
                    f"ledger-hash-{idx}",
                )

            data = json.loads(state_path.read_text(encoding="utf-8"))
            container = data.get("plans", {}).get("history_scored")
            self.assertIsNotNone(container)
            validate_plan_history_scored(container)

            entries = container["entries_by_signature"]["sig-A"]
            self.assertEqual(len(entries), 20)
            # Oldest entry reflects deterministic eviction of earliest scores
            self.assertAlmostEqual(entries[0]["score"], 0.5)
            self.assertAlmostEqual(entries[-1]["score"], 2.4)

            # Hash chaining must be intact
            for prev_entry, entry in zip(entries, entries[1:]):
                self.assertEqual(entry.get("prev_hash"), prev_entry.get("entry_hash"))

            # State hash should be non-empty hex
            state_hash = data.get("state_hash", "")
            self.assertEqual(len(state_hash), 64)
            self.assertTrue(all(ch in "0123456789abcdef" for ch in state_hash))


if __name__ == "__main__":  # pragma: no cover - cli
    unittest.main()
