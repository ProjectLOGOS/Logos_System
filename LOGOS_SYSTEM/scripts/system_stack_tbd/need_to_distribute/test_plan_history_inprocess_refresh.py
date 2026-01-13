#!/usr/bin/env python3
"""Verify in-process plan history refresh reorders plan selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logos.plan_scoring import plan_signature, update_plan_history
from JUNK_DRAWER.scripts.runtime.need_to_distribute.logos_agi_adapter import LogosAgiNexus
from JUNK_DRAWER.scripts.runtime.could_be_dev.start_agent import _plan_history_mean


class PlanHistoryRefreshTests(unittest.TestCase):
    def test_refresh_reorders_candidates_in_process(self) -> None:
        with TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            nexus = LogosAgiNexus(
                enable=True,
                audit_logger=lambda _: None,
                max_compute_ms=50,
                state_dir=str(state_dir),
                repo_sha="testsha",
                mode="stub",
                scp_recovery_mode=False,
            )
            nexus.bootstrap()

            plan_a = {
                "plan_id": "plan-a",
                "objective_class": "STATUS",
                "steps": [{"tool": "mission.status"}],
                "read_only": True,
            }
            plan_b = {
                "plan_id": "plan-b",
                "objective_class": "STATUS",
                "steps": [{"tool": "probe.last"}],
                "read_only": True,
            }
            candidates = [plan_a, plan_b]

            def select_best_plan_id() -> str:
                scored = []
                for idx, cand in enumerate(candidates):
                    sig = plan_signature(cand)
                    mean = _plan_history_mean(nexus.prior_state or {}, sig)
                    scored.append((mean, idx, cand["plan_id"]))
                scored.sort(key=lambda t: (-t[0], t[1]))
                return scored[0][2]

            initial_choice = select_best_plan_id()
            self.assertEqual(initial_choice, "plan-a")

            state_path = state_dir / "scp_state.json"
            update_result = update_plan_history(
                state_path,
                plan_signature(plan_b),
                0.9,
                "report-hash-b",
                "ledger-hash-b",
            )
            nexus.refresh_plan_history(update_result.get("history_container", {}))

            refreshed_choice = select_best_plan_id()
            self.assertEqual(refreshed_choice, "plan-b")

            repeat_choice = select_best_plan_id()
            self.assertEqual(repeat_choice, "plan-b")


if __name__ == "__main__":  # pragma: no cover - cli
    unittest.main()
