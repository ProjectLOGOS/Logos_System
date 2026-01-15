#!/usr/bin/env python3
# LOGOS_HEADER: v1
# updated_utc: 2026-01-14T20:07:43Z
# path: /workspaces/Logos_System/_Dev_Resources/Dev_Scripts/LEGACY_SCRIPTS_TO_EXAMINE/INSPECT_DECIDE/_Dev_Resources/Dev_Scripts/smoke_tests/test_plan_validation_smoke.py
# role: dev_tool
# phase: audit_normalize
# origin: INSPECT_DECIDE
# intended_bucket: REWRITE_PROMOTE
# side_effects: unknown
# entrypoints: unknown
# depends_on: 
# notes: 
# END_LOGOS_HEADER

"""Smoke tests for deterministic plan validation invariants."""

from __future__ import annotations

import unittest

import JUNK_DRAWER.scripts.need_to_distribute._bootstrap as _bootstrap  # noqa: F401

from logos.plan_validation import validate_plan_run


class PlanValidationSmokeTests(unittest.TestCase):
    def _make_plan(self, plan_id: str = "plan-1"):
        return {
            "schema_version": 1,
            "plan_id": plan_id,
            "created_at": "2025-01-01T00:00:00Z",
            "objective": "status check",
            "objective_class": "STATUS",
            "steps": [
                {
                    "step_id": "s1",
                    "index": 0,
                    "tool": "probe.last",
                    "args": {},
                    "status": "DONE",
                    "objective_class": "STATUS",
                },
                {
                    "step_id": "s2",
                    "index": 1,
                    "tool": "retrieve.local",
                    "args": {"query": "status"},
                    "status": "DONE",
                    "objective_class": "STATUS",
                },
            ],
            "current_index": 2,
            "status": "COMPLETED",
            "checkpoints": [],
        }

    def test_happy_path_validation(self) -> None:
        plan = self._make_plan()
        execution = [
            {"tool": "probe.last", "outcome": "SUCCESS", "validation": {"ok": True}},
            {"tool": "retrieve.local", "outcome": "SUCCESS", "validation": {"ok": True}},
        ]
        report = validate_plan_run(plan, execution)
        self.assertTrue(report["run_ok"])
        self.assertTrue(report["invariants_ok"])
        self.assertEqual(report["steps_ok"], 2)

    def test_detects_out_of_order_and_skipped(self) -> None:
        plan = self._make_plan("plan-2")
        plan["steps"][1]["status"] = "SKIPPED"
        execution = [
            {"tool": "retrieve.local", "outcome": "SUCCESS", "validation": {"ok": True}},
            {"tool": "probe.last", "outcome": "SUCCESS", "validation": {"ok": True}},
        ]
        report = validate_plan_run(plan, execution)
        self.assertFalse(report["run_ok"])
        reasons = [reason for r in report["step_reports"] for reason in r.get("reasons", [])]
        self.assertIn("skipped_step_executed", reasons)
        self.assertIn("out_of_order", reasons)

    def test_requires_uip_for_high_impact(self) -> None:
        plan = self._make_plan("plan-3")
        plan["steps"][0]["tool"] = "start_agent"
        execution = [
            {
                "tool": "start_agent",
                "outcome": "SUCCESS",
                "approval_required": True,
                "approval_granted": False,
                "validation": {"ok": True},
            }
        ]
        report = validate_plan_run(plan, execution)
        self.assertFalse(report["run_ok"])
        reasons = [reason for r in report["step_reports"] for reason in r.get("reasons", [])]
        self.assertIn("uip_approval_missing", reasons)


if __name__ == "__main__":  # pragma: no cover - cli
    unittest.main()
