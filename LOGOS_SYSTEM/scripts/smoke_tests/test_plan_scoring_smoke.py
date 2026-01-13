#!/usr/bin/env python3
"""Smoke tests for deterministic plan scoring."""

from __future__ import annotations

import unittest

import JUNK_DRAWER.scripts.need_to_distribute._bootstrap as _bootstrap  # noqa: F401

from logos.plan_scoring import compute_plan_score


class PlanScoringSmokeTests(unittest.TestCase):
    def test_scores_happy_path(self) -> None:
        plan_report = {
            "steps_total": 4,
            "steps_ok": 3,
            "steps_denied": 0,
            "steps_error": 0,
            "invariants_ok": True,
        }
        score, explanation = compute_plan_score(plan_report)
        self.assertAlmostEqual(score, 0.75, places=4)
        self.assertIn("penalties", explanation)
        self.assertAlmostEqual(explanation["penalties"]["denied"], 0.0)

    def test_penalties_and_clamp(self) -> None:
        plan_report = {
            "steps_total": 5,
            "steps_ok": 1,
            "steps_denied": 1,
            "steps_error": 1,
            "invariants_ok": False,
        }
        score, explanation = compute_plan_score(plan_report)
        self.assertEqual(score, 0.0)
        self.assertAlmostEqual(explanation["penalties"]["denied"], 0.2)
        self.assertAlmostEqual(explanation["penalties"]["error"], 0.3)
        self.assertAlmostEqual(explanation["penalties"]["invariants"], 0.2)
        self.assertLess(explanation["raw_score"], 0)


if __name__ == "__main__":  # pragma: no cover - cli
    unittest.main()
