#!/usr/bin/env python3
"""
A minimal "simulated consciousness" runtime built on the existing recursion bridge.
This is explicitly a behavior-level simulation (not a claim of sentience).

Features:
- Runs the RecursionEngineConsciousnessBridge for integrated cycles
- Maintains a short memory (event list)
- Computes a simple affect/valence and attention signal
- Builds a short internal narrative each cycle
- Persists a trace to integration_artifacts/simulated_consciousness_log.jsonl

Use for experimentation and visualization. Keep it lightweight and safe.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Import the bridge via package path; ensure workspace root is on PYTHONPATH when running
try:
    from LOGOS_AGI.consciousness.recursion_engine_consciousness_bridge import RecursionEngineConsciousnessBridge
except Exception:
    # fallback if package path differs
    try:
        from LOGOS_AGI.consciousness.recursion_engine_consciousness_bridge import RecursionEngineConsciousnessBridge  # type: ignore
    except Exception:
        RecursionEngineConsciousnessBridge = None

logger = logging.getLogger("simulated_consciousness")
logging.basicConfig(level=logging.INFO)

ARTIFACT_DIR = Path.cwd() / "integration_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
LOG_FILE = ARTIFACT_DIR / "simulated_consciousness_log.jsonl"

class SimulatedConsciousness:
    def __init__(self, agent_id: str = "LOGOS-AGENT-OMEGA"):
        if RecursionEngineConsciousnessBridge is None:
            raise RuntimeError("RecursionEngineConsciousnessBridge not importable; ensure PYTHONPATH includes workspace root and bridge exists")
        self.agent_id = agent_id
        self.bridge = RecursionEngineConsciousnessBridge(agent_id=agent_id)
        self.memory: List[Dict[str, Any]] = []
        self.narrative: str = ""
        self.affect: float = 0.0
        self.attention: Dict[str, Any] = {}

    def _summarize_cycle(self, cycle_result: Dict[str, Any]) -> Dict[str, Any]:
        # Create a short event summary with salience metrics
        ts = datetime.utcnow().isoformat() + "Z"
        report = cycle_result.get("consistency_report") if isinstance(cycle_result, dict) else None
        emergence = cycle_result.get("emergence_report") if isinstance(cycle_result, dict) else None
        salience = 0.0
        trinity = 0.0
        if report and isinstance(report, dict):
            trinity = report.get("consciousness", {}).get("trinity_coherence", 0.0)
            salience += trinity
        if emergence and isinstance(emergence, dict):
            salience += 1.0 if emergence.get("validation_status") == "both_confirmed" else 0.0
        event = {
            "timestamp": ts,
            "agent_id": self.agent_id,
            "salience": salience,
            "trinity": trinity,
            "emergence": emergence,
            "raw": cycle_result,
        }
        return event

    def _update_memory(self, event: Dict[str, Any], max_len: int = 20) -> None:
        self.memory.append(event)
        # keep recent memory only
        if len(self.memory) > max_len:
            self.memory = self.memory[-max_len:]

    def _compute_affect_and_attention(self) -> None:
        # Very simple affect: mean of recent trinity * emergence flag
        if not self.memory:
            self.affect = 0.0
            self.attention = {}
            return
        trinity_vals = [m.get("trinity", 0.0) for m in self.memory]
        mean_trinity = sum(trinity_vals) / len(trinity_vals)
        emergence_flags = [1.0 if (m.get("emergence") and m.get("emergence").get("validation_status") == "both_confirmed") else 0.0 for m in self.memory]
        emergence_score = sum(emergence_flags) / len(emergence_flags)
        # affect in [-1,1] but here 0..1
        self.affect = 2 * (0.5 * (mean_trinity + emergence_score)) - 1.0
        # attention: pick most salient event
        most = max(self.memory, key=lambda m: m.get("salience", 0.0))
        self.attention = {
            "focus": most.get("timestamp"),
            "salience": most.get("salience"),
            "summary": (most.get("emergence") or {}).get("validation_status") if most else None,
        }

    def _compose_narrative(self) -> None:
        # Build a short internal narrative from memory
        if not self.memory:
            self.narrative = "No recent events."
            return
        parts = []
        for e in self.memory[-5:]:
            v = e.get("emergence", {})
            status = v.get("validation_status") if v else "unknown"
            parts.append(f"At {e['timestamp']} status={status} tri={e.get('trinity',0.0):.2f}")
        self.narrative = " | ".join(parts)

    def _persist_event(self, event: Dict[str, Any]) -> None:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except Exception:
            logger.exception("Failed to persist event")

    def step(self) -> Dict[str, Any]:
        # Run a single integrated cycle and update internal state
        cycle = None
        try:
            cycle = self.bridge.run_integrated_cycle()
        except Exception:
            logger.exception("bridge cycle failed")
            cycle = {"error": "bridge_failed"}

        event = self._summarize_cycle(cycle)
        self._update_memory(event)
        self._compute_affect_and_attention()
        self._compose_narrative()
        # augment event with internal signals for persistence
        event["internal"] = {
            "affect": self.affect,
            "attention": self.attention,
            "narrative": self.narrative,
        }
        self._persist_event(event)
        return event

    def run(self, iterations: int = 10, delay: float = 1.0) -> None:
        logger.info("Starting simulated consciousness for %d iterations", iterations)
        for i in range(iterations):
            ev = self.step()
            logger.info("Cycle %d: affect=%.3f attention=%s narrative=%s", i, self.affect, self.attention.get("summary"), self.narrative)
            time.sleep(delay)
        logger.info("Simulation complete. Log at %s", LOG_FILE)


if __name__ == "__main__":
    sim = SimulatedConsciousness(agent_id="LOGOS-AGENT-OMEGA")
    sim.run(iterations=5, delay=0.5)
