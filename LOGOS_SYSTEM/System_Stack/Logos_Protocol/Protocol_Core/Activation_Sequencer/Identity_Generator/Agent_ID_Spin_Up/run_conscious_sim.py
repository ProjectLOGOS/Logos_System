#!/usr/bin/env python3
"""
Runner for the simulated consciousness runtime.
Usage: python3 scripts/run_conscious_sim.py [iterations] [delay_seconds]
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from Synthetic_Cognition_Protocol.consciousness.simulated_consciousness_runtime import SimulatedConsciousness

if __name__ == '__main__':
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    sim = SimulatedConsciousness(agent_id="LOGOS-AGENT-OMEGA")
    sim.run(iterations=iterations, delay=delay)
