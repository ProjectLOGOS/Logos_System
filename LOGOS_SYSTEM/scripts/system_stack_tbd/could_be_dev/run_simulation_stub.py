#!/usr/bin/env python3
"""Generate example simulation events for sandbox ingestion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
from Logos_AGI.Logos_Agent.creator_packet.reflection_builder.perception_ingestors import ObservationBroker
from sandbox.simulations.gridworld import GridworldSimulation

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of movement commands to execute",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip observation broker ingestion after simulation",
    )
    parser.add_argument(
        "--show-health",
        action="store_true",
        help="Emit observation broker health information",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Write simulation JSONL logs into the provided directory",
    )
    return parser.parse_args()


def _command_script(steps: int) -> List[dict]:
    pattern = ["right", "up"]
    commands: List[dict] = []
    for index in range(steps):
        commands.append({"move": pattern[index % len(pattern)]})
    return commands


def main() -> None:
    args = _parse_args()
    broker = ObservationBroker()

    if args.show_health:
        _emit_health("pre-run", broker)

    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else None
    sim = GridworldSimulation(log_dir=log_dir)
    episode = sim.run_episode(_command_script(args.steps))
    log_path = sim.log_dir / f"{sim.name}_events.jsonl"
    print(
        "[sim] wrote %s events to %s (score=%.2f terminated=%s)"
        % (len(episode.events), log_path, episode.score, episode.terminated)
    )

    if args.no_ingest:
        print("[ingest] skipped (--no-ingest)")
    else:
        observations = broker.gather()
        print("[ingest] gathered %s observations" % len(observations))
        for observation in observations:
            if observation.modality == "simulation":
                print("[ingest] simulation payload %s" % observation.payload)

    if args.show_health:
        _emit_health("post-run", broker)


def _emit_health(stage: str, broker: ObservationBroker) -> None:
    statuses = broker.health_report()
    fragments: List[str] = []
    for status in statuses:
        state = "ready" if status.available else "unavailable"
        fragments.append(f"{status.name}:{state}:{status.detail}")
    summary = " | ".join(fragments) if fragments else "no-ingestors"
    print(f"[health:{stage}] {summary}")


if __name__ == "__main__":
    main()
