"""CLI entrypoint for the Logos identity portal unlocked via LEM discharge."""

from __future__ import annotations

import argparse
import json

from LOGOS_AGI.Logos_Agent.Logos_Core_Recursion_Engine import boot_identity
from LOGOS_AGI.Logos_Agent.ui.lem_portal import open_identity_portal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-id",
        default="LOGOS-AGENT-OMEGA",
        help="Agent identifier used for LEM discharge.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Boot the agent (discharges LEM if necessary) before opening portal.",
    )
    args = parser.parse_args()

    if args.bootstrap:
        boot_identity(agent_id=args.agent_id)

    portal_payload = open_identity_portal()
    print(json.dumps(portal_payload, indent=2))


if __name__ == "__main__":
    main()
