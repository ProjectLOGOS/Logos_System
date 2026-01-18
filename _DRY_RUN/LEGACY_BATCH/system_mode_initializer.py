# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: FORBIDDEN (DRY_RUN_ONLY)
# AUTHORITY: GOVERNED
# INSTALL_STATUS: DRY_RUN_ONLY
# SOURCE_LEGACY: system_mode_initializer.py

"""
DRY-RUN REWRITE

This file is a structural, governed rewrite candidate generated for
rewrite-system validation only. No execution, no side effects.
"""
"""Initialize mission profile for Logos agent subsystems.

This script bifurcates behavior between a stable demo track and an
experimental agentic track. Execute it before launching probes or agent
loops so downstream modules can read the selected profile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPO_DIR.parent
LOGOS_PATH = REPO_ROOT / "external" / "Logos_AGI"
STATE_ROOT = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))
STATE_FILE = STATE_ROOT / "mission_profile.json"

if str(LOGOS_PATH) not in sys.path:
    sys.path.insert(0, str(LOGOS_PATH))

try:
    from System_Operations_Protocol.infrastructure.agent_nexus import (
        set_mission_profile,
    )  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback when nexus is unavailable

    def set_mission_profile(profile: Dict[str, Any]) -> None:
        """Fallback stub that simply writes profile to state."""

        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(
            "[mission] WARN: SOP integration unavailable; wrote mission_profile.json only",
            file=sys.stderr,
        )


MISSION_PROFILES: Dict[str, Dict[str, Any]] = {
    "demo_mode": {
        "label": "DEMO_STABLE",
        "allow_self_modification": False,
        "allow_reflexivity": False,
        "execute_hooks": True,
        "log_detail": "high",
        "override_exit_on_error": True,
        "safe_interfaces_only": True,
        "description": (
            "Stable, observable, reproducible behavior for investor/stakeholder demo. "
            "All operations constrained to verified-safe scope."
        ),
    },
    "experimental_mode": {
        "label": "AGENTIC_EXPERIMENT",
        "allow_self_modification": True,
        "allow_reflexivity": True,
        "execute_hooks": True,
        "log_detail": "maximum",
        "override_exit_on_error": True,
        "safe_interfaces_only": False,
        "description": (
            "Experimental black-box mode with expanded autonomy and deeper probing. "
            "Intended for isolated sandbox only."
        ),
    },
    "agentic_mode": {
        "label": "AGENTIC_MISSION",
        "allow_self_modification": True,
        "allow_reflexivity": True,
        "execute_hooks": True,
        "log_detail": "maximum",
        "override_exit_on_error": False,
        "safe_interfaces_only": False,
        "system_completeness_priority": True,
        "gap_detection_enabled": True,
        "commitment_prioritization_required": True,
        "world_model_snapshot_required": True,
        "description": (
            "Agentic mission posture aligned with commitment ledger governance and "
            "world-model validation requirements. Enables autonomy while keeping "
            "new guardrails engaged."
        ),
    },
}

# Select default profile: change the key to switch module-level default.
DEFAULT_MODE_KEY = "demo_mode"
ACTIVE_MODE = MISSION_PROFILES[DEFAULT_MODE_KEY]


def _persist_profile(profile: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def _resolve_profile(name: str | None) -> Dict[str, Any]:
    if not name:
        return ACTIVE_MODE
    key = name
    if key not in MISSION_PROFILES:
        candidate = f"{name}_mode"
        key = candidate if candidate in MISSION_PROFILES else key
    if key not in MISSION_PROFILES and name:
        lowered = name.lower()
        alias_map = {
            "demo": "demo_mode",
            "experimental": "experimental_mode",
            "agentic": "agentic_mode",
        }
        mapped = alias_map.get(lowered)
        if mapped:
            key = mapped
    if key not in MISSION_PROFILES:
        available = ", ".join(sorted(MISSION_PROFILES))
        raise SystemExit(f"Unknown mission mode '{name}'. Available: {available}")
    return MISSION_PROFILES[key]


def initialize(profile: Dict[str, Any] | None = None) -> None:
    """Apply the selected mission profile and persist it for audits."""

    active = profile or ACTIVE_MODE
    set_mission_profile(active)
    _persist_profile(active)
    print(f"[mission] Mode set to {active['label']}: {active['description']}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select mission mode for Logos agent subsystems"
    )
    parser.add_argument(
        "--mode",
        help="Mission mode name (demo, experimental, agentic, *_mode variants)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    args = _parse_args()
    initialize(_resolve_profile(args.mode))