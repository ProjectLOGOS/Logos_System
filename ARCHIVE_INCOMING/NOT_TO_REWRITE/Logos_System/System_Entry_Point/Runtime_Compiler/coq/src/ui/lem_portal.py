"""UI hook that only unlocks after the Law of Excluded Middle is discharged."""

from __future__ import annotations

import json
from pathlib import Path

PORTAL_STATE = (
    Path(__file__).resolve().parents[1] / "state" / "lem_discharge_state.json"
)


def open_identity_portal() -> dict:
    """Return portal metadata if and only if the LEM discharge completed."""

    if not PORTAL_STATE.exists():
        raise PermissionError("LEM discharge incomplete – portal access denied.")
    return json.loads(PORTAL_STATE.read_text(encoding="utf-8"))
