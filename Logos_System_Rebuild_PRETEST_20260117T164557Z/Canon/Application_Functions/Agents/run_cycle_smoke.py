from __future__ import annotations

import json

from .sample_task import make_sample_task
from ..arp_cycle.cycle_runner import run_arp_cycle


def main() -> int:
    task = make_sample_task(kind="analysis", priority="normal")
    out = run_arp_cycle(task=task)
    print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
