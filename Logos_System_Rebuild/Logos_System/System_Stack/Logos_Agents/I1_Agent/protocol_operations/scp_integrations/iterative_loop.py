# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..scp_transform.transform_registry import TransformRegistry, default_registry
from ..scp_transform.transform_types import TransformOutcome, TransformStep

@dataclass(frozen=True)
class LoopConfig:
    max_iters: int = 3
    stop_on_no_change: bool = True


def run_iterative_stabilization(
    *,
    payload: Any,
    context: Dict[str, Any],
    registry: Optional[TransformRegistry] = None,
    config: Optional[LoopConfig] = None,
) -> TransformOutcome:
    """
    Bounded iterative stabilization loop.
    Applies default safe transforms in a fixed order unless overridden.
    No heavy reasoning. No hallucination. No rebuilds.
    """
    reg = registry or default_registry()
    cfg = config or LoopConfig()

    steps: List[TransformStep] = []
    cur = payload
    changed_any = False

    order = ["normalize", "reframe", "decompose", "annotate"]

    for i in range(max(1, cfg.max_iters)):
        iter_changed = False

        for name in order:
            fn = reg.get(name)
            if not fn:
                continue
            new_cur, step = fn(cur, context)
            steps.append(step)
            if step.applied and new_cur is not cur:
                cur = new_cur
                iter_changed = True
                changed_any = True

        if cfg.stop_on_no_change and not iter_changed:
            break

    score_vector = {
        "coherence": 0.0,
        "conservation": 1.0 if changed_any else 0.8,
        "feasibility": 0.8,
    }

    status = "ok" if changed_any else "partial"
    summary = "Applied safe stabilization transforms." if changed_any else "No eligible transforms changed the payload."

    return TransformOutcome(
        payload=cur,
        steps=steps,
        score_vector=score_vector,
        status=status,
        summary=summary,
    )
