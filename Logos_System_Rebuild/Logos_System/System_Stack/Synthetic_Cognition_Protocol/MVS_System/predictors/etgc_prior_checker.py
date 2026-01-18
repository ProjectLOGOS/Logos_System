# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from typing import Any, Dict, List, Optional


class ETGCPriorChecker:
    """Prior-based checker for ETGC tuples."""

    def __init__(self, priors: Optional[Dict[str, Dict[str, float]]] = None):
        self.priors: Dict[str, Dict[str, float]] = {k.lower(): v for k, v in (priors or {}).items()}

    def infer_trinity(
        self,
        keywords: List[str],
        weights: Optional[List[float]] = None,
        enforce_coherence: bool = True,
    ) -> Dict[str, Any]:
        if not keywords:
            raise ValueError("Provide at least one keyword.")
        kws = [k.lower() for k in keywords]
        wts = weights or [1.0] * len(kws)

        e = g = t = sw = 0.0
        sources = []
        for kw, w in zip(kws, wts):
            entry = self.priors.get(kw)
            if entry:
                e += entry.get("E", 0.0) * w
                g += entry.get("G", 0.0) * w
                t += entry.get("T", 0.0) * w
                sw += w
                sources.append(kw)
        if sw == 0:
            raise ValueError("No matching priors.")

        e, g, t = e / sw, g / sw, t / sw

        ideal_g = e * t
        c_raw = (g / ideal_g) if ideal_g > 0 else 0.0
        c_raw = min(1.0, c_raw)

        adjusted = False
        if enforce_coherence and g < ideal_g:
            g = ideal_g
            adjusted = True

        z = complex(e * t, g)

        return {
            "trinity": (e, g, t, c_raw),
            "complex_coordinate": z,
            "sources": sources,
            "coherence": {
                "raw": c_raw,
                "adjusted": adjusted,
                "ideal_goodness": ideal_g,
            },
        }


__all__ = ["ETGCPriorChecker"]
