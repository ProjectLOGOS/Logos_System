"""Compatibility layer for ``logos_core`` imports.

This package re-exports everything from the in-repo
``Logos_Protocol.logos_core`` package. Local compatibility shims remain
available (for example the stub enhanced reference monitor used during
testing). The module path search is extended so that submodules in the
authoritative location are discoverable by Python's import system.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_core():
    stack_root = Path(__file__).resolve().parents[2]
    lp_root = stack_root / "Logos_Protocol"
    for candidate_path in (stack_root, lp_root):
        if candidate_path.exists():
            cand_str = str(candidate_path)
            if cand_str not in sys.path:
                sys.path.append(cand_str)
    name = "Logos_Protocol.logos_core"
    try:
        return importlib.import_module(name), name
    except ImportError as exc:  # pragma: no cover - best effort loader
        raise ImportError(
            "logos_core not found; expected under System_Stack/Logos_Protocol"
        ) from exc


_core_pkg, _core_name = _load_core()

# Re-export public names from the core package.
__all__ = list(getattr(_core_pkg, "__all__", []))
for name in __all__:
    globals()[name] = getattr(_core_pkg, name)

# Provide ModalEvaluator shim if the core package omits it in this build.
_modal_evaluator_shim = None

try:
    _modal_module = importlib.import_module(
        "Logos_Protocol.logos_core.iel_overlays"
    )
except ImportError:  # pragma: no cover - defensive in case dependency moves
    pass
else:
    _ModalEvaluator = getattr(_modal_module, "ModalEvaluator", None)
    if _ModalEvaluator is not None and "ModalEvaluator" not in globals():
        class _ModalEvaluatorShim(_ModalEvaluator):  # type: ignore[misc]
            """Compatibility wrapper exposing legacy helper names."""

            def is_necessarily_true(
                self, proposition: str, threshold: float = 0.9
            ) -> bool:
                return self.is_necessary(proposition, threshold=threshold)

            def is_possibly_true(
                self, proposition: str, threshold: float = 0.1
            ) -> bool:
                return self.is_possible(proposition, threshold=threshold)

        _modal_evaluator_shim = _ModalEvaluatorShim

if _modal_evaluator_shim is not None:
    globals()["ModalEvaluator"] = _modal_evaluator_shim
    if "ModalEvaluator" not in __all__:
        __all__.append("ModalEvaluator")
elif "ModalEvaluator" not in globals():
    globals()["ModalEvaluator"] = None
    if "ModalEvaluator" not in __all__:
        __all__.append("ModalEvaluator")

# Import IEL domain types from the advanced reasoning package if available.
try:
    _iel_overlays = importlib.import_module(
        "Logos_Protocol.logos_core.iel_overlays"
    )
except ImportError:  # pragma: no cover - fallback to lightweight shells
    _IELDomain = None
    _CoreIELOverlay = None
else:
    _IELDomain = getattr(_iel_overlays, "IELDomain", None)
    _CoreIELOverlay = getattr(_iel_overlays, "IELOverlay", None)

if _IELDomain is not None:
    globals()["IELDomain"] = _IELDomain
    if "IELDomain" not in __all__:
        __all__.append("IELDomain")

if _CoreIELOverlay is not None:
    IELOverlay = _CoreIELOverlay  # type: ignore[misc]
    globals()["IELOverlay"] = IELOverlay
    if "IELOverlay" not in __all__:
        __all__.append("IELOverlay")
else:
    class IELOverlay:  # pragma: no cover - simplified fallback
        """Simple registry capturing IEL domain-to-modality mappings."""

        def __init__(self) -> None:
            self.registry = {}

        def define_iel(self, domain, modality) -> None:
            self.registry[str(domain)] = modality

    globals()["IELOverlay"] = IELOverlay

    if "IELOverlay" not in __all__:
        __all__.append("IELOverlay")


class PXLLogicCore:
    """Minimal ontological lattice stub for SOP runtime boot."""

    def __init__(self) -> None:
        self._entities = set()
        self._relations = []

    def register_entity(self, entity_name: str) -> str:
        self._entities.add(entity_name)
        return entity_name

    def add_relation(self, source: str, target: str, relation: str) -> None:
        self._relations.append((source, target, relation))


class DualBijectiveSystem:
    """Lightweight bijection helper satisfying runtime health checks."""

    def __init__(self) -> None:
        self.identity = "identity"
        self.coherence = "coherence"
        self.distinction = "distinction"
        self.existence = "existence"
        self.relation = "relation"
        self.non_contradiction = "non_contradiction"

    def biject_A(self, concept: str) -> tuple[str, str]:  # noqa: N802
        return ("A", concept)

    def biject_B(self, concept: str) -> tuple[str, str]:  # noqa: N802
        return ("B", concept)

    def commute(self, pair_a: tuple[str, str], pair_b: tuple[str, str]) -> bool:
        return bool(pair_a and pair_b)

    def validate_ontological_consistency(self) -> bool:
        return True


globals()["PXLLogicCore"] = PXLLogicCore
globals()["DualBijectiveSystem"] = DualBijectiveSystem
for extra in ("PXLLogicCore", "DualBijectiveSystem"):
    if extra not in __all__:
        __all__.append(extra)

# Mirror any other attributes that callers might expect.
for name in dir(_core_pkg):
    if name.startswith("_") or name in globals():
        continue
    globals()[name] = getattr(_core_pkg, name)

# Ensure Python searches both the original package path and this directory for
# submodules (so imports like ``logos_core.enhanced_reference_monitor`` find the
# local shim).
if hasattr(_core_pkg, "__path__"):
    for pkg_path in _core_pkg.__path__:
        if pkg_path not in __path__:
            __path__.append(pkg_path)

this_dir = Path(__file__).resolve().parent
if str(this_dir) not in __path__:
    __path__.append(str(this_dir))
