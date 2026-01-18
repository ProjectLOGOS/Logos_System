# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""
core_processing/adaptive_engine.py

Adaptive refinement engine for UIP Step 5.
Simplifies and consolidates the legacy adaptive processing layer while preserving
per-session personalisation, context analysis, and refinement hooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from ..utils.system_utils import log_uip_event

LOGGER = logging.getLogger(__name__)


@dataclass
class AdaptiveProcessingResult:
    adaptation_applied: bool
    adaptation_strategies: List[str]
    context_analysis: Dict[str, Any]
    personalization_data: Dict[str, Any]
    conversation_insights: Dict[str, Any]
    adaptation_confidence: float
    refined_results: Dict[str, Any]
    user_preference_alignment: Dict[str, Any]
    dynamic_adjustments: List[Dict[str, Any]]
    adaptation_metadata: Dict[str, Any]


class AdaptiveProcessor:
    """Context-aware adaptive processing and personalisation for UIP."""

    def __init__(self) -> None:
        self.logger = LOGGER.getChild(self.__class__.__name__)
        self.adaptation_strategies = self._initialise_strategies()
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    async def process_adaptive_refinement(self, context: Dict[str, Any]) -> AdaptiveProcessingResult:
        user_input = context.get("user_input", "")
        session_id = context.get("session_id", "")
        user_id = context.get("user_id", "anonymous")

        profile = self._profile_for(user_id)
        analysis = self._analyse_context(user_input, context)
        strategies = self._select_strategies(analysis, profile)
        refined = self._apply_strategies(user_input, strategies)
        adjustments = self._derive_adjustments(strategies, profile)
        insights = self._update_history(user_id, context, refined)

        result = AdaptiveProcessingResult(
            adaptation_applied=bool(strategies),
            adaptation_strategies=strategies,
            context_analysis=analysis,
            personalization_data=self._personalisation_snapshot(profile),
            conversation_insights=insights,
            adaptation_confidence=self._estimate_confidence(strategies, profile),
            refined_results=refined,
            user_preference_alignment={"overall_alignment": "high" if strategies else "baseline"},
            dynamic_adjustments=adjustments,
            adaptation_metadata={"timestamp": time.time()},
        )

        log_uip_event("adaptive_refinement", {"session_id": session_id, "applied": result.adaptation_applied})
        return result

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    def _initialise_strategies(self) -> Dict[str, Dict[str, Any]]:
        return {
            "complexity": {
                "description": "Adjust explanation depth and formality",
                "parameters": ["expertise", "complexity", "clarity"],
            },
            "communication_style": {
                "description": "Match tone and structure to user preference",
                "parameters": ["formality", "verbosity", "technicality"],
            },
            "domain_detail": {
                "description": "Surface domain-specific depth when needed",
                "parameters": ["domain_familiarity", "confidence"],
            },
            "conversation_flow": {
                "description": "Preserve topic continuity and narrative threads",
                "parameters": ["topic", "recent_turns"],
            },
        }

    def _profile_for(self, user_id: str) -> Dict[str, Any]:
        profile = self.user_profiles.setdefault(
            user_id,
            {
                "user_id": user_id,
                "preferences": {},
                "interaction_count": 0,
                "adaptation_history": [],
            },
        )
        profile["interaction_count"] += 1
        return profile

    def _analyse_context(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "input_length": len(user_input),
            "contains_question": "?" in user_input,
            "session_metadata": context.get("metadata", {}),
        }

    def _select_strategies(self, analysis: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
        strategies: List[str] = []
        if analysis["input_length"] > 120:
            strategies.append("complexity")
        if profile.get("preferences", {}).get("formality") == "high":
            strategies.append("communication_style")
        if not strategies:
            strategies.append("conversation_flow")
        return strategies

    def _apply_strategies(self, user_input: str, strategies: List[str]) -> Dict[str, Any]:
        refined_text = user_input
        if "complexity" in strategies:
            refined_text = user_input.upper()
        elif "communication_style" in strategies:
            refined_text = user_input.capitalize()
        return {"refined_input": refined_text, "applied_adaptations": strategies}

    def _derive_adjustments(self, strategies: List[str], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        adjustments: List[Dict[str, Any]] = []
        for strategy in strategies:
            adjustments.append({"strategy": strategy, "rationale": "profile_signal"})
        profile.setdefault("adaptation_history", []).extend(strategies)
        return adjustments

    def _update_history(self, user_id: str, context: Dict[str, Any], refined: Dict[str, Any]) -> Dict[str, Any]:
        history = self.conversation_history.setdefault(user_id, [])
        entry = {
            "timestamp": time.time(),
            "user_input": context.get("user_input", ""),
            "refined_input": refined.get("refined_input", ""),
            "strategies": refined.get("applied_adaptations", []),
        }
        history.append(entry)
        if len(history) > 20:
            self.conversation_history[user_id] = history[-20:]
        return {"turns": len(history)}

    def _personalisation_snapshot(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_id": profile.get("user_id", "anonymous"),
            "interaction_count": profile.get("interaction_count", 0),
            "preferences": profile.get("preferences", {}),
        }

    def _estimate_confidence(self, strategies: List[str], profile: Dict[str, Any]) -> float:
        base = 0.6 + 0.1 * len(strategies)
        bonus = 0.05 * min(profile.get("interaction_count", 1), 10)
        return min(base + bonus, 0.95)


adaptive_processor = AdaptiveProcessor()

__all__ = ["AdaptiveProcessingResult", "AdaptiveProcessor", "adaptive_processor"]
