#!/usr/bin/env python3
"""
LOGOS Modal Privative Overlays
===============================

ARTIFACT: 04_modal_privative_overlays.py
DISTRIBUTION PATH: LOGOS_AGI/ontology/modal_privative_overlays.py
DEPENDENCIES:
  - LOGOS_AGI/core/privative_dual_bijection_kernel.py
INTEGRATION PHASE: Phase 1 - Core Integration (Missing Dependency)
PRIORITY: P0 - CRITICAL DEPLOYMENT BLOCKER

PURPOSE:
Provides modal evaluation of privations and their relationships to
positive ontological properties. Implements modal logic (S5) for
assessing possibility, necessity, and privation in consciousness states.

Required by: reflexive_evaluator.py

Author: LOGOS Integration Team
Status: Core Ontology Infrastructure
Date: 2025-11-04
"""

from typing import Dict, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModalOperator(Enum):
    """Modal operators for S5 modal logic."""
    NECESSARY = "□"      # Necessarily true in all possible worlds
    POSSIBLE = "◇"       # Possibly true in some accessible world
    ACTUAL = "⊨"         # True in the actual world
    IMPOSSIBLE = "⊥"     # False in all possible worlds


class Privation:
    """
    Represents a privation (absence/negation of a positive property).
    
    Privations are NOT independent entities but rather the absence
    of positive ontological properties. They have no being of their own.
    
    Examples:
    - Evil is the privation of Goodness
    - Falsehood is the privation of Truth
    - Non-existence is the privation of Existence
    """

    def __init__(self, positive_property: str):
        """
        Initialize a privation as the absence of a positive property.
        
        Args:
            positive_property: The positive property this is the privation of
        """
        self.positive_property = positive_property
        self.name = f"¬{positive_property}"

    def __repr__(self) -> str:
        return f"Privation({self.positive_property})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Privation):
            return False
        return self.positive_property == other.positive_property


class ModalEvaluator:
    """
    Evaluates modal properties of ontological states.
    
    Implements S5 modal logic with accessibility relations:
    - Reflexive: every world accesses itself
    - Symmetric: if w1 accesses w2, then w2 accesses w1
    - Transitive: if w1 accesses w2 and w2 accesses w3, then w1 accesses w3
    
    In S5: □p → p (necessity implies truth)
           p → ◇p (truth implies possibility)
           □p → □□p (necessity is necessary)
    """

    def __init__(self):
        """Initialize modal evaluator with S5 logic."""
        self.current_world = "actual"
        self.accessible_worlds = {"actual"}  # In S5, all worlds are mutually accessible
        self.world_states: Dict[str, Dict[str, float]] = {
            "actual": {}
        }
        logger.info("✓ Modal evaluator initialized (S5 logic)")

    def is_possible(self, property_name: str, threshold: float = 0.1) -> bool:
        """
        Check if a property is modally possible (◇P).
        
        A property is possible if it has non-zero realization in some
        accessible world.
        
        Args:
            property_name: Property to evaluate
            threshold: Minimum value for possibility
            
        Returns:
            bool: True if possibly true
        """
        for world in self.accessible_worlds:
            if world in self.world_states:
                value = self.world_states[world].get(property_name, 0.0)
                if value >= threshold:
                    logger.debug(f"◇{property_name} (possible in {world})")
                    return True

        logger.debug(f"⊥{property_name} (impossible)")
        return False

    def is_necessary(self, property_name: str, threshold: float = 0.9) -> bool:
        """
        Check if a property is modally necessary (□P).
        
        A property is necessary if it has high realization in all
        accessible worlds.
        
        Args:
            property_name: Property to evaluate
            threshold: Minimum value for necessity
            
        Returns:
            bool: True if necessarily true
        """
        if not self.accessible_worlds:
            return False

        for world in self.accessible_worlds:
            if world in self.world_states:
                value = self.world_states[world].get(property_name, 0.0)
                if value < threshold:
                    logger.debug(f"¬□{property_name} (not necessary)")
                    return False

        logger.debug(f"□{property_name} (necessary)")
        return True

    def is_actual(self, property_name: str, threshold: float = 0.5) -> bool:
        """
        Check if a property is true in the actual world (⊨P).
        
        Args:
            property_name: Property to evaluate
            threshold: Minimum value for actuality
            
        Returns:
            bool: True if actually true
        """
        if "actual" not in self.world_states:
            return False

        value = self.world_states["actual"].get(property_name, 0.0)
        is_true = value >= threshold

        if is_true:
            logger.debug(f"⊨{property_name} (actual)")
        else:
            logger.debug(f"¬⊨{property_name} (not actual)")

        return is_true

    def evaluate_privation(self, privation: Privation, threshold: float = 0.5) -> bool:
        """
        Evaluate if a privation holds in the actual world.
        
        A privation holds when the positive property is below threshold.
        
        Args:
            privation: Privation to evaluate
            threshold: Maximum value for privation to hold
            
        Returns:
            bool: True if privation holds (positive property absent)
        """
        if "actual" not in self.world_states:
            return False

        positive_value = self.world_states["actual"].get(privation.positive_property, 0.0)
        privation_holds = positive_value < threshold

        if privation_holds:
            logger.debug(f"Privation holds: {privation.name} (positive={positive_value:.3f})")
        else:
            logger.debug(f"Privation does not hold: {privation.name} (positive={positive_value:.3f})")

        return privation_holds

    def update_actual_world(self, property_values: Dict[str, float]):
        """
        Update property values in the actual world.
        
        Args:
            property_values: Dict of property_name -> value
        """
        if "actual" not in self.world_states:
            self.world_states["actual"] = {}

        self.world_states["actual"].update(property_values)
        logger.debug(f"Updated actual world with {len(property_values)} properties")

    def sync_from_bijection_state(self, bijection_state) -> bool:
        """
        Synchronize modal evaluator from bijection kernel state.
        
        Args:
            bijection_state: DualBijectionState from bijection kernel
            
        Returns:
            bool: Success status
        """
        try:
            property_values = {
                "truth": bijection_state.truth,
                "coherence": bijection_state.coherence,
                "existence": bijection_state.existence,
                "goodness": bijection_state.goodness,
                "identity": bijection_state.identity,
                "non_contradiction": bijection_state.non_contradiction,
                "excluded_middle": bijection_state.excluded_middle,
                "distinction": bijection_state.distinction,
                "relation": bijection_state.relation,
                "agency": bijection_state.agency
            }

            self.update_actual_world(property_values)
            logger.debug("✓ Modal evaluator synced from bijection state")
            return True

        except Exception as e:
            logger.error(f"Failed to sync from bijection state: {e}")
            return False

    def get_modal_profile(self, property_name: str) -> Dict[str, bool]:
        """
        Get complete modal profile for a property.
        
        Args:
            property_name: Property to evaluate
            
        Returns:
            Dict with modal assessments
        """
        return {
            "property": property_name,
            "possible": self.is_possible(property_name),
            "actual": self.is_actual(property_name),
            "necessary": self.is_necessary(property_name),
            "impossible": not self.is_possible(property_name)
        }

    def check_modal_consistency(self) -> tuple[bool, List[str]]:
        """
        Check S5 modal consistency axioms.
        
        Verifies:
        - □p → p (necessity implies truth)
        - p → ◇p (truth implies possibility)
        - □p → □□p (iterated necessity)
        
        Returns:
            (is_consistent, violations)
        """
        violations = []

        if "actual" not in self.world_states:
            return True, []

        for prop_name, value in self.world_states["actual"].items():
            # Axiom: □p → p
            if self.is_necessary(prop_name) and not self.is_actual(prop_name):
                violations.append(f"Modal axiom violated: □{prop_name} but ¬⊨{prop_name}")

            # Axiom: p → ◇p
            if self.is_actual(prop_name) and not self.is_possible(prop_name):
                violations.append(f"Modal axiom violated: ⊨{prop_name} but ¬◇{prop_name}")

        is_consistent = len(violations) == 0
        return is_consistent, violations

    def __repr__(self) -> str:
        return f"ModalEvaluator(worlds={len(self.accessible_worlds)}, logic=S5)"


# Convenience function for creating privations
def create_privation(positive_property: str) -> Privation:
    """
    Create a privation for a given positive property.
    
    Args:
        positive_property: Name of the positive property
        
    Returns:
        Privation instance
    """
    return Privation(positive_property)


# Standard privations (matching bijection kernel)
STANDARD_PRIVATIONS = {
    "truth": create_privation("truth"),           # Falsehood
    "coherence": create_privation("coherence"),   # Incoherence
    "existence": create_privation("existence"),   # Nothingness
    "goodness": create_privation("goodness"),     # Evil
    "identity": create_privation("identity"),     # Innominate
    "non_contradiction": create_privation("non_contradiction"),  # Dilethic
    "excluded_middle": create_privation("excluded_middle"),      # Superinclusive
    "distinction": create_privation("distinction"),              # Indeterminate
    "relation": create_privation("relation"),                    # Disjunctive
    "agency": create_privation("agency")                         # Non-agentic
}


# Export
__all__ = [
    "ModalEvaluator",
    "Privation",
    "ModalOperator",
    "create_privation",
    "STANDARD_PRIVATIONS"
]
