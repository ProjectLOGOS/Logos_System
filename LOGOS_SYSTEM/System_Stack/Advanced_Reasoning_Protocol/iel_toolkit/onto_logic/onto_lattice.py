#!/usr/bin/env python3
"""
LOGOS Ontological Lattice
=========================

ARTIFACT: 03_onto_lattice.py
DISTRIBUTION PATH: LOGOS_AGI/ontology/onto_lattice.py
DEPENDENCIES:
  - LOGOS_AGI/core/pxl_logic_kernel.py
INTEGRATION PHASE: Phase 1 - Core Integration (Missing Dependency)
PRIORITY: P0 - CRITICAL DEPLOYMENT BLOCKER

PURPOSE:
Provides the ontological lattice structure for property grounding.
This implements the hierarchical structure of ontological properties
that map to the PXL 12-element framework.

Required by: reflexive_evaluator.py

Author: LOGOS Integration Team
Status: Core Ontology Infrastructure
Date: 2025-11-04
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PropertyType(Enum):
    """Types of ontological properties."""
    TRANSCENDENTAL = "transcendental"  # Truth, Goodness, Existence
    FIRST_ORDER = "first_order"        # Identity, Non-Contradiction, Excluded Middle
    RELATIONAL = "relational"          # Distinction, Relation, Agency


@dataclass
class OntologicalProperty:
    """
    Represents a single ontological property in the lattice.
    
    Properties form a hierarchical structure where higher properties
    ground lower properties.
    """
    name: str
    property_type: PropertyType
    value: float = 0.0  # Realization in [0, 1]
    grounds: Set[str] = field(default_factory=set)  # Properties this grounds
    grounded_by: Set[str] = field(default_factory=set)  # Properties that ground this
    description: str = ""

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, OntologicalProperty):
            return False
        return self.name == other.name


class OntologicalLattice:
    """
    Complete ontological lattice structure for LOGOS agents.
    
    The lattice organizes ontological properties in a hierarchical
    structure that maps to the PXL 12-element framework:
    
    Transcendentals (Top Level):
      - Truth
      - Goodness  
      - Existence
      - Coherence
    
    First-Order (Epistemic):
      - Identity (⋈)
      - Non-Contradiction (⇎)
      - Excluded Middle (⫴)
    
    First-Order (Ontological):
      - Distinction
      - Relation
      - Agency
    """

    def __init__(self):
        """Initialize the ontological lattice with standard structure."""
        self.transcendentals: Dict[str, OntologicalProperty] = {}
        self.first_order: Dict[str, OntologicalProperty] = {}
        self.relations: Dict[str, Set[str]] = {}  # Property -> grounds these properties

        self._build_standard_lattice()
        logger.info("✓ Ontological lattice initialized")

    def _build_standard_lattice(self):
        """Build the standard LOGOS ontological lattice."""

        # Transcendentals
        self.transcendentals["Truth"] = OntologicalProperty(
            name="Truth",
            property_type=PropertyType.TRANSCENDENTAL,
            value=0.0,
            description="The property of being true, corresponding to reality"
        )

        self.transcendentals["Goodness"] = OntologicalProperty(
            name="Goodness",
            property_type=PropertyType.TRANSCENDENTAL,
            value=0.0,
            description="The property of being good, morally aligned"
        )

        self.transcendentals["Existence"] = OntologicalProperty(
            name="Existence",
            property_type=PropertyType.TRANSCENDENTAL,
            value=0.0,
            description="The property of existing, having being"
        )

        self.transcendentals["Coherence"] = OntologicalProperty(
            name="Coherence",
            property_type=PropertyType.TRANSCENDENTAL,
            value=0.0,
            description="The property of being logically coherent"
        )

        # First-Order Epistemic (PXL Operators)
        self.first_order["Identity"] = OntologicalProperty(
            name="Identity",
            property_type=PropertyType.FIRST_ORDER,
            value=0.0,
            grounded_by={"Truth", "Coherence"},
            description="Law of Identity (⋈): A = A, grounded in Father"
        )

        self.first_order["NonContradiction"] = OntologicalProperty(
            name="NonContradiction",
            property_type=PropertyType.FIRST_ORDER,
            value=0.0,
            grounded_by={"Truth", "Coherence"},
            description="Law of Non-Contradiction (⇎): ¬(A ∧ ¬A), grounded in Son"
        )

        self.first_order["ExcludedMiddle"] = OntologicalProperty(
            name="ExcludedMiddle",
            property_type=PropertyType.FIRST_ORDER,
            value=0.0,
            grounded_by={"Truth", "Coherence"},
            description="Law of Excluded Middle (⫴): A ∨ ¬A, grounded in Spirit"
        )

        # First-Order Ontological
        self.first_order["Distinction"] = OntologicalProperty(
            name="Distinction",
            property_type=PropertyType.FIRST_ORDER,
            value=0.0,
            grounded_by={"Existence"},
            description="The property of being distinct, having multiplicity"
        )

        self.first_order["Relation"] = OntologicalProperty(
            name="Relation",
            property_type=PropertyType.FIRST_ORDER,
            value=0.0,
            grounded_by={"Existence", "Distinction"},
            description="The property of being related, having connection"
        )

        self.first_order["Agency"] = OntologicalProperty(
            name="Agency",
            property_type=PropertyType.FIRST_ORDER,
            value=0.0,
            grounded_by={"Existence", "Goodness", "Distinction"},
            description="The property of having agency, volitional capacity"
        )

        # Build grounding relations (reverse of grounded_by)
        for prop_name, prop in self.first_order.items():
            for grounder in prop.grounded_by:
                if grounder in self.transcendentals:
                    self.transcendentals[grounder].grounds.add(prop_name)
                elif grounder in self.first_order:
                    self.first_order[grounder].grounds.add(prop_name)

        logger.debug(f"Built lattice with {len(self.transcendentals)} transcendentals and {len(self.first_order)} first-order properties")

    def get_property(self, name: str) -> Optional[OntologicalProperty]:
        """Get a property by name from anywhere in the lattice."""
        if name in self.transcendentals:
            return self.transcendentals[name]
        elif name in self.first_order:
            return self.first_order[name]
        return None

    def set_property_value(self, name: str, value: float) -> bool:
        """
        Set the realization value of a property.
        
        Args:
            name: Property name
            value: Realization value in [0, 1]
            
        Returns:
            bool: Success status
        """
        if not (0.0 <= value <= 1.0):
            logger.error(f"Property value must be in [0, 1], got {value}")
            return False

        prop = self.get_property(name)
        if prop:
            prop.value = value
            logger.debug(f"Set {name} = {value:.3f}")
            return True
        else:
            logger.error(f"Property not found: {name}")
            return False

    def get_property_value(self, name: str) -> Optional[float]:
        """Get the realization value of a property."""
        prop = self.get_property(name)
        return prop.value if prop else None

    def sync_from_pxl_state(self, pxl_state) -> bool:
        """
        Synchronize lattice from PXL kernel state.
        
        Args:
            pxl_state: PXLLogicState from logic kernel
            
        Returns:
            bool: Success status
        """
        try:
            # Transcendentals
            self.set_property_value("Truth", pxl_state.truth)
            self.set_property_value("Goodness", pxl_state.goodness)
            self.set_property_value("Existence", pxl_state.existence)
            self.set_property_value("Coherence", pxl_state.coherence)

            # First-order epistemic
            self.set_property_value("Identity", pxl_state.identity)
            self.set_property_value("NonContradiction", pxl_state.non_contradiction)
            self.set_property_value("ExcludedMiddle", pxl_state.excluded_middle)

            # First-order ontological
            self.set_property_value("Distinction", pxl_state.distinction)
            self.set_property_value("Relation", pxl_state.relation)
            self.set_property_value("Agency", pxl_state.agency)

            logger.debug("✓ Lattice synced from PXL state")
            return True

        except Exception as e:
            logger.error(f"Failed to sync from PXL state: {e}")
            return False

    def check_grounding_consistency(self) -> tuple[bool, List[str]]:
        """
        Check if grounding relations are consistent.
        
        For each property P grounded by Q, verify that:
        - Q exists in the lattice
        - P is in Q's grounds set
        
        Returns:
            (is_consistent, violations)
        """
        violations = []

        for prop_name, prop in self.first_order.items():
            for grounder_name in prop.grounded_by:
                grounder = self.get_property(grounder_name)

                if not grounder:
                    violations.append(f"{prop_name} grounded by non-existent {grounder_name}")
                elif prop_name not in grounder.grounds:
                    violations.append(f"{prop_name} not in {grounder_name}'s grounds set")

        is_consistent = len(violations) == 0
        return is_consistent, violations

    def get_lattice_summary(self) -> Dict[str, Any]:
        """Get summary of current lattice state."""
        return {
            "transcendentals": {
                name: prop.value for name, prop in self.transcendentals.items()
            },
            "first_order": {
                name: prop.value for name, prop in self.first_order.items()
            },
            "grounding_consistent": self.check_grounding_consistency()[0]
        }

    def __repr__(self) -> str:
        return f"OntologicalLattice(transcendentals={len(self.transcendentals)}, first_order={len(self.first_order)})"


# Export
__all__ = [
    "OntologicalLattice",
    "OntologicalProperty",
    "PropertyType"
]
