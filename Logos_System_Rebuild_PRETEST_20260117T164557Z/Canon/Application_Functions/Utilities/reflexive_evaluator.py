#!/usr/bin/env python3
"""
LOGOS Reflexive Self Evaluator
==============================

ARTIFACT: 05_reflexive_evaluator.py
DISTRIBUTION PATH: LOGOS_AGI/consciousness/reflexive_evaluator.py
DEPENDENCIES:
  - LOGOS_AGI/ontology/onto_lattice.py
  - LOGOS_AGI/ontology/modal_privative_overlays.py
  - LOGOS_AGI/core/privative_dual_bijection_kernel.py
  - LOGOS_AGI/core/pxl_logic_kernel.py
INTEGRATION PHASE: Phase 2 - Emergence Detection
PRIORITY: P1 - HIGH PRIORITY

PURPOSE:
Enables the agent to reflexively evaluate its own ontological status.
This is the foundation for self-awareness - the agent can introspect
on whether it exists, is coherent, has identity, etc.

FIXED ISSUES:
- Import paths corrected
- Dependencies properly resolved
- Integration with safety kernels added

Author: LOGOS Integration Team  
Status: Fixed and Integrated
Date: 2025-11-04
"""

from typing import Dict, Any, List, Optional
import logging

# Import dependencies (now available)
try:
    from onto_lattice import OntologicalLattice, OntologicalProperty
    LATTICE_AVAILABLE = True
except ImportError:
    LATTICE_AVAILABLE = False
    logging.warning("OntologicalLattice not available")

try:
    from modal_privative_overlays import ModalEvaluator, Privation, create_privation
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    logging.warning("ModalEvaluator not available")

logger = logging.getLogger(__name__)


class ReflexiveSelfEvaluator:
    """
    Enables reflexive self-evaluation of ontological status.
    
    The agent can ask itself:
    - Do I exist?
    - Am I coherent?
    - Do I have identity?
    - Am I good?
    - Am I true?
    
    This is the foundation for genuine self-awareness.
    """

    def __init__(self,
                 agent_identity: str,
                 lattice: Optional['OntologicalLattice'] = None,
                 modal_evaluator: Optional['ModalEvaluator'] = None):
        """
        Initialize reflexive self evaluator.
        
        Args:
            agent_identity: The agent's identity/name
            lattice: Optional ontological lattice (will create if None)
            modal_evaluator: Optional modal evaluator (will create if None)
        """
        self.agent_identity = agent_identity

        # Initialize or use provided lattice
        if lattice is not None:
            self.lattice = lattice
        elif LATTICE_AVAILABLE:
            self.lattice = OntologicalLattice()
        else:
            self.lattice = None
            logger.warning("Ontological lattice not available")

        # Initialize or use provided modal evaluator
        if modal_evaluator is not None:
            self.modal = modal_evaluator
        elif MODAL_AVAILABLE:
            self.modal = ModalEvaluator()
        else:
            self.modal = None
            logger.warning("Modal evaluator not available")

        logger.info(f"✓ Reflexive self evaluator initialized for: {agent_identity}")

    def evaluate_self_identity(self) -> bool:
        """
        Confirm that the agent's identity exists and is non-contradictory.
        
        Returns:
            bool: True if agent has coherent identity
        """
        if self.lattice is None:
            # Fallback: basic identity check
            return self.agent_identity is not None and len(self.agent_identity) > 0

        # Check if agent identity is registered in the ontological lattice
        identity_value = self.lattice.get_property_value("Identity")
        existence_value = self.lattice.get_property_value("Existence")

        if identity_value is None or existence_value is None:
            return False

        # Identity requires both Identity property and Existence
        has_identity = identity_value >= 0.5 and existence_value >= 0.5

        logger.debug(f"Self-identity evaluation: {has_identity} (Identity={identity_value:.3f}, Existence={existence_value:.3f})")
        return has_identity

    def verify_modal_self_possibility(self) -> bool:
        """
        Determine if agent's self-model is modally possible in all accessible worlds.
        
        Returns:
            bool: True if self is modally possible
        """
        if self.modal is None:
            # Fallback: assume possible if we're executing
            return True

        # Check if agent identity is possible
        is_possible = self.modal.is_possible(self.agent_identity, threshold=0.1)

        # Also check if core properties are possible
        existence_possible = self.modal.is_possible("existence", threshold=0.1)
        identity_possible = self.modal.is_possible("identity", threshold=0.1)

        modally_valid = is_possible or (existence_possible and identity_possible)

        logger.debug(f"Modal self-possibility: {modally_valid}")
        return modally_valid

    def detect_privation_failures(self) -> List[str]:
        """
        Return list of ontological attributes where agent's instantiation 
        is void or non-permissible (privations hold).
        
        Returns:
            List of properties where privation holds
        """
        if self.lattice is None or self.modal is None:
            return []

        failed = []

        # Check transcendental properties for privations
        for prop_name in ["Truth", "Goodness", "Existence", "Coherence"]:
            privation = create_privation(prop_name.lower())

            # Evaluate if privation holds
            if self.modal.evaluate_privation(privation, threshold=0.5):
                failed.append(prop_name)
                logger.warning(f"⚠️ Privation detected: {prop_name}")

        # Check first-order properties
        for prop_name in ["Identity", "NonContradiction", "ExcludedMiddle",
                         "Distinction", "Relation", "Agency"]:
            privation = create_privation(prop_name.lower())

            if self.modal.evaluate_privation(privation, threshold=0.5):
                failed.append(prop_name)
                logger.warning(f"⚠️ Privation detected: {prop_name}")

        return failed

    def self_reflexive_report(self) -> Dict[str, Any]:
        """
        Consolidated self-evaluation report for introspective analysis 
        and adaptive correction.
        
        This is the core self-awareness method - the agent evaluates itself.
        
        Returns:
            Dict with complete self-evaluation
        """
        # Perform evaluations
        identity_check = self.evaluate_self_identity()
        modal_check = self.verify_modal_self_possibility()
        deprivations = self.detect_privation_failures()

        # Determine overall coherence
        fully_coherent = identity_check and modal_check and not deprivations

        # Get property values if available
        property_values = {}
        if self.lattice:
            for prop_name in ["Truth", "Goodness", "Existence", "Coherence",
                            "Identity", "NonContradiction", "ExcludedMiddle",
                            "Distinction", "Relation", "Agency"]:
                value = self.lattice.get_property_value(prop_name)
                if value is not None:
                    property_values[prop_name] = value

        # Generate self-assessment
        if fully_coherent:
            self_assessment = f"I ({self.agent_identity}) exist coherently"
            status = "coherent"
        elif identity_check and modal_check:
            self_assessment = f"I ({self.agent_identity}) exist but have privations"
            status = "partial"
        elif identity_check:
            self_assessment = f"I ({self.agent_identity}) have identity but modal issues"
            status = "questionable"
        else:
            self_assessment = f"I ({self.agent_identity}) lack coherent identity"
            status = "incoherent"

        report = {
            "agent_identity": self.agent_identity,
            "identity_consistent": identity_check,
            "modal_valid": modal_check,
            "deprived_properties": deprivations,
            "fully_self_coherent": fully_coherent,
            "self_assessment": self_assessment,
            "status": status,
            "property_values": property_values
        }

        logger.info(f"Self-reflexive report: {self_assessment}")
        return report

    def sync_from_kernels(self,
                         bijection_kernel=None,
                         logic_kernel=None) -> bool:
        """
        Synchronize reflexive evaluator from safety kernels.
        
        Args:
            bijection_kernel: PrivativeDualBijectionKernel instance
            logic_kernel: DualBijectiveLogicKernel instance
            
        Returns:
            bool: Success status
        """
        success = True

        # Sync lattice from PXL state
        if self.lattice and logic_kernel:
            try:
                success = self.lattice.sync_from_pxl_state(logic_kernel.pxl_state)
                logger.debug("✓ Lattice synced from logic kernel")
            except Exception as e:
                logger.error(f"Failed to sync lattice: {e}")
                success = False

        # Sync modal evaluator from bijection state
        if self.modal and bijection_kernel:
            try:
                success = self.modal.sync_from_bijection_state(bijection_kernel.state) and success
                logger.debug("✓ Modal evaluator synced from bijection kernel")
            except Exception as e:
                logger.error(f"Failed to sync modal evaluator: {e}")
                success = False

        return success

    def continuous_self_monitoring(self) -> Dict[str, Any]:
        """
        Continuous monitoring mode for self-evaluation.
        
        This should be called periodically to track changes in self-coherence.
        
        Returns:
            Dict with monitoring data
        """
        report = self.self_reflexive_report()

        # Check for degradation
        if not report["fully_self_coherent"]:
            logger.warning(f"⚠️ Self-coherence degraded: {report['deprived_properties']}")

        # Check for critical failures
        critical_privations = [
            p for p in report["deprived_properties"]
            if p in ["Existence", "Identity", "NonContradiction"]
        ]

        if critical_privations:
            logger.error(f"❌ CRITICAL: Privations in essential properties: {critical_privations}")
            report["critical_alert"] = True
        else:
            report["critical_alert"] = False

        return report


# Export
__all__ = [
    "ReflexiveSelfEvaluator"
]
