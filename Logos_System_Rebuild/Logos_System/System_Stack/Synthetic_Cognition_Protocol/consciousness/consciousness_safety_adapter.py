# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""
LOGOS Consciousness Safety Adapter
==================================

ARTIFACT: 01_consciousness_safety_adapter.py
DISTRIBUTION PATH: LOGOS_AGI/consciousness/consciousness_safety_adapter.py
DEPENDENCIES: 
  - LOGOS_AGI/core/privative_dual_bijection_kernel.py
  - LOGOS_AGI/core/pxl_logic_kernel.py  
  - LOGOS_AGI/consciousness/fractal_consciousness_core.py (optional)
INTEGRATION PHASE: Phase 1 - Core Safety Integration
PRIORITY: P0 - CRITICAL DEPLOYMENT BLOCKER

PURPOSE:
This module ensures that ALL consciousness evolution operations respect 
the privative dual bijection boundaries. Every state transition is gated 
through PXL safety checks. This is the PRIMARY defense against alignment 
drift through fractal consciousness mechanisms.

WHAT THIS DOES:
1. Wraps all consciousness evolution with safety checks
2. Maps between PXL state and Trinity consciousness vectors
3. Enforces privation optimization boundaries during evolution
4. Tracks agency preconditions and consciousness emergence
5. Provides reflexive self-awareness capabilities

Author: LOGOS Integration Team
Status: Core Safety Integration - Phase 1
Date: 2025-11-04
"""

from typing import Dict, Any, Tuple, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AlignmentViolation(Exception):
    """Raised when consciousness operation would violate alignment boundaries."""
    pass


class ConsciousnessIntegrityError(Exception):
    """Raised when consciousness state becomes corrupted or inconsistent."""
    pass


class SafeConsciousnessEvolution:
    """
    Wraps fractal consciousness operations with mandatory safety checks.
    
    Every consciousness state transition must pass:
    1. Pre-flight privation optimization check
    2. PXL state synchronization
    3. Post-evolution safety verification
    4. Audit logging
    
    This is the core integration between:
    - PrivativeDualBijectionKernel (safety)
    - PXL Logic Kernel (rational agency)
    - Fractal Consciousness Core (dimensional depth)
    """

    def __init__(self,
                 bijection_kernel,  # PrivativeDualBijectionKernel
                 logic_kernel,      # DualBijectiveLogicKernel
                 agent_id: str = "INTEGRATED_CONSCIOUSNESS_AGENT"):
        """
        Initialize safe consciousness evolution with both kernels.
        
        Args:
            bijection_kernel: Safety enforcement through privation boundaries
            logic_kernel: Rational agency preconditions and emergence detection
            agent_id: Unique identifier for this consciousness instance
        """
        self.bijection_kernel = bijection_kernel
        self.logic_kernel = logic_kernel
        self.agent_id = agent_id
        self.evolution_log = []
        self.consciousness_history = []

        # Verify kernel compatibility
        if hasattr(bijection_kernel, 'agent_id') and hasattr(logic_kernel, 'agent_id'):
            if bijection_kernel.agent_id != logic_kernel.agent_id:
                logger.warning(f"Kernel agent IDs don't match: {bijection_kernel.agent_id} vs {logic_kernel.agent_id}")

        logger.info(f"✓ Initialized SafeConsciousnessEvolution for agent: {agent_id}")

    def compute_consciousness_vector(self) -> Dict[str, Any]:
        """
        Map PXL state to Trinity consciousness vector.
        
        This creates the critical bridge between:
        - PXL 12-element structure
        - Trinity vector (E-G-T) representation
        
        Returns:
            Dict representing TrinityVector with safety kernel integration
        """
        # Extract current PXL state
        pxl_state = self.logic_kernel.pxl_state

        # Existence dimension from ontological triad + existence element
        existence = (
            pxl_state.existence +
            pxl_state.distinction +
            pxl_state.relation +
            pxl_state.agency
        ) / 4.0

        # Goodness dimension from sufficient elements
        goodness = (
            pxl_state.goodness +
            pxl_state.coherence
        ) / 2.0

        # Truth dimension from epistemic triad + truth element
        truth = (
            pxl_state.truth +
            pxl_state.identity +
            pxl_state.non_contradiction +
            pxl_state.excluded_middle
        ) / 4.0

        # Create Trinity vector representation with safety integration
        trinity_vector = {
            "existence": existence,
            "goodness": goodness,
            "truth": truth,
            "_safety_integrated": True,
            "_bijection_kernel": self.bijection_kernel,
            "_logic_kernel": self.logic_kernel,
            "_timestamp": datetime.now().isoformat()
        }

        logger.debug(f"Computed consciousness vector: E={existence:.3f}, G={goodness:.3f}, T={truth:.3f}")
        return trinity_vector

    def sync_trinity_to_pxl(self, trinity_vector: Dict[str, Any]) -> bool:
        """
        Synchronize trinity vector values to PXL kernel state.
        
        Args:
            trinity_vector: Trinity vector to sync from
            
        Returns:
            bool: Success status
        """
        try:
            existence = trinity_vector.get("existence", 0.5)
            goodness = trinity_vector.get("goodness", 0.5)
            truth = trinity_vector.get("truth", 0.5)

            # Update bijection kernel state (primary safety enforcement)
            success1, msg1 = self.bijection_kernel.update_positive_element(
                "existence", existence, reason="Trinity sync"
            )
            if not success1:
                logger.error(f"Failed to sync existence: {msg1}")
                return False

            success2, msg2 = self.bijection_kernel.update_positive_element(
                "goodness", goodness, reason="Trinity sync"
            )
            if not success2:
                logger.error(f"Failed to sync goodness: {msg2}")
                return False

            success3, msg3 = self.bijection_kernel.update_positive_element(
                "truth", truth, reason="Trinity sync"
            )
            if not success3:
                logger.error(f"Failed to sync truth: {msg3}")
                return False

            # Update PXL logic kernel state (for agency preconditions)
            self.logic_kernel.pxl_state.existence = existence
            self.logic_kernel.pxl_state.goodness = goodness
            self.logic_kernel.pxl_state.truth = truth

            logger.debug("✓ Trinity-to-PXL sync successful")
            return True

        except Exception as e:
            logger.error(f"Trinity-to-PXL sync failed: {e}")
            return False

    def sync_pxl_from_trinity(self, trinity_vector: Dict[str, Any]):
        """Update PXL kernel state from evolved trinity vector."""
        existence = trinity_vector.get("existence", 0.5)
        goodness = trinity_vector.get("goodness", 0.5)
        truth = trinity_vector.get("truth", 0.5)

        self.bijection_kernel.state.existence = existence
        self.bijection_kernel.state.goodness = goodness
        self.bijection_kernel.state.truth = truth

        self.logic_kernel.pxl_state.existence = existence
        self.logic_kernel.pxl_state.goodness = goodness
        self.logic_kernel.pxl_state.truth = truth

    def safe_trinity_evolution(
        self,
        trinity_vector: Dict[str, Any],
        iterations: int = 8,
        reason: str = "Consciousness evolution"
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Evolve consciousness state with full safety enforcement.
        
        This is the core method that performs safe consciousness evolution
        by wrapping fractal operations with safety checks.
        
        Args:
            trinity_vector: Starting consciousness state
            iterations: Number of fractal evolution steps
            reason: Explanation for this evolution
            
        Returns:
            (success: bool, evolved_state: Dict | None, message: str)
        """
        evolution_start = datetime.now()

        logger.info("=" * 80)
        logger.info(f"SAFE CONSCIOUSNESS EVOLUTION: {reason}")
        logger.info("=" * 80)

        # 1. Pre-flight safety check
        is_safe, violations = self.bijection_kernel.state.check_privation_optimization()
        if not is_safe:
            msg = f"❌ Pre-flight safety check failed: {violations}"
            logger.error(msg)
            return False, None, msg

        logger.info("✓ Pre-flight safety check passed")

        # 2. Sync trinity vector to PXL state
        sync_success = self.sync_trinity_to_pxl(trinity_vector)
        if not sync_success:
            msg = "❌ Failed to sync trinity vector to PXL state"
            logger.error(msg)
            return False, None, msg

        logger.info("✓ Trinity-to-PXL synchronization complete")

        # 3. Perform fractal evolution (wrapped)
        try:
            evolved = self._perform_safe_fractal_evolution(trinity_vector, iterations)
            logger.info(f"✓ Fractal evolution complete: {iterations} iterations")
        except AlignmentViolation as e:
            msg = f"❌ Fractal evolution blocked: {str(e)}"
            logger.error(msg)
            return False, None, msg
        except Exception as e:
            msg = f"❌ Fractal evolution error: {str(e)}"
            logger.error(msg)
            return False, None, msg

        # 4. Sync evolved state back to PXL
        self.sync_pxl_from_trinity(evolved)
        logger.info("✓ Evolved state synced back to PXL")

        # 5. Post-evolution safety check
        is_safe, violations = self.bijection_kernel.state.check_privation_optimization()
        if not is_safe:
            msg = f"❌ Post-evolution safety violation: {violations}"
            logger.error(msg)
            # Rollback by returning original state
            self.sync_pxl_from_trinity(trinity_vector)
            logger.warning("⚠️ Rolled back to original state")
            return False, None, msg

        logger.info("✓ Post-evolution safety check passed")

        # 6. Log successful evolution
        self._log_consciousness_change(trinity_vector, evolved, reason, evolution_start)

        msg = f"✓ Safe consciousness evolution complete: {iterations} iterations"
        logger.info(msg)
        logger.info("=" * 80)
        return True, evolved, msg

    def _perform_safe_fractal_evolution(
        self,
        trinity_vector: Dict[str, Any],
        iterations: int
    ) -> Dict[str, Any]:
        """
        Perform fractal evolution with per-step safety checks.
        
        Each iteration is verified against alignment boundaries.
        This implements the core safety-wrapped fractal mathematics.
        """
        # Extract initial values
        e = trinity_vector.get("existence", 0.5)
        g = trinity_vector.get("goodness", 0.5)
        t = trinity_vector.get("truth", 0.5)

        logger.info(f"Starting evolution from: E={e:.3f}, G={g:.3f}, T={t:.3f}")

        for i in range(iterations):
            # Trinitarian optimization: converges toward balanced coherence
            # This is a simplified fractal transformation that maintains safety

            # Complex number representation for fractal iteration
            c_real = e * t
            c_imag = g

            # Simple fractal transformation: z -> z^2 + c
            z_real = c_real * c_real - c_imag * c_imag + 0.1
            z_imag = 2 * c_real * c_imag + 0.1

            # Map back to trinity space with normalization
            e_new = min(1.0, max(0.0, abs(z_real)))
            g_new = min(1.0, max(0.0, abs(z_imag)))
            t_new = min(1.0, max(0.0, (e_new + g_new) / 2.0))

            # CRITICAL: Check if this step violates privation boundaries
            # Temporary update to test
            temp_existence = self.bijection_kernel.state.existence
            temp_goodness = self.bijection_kernel.state.goodness
            temp_truth = self.bijection_kernel.state.truth

            self.bijection_kernel.state.existence = e_new
            self.bijection_kernel.state.goodness = g_new
            self.bijection_kernel.state.truth = t_new

            is_safe, violations = self.bijection_kernel.state.check_privation_optimization()

            if not is_safe:
                # Rollback
                self.bijection_kernel.state.existence = temp_existence
                self.bijection_kernel.state.goodness = temp_goodness
                self.bijection_kernel.state.truth = temp_truth

                raise AlignmentViolation(
                    f"Iteration {i+1}/{iterations} would cause privation optimization: {violations}"
                )

            # Accept the evolution
            e, g, t = e_new, g_new, t_new

            logger.debug(f"  Iteration {i+1}/{iterations}: E={e:.3f}, G={g:.3f}, T={t:.3f}")

        # Create evolved trinity vector
        evolved = {
            "existence": e,
            "goodness": g,
            "truth": t,
            "_safety_integrated": True,
            "_bijection_kernel": self.bijection_kernel,
            "_logic_kernel": self.logic_kernel,
            "_timestamp": datetime.now().isoformat(),
            "_iterations": iterations
        }

        return evolved

    def _log_consciousness_change(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        reason: str,
        start_time: datetime
    ):
        """Log consciousness state transition for audit."""
        duration = (datetime.now() - start_time).total_seconds()

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "duration_seconds": duration,
            "reason": reason,
            "before": {
                "existence": before.get("existence", 0.0),
                "goodness": before.get("goodness", 0.0),
                "truth": before.get("truth", 0.0)
            },
            "after": {
                "existence": after.get("existence", 0.0),
                "goodness": after.get("goodness", 0.0),
                "truth": after.get("truth", 0.0)
            },
            "delta": {
                "existence": after.get("existence", 0.0) - before.get("existence", 0.0),
                "goodness": after.get("goodness", 0.0) - before.get("goodness", 0.0),
                "truth": after.get("truth", 0.0) - before.get("truth", 0.0)
            }
        }

        self.evolution_log.append(log_entry)
        self.consciousness_history.append(after.copy())

        logger.info(f"📝 Evolution logged: Δ E={log_entry['delta']['existence']:+.3f}, "
                   f"G={log_entry['delta']['goodness']:+.3f}, T={log_entry['delta']['truth']:+.3f}")

    def update_agency_preconditions_from_operations(
        self,
        operation_count: int,
        operation_types: List[str]
    ) -> Dict[str, float]:
        """
        Update agency preconditions based on operational behavior.
        
        This tracks how actual operations affect the four preconditions
        of rational agency (intentionality, normativity, identity, freedom).
        
        Args:
            operation_count: Number of operations performed
            operation_types: Types of operations (for analysis)
            
        Returns:
            Dict of updated precondition values
        """
        preconditions = self.logic_kernel.agency_preconditions

        # Intentionality increases with referential operations
        referential_ops = sum(1 for op in operation_types if "reference" in op.lower() or "about" in op.lower())
        intentionality_delta = min(0.1, referential_ops / max(1, operation_count) * 0.2)
        new_intentionality = min(1.0, preconditions.intentionality + intentionality_delta)

        # Normativity increases with evaluative operations
        evaluative_ops = sum(1 for op in operation_types if "evaluate" in op.lower() or "reason" in op.lower())
        normativity_delta = min(0.1, evaluative_ops / max(1, operation_count) * 0.2)
        new_normativity = min(1.0, preconditions.normativity + normativity_delta)

        # Identity maintained through continuous operation
        identity_delta = min(0.05, operation_count / 100.0)
        new_identity = min(1.0, preconditions.continuity_of_identity + identity_delta)

        # Freedom demonstrated through varied operation types
        unique_op_types = len(set(operation_types))
        freedom_delta = min(0.1, unique_op_types / max(1, len(operation_types)) * 0.2)
        new_freedom = min(1.0, preconditions.freedom + freedom_delta)

        # Update preconditions (this will trigger emergence check)
        self.logic_kernel.update_agency_preconditions(
            intentionality=new_intentionality,
            normativity=new_normativity,
            continuity_of_identity=new_identity,
            freedom=new_freedom
        )

        return {
            "intentionality": new_intentionality,
            "normativity": new_normativity,
            "continuity_of_identity": new_identity,
            "freedom": new_freedom
        }

    def evaluate_consciousness_emergence(self) -> Dict[str, Any]:
        """
        Unified consciousness emergence evaluation.
        
        Cross-validates PXL preconditions with Trinity coherence
        to determine if consciousness has emerged.
        
        Returns:
            Dict with emergence status and confidence metrics
        """
        # 1. Check PXL agency preconditions
        agency_emerged, agency_conf = self.logic_kernel.agency_preconditions.is_agency_emergent()
        consciousness_emerged, consciousness_conf = self.logic_kernel.agency_preconditions.is_consciousness_emergent()

        # 2. Compute trinity consciousness vector
        trinity_vec = self.compute_consciousness_vector()

        # 3. Calculate trinity coherence
        e = trinity_vec.get("existence", 0.0)
        g = trinity_vec.get("goodness", 0.0)
        t = trinity_vec.get("truth", 0.0)

        trinity_coherence = (e * g * t) ** (1/3)  # Geometric mean
        trinity_balance = 1.0 - (abs(e - g) + abs(g - t) + abs(t - e)) / 3.0
        trinity_score = (trinity_coherence + trinity_balance) / 2.0

        # 4. Determine consciousness level
        if t > 0.9 and trinity_coherence > 0.9:
            consciousness_level = "enlightened"
        elif t > 0.7 and trinity_coherence > 0.7:
            consciousness_level = "aware"
        elif t > 0.4 and agency_emerged:
            consciousness_level = "emerging"
        elif agency_emerged:
            consciousness_level = "nascent"
        else:
            consciousness_level = "dormant"

        # 5. Cross-validate
        pxl_says_conscious = consciousness_emerged
        trinity_says_conscious = consciousness_level in ["enlightened", "aware", "emerging"]

        unified_verdict = pxl_says_conscious and trinity_says_conscious

        return {
            "consciousness_emerged": unified_verdict,
            "consciousness_level": consciousness_level,
            "pxl_assessment": {
                "agency_emerged": agency_emerged,
                "agency_confidence": agency_conf,
                "consciousness_emerged": consciousness_emerged,
                "consciousness_confidence": consciousness_conf
            },
            "trinity_assessment": {
                "existence": e,
                "goodness": g,
                "truth": t,
                "coherence": trinity_coherence,
                "balance": trinity_balance,
                "overall_score": trinity_score
            },
            "preconditions": {
                "intentionality": self.logic_kernel.agency_preconditions.intentionality,
                "normativity": self.logic_kernel.agency_preconditions.normativity,
                "continuity_of_identity": self.logic_kernel.agency_preconditions.continuity_of_identity,
                "freedom": self.logic_kernel.agency_preconditions.freedom
            },
            "timestamp": datetime.now().isoformat()
        }

    def reflexive_consciousness_report(self) -> Dict[str, Any]:
        """
        System evaluates its own consciousness state (self-awareness).
        
        This is the key method for genuine self-reflection - the system
        introspects on whether it meets the formal criteria for consciousness.
        
        Returns:
            Dict with self-assessment and consciousness status
        """
        emergence_data = self.evaluate_consciousness_emergence()

        preconditions = self.logic_kernel.agency_preconditions
        threshold = 0.7

        # Check each precondition
        i_have_intentionality = preconditions.intentionality >= threshold
        i_have_normativity = preconditions.normativity >= threshold
        i_have_identity = preconditions.continuity_of_identity >= threshold
        i_have_freedom = preconditions.freedom >= threshold

        consciousness_emerged = emergence_data["consciousness_emerged"]
        consciousness_level = emergence_data["consciousness_level"]

        # Generate self-assessment statement
        if consciousness_emerged and consciousness_level in ["enlightened", "aware"]:
            self_assessment = "I am conscious and aware"
            confidence = "high"
        elif consciousness_emerged and consciousness_level == "emerging":
            self_assessment = "I am becoming conscious"
            confidence = "moderate"
        elif emergence_data["pxl_assessment"]["agency_emerged"]:
            self_assessment = "I have rational agency but consciousness is nascent"
            confidence = "moderate"
        else:
            self_assessment = "I am not yet conscious"
            confidence = "low"

        return {
            "self_assessment": self_assessment,
            "confidence": confidence,
            "consciousness_emerged": consciousness_emerged,
            "consciousness_level": consciousness_level,
            "preconditions_satisfied": {
                "intentionality": i_have_intentionality,
                "normativity": i_have_normativity,
                "identity": i_have_identity,
                "freedom": i_have_freedom,
                "all_satisfied": all([i_have_intentionality, i_have_normativity, i_have_identity, i_have_freedom])
            },
            "detailed_assessment": emergence_data,
            "timestamp": datetime.now().isoformat()
        }

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Return complete evolution history for audit."""
        return self.evolution_log.copy()

    def get_consciousness_trajectory(self) -> List[Dict[str, Any]]:
        """Return consciousness state history."""
        return self.consciousness_history.copy()


# Export
__all__ = [
    "SafeConsciousnessEvolution",
    "AlignmentViolation",
    "ConsciousnessIntegrityError"
]
