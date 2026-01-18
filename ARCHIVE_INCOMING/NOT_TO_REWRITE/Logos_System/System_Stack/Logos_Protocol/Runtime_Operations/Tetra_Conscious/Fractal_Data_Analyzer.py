"""
System A: Tetrahedral Evaluation Module
LOGOS Passive Runtime - SMP Analysis and Refinement

This module performs recursive multi-engine analysis of Structured Meaning Packets (SMPs)
using four evaluation faces: PXL Logic, IEL Domains, ARP Mathematics, and Cognitive Resistor.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

SYSTEM_A_CONFIG = {
    "max_iterations": 10,
    "stability_threshold": 0.95,
    "min_iterations": 3,
    "convergence_window": 3,
    "parallel_evaluation": True,
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SMP:
    """Structured Meaning Packet - input to System A"""
    id: str
    content: Dict[str, Any]
    priority: int = 5
    source_agent: Optional[str] = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineReport:
    """Report from a single evaluation face"""
    engine_name: str
    score: float  # 0.0 to 1.0
    passed: bool
    details: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RefinedSMP:
    """Output from System A - ready for System B"""
    original_smp: SMP
    refined_content: Dict[str, Any]
    iterations: int
    stability_score: float
    pxl_report: EngineReport
    iel_report: EngineReport
    arp_report: EngineReport
    resistor_report: EngineReport
    convergence_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# INTEGRATED STUBS FOR SYSTEM A (FROM STUB_FILLERS_PROTOTYPE.PY)
# ============================================================================

import hashlib

class SMPSynthesizer:
    """
    Synthesizes refinements from multiple engine reports.
    
    NON-NATIVE STUB: This logic should eventually be integrated into
    the LOGOS core reasoning system, possibly as part of SCP or ARP.
    
    REPLACEMENT PATH: 
    - Could be part of SCP's cognitive synthesis
    - Could be ARP's recursive refinement logic
    - Might integrate with Trinity framework dialectics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("stub.smp_synthesizer")
    
    def synthesize(
        self,
        smp_content: Dict[str, Any],
        pxl_report: Any,
        iel_report: Any,
        arp_report: Any,
        resistor_report: Any
    ) -> Dict[str, Any]:
        """
        Synthesize refinements from all four evaluation faces.
        
        Algorithm:
        1. Apply logical corrections from PXL
        2. Apply domain adjustments from IEL
        3. Apply mathematical refinements from ARP
        4. Apply confidence adjustments from Resistor
        5. Merge and resolve conflicts
        """
        refined = dict(smp_content)
        
        # Track what was modified
        modifications = []
        
        # 1. PXL Logical Corrections
        pxl_score = getattr(pxl_report, 'score', 0.9)
        if pxl_score < 0.9:
            # Reduce confidence for logical issues
            if 'confidence' in refined:
                old_conf = refined['confidence']
                refined['confidence'] *= 0.95
                modifications.append(f"PXL: confidence {old_conf:.3f} → {refined['confidence']:.3f}")
        
        # 2. IEL Domain Adjustments
        iel_details = getattr(iel_report, 'details', {})
        consensus = iel_details.get('consensus_strength', 0.9)
        if consensus < 0.85:
            # Add domain context warning
            refined['_domain_context_warnings'] = [
                f"Low IEL consensus ({consensus:.2f})",
                "Multiple domain disagreements detected"
            ]
            modifications.append(f"IEL: added domain warnings (consensus={consensus:.2f})")
        
        # 3. ARP Mathematical Refinements
        arp_details = getattr(arp_report, 'details', {})
        if 'constructive_witness' in arp_details:
            refined['_mathematical_grounding'] = {
                'witness_available': True,
                'category_validated': arp_details.get('category_theory_check') == 'passed'
            }
            modifications.append("ARP: added mathematical grounding metadata")
        
        # 4. Cognitive Resistor Adjustments
        resistor_details = getattr(resistor_report, 'details', {})
        confidence_adj = resistor_details.get('confidence_adjustment', 0.0)
        if confidence_adj != 0.0 and 'confidence' in refined:
            old_conf = refined.get('confidence', 1.0)
            refined['confidence'] = max(0.0, min(1.0, old_conf + confidence_adj))
            modifications.append(
                f"Resistor: confidence {old_conf:.3f} → {refined['confidence']:.3f} "
                f"(adj={confidence_adj:+.3f})"
            )
        
        # Add synthesis metadata
        refined['_synthesis_metadata'] = {
            'modifications': modifications,
            'engine_scores': {
                'pxl': pxl_score,
                'iel': consensus,
                'arp': getattr(arp_report, 'score', 0.9),
                'resistor': getattr(resistor_report, 'score', 0.85)
            }
        }
        
        self.logger.debug(f"Synthesized {len(modifications)} refinements")
        return refined

class StabilityCalculator:
    """
    Calculates stability scores for iterative refinement.
    
    NON-NATIVE STUB: This should integrate with SCP's convergence tracking
    and possibly MVS (Modal Vector Space) metrics.
    
    REPLACEMENT PATH:
    - Integrate with SCP convergence metrics system
    - Use MVS dual bijection for stability measurement
    - Connect to fractal orbital analytics
    """
    
    def __init__(self, convergence_window: int = 3):
        self.convergence_window = convergence_window
        self.logger = logging.getLogger("stub.stability_calculator")
    
    def calculate(
        self,
        previous_content: Dict[str, Any],
        current_content: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate stability score: 0.0 (unstable) to 1.0 (stable)
        
        Factors:
        1. Content hash comparison (semantic unchanged)
        2. Score convergence (all engines stabilizing)
        3. Modification velocity (rate of change decreasing)
        4. Structural similarity (key preservation)
        """
        stability_factors = {}
        
        # Factor 1: Content Stability
        prev_hash = self._content_hash(previous_content)
        curr_hash = self._content_hash(current_content)
        content_stable = (prev_hash == curr_hash)
        stability_factors['content'] = 1.0 if content_stable else 0.5
        
        # Factor 2: Score Convergence
        if len(history) >= self.convergence_window:
            recent_scores = []
            for step in history[-self.convergence_window:]:
                total = (
                    step.get('pxl_score', 0) +
                    step.get('iel_score', 0) +
                    step.get('arp_score', 0) +
                    step.get('resistor_score', 0)
                )
                recent_scores.append(total)
            
            score_variance = max(recent_scores) - min(recent_scores)
            stability_factors['score_convergence'] = max(0.0, 1.0 - (score_variance * 2))
        else:
            stability_factors['score_convergence'] = 0.6
        
        # Factor 3: Modification Velocity
        if len(history) >= 2:
            recent_changes = []
            for i in range(len(history) - 1):
                change = abs(history[i+1].get('stability_score', 0) - 
                           history[i].get('stability_score', 0))
                recent_changes.append(change)
            
            avg_velocity = sum(recent_changes) / len(recent_changes) if recent_changes else 1.0
            stability_factors['velocity'] = max(0.0, 1.0 - (avg_velocity * 5))
        else:
            stability_factors['velocity'] = 0.5
        
        # Factor 4: Structural Similarity
        prev_keys = set(previous_content.keys())
        curr_keys = set(current_content.keys())
        key_overlap = len(prev_keys & curr_keys) / max(len(prev_keys | curr_keys), 1)
        stability_factors['structure'] = key_overlap
        
        # Weighted average
        weights = {
            'content': 0.4,
            'score_convergence': 0.3,
            'velocity': 0.2,
            'structure': 0.1
        }
        
        stability = sum(
            stability_factors[k] * weights[k] 
            for k in weights
        )
        
        self.logger.debug(
            f"Stability: {stability:.3f} "
            f"(content={stability_factors['content']:.2f}, "
            f"scores={stability_factors['score_convergence']:.2f}, "
            f"velocity={stability_factors['velocity']:.2f}, "
            f"structure={stability_factors['structure']:.2f})"
        )
        
        return stability
    
    def _content_hash(self, content: Dict[str, Any]) -> str:
        """Hash content for comparison, ignoring metadata"""
        filtered = {
            k: v for k, v in content.items()
            if not k.startswith('_')
        }
        canonical = json.dumps(filtered, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

# ============================================================================
# EVALUATION FACES (STUBS WITH INTEGRATION POINTS)
# ============================================================================

class PXLLogicFace:
    """
    Face 1: Classical & Modal Logic Evaluation
    
    INTEGRATION POINT: 
    - Should import from PXL Core module (proof-theoretic validation)
    - Location: Based on repo structure, likely in Advanced_Reasoning_Protocol
    - Expected interface: PXL core reasoning engines for modal logic
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("system_a.pxl_face")
        
        # STUB: Import PXL logic engine
        # TODO: Replace with actual PXL core import
        # Expected: from logos.arp.pxl_core import PXLReasoningEngine
        self.pxl_engine = None  # NEEDS MANUAL REVIEW: PXL core integration point
        
    async def analyze(self, smp: SMP) -> EngineReport:
        """
        Analyze SMP for logical consistency, modal operator validity,
        privation-as-negation, and formal proof requirements.
        """
        self.logger.debug(f"PXL analyzing SMP {smp.id}")
        
        # STUB: Replace with actual PXL logic analysis
        if self.pxl_engine is None:
            # Fallback stub logic
            score = 0.90  # Mock score
            passed = True
            details = {
                "consistency_check": "passed",
                "modal_operators": "valid",
                "privation_analysis": "no_violations",
                "proof_requirements": "satisfied",
                "note": "STUB: Replace with actual PXL core validation"
            }
            warnings = ["PXL engine not loaded - using stub"]
        else:
            # TODO: Actual PXL analysis
            # result = await self.pxl_engine.validate(smp.content)
            # score = result.score
            # passed = result.passed
            # details = result.details
            # warnings = result.warnings
            pass
        
        return EngineReport(
            engine_name="PXL_Logic",
            score=score,
            passed=passed,
            details=details,
            warnings=warnings,
            recommendations=[]
        )


class IELDomainFace:
    """
    Face 2: Internal Emergent Logics (18 Domains)
    
    INTEGRATION POINT:
    - Should import from IEL domain modules
    - Location: Advanced_Reasoning_Protocol/iel_domains/
    - Expected: 18 domain-specific reasoning modules
    - Interface: Domain-specific semantic validators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("system_a.iel_face")
        self.domain_count = 18
        
        # STUB: Import IEL domain engines
        # TODO: Replace with actual IEL imports
        # Expected: from logos.arp.iel_domains import IELDomainRegistry
        self.iel_registry = None  # NEEDS MANUAL REVIEW: IEL domain integration
        
    async def analyze(self, smp: SMP) -> EngineReport:
        """
        Analyze SMP across all 18 IEL domains for semantic integrity,
        domain-specific plausibility, and cross-domain coherence.
        """
        self.logger.debug(f"IEL analyzing SMP {smp.id} across {self.domain_count} domains")
        
        # STUB: Replace with actual IEL domain analysis
        if self.iel_registry is None:
            # Fallback stub logic
            domains_evaluated = 18
            domains_passed = 17
            consensus_strength = domains_passed / domains_evaluated
            score = consensus_strength
            passed = score >= 0.75  # Minimum 75% domain consensus
            
            details = {
                "domains_evaluated": domains_evaluated,
                "domains_passed": domains_passed,
                "consensus_strength": consensus_strength,
                "domain_breakdown": {
                    # STUB: Mock domain results
                    "passed": ["domain_1", "domain_2", "...domain_17"],
                    "failed": ["domain_18"],
                    "not_applicable": []
                },
                "note": "STUB: Replace with actual IEL domain validation"
            }
            warnings = ["IEL registry not loaded - using stub"]
        else:
            # TODO: Actual IEL analysis
            # results = await self.iel_registry.evaluate_all_domains(smp.content)
            # score = results.consensus_strength
            # passed = results.quorum_satisfied
            # details = results.to_dict()
            # warnings = results.warnings
            pass
        
        return EngineReport(
            engine_name="IEL_Domains",
            score=score,
            passed=passed,
            details=details,
            warnings=warnings,
            recommendations=[]
        )


class ARPMathematicalFace:
    """
    Face 3: Advanced Reasoning Protocol - Mathematical Components
    
    INTEGRATION POINT:
    - Should import from ARP mathematical foundations
    - Location: Advanced_Reasoning_Protocol/mathematical_foundations/
    - Expected: Category theory, type theory, number theory modules
    - Interface: Mathematical soundness validators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("system_a.arp_face")
        
        # STUB: Import ARP mathematical engines
        # TODO: Replace with actual ARP math imports
        # Expected: from logos.arp.mathematical_foundations import MathematicalValidator
        self.math_validator = None  # NEEDS MANUAL REVIEW: ARP math integration
        
    async def analyze(self, smp: SMP) -> EngineReport:
        """
        Analyze SMP for mathematical coherence, category-theoretic structure,
        type consistency, and formal transformation validity.
        """
        self.logger.debug(f"ARP Math analyzing SMP {smp.id}")
        
        # STUB: Replace with actual ARP mathematical analysis
        if self.math_validator is None:
            # Fallback stub logic
            score = 0.95
            passed = True
            details = {
                "category_theory_check": "passed",
                "type_consistency": "valid",
                "formal_transformations": "sound",
                "constructive_witness": "available",
                "note": "STUB: Replace with actual ARP mathematical validation"
            }
            warnings = ["ARP math validator not loaded - using stub"]
        else:
            # TODO: Actual ARP math analysis
            # result = await self.math_validator.validate(smp.content)
            # score = result.soundness_score
            # passed = result.mathematically_sound
            # details = result.details
            # warnings = result.warnings
            pass
        
        return EngineReport(
            engine_name="ARP_Mathematics",
            score=score,
            passed=passed,
            details=details,
            warnings=warnings,
            recommendations=[]
        )


class CognitiveResistor:
    """
    Face 4: Adversarial Critique Layer
    
    INTEGRATION POINT:
    - Should integrate with SCP (Synthetic Cognition Protocol)
    - Location: Synthetic_Cognition_Protocol/
    - Expected: Critical analysis, bias detection, fallacy identification
    - Philosophy: Removes unjustified confidence without adding content
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("system_a.cognitive_resistor")
        
        # STUB: Import cognitive critique engine
        # TODO: Replace with actual SCP integration
        # Expected: from logos.scp.critical_analysis import CognitiveResistor
        self.critique_engine = None  # NEEDS MANUAL REVIEW: SCP integration point
        
    async def analyze(self, smp: SMP, prior_reports: List[EngineReport]) -> EngineReport:
        """
        Perform adversarial critique: detect fallacies, circular reasoning,
        confirmation bias, overconfidence, and edge cases.
        
        Args:
            smp: The SMP being evaluated
            prior_reports: Reports from PXL, IEL, and ARP faces
        """
        self.logger.debug(f"Cognitive Resistor critiquing SMP {smp.id}")
        
        # STUB: Replace with actual cognitive resistance analysis
        if self.critique_engine is None:
            # Fallback stub logic
            # Check for overly high scores (potential overconfidence)
            avg_score = sum(r.score for r in prior_reports) / len(prior_reports)
            overconfidence_detected = avg_score > 0.95
            
            score = 0.85  # Resistance effectiveness
            passed = True
            details = {
                "fallacy_detection": "no_fallacies_detected",
                "circular_reasoning": "none",
                "confirmation_bias": "low_risk",
                "overconfidence_check": "flagged" if overconfidence_detected else "acceptable",
                "edge_cases_tested": ["boundary_condition_1", "extreme_value_2"],
                "confidence_adjustment": -0.05 if overconfidence_detected else 0.0,
                "note": "STUB: Replace with actual cognitive resistance analysis"
            }
            warnings = ["Cognitive resistor not loaded - using stub"]
            if overconfidence_detected:
                warnings.append("Potential overconfidence detected in prior evaluations")
        else:
            # TODO: Actual cognitive resistance
            # result = await self.critique_engine.critique(smp.content, prior_reports)
            # score = result.critique_effectiveness
            # passed = result.no_critical_issues
            # details = result.details
            # warnings = result.warnings
            pass
        
        return EngineReport(
            engine_name="Cognitive_Resistor",
            score=score,
            passed=passed,
            details=details,
            warnings=warnings,
            recommendations=[]
        )


# ============================================================================
# TETRAHEDRAL EVALUATOR - MAIN ORCHESTRATOR
# ============================================================================

class TetrahedralEvaluator:
    """
    Main orchestrator for System A tetrahedral evaluation.
    Manages iterative refinement across four evaluation faces.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**SYSTEM_A_CONFIG, **(config or {})}
        self.logger = logging.getLogger("system_a.tetrahedral")
        
        # Initialize four faces
        self.pxl_face = PXLLogicFace(config)
        self.iel_face = IELDomainFace(config)
        self.arp_face = ARPMathematicalFace(config)
        self.resistor = CognitiveResistor(config)
        
    async def evaluate(self, smp: SMP) -> RefinedSMP:
        """
        Main evaluation loop: iteratively refine SMP through four faces
        until stability threshold reached or max iterations exceeded.
        """
        self.logger.info(f"Starting tetrahedral evaluation for SMP {smp.id}")
        
        current_smp = smp
        convergence_history = []
        
        for iteration in range(self.config["max_iterations"]):
            self.logger.debug(f"Iteration {iteration + 1}/{self.config['max_iterations']}")
            
            # Evaluate through all four faces
            if self.config["parallel_evaluation"]:
                # Parallel evaluation for speed
                pxl_task = self.pxl_face.analyze(current_smp)
                iel_task = self.iel_face.analyze(current_smp)
                arp_task = self.arp_face.analyze(current_smp)
                
                pxl_report, iel_report, arp_report = await asyncio.gather(
                    pxl_task, iel_task, arp_task
                )
            else:
                # Sequential evaluation
                pxl_report = await self.pxl_face.analyze(current_smp)
                iel_report = await self.iel_face.analyze(current_smp)
                arp_report = await self.arp_face.analyze(current_smp)
            
            # Cognitive resistor evaluates after other faces
            resistor_report = await self.resistor.analyze(
                current_smp, 
                [pxl_report, iel_report, arp_report]
            )
            
            # Synthesize refinements
            refined_content = self._synthesize_reports(
                current_smp,
                pxl_report,
                iel_report,
                arp_report,
                resistor_report
            )
            
            # Calculate stability
            stability_score = self._calculate_stability(
                current_smp.content,
                refined_content,
                convergence_history
            )
            
            # Record convergence step
            convergence_history.append({
                "iteration": iteration + 1,
                "stability_score": stability_score,
                "pxl_score": pxl_report.score,
                "iel_score": iel_report.score,
                "arp_score": arp_report.score,
                "resistor_score": resistor_report.score,
            })
            
            # Check for convergence
            if (iteration >= self.config["min_iterations"] - 1 and 
                stability_score >= self.config["stability_threshold"]):
                self.logger.info(
                    f"Convergence reached at iteration {iteration + 1}, "
                    f"stability: {stability_score:.3f}"
                )
                break
            
            # Update for next iteration
            current_smp = SMP(
                id=smp.id,
                content=refined_content,
                priority=smp.priority,
                source_agent=smp.source_agent,
                timestamp_utc=smp.timestamp_utc,
                metadata={**smp.metadata, "iteration": iteration + 1}
            )
        
        # Build final refined SMP
        refined_smp = RefinedSMP(
            original_smp=smp,
            refined_content=refined_content,
            iterations=iteration + 1,
            stability_score=stability_score,
            pxl_report=pxl_report,
            iel_report=iel_report,
            arp_report=arp_report,
            resistor_report=resistor_report,
            convergence_history=convergence_history
        )
        
        self.logger.info(
            f"Tetrahedral evaluation complete for SMP {smp.id}: "
            f"{iteration + 1} iterations, stability {stability_score:.3f}"
        )
        
        return refined_smp
    
    def _synthesize_reports(
        self,
        smp: SMP,
        pxl_report: EngineReport,
        iel_report: EngineReport,
        arp_report: EngineReport,
        resistor_report: EngineReport
    ) -> Dict[str, Any]:
        """
        Synthesize refinements from all four face reports.
        
        NEEDS MANUAL REVIEW: This is the core refinement logic.
        Should incorporate:
        - PXL logical corrections
        - IEL domain-specific adjustments
        - ARP mathematical refinements
        - Resistor confidence adjustments
        """
        # Integrated stub: Use SMPSynthesizer
        synthesizer = SMPSynthesizer(self.config)
        refined_content = synthesizer.synthesize(smp.content, pxl_report, iel_report, arp_report, resistor_report)
        
        return refined_content
    
    def _calculate_stability(
        self,
        previous_content: Dict[str, Any],
        current_content: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate stability score based on content changes and score convergence.
        
        Returns: 0.0 (unstable) to 1.0 (fully stable)
        """
        # Integrated stub: Use StabilityCalculator
        calculator = StabilityCalculator(self.config["convergence_window"])
        stability_score = calculator.calculate(previous_content, current_content, history)
        
        return stability_score


# ============================================================================
# MODULE INTERFACE
# ============================================================================

# Singleton evaluator instance
_tetrahedral_evaluator: Optional[TetrahedralEvaluator] = None

def initialize_system_a(config: Optional[Dict[str, Any]] = None) -> TetrahedralEvaluator:
    """Initialize System A module (call once at startup)"""
    global _tetrahedral_evaluator
    _tetrahedral_evaluator = TetrahedralEvaluator(config)
    logging.info("System A: Tetrahedral Evaluator initialized")
    return _tetrahedral_evaluator


async def evaluate_smp(smp: SMP) -> RefinedSMP:
    """Main entry point for System A evaluation"""
    if _tetrahedral_evaluator is None:
        raise RuntimeError("System A not initialized. Call initialize_system_a() first.")
    return await _tetrahedral_evaluator.evaluate(smp)


# ============================================================================
# TESTING / DEVELOPMENT
# ============================================================================

async def _test_system_a():
    """Test harness for System A development"""
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize
    evaluator = initialize_system_a()
    
    # Create test SMP
    test_smp = SMP(
        id="test_smp_001",
        content={
            "proposition": "All rational numbers are countable",
            "domain": "mathematics",
            "confidence": 0.95,
            "evidence": ["Cantor's diagonal argument", "Bijection with naturals"]
        },
        priority=8,
        source_agent="test_harness"
    )
    
    # Evaluate
    refined = await evaluator.evaluate(test_smp)
    
    # Print results
    print("\n" + "="*80)
    print("SYSTEM A EVALUATION RESULTS")
    print("="*80)
    print(f"SMP ID: {refined.original_smp.id}")
    print(f"Iterations: {refined.iterations}")
    print(f"Stability Score: {refined.stability_score:.3f}")
    print(f"\nPXL Logic: {refined.pxl_report.score:.3f} - {refined.pxl_report.passed}")
    print(f"IEL Domains: {refined.iel_report.score:.3f} - {refined.iel_report.passed}")
    print(f"ARP Math: {refined.arp_report.score:.3f} - {refined.arp_report.passed}")
    print(f"Cognitive Resistor: {refined.resistor_report.score:.3f} - {refined.resistor_report.passed}")
    print(f"\nRefined Content:")
    print(json.dumps(refined.refined_content, indent=2))
    print("="*80)


if __name__ == "__main__":
    asyncio.run(_test_system_a())