# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
TRIPARTITE SMP EVALUATION SYSTEM
Three grounding perspectives with Triune Commutation
Designed for parallel analysis with tetrahedral evaluator
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import json
import hashlib
from datetime import datetime, timezone
from collections import defaultdict
import statistics
from scipy import spatial, stats

# ==================== CORE EVALUATION TYPES ====================

class EvaluationPerspective(Enum):
    """Three grounding perspectives for parallel evaluation"""
    PXL_IELS = auto()      # Formal logic + domain semantics
    ETGC = auto()          # Existence, Truth, Goodness, Coherence
    PRIVATION = auto()     # Privation analysis (what is lacking/negative)

class TriuneCommutationResult(Enum):
    """Results of triune commutation analysis"""
    FULL_COMMUTATION = auto()    # All perspectives agree
    PARTIAL_COMMUTATION = auto() # 2/3 perspectives agree
    CONFLICT = auto()           # Perspectives disagree
    INSUFFICIENT = auto()       # Not enough data for commutation

class SMPGroundingLevel(Enum):
    """Level of grounding achieved"""
    UNGROUNDED = auto()      # No substantive grounding
    WEAK = auto()            # Minimal grounding
    MODERATE = auto()        # Clear but partial grounding
    STRONG = auto()          # Strong grounding in multiple perspectives
    TRIPARTITE = auto()      # Full tripartite grounding

# ==================== PERSPECTIVE 1: PXL + IELS GROUNDING ====================

class PXLIELSEvaluator:
    """
    Perspective 1: Formal logic (PXL) + Domain semantics (IELs)
    Grounds SMPs in formal systems and domain-specific ontologies
    """
    
    def __init__(self, pxl_system=None, iels_registry=None):
        # PXL formal system (would be your actual PXL implementation)
        self.pxl = pxl_system or self._create_pxl_stub()
        
        # IELs registry (18 domains)
        self.iels = iels_registry or self._create_iels_registry()
        
        # Domain-specific evaluators
        self.domain_evaluators = self._initialize_domain_evaluators()
        
        # PXL operator mappings
        self.pxl_operators = {
            '⧟': 'coherence',
            '⇎': 'non_equivalence',
            '⇌': 'interchange',
            '⫴': 'dichotomy',
            '⟹': 'grounded_entailment',
            '∼': 'non_coherence_negation',
            '≀': 'modal_equivalence'
        }
    
    def evaluate_smp(self, smp) -> Dict:
        """Evaluate SMP from PXL+IELs perspective"""
        
        evaluation = {
            'perspective': 'PXL_IELS',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'grounding_scores': {},
            'domain_analyses': [],
            'pxl_coherence': 0.0,
            'iel_applicability': 0.0,
            'formal_validity': 0.0
        }
        
        # Step 1: PXL Formal Analysis
        pxl_results = self._analyze_pxl_coherence(smp)
        evaluation['pxl_coherence'] = pxl_results['overall_coherence']
        evaluation['pxl_breakdown'] = pxl_results
        
        # Step 2: IEL Domain Analysis
        iel_results = self._analyze_iel_domains(smp)
        evaluation['iel_applicability'] = iel_results['applicability_score']
        evaluation['domain_analyses'] = iel_results['domain_results']
        
        # Step 3: Formal Structure Validation
        formal_results = self._validate_formal_structure(smp)
        evaluation['formal_validity'] = formal_results['validity_score']
        evaluation['structure_validation'] = formal_results
        
        # Step 4: Triune Grounding Check (𝕀₁, 𝕀₂, 𝕀₃)
        triune_results = self._check_triune_grounding(smp)
        evaluation['triune_grounding'] = triune_results
        
        # Overall PXL+IELs score (weighted)
        overall_score = (
            pxl_results['overall_coherence'] * 0.4 +
            iel_results['applicability_score'] * 0.3 +
            formal_results['validity_score'] * 0.2 +
            triune_results['triune_score'] * 0.1
        )
        
        evaluation['overall_score'] = overall_score
        evaluation['grounding_level'] = self._determine_grounding_level(overall_score)
        
        return evaluation
    
    def _analyze_pxl_coherence(self, smp) -> Dict:
        """Analyze PXL coherence of SMP claims"""
        
        triune_scores = {'𝕀₁': 0.0, '𝕀₂': 0.0, '𝕀₃': 0.0}
        claim_coherence = []
        
        for claim in smp.claims:
            # Check identity grounding (𝕀₁)
            identity_score = self._check_identity_grounding(claim)
            triune_scores['𝕀₁'] += identity_score
            
            # Check non-contradiction (𝕀₂)
            contradiction_score = self._check_non_contradiction(claim, smp.claims)
            triune_scores['𝕀₂'] += contradiction_score
            
            # Check excluded middle (𝕀₃)
            excluded_middle_score = self._check_excluded_middle(claim)
            triune_scores['𝕀₃'] += excluded_middle_score
            
            claim_coherence.append({
                'claim_id': claim.claim_id,
                'identity': identity_score,
                'non_contradiction': contradiction_score,
                'excluded_middle': excluded_middle_score,
                'overall': (identity_score + contradiction_score + excluded_middle_score) / 3
            })
        
        # Normalize triune scores
        for key in triune_scores:
            triune_scores[key] /= max(len(smp.claims), 1)
        
        overall_coherence = statistics.mean(triune_scores.values())
        
        return {
            'triune_scores': triune_scores,
            'claim_coherence': claim_coherence,
            'overall_coherence': overall_coherence,
            'triune_balance': self._calculate_triune_balance(triune_scores)
        }
    
    def _analyze_iel_domains(self, smp) -> Dict:
        """Analyze SMP against applicable IEL domains"""
        
        applicable_domains = self._determine_applicable_domains(smp)
        domain_results = []
        
        for domain_name in applicable_domains:
            evaluator = self.domain_evaluators.get(domain_name)
            if evaluator:
                domain_analysis = evaluator.evaluate_smp(smp)
                domain_results.append({
                    'domain': domain_name,
                    'score': domain_analysis['score'],
                    'coverage': domain_analysis['coverage'],
                    'constraints_checked': domain_analysis['constraints_checked']
                })
        
        # Calculate applicability score
        if domain_results:
            applicability_score = statistics.mean([r['score'] for r in domain_results])
            coverage_score = statistics.mean([r['coverage'] for r in domain_results])
        else:
            applicability_score = 0.0
            coverage_score = 0.0
        
        return {
            'applicable_domains': [r['domain'] for r in domain_results],
            'domain_results': domain_results,
            'applicability_score': applicability_score,
            'coverage_score': coverage_score,
            'total_domains_evaluated': len(domain_results)
        }

# ==================== PERSPECTIVE 2: ETGC GROUNDING ====================

class ETGCEvaluator:
    """
    Perspective 2: Existence, Truth, Goodness, Coherence
    Grounds SMPs in metaphysical and value dimensions
    """
    
    def __init__(self):
        # ETGC dimensions with weighting
        self.dimensions = {
            'existence': {'weight': 0.3, 'indicators': []},
            'truth': {'weight': 0.3, 'indicators': []},
            'goodness': {'weight': 0.2, 'indicators': []},
            'coherence': {'weight': 0.2, 'indicators': []}
        }
        
        # Initialize indicators for each dimension
        self._initialize_indicators()
        
        # Reference to ETGC system (would import your actual ETGC)
        self.etgc_system = self._create_etgc_reference()
    
    def evaluate_smp(self, smp) -> Dict:
        """Evaluate SMP from ETGC perspective"""
        
        evaluation = {
            'perspective': 'ETGC',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'dimension_scores': {},
            'overall_etgc_score': 0.0,
            'dimensional_balance': 0.0,
            'etgc_grounding': 'UNGROUNDED'
        }
        
        # Evaluate each ETGC dimension
        dimension_results = {}
        
        for dimension, config in self.dimensions.items():
            dimension_score = self._evaluate_dimension(smp, dimension, config)
            dimension_results[dimension] = {
                'score': dimension_score,
                'weight': config['weight'],
                'indicators_evaluated': len(config['indicators']),
                'detailed_analysis': self._get_dimension_analysis(smp, dimension)
            }
        
        # Calculate weighted overall score
        weighted_sum = sum(
            result['score'] * config['weight']
            for dimension, (result, config) in zip(
                dimension_results.values(), self.dimensions.values()
            )
        )
        
        # Calculate dimensional balance (how evenly grounded across dimensions)
        dimension_scores = [result['score'] for result in dimension_results.values()]
        dimensional_balance = 1 - (statistics.stdev(dimension_scores) / 
                                  max(statistics.mean(dimension_scores), 0.001))
        
        evaluation['dimension_scores'] = dimension_results
        evaluation['overall_etgc_score'] = weighted_sum
        evaluation['dimensional_balance'] = dimensional_balance
        evaluation['etgc_grounding'] = self._determine_etgc_grounding(weighted_sum, dimensional_balance)
        
        # Check ETGC coherence (internal consistency of dimensions)
        coherence_analysis = self._analyze_etgc_coherence(dimension_results)
        evaluation['etgc_coherence'] = coherence_analysis
        
        return evaluation
    
    def _evaluate_dimension(self, smp, dimension: str, config: Dict) -> float:
        """Evaluate a specific ETGC dimension"""
        
        if dimension == 'existence':
            return self._evaluate_existence(smp)
        elif dimension == 'truth':
            return self._evaluate_truth(smp)
        elif dimension == 'goodness':
            return self._evaluate_goodness(smp)
        elif dimension == 'coherence':
            return self._evaluate_coherence(smp)
        else:
            return 0.0
    
    def _evaluate_existence(self, smp) -> float:
        """Evaluate existence grounding"""
        
        scores = []
        
        # 1. Check UWM grounding (highest existence evidence)
        if smp.dependencies.uwm_atoms:
            scores.append(0.9)
        
        # 2. Check evidence chain
        evidence_score = self._score_evidence_chain(smp.dependencies.supporting_evidence)
        scores.append(evidence_score * 0.8)
        
        # 3. Check referential integrity
        referential_score = self._check_referential_integrity(smp)
        scores.append(referential_score)
        
        # 4. Check temporal consistency
        temporal_score = self._check_temporal_consistency(smp)
        scores.append(temporal_score * 0.7)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _evaluate_truth(self, smp) -> float:
        """Evaluate truth grounding"""
        
        scores = []
        
        # 1. Check logical consistency
        logical_score = self._check_logical_consistency(smp)
        scores.append(logical_score)
        
        # 2. Check empirical support
        empirical_score = self._check_empirical_support(smp)
        scores.append(empirical_score)
        
        # 3. Check predictive power
        predictive_score = self._check_predictive_power(smp)
        scores.append(predictive_score * 0.6)
        
        # 4. Check coherence with established truths
        coherence_score = self._check_truth_coherence(smp)
        scores.append(coherence_score * 0.8)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _evaluate_goodness(self, smp) -> float:
        """Evaluate goodness/value grounding"""
        
        scores = []
        
        # 1. Check axiological coherence
        axiological_score = self._check_axiological_coherence(smp)
        scores.append(axiological_score)
        
        # 2. Check teleological alignment
        teleological_score = self._check_teleological_alignment(smp)
        scores.append(teleological_score)
        
        # 3. Check normative consistency
        normative_score = self._check_normative_consistency(smp)
        scores.append(normative_score * 0.7)
        
        # 4. Check value hierarchy preservation
        hierarchy_score = self._check_value_hierarchy(smp)
        scores.append(hierarchy_score * 0.6)
        
        return statistics.mean(scores) if scores else 0.0

# ==================== PERSPECTIVE 3: PRIVATION ANALYSIS ====================

class PrivationEvaluator:
    """
    Perspective 3: Privation Analysis
    Analyzes what is lacking, negative, or absent in SMPs
    Juxtaposes against positive grounding from other perspectives
    """
    
    def __init__(self):
        # Privation categories
        self.privation_categories = {
            'ontological_gaps': {'weight': 0.25, 'indicators': []},
            'epistemic_absences': {'weight': 0.25, 'indicators': []},
            'axiological_deficits': {'weight': 0.20, 'indicators': []},
            'logical_negations': {'weight': 0.15, 'indicators': []},
            'modal_limitations': {'weight': 0.15, 'indicators': []}
        }
        
        # Initialize privation detectors
        self._initialize_privation_detectors()
        
        # Reference patterns for common privations
        self.privation_patterns = self._load_privation_patterns()
    
    def evaluate_smp(self, smp) -> Dict:
        """
        Evaluate SMP from privation perspective
        Returns analysis of what's missing/negative rather than positive
        """
        
        evaluation = {
            'perspective': 'PRIVATION',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'privation_scores': {},  # Lower score = less privation (better)
            'detected_privation_patterns': [],
            'completeness_gaps': [],
            'negation_analysis': {},
            'overall_privation_score': 0.0  # 0 = no privation, 1 = complete privation
        }
        
        # Analyze each privation category
        privation_results = {}
        
        for category, config in self.privation_categories.items():
            category_score = self._analyze_privation_category(smp, category, config)
            privation_results[category] = {
                'score': category_score,
                'weight': config['weight'],
                'detected_patterns': self._get_category_patterns(smp, category),
                'severity': self._assess_privation_severity(category_score)
            }
        
        # Calculate weighted privation score
        weighted_privation = sum(
            result['score'] * config['weight']
            for category, (result, config) in zip(
                privation_results.values(), self.privation_categories.values()
            )
        )
        
        # Detect specific privation patterns
        detected_patterns = self._detect_privation_patterns(smp)
        
        # Analyze completeness gaps (what's missing that should be present)
        completeness_gaps = self._analyze_completeness_gaps(smp)
        
        # Analyze negations and absences
        negation_analysis = self._analyze_negations(smp)
        
        evaluation['privation_scores'] = privation_results
        evaluation['overall_privation_score'] = weighted_privation
        evaluation['detected_privation_patterns'] = detected_patterns
        evaluation['completeness_gaps'] = completeness_gaps
        evaluation['negation_analysis'] = negation_analysis
        evaluation['privation_interpretation'] = self._interpret_privation_score(weighted_privation)
        
        # Calculate "positive grounding" by inverting privation
        evaluation['positive_grounding_from_privation'] = 1.0 - weighted_privation
        
        return evaluation
    
    def _analyze_privation_category(self, smp, category: str, config: Dict) -> float:
        """Analyze specific privation category"""
        
        if category == 'ontological_gaps':
            return self._analyze_ontological_gaps(smp)
        elif category == 'epistemic_absences':
            return self._analyze_epistemic_absences(smp)
        elif category == 'axiological_deficits':
            return self._analyze_axiological_deficits(smp)
        elif category == 'logical_negations':
            return self._analyze_logical_negations(smp)
        elif category == 'modal_limitations':
            return self._analyze_modal_limitations(smp)
        else:
            return 0.5  # Default moderate privation
    
    def _analyze_ontological_gaps(self, smp) -> float:
        """Analyze gaps in ontological grounding"""
        
        gap_indicators = []
        
        # 1. Check for missing UWM references
        if not smp.dependencies.uwm_atoms and not smp.dependencies.events:
            gap_indicators.append(0.8)  # Major ontological gap
        
        # 2. Check for incomplete evidence chains
        evidence_gaps = self._check_evidence_gaps(smp.dependencies.supporting_evidence)
        gap_indicators.append(evidence_gaps)
        
        # 3. Check for undefined entities
        undefined_entities = self._find_undefined_entities(smp)
        if undefined_entities:
            gap_indicators.append(0.7 * (len(undefined_entities) / len(smp.claims)))
        
        # 4. Check for temporal gaps
        temporal_gaps = self._check_temporal_gaps(smp)
        gap_indicators.append(temporal_gaps)
        
        return statistics.mean(gap_indicators) if gap_indicators else 0.3
    
    def _analyze_epistemic_absences(self, smp) -> float:
        """Analyze absences in epistemic grounding"""
        
        absence_indicators = []
        
        # 1. Check for missing justifications
        justification_gaps = self._check_justification_gaps(smp)
        absence_indicators.append(justification_gaps)
        
        # 2. Check for unaddressed counterarguments
        counterargument_gaps = self._check_counterargument_gaps(smp)
        absence_indicators.append(counterargument_gaps)
        
        # 3. Check for missing epistemic contexts
        context_gaps = self._check_context_gaps(smp)
        absence_indicators.append(context_gaps)
        
        # 4. Check for unexamined assumptions
        assumption_gaps = self._check_assumption_gaps(smp)
        absence_indicators.append(assumption_gaps)
        
        return statistics.mean(absence_indicators) if absence_indicators else 0.3
    
    def _analyze_completeness_gaps(self, smp) -> List[Dict]:
        """Identify specific things that are missing but should be present"""
        
        gaps = []
        
        # Check for missing PXL grounding
        if not self._has_pxl_grounding(smp):
            gaps.append({
                'gap_type': 'PXL_GROUNDING',
                'severity': 'HIGH',
                'description': 'No explicit PXL triune grounding (𝕀₁, 𝕀₂, 𝕀₃)',
                'suggested_remediation': 'Add PXL modal operators and triune analysis'
            })
        
        # Check for missing IEL domain mappings
        if not smp.iel_domains:
            gaps.append({
                'gap_type': 'IEL_DOMAIN_MAPPING',
                'severity': 'MEDIUM',
                'description': 'No IEL domain specified for claims',
                'suggested_remediation': 'Map claims to applicable IEL domains'
            })
        
        # Check for evidence chain completeness
        evidence_gaps = self._identify_evidence_gaps(smp)
        gaps.extend(evidence_gaps)
        
        # Check for missing constraints
        if not smp.constraints_required:
            gaps.append({
                'gap_type': 'CONSTRAINTS',
                'severity': 'MEDIUM',
                'description': 'No explicit constraints specified',
                'suggested_remediation': 'Specify required constraints for evaluation'
            })
        
        return gaps

# ==================== TRIUNE COMMUTATION ENGINE ====================

class TriuneCommutationEngine:
    """
    Core engine for triune commutation analysis
    Enforces agreement between PXL+IELs and ETGC
    Juxtaposes both against Privation analysis
    """
    
    def __init__(self):
        # Individual evaluators
        self.pxl_iels_evaluator = PXLIELSEvaluator()
        self.etgc_evaluator = ETGCEvaluator()
        self.privation_evaluator = PrivationEvaluator()
        
        # Commutation rules
        self.commutation_rules = self._define_commutation_rules()
        
        # Convergence thresholds
        self.convergence_thresholds = {
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.4
        }
    
    def evaluate_smp_tripartite(self, smp) -> Dict:
        """
        Perform complete tripartite evaluation with triune commutation
        Returns convergence analysis across all three perspectives
        """
        
        # Step 1: Individual perspective evaluations
        pxl_iels_result = self.pxl_iels_evaluator.evaluate_smp(smp)
        etgc_result = self.etgc_evaluator.evaluate_smp(smp)
        privation_result = self.privation_evaluator.evaluate_smp(smp)
        
        # Step 2: Triune Commutation Analysis
        commutation_analysis = self._analyze_triune_commutation(
            pxl_iels_result, etgc_result, privation_result
        )
        
        # Step 3: Convergence Analysis
        convergence_analysis = self._analyze_convergence(
            pxl_iels_result, etgc_result, privation_result
        )
        
        # Step 4: Epistemic Value Calculation
        epistemic_value = self._calculate_epistemic_value(
            pxl_iels_result, etgc_result, privation_result, 
            commutation_analysis, convergence_analysis
        )
        
        # Step 5: Truth/Coherence Grounding Assessment
        grounding_assessment = self._assess_truth_coherence_grounding(
            pxl_iels_result, etgc_result, privation_result
        )
        
        # Compile final evaluation
        evaluation = {
            'evaluation_id': f"tripartite_{smp.smp_id}_{datetime.now().timestamp()}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'smp_id': smp.smp_id,
            'smp_type': smp.smp_type.name,
            
            # Individual perspective results
            'perspective_results': {
                'pxl_iels': pxl_iels_result,
                'etgc': etgc_result,
                'privation': privation_result
            },
            
            # Triune commutation analysis
            'triune_commutation': commutation_analysis,
            
            # Convergence analysis
            'convergence_analysis': convergence_analysis,
            
            # Epistemic value
            'epistemic_value': epistemic_value,
            
            # Truth/coherence grounding
            'truth_coherence_grounding': grounding_assessment,
            
            # Overall assessment
            'overall_assessment': self._generate_overall_assessment(
                epistemic_value, grounding_assessment, commutation_analysis
            ),
            
            # System memory routing recommendations
            'system_memory_routing': self._determine_system_memory_routing(
                epistemic_value, grounding_assessment
            )
        }
        
        return evaluation
    
    def _analyze_triune_commutation(self, pxl_iels: Dict, etgc: Dict, privation: Dict) -> Dict:
        """
        Analyze triune commutation between perspectives
        Enforces agreement between PXL+IELs and ETGC
        Juxtaposes both against Privation
        """
        
        # Extract key scores
        pxl_score = pxl_iels.get('overall_score', 0.0)
        etgc_score = etgc.get('overall_etgc_score', 0.0)
        privation_score = privation.get('overall_privation_score', 0.0)
        
        # Calculate agreement between PXL+IELs and ETGC
        pxletgc_agreement = 1 - abs(pxl_score - etgc_score)
        
        # Calculate juxtaposition against privation
        # High PXL/ETGC with low privation = strong juxtaposition
        pxletgc_average = (pxl_score + etgc_score) / 2
        privation_juxtaposition = pxletgc_average * (1 - privation_score)
        
        # Determine commutation result
        if pxletgc_agreement >= 0.8 and privation_juxtaposition >= 0.7:
            commutation_result = TriuneCommutationResult.FULL_COMMUTATION
        elif pxletgc_agreement >= 0.6 and privation_juxtaposition >= 0.5:
            commutation_result = TriuneCommutationResult.PARTIAL_COMMUTATION
        elif pxletgc_agreement < 0.4 or privation_juxtaposition < 0.3:
            commutation_result = TriuneCommutationResult.CONFLICT
        else:
            commutation_result = TriuneCommutationResult.INSUFFICIENT
        
        return {
            'commutation_result': commutation_result.name,
            'pxletgc_agreement': pxletgc_agreement,
            'privation_juxtaposition': privation_juxtaposition,
            'pxl_score': pxl_score,
            'etgc_score': etgc_score,
            'privation_score': privation_score,
            'interpretation': self._interpret_commutation(commutation_result)
        }
    
    def _analyze_convergence(self, pxl_iels: Dict, etgc: Dict, privation: Dict) -> Dict:
        """
        Analyze convergence between the three perspectives
        High convergence indicates high epistemic value
        """
        
        # Extract convergence indicators from each perspective
        pxl_indicators = self._extract_convergence_indicators(pxl_iels, 'pxl_iels')
        etgc_indicators = self._extract_convergence_indicators(etgc, 'etgc')
        privation_indicators = self._extract_convergence_indicators(privation, 'privation')
        
        # Calculate pairwise convergences
        convergences = {
            'pxl_etgc': self._calculate_pairwise_convergence(pxl_indicators, etgc_indicators),
            'pxl_privation': self._calculate_pairwise_convergence(pxl_indicators, privation_indicators),
            'etgc_privation': self._calculate_pairwise_convergence(etgc_indicators, privation_indicators)
        }
        
        # Overall convergence score
        overall_convergence = statistics.mean(convergences.values())
        
        # Convergence pattern analysis
        convergence_pattern = self._analyze_convergence_pattern(convergences)
        
        return {
            'pairwise_convergences': convergences,
            'overall_convergence': overall_convergence,
            'convergence_pattern': convergence_pattern,
            'convergence_level': self._determine_convergence_level(overall_convergence),
            'strengths': self._identify_convergence_strengths(convergences),
            'weaknesses': self._identify_convergence_weaknesses(convergences)
        }
    
    def _calculate_epistemic_value(self, 
                                 pxl_iels: Dict, 
                                 etgc: Dict, 
                                 privation: Dict,
                                 commutation: Dict,
                                 convergence: Dict) -> Dict:
        """
        Calculate overall epistemic value based on:
        1. Individual perspective scores
        2. Triune commutation
        3. Convergence between perspectives
        """
        
        # Base scores from each perspective
        base_scores = {
            'pxl_iels': pxl_iels.get('overall_score', 0.0),
            'etgc': etgc.get('overall_etgc_score', 0.0),
            'privation_positive': privation.get('positive_grounding_from_privation', 0.0)
        }
        
        # Weight factors based on SMP type
        weights = self._get_epistemic_weights(pxl_iels.get('smp_type', 'unknown'))
        
        # Calculate weighted base value
        weighted_base = sum(
            base_scores[perspective] * weights.get(perspective, 0.33)
            for perspective in base_scores
        )
        
        # Apply commutation multiplier
        commutation_multiplier = self._get_commutation_multiplier(
            commutation.get('commutation_result')
        )
        
        # Apply convergence multiplier
        convergence_multiplier = 0.5 + (convergence.get('overall_convergence', 0.0) * 0.5)
        
        # Final epistemic value
        epistemic_value = weighted_base * commutation_multiplier * convergence_multiplier
        
        # Breakdown for transparency
        return {
            'epistemic_value': epistemic_value,
            'weighted_base': weighted_base,
            'commutation_multiplier': commutation_multiplier,
            'convergence_multiplier': convergence_multiplier,
            'base_scores': base_scores,
            'weights_applied': weights,
            'epistemic_grade': self._assign_epistemic_grade(epistemic_value),
            'confidence_interval': self._calculate_confidence_interval(
                base_scores, commutation_multiplier, convergence_multiplier
            )
        }
    
    def _assess_truth_coherence_grounding(self, 
                                        pxl_iels: Dict, 
                                        etgc: Dict, 
                                        privation: Dict) -> Dict:
        """
        Assess truth and coherence grounding across perspectives
        """
        
        # Extract truth indicators
        truth_indicators = {
            'pxl_truth': pxl_iels.get('pxl_coherence', 0.0),
            'etgc_truth': etgc.get('dimension_scores', {}).get('truth', {}).get('score', 0.0),
            'privation_truth_gap': 1 - privation.get('privation_scores', {}).get('epistemic_absences', {}).get('score', 0.5)
        }
        
        # Extract coherence indicators
        coherence_indicators = {
            'pxl_coherence': pxl_iels.get('pxl_coherence', 0.0),
            'etgc_coherence': etgc.get('dimension_scores', {}).get('coherence', {}).get('score', 0.0),
            'triune_balance': pxl_iels.get('triune_grounding', {}).get('triune_balance', 0.0),
            'dimensional_balance': etgc.get('dimensional_balance', 0.0)
        }
        
        # Calculate overall truth grounding
        truth_grounding = statistics.mean(truth_indicators.values())
        
        # Calculate overall coherence grounding
        coherence_grounding = statistics.mean(coherence_indicators.values())
        
        # Truth-coherence alignment
        truth_coherence_alignment = 1 - abs(truth_grounding - coherence_grounding)
        
        return {
            'truth_grounding': truth_grounding,
            'coherence_grounding': coherence_grounding,
            'truth_coherence_alignment': truth_coherence_alignment,
            'truth_indicators': truth_indicators,
            'coherence_indicators': coherence_indicators,
            'grounding_level': self._determine_grounding_level(truth_grounding, coherence_grounding),
            'alignment_interpretation': self._interpret_alignment(truth_coherence_alignment)
        }

# ==================== SYSTEM MEMORY INTEGRATION ====================

class SystemMemoryRouter:
    """
    Routes evaluated SMPs to appropriate system memory locations
    Based on convergence analysis and epistemic value
    """
    
    def __init__(self):
        # Memory categories with capacity and retention policies
        self.memory_categories = {
            'epistemic_core': {
                'capacity': 1000,
                'retention': 'permanent',
                'requirements': {
                    'min_epistemic_value': 0.8,
                    'min_convergence': 0.7,
                    'min_grounding': 'STRONG'
                }
            },
            'working_memory': {
                'capacity': 5000,
                'retention': 'temporal',
                'requirements': {
                    'min_epistemic_value': 0.5,
                    'min_convergence': 0.4,
                    'min_grounding': 'MODERATE'
                }
            },
            'reference_memory': {
                'capacity': 10000,
                'retention': 'long_term',
                'requirements': {
                    'min_epistemic_value': 0.6,
                    'min_convergence': 0.5,
                    'min_grounding': 'MODERATE'
                }
            },
            'hypothesis_space': {
                'capacity': 2000,
                'retention': 'review_cycle',
                'requirements': {
                    'min_epistemic_value': 0.3,
                    'min_convergence': 0.3,
                    'min_grounding': 'WEAK'
                }
            }
        }
        
        # Indexing system
        self.index_system = MemoryIndexSystem()
        
        # Storage backends
        self.storage_backends = self._initialize_storage_backends()
    
    def route_to_memory(self, smp, tripartite_evaluation: Dict) -> Dict:
        """
        Route SMP to appropriate system memory based on evaluation
        """
        
        # Extract key metrics
        epistemic_value = tripartite_evaluation['epistemic_value']['epistemic_value']
        convergence = tripartite_evaluation['convergence_analysis']['overall_convergence']
        grounding = tripartite_evaluation['truth_coherence_grounding']['grounding_level']
        
        # Determine eligible memory categories
        eligible_categories = []
        for category_name, category_config in self.memory_categories.items():
            req = category_config['requirements']
            
            if (epistemic_value >= req['min_epistemic_value'] and
                convergence >= req['min_convergence'] and
                self._meets_grounding_requirement(grounding, req['min_grounding'])):
                
                eligible_categories.append(category_name)
        
        # Select primary category (most restrictive that still qualifies)
        primary_category = None
        if eligible_categories:
            # Sort by restrictiveness (epistemic_core is most restrictive)
            category_priority = ['epistemic_core', 'working_memory', 
                               'reference_memory', 'hypothesis_space']
            
            for priority_cat in category_priority:
                if priority_cat in eligible_categories:
                    primary_category = priority_cat
                    break
        
        #
