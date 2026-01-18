# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
STRUCTURED MEANING PACKET (SMP) GENERATION & PROCESSING SYSTEM
Implements the "Information Explosion" pattern with dual enrichment pipelines
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import uuid
import itertools
from collections import defaultdict, deque
import networkx as nx
from scipy import spatial, stats
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import zlib
import msgpack

# ==================== SMP CORE TYPES ====================

class SMPType(Enum):
    """SMP types from blueprint"""
    GROUNDED = auto()      # Highest trust - from UWM events
    ANALYTIC = auto()      # From UWM state analysis
    HYPOTHESIS = auto()    # Agent proposals (low confidence)
    DIAGNOSTIC = auto()   # System-generated meta

class SMPStatus(Enum):
    """SMP lifecycle status"""
    RAW = auto()           # Just generated, unevaluated
    ENRICHING = auto()     # In enrichment pipeline
    EVALUATING = auto()    # In tetrahedral evaluation
    CONSTRAINT_CHECK = auto()  # In constraint pipeline
    APPROVED = auto()      # Passed all checks
    REJECTED = auto()      # Failed evaluation
    ARCHIVED = auto()      # Superseded or expired
    CONFLICT = auto()      # In conflict resolution

class SMPConfidence(Enum):
    """Confidence levels with numeric mapping"""
    SPECULATIVE = (0.1, 0.3)   # Hypothesis
    TENTATIVE = (0.3, 0.6)     # Early analysis
    MODERATE = (0.6, 0.8)      # Good evidence
    HIGH = (0.8, 0.95)         # Strong evidence
    CERTAIN = (0.95, 1.0)      # Grounded in UWM

# ==================== SMP CORE STRUCTURE ====================

@dataclass
class SMPClaim:
    """Individual claim within an SMP"""
    claim_id: str
    predicate: str
    subject: str
    object: str
    modality: str  # "necessity", "possibility", "actuality"
    confidence: float  # 0.0-1.0
    evidence: List[Dict[str, Any]]
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'claim_id': self.claim_id,
            'predicate': self.predicate,
            'subject': self.subject,
            'object': self.object,
            'modality': self.modality,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'qualifiers': self.qualifiers
        }

@dataclass
class SMPDependencies:
    """Explicit dependencies as per blueprint"""
    uwm_atoms: List[str]  # References to UWM atoms
    events: List[str]     # References to UWM events
    parent_smps: List[str] # References to parent SMPs
    supporting_evidence: List[Dict[str, Any]]
    
    def is_grounded(self) -> bool:
        """Check if SMP is grounded in UWM (highest trust)"""
        return len(self.uwm_atoms) > 0 or len(self.events) > 0

@dataclass
class StructuredMeaningPacket:
    """
    Canonical SMP structure as per blueprint
    "A bounded, auditable unit of meaning"
    """
    # Core identification
    smp_id: str  # Format: "SMP:{hash}"
    smp_type: SMPType
    generation_timestamp: datetime
    
    # Content
    claims: List[SMPClaim]
    scope: str  # What this SMP applies to
    summary: str  # Human-readable summary
    
    # Dependencies (CRITICAL: "No SMP exists without explicit dependencies")
    dependencies: SMPDependencies
    
    # Confidence and provenance
    confidence_score: float  # 0.0-1.0 aggregate
    proposed_by: str  # "system" or agent_id
    
    # Processing metadata
    status: SMPStatus = SMPStatus.RAW
    enrichment_rounds: int = 0
    evaluation_history: List[Dict] = field(default_factory=list)
    
    # Constraints and requirements
    constraints_required: List[str] = field(default_factory=list)
    iel_domains: List[str] = field(default_factory=list)  # Which IEL domains apply
    
    # System fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def __post_init__(self):
        # Ensure SMP ID format
        if not self.smp_id.startswith("SMP:"):
            self.smp_id = f"SMP:{self.smp_id}"
        
        # Compute initial confidence if not provided
        if self.confidence_score == 0.0:
            self.confidence_score = self._compute_initial_confidence()
    
    @property
    def id_hash(self) -> str:
        """Unique hash based on content and dependencies"""
        content = json.dumps({
            'claims': [c.to_dict() for c in self.claims],
            'dependencies': {
                'uwm_atoms': self.dependencies.uwm_atoms,
                'events': self.dependencies.events
            },
            'scope': self.scope,
            'timestamp': self.generation_timestamp.isoformat()
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _compute_initial_confidence(self) -> float:
        """Compute initial confidence based on type and dependencies"""
        base_confidence = {
            SMPType.GROUNDED: 0.9,
            SMPType.ANALYTIC: 0.7,
            SMPType.HYPOTHESIS: 0.4,
            SMPType.DIAGNOSTIC: 0.6
        }.get(self.smp_type, 0.5)
        
        # Boost for UWM grounding
        if self.dependencies.is_grounded():
            base_confidence = min(1.0, base_confidence + 0.15)
        
        # Adjust based on evidence quality
        evidence_score = self._score_evidence_quality()
        return (base_confidence + evidence_score) / 2
    
    def _score_evidence_quality(self) -> float:
        """Score the quality of supporting evidence"""
        if not self.dependencies.supporting_evidence:
            return 0.3
        
        scores = []
        for evidence in self.dependencies.supporting_evidence:
            # Score based on evidence type
            evidence_type = evidence.get('type', 'unknown')
            type_scores = {
                'uwm_event': 1.0,
                'uwm_atom': 0.9,
                'approved_smp': 0.8,
                'external_verification': 0.7,
                'agent_testimony': 0.5,
                'inference': 0.4
            }
            scores.append(type_scores.get(evidence_type, 0.3))
        
        return np.mean(scores) if scores else 0.3
    
    def to_canonical_dict(self) -> Dict:
        """Convert to canonical JSON format per blueprint"""
        return {
            'smp_id': self.smp_id,
            'smp_type': self.smp_type.name.lower(),
            'claims': [c.to_dict() for c in self.claims],
            'scope': self.scope,
            'confidence': f"{self.confidence_score:.2f}",
            'dependencies': {
                'uwm_atoms': self.dependencies.uwm_atoms,
                'events': self.dependencies.events,
                'parent_smps': self.dependencies.parent_smps
            },
            'constraints_required': self.constraints_required,
            'proposed_by': self.proposed_by,
            'timestamp': self.generation_timestamp.isoformat(),
            'iel_domains': self.iel_domains,
            'metadata': {
                'enrichment_rounds': self.enrichment_rounds,
                'status': self.status.name,
                'id_hash': self.id_hash,
                'version': self.version
            }
        }

# ==================== SMP GENERATOR SYSTEM ====================

class SMPGenerator:
    """
    Generates SMPs from four controlled sources per blueprint
    Implements "information explosion" through dual enrichment
    """
    
    def __init__(self, uwm_interface=None):
        self.uwm = uwm_interface
        self.generation_counter = 0
        
        # Enrichment systems (System A and System B from blueprint)
        self.enrichment_system_a = TetrahedralEnrichment()
        self.enrichment_system_b = ConstraintEnrichment()
        
        # Tracking
        self.generated_smps: Dict[str, StructuredMeaningPacket] = {}
        self.generation_log: List[Dict] = []
        
        # Caches for efficiency
        self.pattern_cache: Dict[str, List] = {}
        self.template_cache: Dict[str, Dict] = {}
    
    def generate_from_uwm_event(self, uwm_event: Dict) -> List[StructuredMeaningPacket]:
        """
        4.1 UWM Event–Driven SMPs (Primary)
        "Every Phase-5-approved world event can spawn SMPs"
        """
        smps = []
        
        # Extract event metadata
        event_type = uwm_event.get('event_type', 'unknown')
        event_target = uwm_event.get('target', {})
        evidence = uwm_event.get('evidence', {})
        
        # Generate appropriate SMPs based on event type
        if event_type == 'CREATE':
            smps.extend(self._generate_create_smps(event_target, evidence))
        elif event_type == 'MOVE':
            smps.extend(self._generate_move_smps(event_target, evidence))
        elif event_type == 'RENAME':
            smps.extend(self._generate_rename_smps(event_target, evidence))
        elif event_type == 'MERGE':
            smps.extend(self._generate_merge_smps(event_target, evidence))
        elif event_type == 'DEPRECATE':
            smps.extend(self._generate_deprecate_smps(event_target, evidence))
        
        # Mark all as GROUNDED type
        for smp in smps:
            smp.smp_type = SMPType.GROUNDED
        
        self._log_generation('uwm_event', len(smps), event_type)
        return smps
    
    def generate_from_uwm_state(self, 
                               uwm_snapshot: Dict,
                               analysis_type: str = 'periodic') -> List[StructuredMeaningPacket]:
        """
        4.2 UWM State–Derived SMPs
        "Periodic scans over UWM may produce SMPs"
        """
        smps = []
        
        # Dependency cluster analysis
        if analysis_type in ['dependency', 'full']:
            smps.extend(self._analyze_dependency_clusters(uwm_snapshot))
        
        # Invariant confirmation
        if analysis_type in ['invariant', 'full']:
            smps.extend(self._check_invariants(uwm_snapshot))
        
        # Drift detection
        if analysis_type in ['drift', 'full']:
            smps.extend(self._detect_drift(uwm_snapshot))
        
        # Unused structure detection
        if analysis_type in ['unused', 'full']:
            smps.extend(self._find_unused_structures(uwm_snapshot))
        
        # Mark all as ANALYTIC type
        for smp in smps:
            smp.smp_type = SMPType.ANALYTIC
        
        self._log_generation('uwm_state', len(smps), analysis_type)
        return smps
    
    def generate_agent_hypothesis(self,
                                 agent_id: str,
                                 hypothesis_data: Dict,
                                 supporting_uwm_refs: List[str]) -> StructuredMeaningPacket:
        """
        4.3 Agent-Proposed SMPs (Hypotheses)
        "Agents may propose interpretations, predictions, optimizations, warnings"
        """
        
        # Validate agent has necessary permissions
        if not self._validate_agent_permissions(agent_id):
            raise PermissionError(f"Agent {agent_id} not authorized to propose SMPs")
        
        # Extract hypothesis components
        claims = self._extract_hypothesis_claims(hypothesis_data)
        
        # Create dependencies
        dependencies = SMPDependencies(
            uwm_atoms=supporting_uwm_refs,
            events=[],
            parent_smps=[],
            supporting_evidence=hypothesis_data.get('evidence', [])
        )
        
        # Create SMP
        smp = StructuredMeaningPacket(
            smp_id=self._generate_smp_id(),
            smp_type=SMPType.HYPOTHESIS,
            generation_timestamp=datetime.now(timezone.utc),
            claims=claims,
            scope=hypothesis_data.get('scope', 'unknown'),
            summary=hypothesis_data.get('summary', 'Agent hypothesis'),
            dependencies=dependencies,
            confidence_score=hypothesis_data.get('confidence', 0.4),
            proposed_by=agent_id,
            status=SMPStatus.RAW,
            constraints_required=hypothesis_data.get('constraints', []),
            iel_domains=hypothesis_data.get('iel_domains', [])
        )
        
        self._log_generation('agent_hypothesis', 1, agent_id)
        return smp
    
    def generate_diagnostic_smps(self,
                                system_state: Dict,
                                issue_types: List[str] = None) -> List[StructuredMeaningPacket]:
        """
        4.4 System-Generated SMPs (Meta)
        "The system itself can generate SMPs: conflict reports, coherence gaps"
        """
        smps = []
        
        issue_types = issue_types or ['conflict', 'coherence_gap', 'ambiguity']
        
        if 'conflict' in issue_types:
            smps.extend(self._detect_conflicts(system_state))
        
        if 'coherence_gap' in issue_types:
            smps.extend(self._detect_coherence_gaps(system_state))
        
        if 'ambiguity' in issue_types:
            smps.extend(self._detect_ambiguities(system_state))
        
        # Mark all as DIAGNOSTIC type
        for smp in smps:
            smp.smp_type = SMPType.DIAGNOSTIC
        
        self._log_generation('diagnostic', len(smps), str(issue_types))
        return smps
    
    def _generate_create_smps(self, target: Dict, evidence: Dict) -> List[StructuredMeaningPacket]:
        """Generate SMPs for CREATE events"""
        smps = []
        
        # SMP: New entity exists
        claims = [
            SMPClaim(
                claim_id=f"claim_create_{self.generation_counter}_{i}",
                predicate="exists",
                subject=target.get('identity', 'unknown'),
                object="world",
                modality="actuality",
                confidence=0.95,
                evidence=[evidence]
            )
        ]
        
        smp = StructuredMeaningPacket(
            smp_id=self._generate_smp_id(),
            smp_type=SMPType.GROUNDED,
            generation_timestamp=datetime.now(timezone.utc),
            claims=claims,
            scope=target.get('scope', 'system'),
            summary=f"New entity created: {target.get('identity', 'unknown')}",
            dependencies=SMPDependencies(
                uwm_atoms=[target.get('atom_id', '')],
                events=[evidence.get('event_id', '')],
                parent_smps=[],
                supporting_evidence=[evidence]
            ),
            confidence_score=0.95,
            proposed_by="system"
        )
        
        smps.append(smp)
        
        # Additional SMPs based on entity type
        entity_type = target.get('type', '')
        if entity_type == 'module':
            smps.extend(self._generate_module_specific_smps(target, evidence))
        elif entity_type == 'protocol':
            smps.extend(self._generate_protocol_specific_smps(target, evidence))
        
        return smps
    
    def _generate_smp_id(self) -> str:
        """Generate unique SMP ID"""
        self.generation_counter += 1
        timestamp = datetime.now(timezone.utc).timestamp()
        unique = hashlib.sha256(f"{timestamp}_{self.generation_counter}".encode()).hexdigest()[:16]
        return f"SMP:{unique}"

# ==================== DUAL ENRICHMENT SYSTEM ====================

class TetrahedralEnrichment:
    """
    System A from blueprint: Tetrahedral Evaluation
    Enriches SMPs through four "faces" of analysis
    """
    
    def __init__(self):
        self.faces = {
            'logic': LogicFace(),
            'domain': DomainFace(),
            'math': MathFace(),
            'critic': CriticFace()
        }
        
        self.enrichment_history = defaultdict(list)
        
    def enrich_smp(self, smp: StructuredMeaningPacket) -> StructuredMeaningPacket:
        """
        Enrich SMP through tetrahedral analysis
        Returns enriched SMP with added granularity and precision
        """
        original_id = smp.smp_id
        
        # Apply each face sequentially
        enriched_claims = smp.claims.copy()
        new_claims = []
        
        for face_name, face in self.faces.items():
            face_result = face.analyze(smp)
            
            # Add new claims discovered by this face
            new_claims.extend(face_result.get('new_claims', []))
            
            # Refine existing claims
            refined = face_result.get('refined_claims', [])
            for refined_claim in refined:
                # Update or replace existing claims
                enriched_claims = self._merge_claims(enriched_claims, refined_claim)
        
        # Add entirely new claims
        enriched_claims.extend(new_claims)
        
        # Create enriched SMP
        enriched_smp = StructuredMeaningPacket(
            smp_id=f"{smp.smp_id}_A{smp.enrichment_rounds}",
            smp_type=smp.smp_type,
            generation_timestamp=datetime.now(timezone.utc),
            claims=enriched_claims,
            scope=smp.scope,
            summary=f"{smp.summary} [Enriched by System A]",
            dependencies=smp.dependencies,
            confidence_score=self._recompute_confidence(smp, enriched_claims),
            proposed_by=smp.proposed_by,
            status=SMPStatus.ENRICHING,
            enrichment_rounds=smp.enrichment_rounds + 1,
            evaluation_history=smp.evaluation_history.copy(),
            constraints_required=smp.constraints_required,
            iel_domains=smp.iel_domains,
            metadata={
                **smp.metadata,
                'enriched_by': 'System_A',
                'original_smp': original_id,
                'new_claims_added': len(new_claims),
                'claims_refined': len(enriched_claims) - len(smp.claims)
            }
        )
        
        # Log enrichment
        self.enrichment_history[original_id].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system': 'A',
            'round': smp.enrichment_rounds + 1,
            'changes': {
                'new_claims': len(new_claims),
                'refined_claims': len(enriched_claims) - len(smp.claims)
            }
        })
        
        return enriched_smp
    
    class LogicFace:
        """PXL logic analysis face"""
        def analyze(self, smp):
            # Analyze claims for logical consistency
            # Apply PXL triune grounding
            new_claims = []
            refined_claims = []
            
            for claim in smp.claims:
                # Check 𝕀₁ grounding
                if not self._check_identity_grounding(claim):
                    refined = self._add_identity_qualifiers(claim)
                    refined_claims.append(refined)
                
                # Generate logical implications
                implications = self._generate_logical_implications(claim)
                new_claims.extend(implications)
            
            return {
                'new_claims': new_claims,
                'refined_claims': refined_claims,
                'logic_score': self._compute_logic_score(smp)
            }

class ConstraintEnrichment:
    """
    System B from blueprint: Constraint Pipeline
    Enriches SMPs through constraint checking and optimization
    """
    
    def __init__(self):
        self.pipeline_stages = {
            'sign': SignStage(),      # Truth/identity
            'mind': MindStage(),      # Coherence/structure
            'bridge': BridgeStage(),  # Semantic plausibility
            'mesh': MeshStage(),      # Holistic integration
            'tot': TOTStage(),        # Optimization
            'tlm': TLMStage()         # Locking
        }
        
        self.constraint_violations = defaultdict(list)
        
    def enrich_smp(self, smp: StructuredMeaningPacket) -> StructuredMeaningPacket:
        """
        Enrich SMP through constraint pipeline
        Adds constraint-aware refinements
        """
        
        enriched_claims = smp.claims.copy()
        constraint_checks = []
        
        # Process through each pipeline stage
        for stage_name, stage in self.pipeline_stages.items():
            stage_result = stage.process(smp, enriched_claims)
            
            # Apply stage-specific enrichments
            if stage_result.get('enrichments'):
                for enrichment in stage_result['enrichments']:
                    enriched_claims = self._apply_enrichment(enriched_claims, enrichment)
            
            # Record constraint checks
            if stage_result.get('checks'):
                constraint_checks.extend(stage_result['checks'])
            
            # Stop if fatal constraint violation
            if stage_result.get('fatal_violation'):
                smp.status = SMPStatus.REJECTED
                smp.metadata['rejection_reason'] = f"{stage_name}: {stage_result['fatal_violation']}"
                return smp
        
        # Create enriched SMP
        enriched_smp = StructuredMeaningPacket(
            smp_id=f"{smp.smp_id}_B{smp.enrichment_rounds}",
            smp_type=smp.smp_type,
            generation_timestamp=datetime.now(timezone.utc),
            claims=enriched_claims,
            scope=smp.scope,
            summary=f"{smp.summary} [Constraint-enriched by System B]",
            dependencies=smp.dependencies,
            confidence_score=self._adjust_confidence_for_constraints(smp, constraint_checks),
            proposed_by=smp.proposed_by,
            status=SMPStatus.ENRICHING,
            enrichment_rounds=smp.enrichment_rounds + 1,
            evaluation_history=smp.evaluation_history + [{
                'system': 'B',
                'constraint_checks': constraint_checks,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }],
            constraints_required=smp.constraints_required,
            iel_domains=smp.iel_domains,
            metadata={
                **smp.metadata,
                'enriched_by': 'System_B',
                'constraint_checks_passed': len([c for c in constraint_checks if c['passed']]),
                'constraint_checks_failed': len([c for c in constraint_checks if not c['passed']]),
                'pipeline_stages_completed': list(self.pipeline_stages.keys())
            }
        )
        
        return enriched_smp

# ==================== INFORMATION EXPLOSION ORCHESTRATOR ====================

class InformationExplosionOrchestrator:
    """
    Orchestrates the back-and-forth enrichment between System A and System B
    Implements the "information explosion" pattern
    """
    
    def __init__(self, 
                 max_rounds: int = 5,
                 confidence_threshold: float = 0.8):
        
        self.system_a = TetrahedralEnrichment()
        self.system_b = ConstraintEnrichment()
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
        
        # Tracking
        self.explosion_log = []
        self.volume_metrics = {
            'total_claims_generated': 0,
            'unique_claims_final': 0,
            'enrichment_rounds_completed': 0,
            'confidence_gain_per_round': []
        }
        
    def process_smp(self, 
                   raw_smp: StructuredMeaningPacket,
                   target_confidence: float = None) -> StructuredMeaningPacket:
        """
        Process SMP through multiple rounds of dual enrichment
        Implements: "gains granularity, volume, and precision"
        """
        
        if target_confidence is None:
            target_confidence = self.confidence_threshold
        
        current_smp = raw_smp
        round_number = 0
        
        self.explosion_log.append({
            'start': datetime.now(timezone.utc).isoformat(),
            'original_smp': raw_smp.smp_id,
            'initial_claims': len(raw_smp.claims),
            'initial_confidence': raw_smp.confidence_score
        })
        
        while (round_number < self.max_rounds and 
               current_smp.confidence_score < target_confidence and
               current_smp.status != SMPStatus.REJECTED):
            
            round_number += 1
            
            # Round A: System A enrichment
            smp_a = self.system_a.enrich_smp(current_smp)
            
            # Round B: System B enrichment (on A's output)
            smp_b = self.system_b.enrich_smp(smp_a)
            
            # Check for convergence or divergence
            convergence = self._check_convergence(current_smp, smp_b)
            
            if convergence.get('diverged'):
                # Information explosion detected - good!
                current_smp = smp_b
                self._log_explosion_round(round_number, 'explosion', 
                                        len(current_smp.claims), 
                                        current_smp.confidence_score)
            elif convergence.get('stagnant'):
                # No new information - stop
                self._log_explosion_round(round_number, 'stagnant',
                                        len(current_smp.claims),
                                        current_smp.confidence_score)
                break
            else:
                # Steady enrichment
                current_smp = smp_b
                self._log_explosion_round(round_number, 'enriched',
                                        len(current_smp.claims),
                                        current_smp.confidence_score)
        
        # Final processing
        final_smp = self._finalize_smp(current_smp, round_number)
        
        # Update metrics
        self.volume_metrics['total_claims_generated'] += len(final_smp.claims)
        self.volume_metrics['unique_claims_final'] = len(set(
            c.claim_id for c in final_smp.claims
        ))
        self.volume_metrics['enrichment_rounds_completed'] += round_number
        
        if round_number > 0:
            confidence_gain = final_smp.confidence_score - raw_smp.confidence_score
            self.volume_metrics['confidence_gain_per_round'].append(
                confidence_gain / round_number
            )
        
        return final_smp
    
    def _check_convergence(self, 
                          before: StructuredMeaningPacket,
                          after: StructuredMeaningPacket) -> Dict:
        """Check if enrichment is converging, diverging, or stagnant"""
        
        claim_change = len(after.claims) - len(before.claims)
        confidence_change = after.confidence_score - before.confidence_score
        
        # Information explosion: Significant new claims
        if claim_change > len(before.claims) * 0.5:  # 50% growth
            return {'diverged': True, 'growth_rate': claim_change / len(before.claims)}
        
        # Stagnant: Minimal change
        if abs(claim_change) <= 2 and abs(confidence_change) < 0.05:
            return {'stagnant': True, 'reason': 'minimal_change'}
        
        # Steady enrichment
        return {'enriching': True, 'claim_change': claim_change, 'confidence_change': confidence_change}
    
    def _finalize_smp(self, smp: StructuredMeaningPacket, rounds: int) -> StructuredMeaningPacket:
        """Prepare SMP for evaluation pipeline"""
        
        final_smp = StructuredMeaningPacket(
            smp_id=f"{smp.smp_id}_FINAL",
            smp_type=smp.smp_type,
            generation_timestamp=datetime.now(timezone.utc),
            claims=smp.claims,
            scope=smp.scope,
            summary=f"{smp.summary} [Explosion processed: {rounds} rounds]",
            dependencies=smp.dependencies,
            confidence_score=smp.confidence_score,
            proposed_by=smp.proposed_by,
            status=SMPStatus.EVALUATING,  # Ready for tetrahedral evaluation
            enrichment_rounds=smp.enrichment_rounds,
            evaluation_history=smp.evaluation_history,
            constraints_required=smp.constraints_required,
            iel_domains=smp.iel_domains,
            metadata={
                **smp.metadata,
                'information_explosion_rounds': rounds,
                'final_claim_count': len(smp.claims),
                'explosion_processed': True,
                'ready_for_evaluation': True
            }
        )
        
        return final_smp

# ==================== SMP EVALUATION PIPELINE ====================

class SMPEvaluationPipeline:
    """
    6. SMP Evaluation Pipeline (Connection to Earlier Blueprint)
    Every SMP goes through tetrahedral evaluation and constraint pipeline
    """
    
    def __init__(self):
        self.tetrahedral_evaluator = TetrahedralEvaluator()
        self.constraint_pipeline = ConstraintPipeline()
        self.approval_gate = ApprovalGate()
        
        # Routing destinations per blueprint section 7
        self.destination_routers = {
            'system_memory': SystemMemoryRouter(),
            'planning_substrates': PlanningRouter(),
            'policy_tables': PolicyRouter(),
            'agent_advisory': AdvisoryRouter()
        }
    
    def evaluate_smp(self, smp: StructuredMeaningPacket) -> Dict:
        """
        Full evaluation pipeline for an SMP
        Returns evaluation results and routing decisions
        """
        
        # Step 1: Tetrahedral Evaluation (System A from earlier blueprint)
        tetrahedral_result = self.tetrahedral_evaluator.evaluate(smp)
        
        if not tetrahedral_result['approved']:
            return {
                'status': 'REJECTED',
                'stage': 'tetrahedral_evaluation',
                'reasons': tetrahedral_result['violations'],
                'smp': smp
            }
        
        # Step 2: Constraint Pipeline (System B from earlier blueprint)
        constraint_result = self.constraint_pipeline.process(smp)
        
        if not constraint_result['passed']:
            return {
                'status': 'REJECTED',
                'stage': 'constraint_pipeline',
                'reasons': constraint_result['failures'],
                'smp': smp
            }
        
        # Step 3: Final Approval Gate
        approval_result = self.approval_gate.check(smp, tetrahedral_result, constraint_result)
        
        if approval_result['approved']:
            smp.status = SMPStatus.APPROVED
            smp.confidence_score = approval_result['final_confidence']
            
            # Determine routing per blueprint section 7
            routing = self._determine_routing(smp, approval_result)
            
            return {
                'status': 'APPROVED',
                'smp': smp,
                'routing': routing,
                'evaluation_summary': {
                    'tetrahedral_score': tetrahedral_result['score'],
                    'constraint_score': constraint_result['score'],
                    'final_confidence': approval_result['final_confidence'],
                    'recommended_actions': approval_result['recommendations']
                }
            }
        else:
            smp.status = SMPStatus.REJECTED
            return {
                'status': 'REJECTED',
                'stage': 'approval_gate',
                'reasons': approval_result['rejection_reasons'],
                'smp': smp
            }
    
    def _determine_routing(self, 
                          smp: StructuredMeaningPacket,
                          approval_result: Dict) -> Dict:
        """
        7. What Happens to Approved SMPs
        Route to appropriate destinations
        """
        
        routing = {
            'destinations': [],
            'primary_destination': None
        }
        
        # Determine primary destination based on SMP type
        if smp.smp_type == SMPType.GROUNDED:
            # Grounded SMPs go to system memory and planning
            routing['destinations'].extend(['system_memory', 'planning_substrates'])
            routing['primary_destination'] = 'system_memory'
        
        elif smp.smp_type == SMPType.ANALYTIC:
            # Analytic SMPs go to policy tables and advisory
            routing['destinations'].extend(['policy_tables', 'agent_advisory'])
            routing['primary_destination'] = 'policy_tables'
        
        elif smp.smp_type == SMPType.DIAGNOSTIC:
            # Diagnostic SMPs go to advisory and may trigger alerts
            routing['destinations'].extend(['agent_advisory'])
            routing['primary_destination'] = 'agent_advisory'
            
            if approval_result.get('requires_alert'):
                routing['destinations'].append('alert_system')
        
        elif smp.smp_type == SMPType.HYPOTHESIS:
            # Hypotheses go to advisory for review
            routing['destinations'].extend(['agent_advisory'])
            routing['primary_destination'] = 'agent_advisory'
            
            # High-confidence hypotheses may also go to planning
            if smp.confidence_score > 0.7:
                routing['destinations'].append('planning_substrates')
        
        # Execute routing
        for dest in routing['destinations']:
            if dest in self.destination_routers:
                self.destination_routers[dest].route(smp, approval_result)
        
        return routing

# ==================== COMPLETE SMP SYSTEM INTEGRATION ====================

class StructuredMeaningSystem:
    """
    Complete SMP system integrating all components
    Follows blueprint architecture end-to-end
    """
    
    def __init__(self, uwm_interface=None):
        # Core components
        self.generator = SMPGenerator(uwm_interface)
        self.explosion_orchestrator = InformationExplosionOrchestrator()
        self.evaluation_pipeline = SMPEvaluationPipeline()
        
        # Storage
        self.smp_store = SMPStore()
        self.evaluation_log = EvaluationLog()
        
        # Agent interface
        self.agent_interface = AgentSMPInterface(self)
        
        # Metrics
        self.metrics = {
            'smps_generated': 0,
            'smps_approved': 0,
            'smps_rejected': 0,
            'total_claims_processed': 0,
            'average_enrichment_rounds': 0,
            'confidence_distribution': defaultdict(int)
        }
    
    def process_uwm_event(self, uwm_event: Dict) -> List[Dict]:
        """
        Process UWM event through full SMP pipeline
        Returns processed SMPs with status
        """
        # Step 1: Generate SMPs from UWM event
        raw_smps = self.generator.generate_from_uwm_event(uwm_event)
        
        results = []
        for raw_smp in raw_smps:
            # Step 2: Information explosion enrichment
            enriched_smp = self.explosion_orchestrator.process_smp(raw_smp)
            
            # Step 3: Evaluation pipeline
            evaluation_result = self.evaluation_pipeline.evaluate_smp(enriched_smp)
            
            # Step 4: Store results
            if evaluation_result['status'] == 'APPROVED':
                self.smp_store.store_approved(enriched_smp)
                self.metrics['smps_approved'] += 1
            else:
                self.smp_store.store_rejected(enriched_smp, evaluation_result)
                self.metrics['smps_rejected'] += 1
            
            # Update metrics
            self.metrics['smps_generated'] += 1
            self.metrics['total_claims_processed'] += len(enriched_smp.claims)
            
            # Round confidence to nearest 0.1 for distribution
            conf_bucket = round(enriched_smp.confidence_score * 10) / 10
            self.metrics['confidence_distribution'][conf_bucket] += 1
            
            results.append(evaluation_result)
        
        return results
    
