"""
THE COGNITIVE RESISTOR
Tetrahedral Base for PXL-Grounded Critical Analysis

Purpose: To prevent epistemic closure by systematically challenging
PXL's own assumptions and conclusions using PXL's own principles.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

# ==================== CRITIC ARCHETYPES ====================

class CriticMode(Enum):
    """Modes of critical engagement with PXL"""
    INTERNAL_COHERENCE = "internal_coherence"      # Tests PXL's self-consistency
    EXTERNAL_CORRESPONDENCE = "external_correspondence" # Tests alignment with reality
    PRAGMATIC_EFFECTIVENESS = "pragmatic_effectiveness" # Tests practical utility
    METAPHYSICAL_ADEQUACY = "metaphysical_adequacy"     # Tests metaphysical completeness
    TRINUNE_SELF_CRITIQUE = "triune_self_critique"      # Tests each person against others

class CriticArchetype(Enum):
    """Personae for different critical perspectives"""
    THE_SCEPTIC = "sceptic"          # Demands justification, questions foundations
    THE_PRAGMATIST = "pragmatist"    # Tests practical consequences
    THE_MYSTIC = "mystic"            # Questions conceptual boundaries
    THE_DIALECTICIAN = "dialectician" # Seeks synthesis through contradiction
    THE_FORMALIST = "formalist"      # Checks mathematical/formal rigor
    THE_PHENOMENOLOGIST = "phenomenologist" # Tests against lived experience
    THE_NOMINALIST = "nominalist"    # Questions ontological commitments
    THE_HERMENEUT = "hermeneut"      # Tests interpretative adequacy

# ==================== CRITICAL CHALLENGE TYPES ====================

@dataclass
class CriticalChallenge:
    """A specific challenge to PXL-grounded reasoning"""
    id: str
    challenge_type: str
    target_aspect: str  # Which PXL aspect is challenged
    formulation: str    # Natural language challenge
    formal_expression: Optional[str] = None  # PXL-formalized challenge
    grounding_persona: CriticArchetype = CriticArchetype.THE_SCEPTIC
    severity: float = 0.5  # 0.0-1.0 impact on coherence
    resolution_status: str = "pending"
    
    # PXL grounding for the challenge itself
    pxl_grounding: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.challenge_type,
            "target": self.target_aspect,
            "formulation": self.formulation,
            "formal": self.formal_expression,
            "persona": self.grounding_persona.value,
            "severity": self.severity,
            "status": self.resolution_status,
            "pxl_grounding": self.pxl_grounding
        }

# ==================== THE CRITICAL LENS SYSTEM ====================

class CriticalLens:
    """
    A specific critical perspective that can be applied to PXL analyses
    Each lens embodies a different philosophical challenge
    """
    
    def __init__(self, archetype: CriticArchetype):
        self.archetype = archetype
        self.mode = self._determine_mode(archetype)
        self.challenge_templates = self._load_templates(archetype)
        
    def _determine_mode(self, archetype: CriticArchetype) -> CriticMode:
        """Determine primary mode for each archetype"""
        mapping = {
            CriticArchetype.THE_SCEPTIC: CriticMode.INTERNAL_COHERENCE,
            CriticArchetype.THE_FORMALIST: CriticMode.INTERNAL_COHERENCE,
            CriticArchetype.THE_PRAGMATIST: CriticMode.PRAGMATIC_EFFECTIVENESS,
            CriticArchetype.THE_PHENOMENOLOGIST: CriticMode.EXTERNAL_CORRESPONDENCE,
            CriticArchetype.THE_NOMINALIST: CriticMode.METAPHYSICAL_ADEQUACY,
            CriticArchetype.THE_MYSTIC: CriticMode.TRINUNE_SELF_CRITIQUE,
            CriticArchetype.THE_DIALECTICIAN: CriticMode.TRINUNE_SELF_CRITIQUE,
            CriticArchetype.THE_HERMENEUT: CriticMode.EXTERNAL_CORRESPONDENCE
        }
        return mapping.get(archetype, CriticMode.INTERNAL_COHERENCE)
    
    def _load_templates(self, archetype: CriticArchetype) -> Dict[str, List[str]]:
        """Load challenge templates for this archetype"""
        templates = {
            "internal_coherence": [
                "How does this proposition maintain self-identity under modal variation?",
                "What hidden contradictions might emerge under different interpretations?",
                "Does this claim preserve coherence across all three triune persons?"
            ],
            "metaphysical_adequacy": [
                "What ontological commitments are being smuggled in unexamined?",
                "Could this be reformulated without the metaphysical baggage?",
                "What alternative metaphysical frameworks could account for this?"
            ],
            "pragmatic_effectiveness": [
                "What practical difference does accepting this claim make?",
                "How would rejecting this claim change observable outcomes?",
                "What action constraints does this proposition actually impose?"
            ],
            "triune_self_critique": [
                "Does 𝕀₁'s emphasis on identity obscure important distinctions?",
                "Does 𝕀₂'s prohibition of contradiction suppress legitimate paradoxes?",
                "Does 𝕀₃'s excluded middle exclude meaningful middle grounds?"
            ]
        }
        
        # Archetype-specific emphasis
        emphasis = {
            CriticArchetype.THE_SCEPTIC: ["internal_coherence", "metaphysical_adequacy"],
            CriticArchetype.THE_PRAGMATIST: ["pragmatic_effectiveness"],
            CriticArchetype.THE_MYSTIC: ["triune_self_critique"],
            CriticArchetype.THE_DIALECTICIAN: ["triune_self_critique", "internal_coherence"],
            CriticArchetype.THE_FORMALIST: ["internal_coherence"],
            CriticArchetype.THE_PHENOMENOLOGIST: ["pragmatic_effectiveness"],
            CriticArchetype.THE_NOMINALIST: ["metaphysical_adequacy"],
            CriticArchetype.THE_HERMENEUT: ["internal_coherence", "pragmatic_effectiveness"]
        }
        
        # Combine templates based on archetype emphasis
        archetype_templates = {}
        for focus in emphasis.get(archetype, ["internal_coherence"]):
            if focus in templates:
                archetype_templates[focus] = templates[focus]
                
        return archetype_templates
    
    def generate_challenge(self, pxl_analysis: Dict, 
                          target_aspect: str = None) -> CriticalChallenge:
        """Generate a specific challenge based on PXL analysis"""
        
        # Determine challenge type based on analysis weaknesses
        challenge_type = self._identify_vulnerability(pxl_analysis)
        
        # Select template
        templates = self.challenge_templates.get(challenge_type, [])
        if not templates:
            templates = self.challenge_templates.get("internal_coherence", [])
            
        formulation = np.random.choice(templates) if templates else "Is this claim justified?"
        
        # Formalize if possible
        formal = self._formalize_challenge(formulation, pxl_analysis)
        
        # Create challenge
        challenge_id = f"critic_{self.archetype.value}_{datetime.now().timestamp()}"
        
        return CriticalChallenge(
            id=challenge_id,
            challenge_type=challenge_type,
            target_aspect=target_aspect or self._identify_target(pxl_analysis),
            formulation=formulation,
            formal_expression=formal,
            grounding_persona=self.archetype,
            severity=self._calculate_severity(pxl_analysis, challenge_type)
        )
    
    def _identify_vulnerability(self, pxl_analysis: Dict) -> str:
        """Identify the most vulnerable aspect of the PXL analysis"""
        scores = {}
        
        # Check universal filter results
        uf = pxl_analysis.get("universal_filter", {})
        if uf:
            if not uf.get("all_persons_coherent", True):
                scores["triune_self_critique"] = 0.9
            if not uf.get("grounding_complete", True):
                scores["internal_coherence"] = 0.7
        
        # Check paradox status
        if not pxl_analysis.get("paradox_free", True):
            scores["internal_coherence"] = max(scores.get("internal_coherence", 0), 0.8)
            
        # Check grounding completeness
        grounding = pxl_analysis.get("grounding", {})
        if grounding:
            identity_score = grounding.get("identity", {}).get("score", 0)
            if identity_score < 0.7:
                scores["metaphysical_adequacy"] = 0.6
                
        # Default to internal coherence
        if not scores:
            scores["internal_coherence"] = 0.5
            
        # Return highest vulnerability
        return max(scores.items(), key=lambda x: x[1])[0]

# ==================== THE TETRAHEDRAL CRITIC ENGINE ====================

class TetrahedralCritic:
    """
    The fourth face of the tetrahedron - systematically applies
    criticism to PXL-grounded analyses from multiple perspectives.
    
    Architecture: Four interconnected critical modes forming a tetrahedron:
    1. Internal Coherence Critic (Top vertex - tests self-consistency)
    2. External Correspondence Critic 
    3. Pragmatic Effectiveness Critic
    4. Triune Self-Critique Critic (Base - tests PXL's own foundations)
    """
    
    def __init__(self, pxl_system):
        self.pxl = pxl_system
        
        # Initialize all critic archetypes
        self.critic_lenses = {
            archetype: CriticalLens(archetype) 
            for archetype in CriticArchetype
        }
        
        # Challenge registry
        self.active_challenges: Dict[str, CriticalChallenge] = {}
        self.resolved_challenges: Dict[str, Dict] = {}
        
        # Critical memory - tracks patterns in successful/unsuccessful critiques
        self.critical_patterns = {
            "effective_challenges": [],  # Challenges that revealed genuine issues
            "resilient_aspects": [],    # PXL aspects that withstand criticism
            "blind_spots": [],          # Areas where criticism is consistently ineffective
            "dialectical_syntheses": [] # Successful integrations of critique
        }
        
        # Critical energy level - determines intensity of critique
        self.critical_energy = 0.7  # 0.0-1.0, starts moderately critical
        
    def apply_critique(self, pxl_analysis: Dict, 
                      content: Any = None,
                      intensity: float = None) -> Dict:
        """
        Apply systematic critique to a PXL analysis
        Returns critique results and potentially modified analysis
        """
        if intensity is None:
            intensity = self.critical_energy
            
        # Phase 1: Generate challenges from all perspectives
        challenges = self._generate_comprehensive_challenges(pxl_analysis, intensity)
        
        # Phase 2: Apply PXL to the challenges themselves (meta-critique)
        validated_challenges = self._validate_challenges_against_pxl(challenges)
        
        # Phase 3: Assess impact on original analysis
        impact_assessment = self._assess_critical_impact(pxl_analysis, validated_challenges)
        
        # Phase 4: Generate dialectical synthesis if needed
        synthesis = None
        if impact_assessment["requires_synthesis"]:
            synthesis = self._generate_dialectical_synthesis(
                pxl_analysis, validated_challenges, impact_assessment
            )
            
        # Phase 5: Update critical patterns
        self._update_critical_patterns(validated_challenges, impact_assessment, synthesis)
        
        # Phase 6: Adjust critical energy based on outcomes
        self._adjust_critical_energy(impact_assessment)
        
        return {
            "original_analysis": pxl_analysis,
            "challenges_generated": len(challenges),
            "validated_challenges": [c.to_dict() for c in validated_challenges],
            "impact_assessment": impact_assessment,
            "dialectical_synthesis": synthesis,
            "critical_energy_adjustment": self.critical_energy,
            "pattern_updates": self._get_pattern_updates_summary(),
            "tetrahedral_balance": self._calculate_tetrahedral_balance()
        }
    
    def _generate_comprehensive_challenges(self, pxl_analysis: Dict, 
                                         intensity: float) -> List[CriticalChallenge]:
        """Generate challenges from multiple critical perspectives"""
        challenges = []
        
        # Determine which archetypes to activate based on intensity
        active_archetypes = self._select_archetypes(intensity)
        
        for archetype in active_archetypes:
            lens = self.critic_lenses[archetype]
            
            # Generate primary challenge
            primary = lens.generate_challenge(pxl_analysis)
            challenges.append(primary)
            
            # Generate secondary challenges if intensity high
            if intensity > 0.7:
                secondary = lens.generate_challenge(
                    pxl_analysis, 
                    target_aspect=self._identify_secondary_target(pxl_analysis, primary)
                )
                challenges.append(secondary)
                
        return challenges
    
    def _validate_challenges_against_pxl(self, challenges: List[CriticalChallenge]) -> List[CriticalChallenge]:
        """Apply PXL's own criteria to validate the challenges themselves"""
        validated = []
        
        for challenge in challenges:
            # Check if challenge is itself PXL-coherent
            if challenge.formal_expression:
                # Analyze the challenge as a PXL proposition
                challenge_analysis = self.pxl.analyze_proposition(
                    challenge.formal_expression,
                    {"source": "cognitive_resistor", "challenge_id": challenge.id}
                )
                
                # Only accept challenges that pass basic PXL coherence
                if challenge_analysis.get("paradox_free", False):
                    challenge.pxl_grounding = {
                        "coherent": True,
                        "analysis": challenge_analysis,
                        "universal_filter": challenge_analysis.get("universal_filter", {})
                    }
                    challenge.resolution_status = "pxl_validated"
                    validated.append(challenge)
                else:
                    challenge.resolution_status = "pxl_incoherent"
                    # Store even incoherent challenges for pattern learning
                    self._store_incoherent_challenge(challenge, challenge_analysis)
            else:
                # Non-formalized challenges get provisional validation
                challenge.resolution_status = "provisional"
                validated.append(challenge)
                
        return validated
    
    def _assess_critical_impact(self, pxl_analysis: Dict, 
                              challenges: List[CriticalChallenge]) -> Dict:
        """Assess how challenges impact the original analysis"""
        
        impact_scores = {
            "identity_undermined": 0.0,     # 𝕀₁ impact
            "contradiction_exposed": 0.0,   # 𝕀₂ impact  
            "bivalence_challenged": 0.0,    # 𝕀₃ impact
            "grounding_weakened": 0.0,      # Overall grounding impact
            "practical_consequences": 0.0,  # Pragmatic impact
        }
        
        effective_challenges = []
        
        for challenge in challenges:
            # Assess impact based on challenge type and severity
            challenge_impact = self._calculate_challenge_impact(challenge, pxl_analysis)
            
            # Aggregate impacts
            for key in impact_scores:
                impact_scores[key] = max(impact_scores[key], challenge_impact.get(key, 0))
                
            # Track effective challenges
            if challenge_impact.get("effective", False):
                effective_challenges.append({
                    "challenge": challenge.to_dict(),
                    "impact": challenge_impact
                })
                
        # Determine if synthesis is required
        requires_synthesis = any(score > 0.6 for score in impact_scores.values())
        
        # Calculate overall critical impact
        max_impact = max(impact_scores.values()) if impact_scores else 0
        overall_impact = np.mean(list(impact_scores.values())) if impact_scores else 0
        
        return {
            "impact_scores": impact_scores,
            "max_impact": max_impact,
            "overall_impact": overall_impact,
            "effective_challenges": effective_challenges,
            "requires_synthesis": requires_synthesis,
            "analysis_integrity": 1.0 - overall_impact,  # How much survives criticism
            "recommendation": self._generate_recommendation(impact_scores, overall_impact)
        }
    
    def _generate_dialectical_synthesis(self, original_analysis: Dict,
                                       challenges: List[CriticalChallenge],
                                       impact: Dict) -> Dict:
        """
        Generate a dialectical synthesis that integrates valid criticism
        while preserving PXL's core principles
        """
        
        # Extract valid insights from challenges
        valid_insights = []
        for challenge in challenges:
            if challenge.resolution_status == "pxl_validated":
                insight = self._extract_insight_from_challenge(challenge)
                if insight:
                    valid_insights.append(insight)
                    
        if not valid_insights:
            return {"status": "no_valid_insights", "synthesis": original_analysis}
            
        # Apply triune synthesis process
        synthesis_stages = {
            "thesis": original_analysis,
            "antithesis": {
                "challenges": [c.to_dict() for c in challenges],
                "impact_assessment": impact,
                "valid_insights": valid_insights
            },
            "synthesis": self._perform_triune_synthesis(original_analysis, valid_insights)
        }
        
        # Test synthesis against PXL
        synthesis_analysis = self.pxl.analyze_proposition(
            str(synthesis_stages["synthesis"]),
            {"source": "dialectical_synthesis", "stage": "post_critique"}
        )
        
        synthesis_stages["synthesis_validation"] = synthesis_analysis
        
        return synthesis_stages
    
    def _perform_triune_synthesis(self, original: Dict, insights: List[Dict]) -> Dict:
        """Perform synthesis respecting triune structure"""
        
        synthesis = original.copy()
        
        # 𝕀₁ Synthesis: Preserve identity while incorporating distinctions
        if "identity_issues" in [i.get("type") for i in insights]:
            synthesis["grounding"] = synthesis.get("grounding", {})
            synthesis["grounding"]["identity"] = {
                **synthesis["grounding"].get("identity", {}),
                "synthesized": True,
                "incorporated_critique": True,
                "enhanced_distinctions": True
            }
            
        # 𝕀₂ Synthesis: Clarify boundaries without creating false contradictions
        if "contradiction_issues" in [i.get("type") for i in insights]:
            # Add nuance to apparent contradictions
            synthesis["contradiction_handling"] = {
                "explicit_boundaries": True,
                "recognized_pseudo_contradictions": True,
                "dialectical_resolution_applied": True
            }
            
        # 𝕀₃ Synthesis: Acknowledge middle grounds while maintaining bivalence
        if "bivalence_issues" in [i.get("type") for i in insights]:
            synthesis["modal_structure"] = {
                **synthesis.get("modal_structure", {}),
                "acknowledged_gradients": True,
                "preserved_bivalence_core": True,
                "contextual_applications": True
            }
            
        synthesis["meta_status"] = {
            "dialectically_synthesized": True,
            "critique_integrated": True,
            "synthesis_timestamp": datetime.now().isoformat(),
            "original_coherence_preserved": original.get("coherence_score", 0)
        }
        
        return synthesis

# ==================== ADVANCED CRITICAL FUNCTIONS ====================

class SelfReferentialCritic:
    """
    Specialized critic that applies PXL to PXL itself
    Tests the formal system's own coherence and limitations
    """
    
    def __init__(self, pxl_system):
        self.pxl = pxl_system
        
    def critique_pxl_foundations(self) -> Dict:
        """Apply PXL's own criteria to PXL's foundations"""
        
        critiques = []
        
        # Test each axiom against PXL's own criteria
        for i, axiom in enumerate(self.pxl.axioms):
            critique = self._critique_axiom(axiom, i)
            critiques.append(critique)
            
        # Test triune structure coherence
        triune_critique = self._critique_triune_structure()
        critiques.append(triune_critique)
        
        # Test paradox resolution consistency
        paradox_critique = self._critique_paradox_resolutions()
        critiques.append(paradox_critique)
        
        return {
            "axiom_critiques": critiques,
            "overall_coherence": self._calculate_pxl_self_coherence(critiques),
            "recommendations": self._generate_foundation_recommendations(critiques)
        }
    
    def _critique_axiom(self, axiom: Dict, index: int) -> Dict:
        """Critique a specific PXL axiom using PXL"""
        
        # Formulate axiom as proposition
        axiom_proposition = f"Axiom {index}: {axiom.get('formal', '')}"
        
        # Apply PXL analysis to itself
        analysis = self.pxl.analyze_proposition(
            axiom_proposition,
            {"source": "self_referential_critique", "axiom_index": index}
        )
        
        # Check for self-referential issues
        self_referential = self._detect_self_reference(axiom_proposition)
        
        # Check if axiom justifies its own application
        circular_justification = self._check_circular_justification(axiom)
        
        return {
            "axiom_index": index,
            "axiom_name": axiom.get("name", ""),
            "pxl_analysis": analysis,
            "self_referential": self_referential,
            "circular_justification": circular_justification,
            "coherence_score": analysis.get("coherence_score", 0),
            "recommendation": "retain" if analysis.get("paradox_free", False) else "reexamine"
        }

class PragmaticEffectivenessValidator:
    """
    Validates PXL analyses against practical effectiveness criteria
    Ensures metaphysical rigor doesn't come at cost of practical utility
    """
    
    def __init__(self):
        self.effectiveness_metrics = {
            "decision_guidance": 0.0,      # Helps make decisions
            "explanation_power": 0.0,      # Explains phenomena
            "prediction_accuracy": 0.0,    # Predicts outcomes
            "problem_solving": 0.0,        # Solves practical problems
            "communicability": 0.0,        # Can be communicated effectively
            "actionable_insights": 0.0     # Leads to actionable insights
        }
        
    def validate_effectiveness(self, pxl_analysis: Dict, 
                             context: Dict = None) -> Dict:
        """Validate the practical effectiveness of a PXL analysis"""
        
        scores = {}
        
        # Extract actionable content
        actionable = self._extract_actionable_content(pxl_analysis)
        
        # Test decision guidance
        if actionable:
            scores["decision_guidance"] = self._assess_decision_guidance(actionable)
            scores["actionable_insights"] = len(actionable) / 10.0  # Normalized
            
        # Test explanation power
        scores["explanation_power"] = self._assess_explanation_power(pxl_analysis)
        
        # Test communicability
        scores["communicability"] = self._assess_communicability(pxl_analysis)
        
        # Overall effectiveness
        overall = np.mean(list(scores.values())) if scores else 0
        
        return {
            "effectiveness_scores": scores,
            "overall_effectiveness": overall,
            "actionable_content": actionable,
            "pragmatic_status": "effective" if overall > 0.6 else "ineffective",
            "recommendations": self._generate_pragmatic_recommendations(scores, pxl_analysis)
        }

# ==================== INTEGRATION WITH FRACTAL MEMORY ====================

class CriticalMemoryIntegrator:
    """
    Integrates critical insights back into fractal memory
    Ensures criticism leads to system improvement, not just negation
    """
    
    def __init__(self, fractal_orchestrator):
        self.orchestrator = fractal_orchestrator
        self.critical_insights = []
        
    def integrate_critique(self, node_id: str, 
                          critique_results: Dict) -> Dict:
        """Integrate critique results back into fractal memory"""
        
        if node_id not in self.orchestrator.memory_graph:
            return {"error": f"Node {node_id} not found"}
            
        node = self.orchestrator.memory_graph[node_id]
        
        # Store critique with node
        node.pxl_analysis["critiques"] = node.pxl_analysis.get("critiques", [])
        node.pxl_analysis["critiques"].append(critique_results)
        
        # Update coherence score based on critique
        original_coherence = node.coherence_score
        impact = critique_results.get("impact_assessment", {}).get("overall_impact", 0)
        synthesis = critique_results.get("dialectical_synthesis")
        
        if synthesis and synthesis.get("synthesis_validation", {}).get("coherence_score", 0) > original_coherence:
            # Synthesis improved coherence
            node.coherence_score = synthesis["synthesis_validation"]["coherence_score"]
            node.pxl_analysis = synthesis["synthesis"]
            improvement = node.coherence_score - original_coherence
        else:
            # Critique validated existing coherence
            node.coherence_score = original_coherence * (1.0 - impact * 0.3)
            improvement = 0
            
        # Create critical insight node
        insight_id = f"insight_{node_id}_{datetime.now().timestamp()}"
        insight_node = FractalNode(
            id=insight_id,
            content={
                "source_critique": critique_results,
                "target_node": node_id,
                "improvement": improvement,
                "insight_type": "critical_integration"
            },
            pxl_analysis=critique_results,
            connections={node_id},
            depth=node.depth + 1
        )
        
        self.orchestrator.memory_graph[insight_id] = insight_node
        node.connections.add(insight_id)
        
        # Update critical patterns in orchestrator
        self._update_critical_patterns(critique_results, improvement)
        
        return {
            "node_updated": node_id,
            "new_coherence": node.coherence_score,
            "improvement": improvement,
            "insight_node_created": insight_id,
            "critical_integration_complete": True
        }

# ==================== MAIN CRITIC DEMONSTRATION ====================

def demonstrate_cognitive_resistor(pxl_system, fractal_orchestrator):
    """Demonstrate the tetrahedral critic in action"""
    
    print("\n" + "="*70)
    print("COGNITIVE RESISTOR DEMONSTRATION")
    print("Tetrahedral Base for PXL-Grounded Critical Analysis")
    print("="*70)
    
    # Initialize critic system
    print("\n1. Initializing Tetrahedral Critic...")
    critic = TetrahedralCritic(pxl_system)
    
    # Initialize specialized critics
    self_critic = SelfReferentialCritic(pxl_system)
    pragmatic_validator = PragmaticEffectivenessValidator()
    memory_integrator = CriticalMemoryIntegrator(fractal_orchestrator)
    
    # Test proposition from MESH argument
    test_proposition = "Logical structures exist necessarily as transcendental preconditions"
    print(f"\n2. Testing Proposition: {test_proposition}")
    
    # Get PXL analysis
    pxl_analysis = pxl_system.analyze_proposition(
        test_proposition,
        {"source": "critic_demo", "proposition_type": "metaphysical"}
    )
    
    print(f"   Initial PXL Analysis Complete")
    print(f"   Coherence Score: {pxl_analysis.get('coherence_score', 0):.2f}")
    print(f"   Paradox Free: {pxl_analysis.get('paradox_free', False)}")
    
    # Apply comprehensive critique
    print("\n3. Applying Tetrahedral Critique...")
    critique_results = critic.apply_critique(pxl_analysis, test_proposition)
    
    print(f"   Challenges Generated: {critique_results['challenges_generated']}")
    print(f"   Validated Challenges: {len(critique_results['validated_challenges'])}")
    
    impact = critique_results["impact_assessment"]
    print(f"   Overall Critical Impact: {impact['overall_impact']:.2f}")
    print(f"   Analysis Integrity: {impact['analysis_integrity']:.2f}")
    
    if critique_results.get("dialectical_synthesis"):
        print(f"   Dialectical Synthesis Required and Generated")
        synthesis = critique_results["dialectical_synthesis"]
        if synthesis.get("synthesis_validation"):
            synth_score = synthesis["synthesis_validation"].get("coherence_score", 0)
            print(f"   Synthesis Coherence: {synth_score:.2f}")
    
    # Apply self-referential critique to PXL itself
    print("\n4. Applying Self-Referential Critique to PXL Foundations...")
    self_critique = self_critic.critique_pxl_foundations()
    
    print(f"   Axioms Critiqued: {len(self_critique.get('axiom_critiques', []))}")
    print(f"   PXL Self-Coherence: {self_critique.get('overall_coherence', 0):.2f}")
    
    # Validate pragmatic effectiveness
    print("\n5. Validating Pragmatic Effectiveness...")
    pragmatic_validation = pragmatic_validator.validate_effectiveness(pxl_analysis)
    
    print(f"   Overall Effectiveness: {pragmatic_validation['overall_effectiveness']:.2f}")
    print(f"   Pragmatic Status: {pragmatic_validation['pragmatic_status']}")
    
    # Display critical archetypes in action
    print("\n6. Critical Archetypes Deployed:")
    for challenge in critique_results["validated_challenges"][:3]:  # Show first 3
        archetype = challenge.get("persona", "unknown")
        formulation = challenge.get("formulation", "")[:60] + "..."
        print(f"   • {archetype.upper()}: {formulation}")
    
    # Show tetrahedral balance
    print("\n7. Tetrahedral Balance Assessment:")
    balance = critique_results["tetrahedral_balance"]
    for dimension, score in balance.items():
        print(f"   {dimension}: {score:.2f}")
    
    # Integration recommendations
    print("\n8. Integration Recommendations:")
    if impact["recommendation"]:
        print(f"   Primary: {impact['recommendation']}")
    
    if critique_results.get("dialectical_synthesis"):
        print(f"   Secondary: Integrate dialectical synthesis")
        
    if pragmatic_validation["pragmatic_status"] == "ineffective":
        print(f"   Tertiary: Enhance pragmatic effectiveness")
    
    print("\n" + "="*70)
    print("COGNITIVE RESISTOR DEMONSTRATION COMPLETE")
    print("="*70)
    
    return {
        "critique_results": critique_results,
        "self_critique": self_critique,
        "pragmatic_validation": pragmatic_validation,
        "critical_energy": critic.critical_energy,
        "tetrahedral_balance": balance
    }

# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # This would integrate with your existing PXL system
    print("Tetrahedral Cognitive Resistor System")
    print("Designed as the critical base for PXL-grounded reasoning")
    print("\nKey Features:")
    print("1. Eight Critical Archetypes for comprehensive critique")
    print("2. Tetrahedral structure: Internal, External, Pragmatic, Self-Critique")  
    print("3. PXL-grounded challenges (critiques must be PXL-coherent)")
    print("4. Dialectical synthesis generation")
    print("5. Self-referential critique of PXL itself")
    print("6. Pragmatic effectiveness validation")
    print("7. Fractal memory integration")
    print("8. Adaptive critical energy based on effectiveness")
    
    # To integrate with your existing system:
    """
    1. Initialize with your PXL system:
       critic = TetrahedralCritic(your_pxl_system)
    
    2. Apply to any PXL analysis:
       critique = critic.apply_critique(pxl_analysis, content)
    
    3. Integrate results:
       integrator = CriticalMemoryIntegrator(your_fractal_orchestrator)
       integration = integrator.integrate_critique(node_id, critique)
    
    4. For self-critique of PXL:
       self_critique = SelfReferentialCritic(your_pxl_system).critique_pxl_foundations()
    """