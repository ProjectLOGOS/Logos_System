# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
"""
LOGOS Agent System (LAS) - Core Agent Architecture
=================================================

Independent agent system that orchestrates all LOGOS protocols.
Acts as the "will" behind the system, controlling when and how
UIP, SCP, and SOP systems activate and interact.

Architecture:
- SystemAgent: Internal system orchestration and decision-making
- UserAgent: Represents external human/system users  
- ExternalAgent: Handles API/service-to-service interactions
- AgentRegistry: Manages agent lifecycle and capabilities
- ProtocolOrchestrator: Controls protocol activation and coordination

Key Principles:
- Agents exist ABOVE protocols (agents use protocols, not contained by them)
- SystemAgent is the "cognitive will" that drives autonomous behavior
- Clean separation: Agents = Intelligence, Protocols = Execution
- Event-driven activation minimizes computational overhead
"""

import asyncio
import os
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .boot_system import SystemRuntime, initialize_runtime

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the LOGOS system"""
    SYSTEM = "system_agent"           # Internal system orchestration
    USER = "user_agent"               # External human users
    EXTERNAL = "external_agent"       # API/service interactions
    COGNITIVE = "cognitive_agent"     # Specialized cognitive processing


class ProtocolType(Enum):
    """Available protocols that agents can activate"""
    UIP = "user_interaction_protocol"
    SCP = "advanced_general_protocol"
    SOP = "system_operations_protocol"


@dataclass
class AgentCapabilities:
    """Defines what an agent can do"""
    can_activate_uip: bool = False
    can_access_scp: bool = False
    can_monitor_sop: bool = False
    can_create_agents: bool = False
    max_concurrent_operations: int = 10
    priority_level: int = 5  # 1-10, 10 = highest


@dataclass
class AgentContext:
    """Runtime context for agent operations"""
    session_id: str
    correlation_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    active_operations: Set[str] = field(default_factory=set)


class BaseAgent(ABC):
    """Abstract base class for all LOGOS agents"""

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        capabilities: AgentCapabilities,
        protocol_orchestrator: 'ProtocolOrchestrator'
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.protocol_orchestrator = protocol_orchestrator
        self.context = AgentContext(
            session_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4())
        )
        self.active = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent and its capabilities"""
        pass

    @abstractmethod
    async def execute_primary_function(self) -> None:
        """Execute the agent's main operational loop"""
        pass

    async def activate_uip(
        self,
        input_data: Dict[str, Any],
        processing_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Activate UIP processing as this agent"""
        if not self.capabilities.can_activate_uip:
            raise PermissionError(f"Agent {self.agent_id} cannot activate UIP")

        return await self.protocol_orchestrator.activate_uip_for_agent(
            agent=self,
            input_data=input_data,
            config=processing_config
        )

    async def access_scp(
        self,
        cognitive_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Access SCP cognitive processing capabilities"""
        if not self.capabilities.can_access_scp:
            raise PermissionError(f"Agent {self.agent_id} cannot access SCP")

        return await self.protocol_orchestrator.route_to_scp(
            agent=self,
            request=cognitive_request
        )

    async def monitor_sop(self) -> Dict[str, Any]:
        """Monitor SOP system status and metrics"""
        if not self.capabilities.can_monitor_sop:
            raise PermissionError(f"Agent {self.agent_id} cannot monitor SOP")

        return await self.protocol_orchestrator.get_sop_status(agent=self)


class SystemAgent(BaseAgent):
    """
    Primary system agent - the 'will' behind LOGOS
    
    Responsibilities:
    - Autonomous system decision-making
    - Protocol orchestration and coordination
    - Cognitive processing initiation
    - System optimization and learning
    - Background task management
    """

    def __init__(self, protocol_orchestrator: 'ProtocolOrchestrator'):
        capabilities = AgentCapabilities(
            can_activate_uip=True,
            can_access_scp=True,
            can_monitor_sop=True,
            can_create_agents=True,
            max_concurrent_operations=100,
            priority_level=10  # Highest priority
        )

        super().__init__(
            agent_id="SYSTEM_AGENT_PRIMARY",
            agent_type=AgentType.SYSTEM,
            capabilities=capabilities,
            protocol_orchestrator=protocol_orchestrator
        )

        # Initialize PXL Modal Logic System
        import runtime_operations
        self.pxl_evaluator = runtime_operations.ModalEvaluator()
        self.pxl_core = runtime_operations.PXLLogicCore()
        self.ontological_lattice = runtime_operations.OntologicalLattice()
        self.iel_overlay = runtime_operations.IELOverlay()
        self.dual_bijective = runtime_operations.DualBijectiveSystem()
        self.reflexive_evaluator = runtime_operations.ReflexiveSelfEvaluator(
            agent_identity="SYSTEM_AGENT_PRIMARY",
            lattice=self.ontological_lattice
        )

        self.autonomous_processing_enabled = True
        self.learning_cycle_active = False
        self.background_tasks: Set[asyncio.Task] = set()

        # Initialize PXL knowledge base
        self._initialize_pxl_knowledge_base()

    def _initialize_pxl_knowledge_base(self):
        """Initialize PXL modal logic knowledge base with core axioms."""

        # Initialize ontological lattice and register core entities
        self._initialize_ontological_lattice()

        # Initialize IEL domains with Trinity Logic mappings
        self._initialize_iel_domains()

        # Core Trinity Logic axioms (E-G-T)
        self.pxl_knowledge = {
            # Existence axioms
            "existence_necessity": self.pxl_evaluator.is_necessarily_true(
                "existence is necessary for all entities"
            ),
            # Goodness axioms
            "goodness_optimization": self.pxl_evaluator.is_possibly_true(
                "goodness drives optimization"
            ),
            # Truth axioms
            "truth_verification": self.pxl_evaluator.is_necessarily_true(
                "truth requires verification"
            ),
            # Modal reasoning axioms
            "modal_consistency": self.pxl_evaluator.is_necessarily_true(
                "modal operators maintain consistency"
            )
        }

        # Initialize privative logic for property negation (simplified)
        self.privative_properties = {
            "consciousness": "unconscious",
            "autonomy": "heteronomy",
            "truth": "falsehood",
            "existence": "nonexistence"
        }

    def _initialize_ontological_lattice(self):
        """Initialize the ontological lattice with core LOGOS entities."""
        # Register core system entities
        self.system_entity = self.pxl_core.register_entity("SYSTEM_AGENT")
        self.consciousness_entity = self.pxl_core.register_entity("CONSCIOUSNESS_CORE")
        self.reasoning_entity = self.pxl_core.register_entity("REASONING_ENGINE")
        self.protocol_entity = self.pxl_core.register_entity("PROTOCOL_ORCHESTRATOR")

        # Establish core relations
        self.pxl_core.add_relation(self.system_entity, self.consciousness_entity, "CONTROLS")
        self.pxl_core.add_relation(self.system_entity, self.reasoning_entity, "USES")
        self.pxl_core.add_relation(self.system_entity, self.protocol_entity, "ORCHESTRATES")
        self.pxl_core.add_relation(self.consciousness_entity, self.reasoning_entity, "INFORMS")

    def _initialize_iel_domains(self):
        """Initialize IEL domains with Trinity Logic mappings."""
        import runtime_operations

        # Map Trinity Logic to IEL domains (simplified)
        trinity_mappings = {
            runtime_operations.IELDomain.EXISTENCE: "NECESSARY",
            runtime_operations.IELDomain.GOODNESS: "POSSIBLE",
            runtime_operations.IELDomain.TRUTH: "NECESSARY",
            runtime_operations.IELDomain.COHERENCE: "NECESSARY",
            runtime_operations.IELDomain.IDENTITY: "NECESSARY",
            runtime_operations.IELDomain.NON_CONTRADICTION: "NECESSARY",
            runtime_operations.IELDomain.EXCLUDED_MIDDLE: "NECESSARY",
            runtime_operations.IELDomain.DISTINCTION: "NECESSARY",
            runtime_operations.IELDomain.RELATION: "POSSIBLE",
            runtime_operations.IELDomain.AGENCY: "CONTINGENT"
        }

        # Register all domains with their modal operators
        for domain, modality in trinity_mappings.items():
            self.iel_overlay.define_iel(domain, modality)

    def evaluate_modal_reasoning(
        self,
        proposition: str,
        modality,
        context_worlds: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate proposition using integrated PXL modal logic system.
        
        Args:
            proposition: The proposition to evaluate
            modality: Modal operator (NECESSARY, POSSIBLE, CONTINGENT, IMPOSSIBLE)
            context_worlds: Optional list of possible worlds for evaluation
            
        Returns:
            Dict containing evaluation results and reasoning trace
        """

        # Evaluate based on modality type using ModalEvaluator
        if modality == "NECESSARY":
            evaluation = self.pxl_evaluator.is_necessarily_true(proposition)
        elif modality == "POSSIBLE":
            evaluation = self.pxl_evaluator.is_possibly_true(proposition)
        elif modality == "IMPOSSIBLE":
            evaluation = not self.pxl_evaluator.is_possibly_true(proposition)
        elif modality == "CONTINGENT":
            # Contingent means possible but not necessary
            evaluation = (self.pxl_evaluator.is_possibly_true(proposition) and
                         not self.pxl_evaluator.is_necessarily_true(proposition))
        else:
            evaluation = False

        # Apply Trinity Logic validation (E-G-T operators)
        trinity_validation = self._apply_trinity_validation(proposition, evaluation)

        # Check privative properties if relevant
        privative_checks = self._check_privative_properties(proposition)

        # Apply dual bijective logic validation
        bijective_validation = self._apply_bijective_validation(proposition)

        # Apply reflexive self-evaluation
        reflexive_validation = self.reflexive_evaluator.self_reflexive_report()

        return {
            "proposition": proposition,
            "modality": modality,
            "evaluation": evaluation,
            "trinity_validation": trinity_validation,
            "privative_checks": privative_checks,
            "bijective_validation": bijective_validation,
            "reflexive_validation": reflexive_validation,
            "confidence": self._calculate_integrated_confidence(
                evaluation, trinity_validation, bijective_validation, reflexive_validation
            )
        }

    def _apply_trinity_validation(
        self,
        proposition: str,
        modal_evaluation: bool
    ) -> Dict[str, Any]:
        """Apply Dual Bijective Logic validation to modal evaluation (replaces Trinity Logic)."""
        # Use dual bijective system for ontological validation
        # Map Trinity operators to bijective primitives:
        # E (Existence) -> Existence primitive via biject_B(Distinction)
        # G (Goodness) -> Goodness primitive via biject_B(Relation)
        # T (Truth) -> Truth primitive via biject_A(NonContradiction)

        try:
            # Validate ontological consistency using bijective mappings
            ontological_consistent = self.dual_bijective.validate_ontological_consistency()

            # Check individual primitive mappings
            existence_valid = self.dual_bijective.biject_B(self.dual_bijective.distinction) is not None
            goodness_valid = self.dual_bijective.biject_B(self.dual_bijective.relation) is not None
            truth_valid = self.dual_bijective.biject_A(self.dual_bijective.non_contradiction) is not None

            # Additional bijective commutation check
            commutation_valid = self.dual_bijective.commute(
                (self.dual_bijective.identity, self.dual_bijective.coherence),
                (self.dual_bijective.distinction, self.dual_bijective.existence)
            )

            return {
                "existence_valid": existence_valid,
                "goodness_valid": goodness_valid,
                "truth_valid": truth_valid,
                "ontological_consistent": ontological_consistent,
                "commutation_valid": commutation_valid,
                "trinity_consistent": all([existence_valid, goodness_valid, truth_valid, commutation_valid])  # Backward compatibility
            }

        except Exception as e:
            self.logger.warning(f"Dual bijective validation failed: {e}")
            # Fallback to basic validation
            return {
                "existence_valid": False,
                "goodness_valid": False,
                "truth_valid": modal_evaluation,  # Basic truth check
                "ontological_consistent": False,
                "commutation_valid": False,
                "trinity_consistent": False
            }

    def _check_privative_properties(self, proposition: str) -> Dict[str, Any]:
        """Check privative logic properties in the proposition."""
        checks = {}

        for property_name, privative_op in self.privative_properties.items():
            if property_name.lower() in proposition.lower():
                # Apply privative logic check
                privative_result = privative_op.check_property(property_name)
                checks[property_name] = privative_result

        return checks

    def _apply_bijective_validation(self, proposition: str) -> Dict[str, Any]:
        """Apply dual bijective logic validation to proposition."""
        # Extract key concepts from proposition for bijective mapping
        concepts = self._extract_concepts_from_proposition(proposition)

        bijective_results = {}
        for concept in concepts:
            # Try bijective mappings A and B
            mapping_a = self.dual_bijective.biject_A(
                self.dual_bijective.identity if "identity" in concept.lower()
                else self.dual_bijective.non_contradiction
            )
            mapping_b = self.dual_bijective.biject_B(
                self.dual_bijective.distinction if "distinct" in concept.lower()
                else self.dual_bijective.relation
            )

            bijective_results[concept] = {
                "mapping_a": str(mapping_a) if mapping_a else None,
                "mapping_b": str(mapping_b) if mapping_b else None,
                "commutes": self.dual_bijective.commute(
                    (self.dual_bijective.identity, mapping_a) if mapping_a else (self.dual_bijective.identity, self.dual_bijective.coherence),
                    (self.dual_bijective.distinction, mapping_b) if mapping_b else (self.dual_bijective.distinction, self.dual_bijective.existence)
                )
            }

        return {
            "concepts_mapped": bijective_results,
            "bijective_consistent": all(result["commutes"] for result in bijective_results.values())
        }

    def _extract_concepts_from_proposition(self, proposition: str) -> List[str]:
        """Extract ontological concepts from proposition for bijective validation."""
        concepts = []
        key_terms = ["identity", "existence", "truth", "goodness", "coherence",
                    "consciousness", "autonomy", "relation", "agency"]

        for term in key_terms:
            if term.lower() in proposition.lower():
                concepts.append(term)

        return concepts if concepts else ["existence"]  # Default fallback

    def _calculate_integrated_confidence(
        self,
        modal_eval: bool,
        trinity_validation: Dict[str, Any],
        bijective_validation: Dict[str, Any],
        reflexive_validation: Dict[str, Any]
    ) -> float:
        """Calculate confidence score integrating all logic systems (prioritizing dual bijective)."""
        base_confidence = 0.8 if modal_eval else 0.6

        # Primary boost: Dual bijective ontological consistency (replaces Trinity as source of truth)
        if trinity_validation.get("ontological_consistent", False):
            base_confidence += 0.12  # Higher weight than Trinity

        # Secondary boost: Bijective commutation validation
        if trinity_validation.get("commutation_valid", False):
            base_confidence += 0.08

        # Additional boost for individual primitive validations
        primitive_boost = 0.0
        if trinity_validation.get("existence_valid", False):
            primitive_boost += 0.02
        if trinity_validation.get("goodness_valid", False):
            primitive_boost += 0.02
        if trinity_validation.get("truth_valid", False):
            primitive_boost += 0.02
        base_confidence += min(primitive_boost, 0.05)  # Cap individual boosts

        # Boost for bijective consistency (secondary validation)
        if bijective_validation.get("bijective_consistent", False):
            base_confidence += 0.06

        # Boost for reflexive self-coherence
        if reflexive_validation.get("fully_self_coherent", False):
            base_confidence += 0.07

        return min(base_confidence, 1.0)

    async def _apply_modal_reasoning_to_processing(
        self,
        uip_results: Dict[str, Any],
        scp_results: Dict[str, Any],
        trigger_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply PXL modal reasoning to cognitive processing results.
        
        Evaluates key propositions from processing results using modal logic
        to enhance decision-making and validation.
        """

        modal_evaluations = []

        # Extract key propositions from results
        propositions = self._extract_key_propositions(uip_results, scp_results, trigger_data)

        for proposition in propositions:
            # Evaluate necessity - is this conclusion required?
            necessity_eval = self.evaluate_modal_reasoning(
                proposition, "NECESSARY"
            )

            # Evaluate possibility - are alternative conclusions possible?
            possibility_eval = self.evaluate_modal_reasoning(
                proposition, "POSSIBLE"
            )

            # Evaluate contingency - is this situation contingent?
            contingency_eval = self.evaluate_modal_reasoning(
                proposition, "CONTINGENT"
            )

            evaluation = {
                "proposition": proposition,
                "necessity": necessity_eval,
                "possibility": possibility_eval,
                "contingency": contingency_eval,
                "recommended_action": self._determine_modal_action(
                    necessity_eval, possibility_eval, contingency_eval
                )
            }

            modal_evaluations.append(evaluation)

            self.logger.info(f"🔍 Modal evaluation: {proposition[:50]}... -> {evaluation['recommended_action']}")

        return {
            "modal_evaluations": modal_evaluations,
            "overall_confidence": self._calculate_overall_modal_confidence(modal_evaluations),
            "trinity_alignment": self._assess_trinity_alignment(modal_evaluations)
        }

    def _extract_key_propositions(
        self,
        uip_results: Dict[str, Any],
        scp_results: Dict[str, Any],
        trigger_data: Dict[str, Any]
    ) -> List[str]:
        """Extract key propositions from processing results for modal evaluation."""
        propositions = []

        # Extract from trigger data
        if "type" in trigger_data:
            propositions.append(f"The system should address {trigger_data['type']} issues")

        # Extract from UIP results
        if "analysis" in uip_results:
            analysis = uip_results["analysis"]
            if isinstance(analysis, dict):
                if "key_insights" in analysis:
                    for insight in analysis["key_insights"][:3]:  # Limit to top 3
                        propositions.append(str(insight))

        # Extract from SCP results
        if "enhancements" in scp_results:
            enhancements = scp_results["enhancements"]
            if isinstance(enhancements, dict):
                if "causal_chains" in enhancements:
                    for chain in enhancements["causal_chains"][:2]:  # Limit to top 2
                        propositions.append(f"Causal relationship: {str(chain)}")

        # Default propositions if none extracted
        if not propositions:
            propositions = [
                "The system maintains operational integrity",
                "Learning opportunities exist for improvement",
                "Autonomous processing yields beneficial outcomes"
            ]

        return propositions

    def _determine_modal_action(
        self,
        necessity_eval: Dict[str, Any],
        possibility_eval: Dict[str, Any],
        contingency_eval: Dict[str, Any]
    ) -> str:
        """Determine recommended action based on modal evaluations."""
        necessity_conf = necessity_eval.get("confidence", 0)
        possibility_conf = possibility_eval.get("confidence", 0)
        contingency_conf = contingency_eval.get("confidence", 0)

        # High necessity + high contingency = requires immediate action
        if necessity_conf > 0.8 and contingency_conf > 0.7:
            return "immediate_action_required"

        # High possibility + low necessity = monitor and evaluate
        elif possibility_conf > 0.8 and necessity_conf < 0.6:
            return "monitor_and_evaluate"

        # Balanced evaluations = proceed with caution
        else:
            return "proceed_with_caution"

    def _calculate_overall_modal_confidence(self, evaluations: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence from modal evaluations."""
        if not evaluations:
            return 0.5

        total_confidence = 0
        for eval in evaluations:
            # Average the confidence scores from different modalities
            necessity_conf = eval.get("necessity", {}).get("confidence", 0)
            possibility_conf = eval.get("possibility", {}).get("confidence", 0)
            contingency_conf = eval.get("contingency", {}).get("confidence", 0)

            avg_conf = (necessity_conf + possibility_conf + contingency_conf) / 3
            total_confidence += avg_conf

        return total_confidence / len(evaluations)

    def _assess_trinity_alignment(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess how well evaluations align with Trinity Logic (E-G-T)."""
        existence_aligned = 0
        goodness_aligned = 0
        truth_aligned = 0

        for eval in evaluations:
            necessity = eval.get("necessity", {})
            if necessity.get("trinity_validation", {}).get("existence_valid", False):
                existence_aligned += 1
            if necessity.get("trinity_validation", {}).get("goodness_valid", False):
                goodness_aligned += 1
            if necessity.get("trinity_validation", {}).get("truth_valid", False):
                truth_aligned += 1

        total_evals = len(evaluations)

        return {
            "existence_alignment": existence_aligned / total_evals if total_evals > 0 else 0,
            "goodness_alignment": goodness_aligned / total_evals if total_evals > 0 else 0,
            "truth_alignment": truth_aligned / total_evals if total_evals > 0 else 0,
            "trinity_consistent": all([
                existence_aligned / total_evals > 0.7 if total_evals > 0 else False,
                goodness_aligned / total_evals > 0.7 if total_evals > 0 else False,
                truth_aligned / total_evals > 0.7 if total_evals > 0 else False
            ])
        }

    async def initialize(self) -> bool:
        """Initialize the system agent"""
        try:
            self.logger.info("Initializing LOGOS System Agent...")

            # Verify protocol access
            await self._verify_protocol_connections()

            # Start autonomous processing
            await self._start_autonomous_processing()

            # Initialize learning systems
            await self._initialize_learning_systems()

            self.active = True
            self.logger.info("✅ LOGOS System Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"System Agent initialization failed: {e}")
            return False

    async def execute_primary_function(self) -> None:
        """
        Main system agent operational loop
        
        Responsibilities:
        - Monitor system state continuously
        - Initiate cognitive processing cycles
        - Orchestrate learning and optimization
        - Handle autonomous decision-making
        """
        self.logger.info("🤖 System Agent primary function activated")

        while self.active:
            try:
                # Monitor system health
                await self._monitor_system_health()

                # Check for Trinitarian convergence responses
                await self._check_trinitarian_convergence_responses()

                # Check for autonomous processing opportunities
                await self._check_autonomous_processing_opportunities()

                # Execute learning cycles
                await self._execute_learning_cycle()

                # Optimize system performance
                await self._optimize_system_performance()

                # Clean up completed background tasks
                await self._cleanup_background_tasks()

                # Wait before next cycle
                await asyncio.sleep(1.0)  # 1Hz system agent cycle

            except Exception as e:
                self.logger.error(f"System Agent cycle error: {e}")
                await asyncio.sleep(5.0)  # Error recovery delay

    async def _monitor_system_health(self) -> None:
        """Lightweight health sampling hook for demo runtime."""
        await asyncio.sleep(0)

    async def _check_autonomous_processing_opportunities(self) -> None:
        """Placeholder hook; autonomous triggers are external in the demo."""
        await asyncio.sleep(0)

    async def initiate_cognitive_processing(
        self,
        trigger_data: Dict[str, Any],
        processing_type: str = "autonomous"
    ) -> Dict[str, Any]:
        """
        Initiate cognitive processing cycle
        
        Process:
        1. Analyze trigger data through UIP
        2. Send results to SCP for deep processing  
        3. Integrate learning back into system
        4. Update SOP with performance metrics
        """
        self.logger.info(f"🧠 Initiating cognitive processing: {processing_type}")

        # Step 1: Process through UIP for base analysis
        uip_config = {
            "agent_type": "system",
            "processing_depth": "comprehensive",
            "return_format": "pxl_iel_trinity_dataset",
            "enable_learning_capture": True
        }

        uip_results = await self.activate_uip(
            input_data=trigger_data,
            processing_config=uip_config
        )

        # Step 2: Send to SCP for cognitive enhancement
        scp_request = {
            "base_analysis": uip_results,
            "processing_type": "infinite_recursive",
            "enhancement_targets": [
                "novel_insight_generation",
                "causal_chain_analysis",
                "modal_inference_expansion",
                "creative_hypothesis_generation"
            ]
        }

        scp_results = await self.access_scp(scp_request)

        # Step 2.5: Apply PXL Modal Reasoning Evaluation
        modal_evaluation = await self._apply_modal_reasoning_to_processing(
            uip_results, scp_results, trigger_data
        )

        # Step 3: Integrate learning into system
        await self._integrate_cognitive_learning(uip_results, scp_results)

        return {
            "processing_id": str(uuid.uuid4()),
            "trigger_data": trigger_data,
            "uip_analysis": uip_results,
            "scp_enhancement": scp_results,
            "modal_evaluation": modal_evaluation,
            "learning_integrated": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _verify_protocol_connections(self) -> None:
        """Verify connections to all protocols"""
        self.logger.info("Verifying protocol connections...")

        # Test SOP connection
        sop_status = await self.monitor_sop()
        if not sop_status.get("healthy", False):
            raise RuntimeError("SOP connection failed")

        # Test SCP connection
        scp_status = await self.protocol_orchestrator.get_scp_status(self)
        if not scp_status.get("connected", False):
            raise RuntimeError("SCP connection failed")

        self.logger.info("✅ All protocol connections verified")

    async def _start_autonomous_processing(self) -> None:
        """Start autonomous background processing"""
        if self.autonomous_processing_enabled:
            task = asyncio.create_task(self._autonomous_processing_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _initialize_learning_systems(self) -> None:
        """Initialize learning subsystems (noop in demo runtime)."""
        self.logger.info("Learning systems initialization skipped (demo mode)")

    async def _autonomous_processing_loop(self) -> None:
        """Continuous autonomous processing loop"""
        self.logger.info("🔄 Autonomous processing loop started")

        while self.active and self.autonomous_processing_enabled:
            try:
                # Look for autonomous processing triggers
                triggers = await self._identify_processing_triggers()

                for trigger in triggers:
                    await self.initiate_cognitive_processing(
                        trigger_data=trigger,
                        processing_type="autonomous"
                    )

                await asyncio.sleep(10.0)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Autonomous processing error: {e}")
                await asyncio.sleep(30.0)  # Error recovery

    async def _identify_processing_triggers(self) -> List[Dict[str, Any]]:
        """Identify opportunities for autonomous cognitive processing"""
        triggers = []

        # Check system metrics for optimization opportunities
        sop_metrics = await self.monitor_sop()
        if sop_metrics.get("performance_degradation", False):
            triggers.append({
                "type": "performance_optimization",
                "source": "sop_metrics",
                "data": sop_metrics
            })

        # Check for new external data
        # (This would connect to external data sources)

        # Check for learning opportunities
        # (This would analyze recent interactions for patterns)

        return triggers

    async def _integrate_cognitive_learning(
        self,
        uip_results: Dict[str, Any],
        scp_results: Dict[str, Any]
    ) -> None:
        """Integrate learning from cognitive processing"""
        self.logger.info("📚 Integrating cognitive learning...")

        # Extract insights from SCP processing
        insights = scp_results.get("insights", [])

        # Update system knowledge base
        # (Implementation depends on knowledge storage system)

        # Notify SOP of learning updates
        learning_summary = {
            "insights_count": len(insights),
            "processing_quality": scp_results.get("quality_metrics", {}),
            "system_improvements": scp_results.get("system_improvements", [])
        }

        # This would notify SOP of the learning
        # await self.protocol_orchestrator.notify_sop_learning(learning_summary)

    async def _check_trinitarian_convergence_responses(self) -> None:
        """Check for Trinitarian convergence JSON responses from protocols"""
        import os
        import json

        nexus_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'nexus')
        os.makedirs(nexus_dir, exist_ok=True)

        convergence_files = {
            'sop_convergence.json': 'System Operations Protocol',
            'arp_convergence.json': 'Advanced Reasoning Protocol',
            'scp_convergence.json': 'Synthetic Cognition Protocol'
        }

        for filename, protocol_name in convergence_files.items():
            filepath = os.path.join(nexus_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        response_data = json.load(f)

                    self.logger.info(f"📥 Received Trinitarian convergence response from {protocol_name}")
                    self.logger.info(f"   Content: {response_data}")

                    # Process the convergence response
                    await self._process_convergence_response(protocol_name, response_data)

                    # Remove the file after processing
                    os.remove(filepath)
                    self.logger.info(f"   ✅ Processed and removed {filename}")

                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Invalid JSON in {filename}: {e}")
                except Exception as e:
                    self.logger.error(f"❌ Error processing {filename}: {e}")

    async def _process_convergence_response(self, protocol_name: str, response_data: Dict[str, Any]) -> None:
        """Process a Trinitarian convergence response from a protocol"""
        # Expected content based on user specification:
        # SOP: "In the beginning was the Word."
        # ARP: "And the Word was with God"
        # SCP: "And the Word was God"

        content = response_data.get('content', '')

        if protocol_name == 'System Operations Protocol':
            expected = "In the beginning was the Word."
        elif protocol_name == 'Advanced Reasoning Protocol':
            expected = "And the Word was with God"
        elif protocol_name == 'Synthetic Cognition Protocol':
            expected = "And the Word was God"
        else:
            expected = "Unknown protocol"

        if content == expected:
            self.logger.info(f"✅ {protocol_name} convergence response validated: '{content}'")

            # Here you could add logic to integrate the convergence responses
            # For example, update system state, trigger further processing, etc.

        else:
            self.logger.warning(f"⚠️ {protocol_name} response mismatch. Expected: '{expected}', Got: '{content}'")

    async def _execute_learning_cycle(self) -> None:
        """Execute periodic learning and optimization"""
        if not self.learning_cycle_active:
            self.learning_cycle_active = True

            try:
                # Analyze recent system performance
                # Generate optimization recommendations
                # Apply approved optimizations
                pass

            finally:
                self.learning_cycle_active = False

    async def _optimize_system_performance(self) -> None:
        """Optimize system performance based on metrics"""
        # Implementation would analyze metrics and apply optimizations
        pass

    async def _cleanup_background_tasks(self) -> None:
        """Clean up completed background tasks"""
        completed_tasks = {task for task in self.background_tasks if task.done()}
        self.background_tasks -= completed_tasks


class UserAgent(BaseAgent):
    """
    Represents external human users interacting with LOGOS
    """

    def __init__(self, user_id: str, protocol_orchestrator: 'ProtocolOrchestrator'):
        capabilities = AgentCapabilities(
            can_activate_uip=True,
            can_access_scp=False,  # Users don't directly access SCP
            can_monitor_sop=False,
            can_create_agents=False,
            max_concurrent_operations=5,
            priority_level=7  # High priority for user experience
        )

        super().__init__(
            agent_id=f"USER_AGENT_{user_id}",
            agent_type=AgentType.USER,
            capabilities=capabilities,
            protocol_orchestrator=protocol_orchestrator
        )

        self.user_id = user_id
        self.interaction_history: List[Dict[str, Any]] = []

    async def initialize(self) -> bool:
        """Initialize user agent"""
        self.active = True
        self.logger.info(f"User Agent initialized for user: {self.user_id}")
        return True

    async def execute_primary_function(self) -> None:
        """User agents are reactive - no continuous loop needed"""
        pass

    async def process_user_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input through LOGOS system
        
        Flow:
        1. Activate UIP for user processing
        2. System Agent receives copy for learning
        3. Return formatted response to user
        """

        input_data = {
            "user_input": user_input,
            "user_id": self.user_id,
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Process through UIP with user-specific configuration
        uip_config = {
            "agent_type": "user",
            "response_format": "human_readable",
            "enable_explanation": True,
            "privacy_mode": True
        }

        results = await self.activate_uip(
            input_data=input_data,
            processing_config=uip_config
        )

        # Store interaction history
        self.interaction_history.append({
            "input": user_input,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return results


class ProtocolOrchestrator:
    """
    Orchestrates protocol activation and coordination
    
    Responsibilities:
    - Manage protocol lifecycle (SOP/SCP always on, UIP on-demand)
    - Route agent requests to appropriate protocols
    - Coordinate cross-protocol communication
    - Handle protocol resource management
    """

    def __init__(self, *, scp_mode: Optional[str] = None):
        self.sop_system = None
        self.scp_system = None
        self.uip_system = None
        self.active_protocols: Set[ProtocolType] = set()
        self.runtime: Optional[SystemRuntime] = None
        self.scp_mode = (scp_mode or os.environ.get("LOGOS_SCP_MODE", "local")).lower()

    async def initialize_protocols(self) -> bool:
        """
        Initialize all protocols according to architecture:
        - SOP: Always running
        - SCP: Always running  
        - UIP: Dormant until needed
        """
        try:
            self.runtime = await initialize_runtime()
            self.active_protocols.add(ProtocolType.SOP)
            await self._initialize_scp_transport()
            logger.info("✅ Protocol orchestrator initialized with live runtime")
            return True

        except Exception as e:
            logger.error(f"Protocol initialization failed: {e}")
            return False

    async def _initialize_scp_transport(self) -> None:
        runtime = self._require_runtime()
        mode = self.scp_mode
        if mode == "off":
            self.scp_system = None
            logger.warning("SCP transport disabled via configuration")
            return
        if mode == "local":
            from .local_scp import LocalSCPTransport

            self.scp_system = LocalSCPTransport(runtime)
            await self.scp_system.connect()
            self.active_protocols.add(ProtocolType.SCP)
            logger.info("✅ Local SCP transport connected")
            return

        logger.warning("Unknown SCP mode '%s'; SCP transport not started", mode)
        self.scp_system = None

    async def activate_uip_for_agent(
        self,
        agent: BaseAgent,
        input_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Activate UIP processing for specific agent
        
        This is where UIP spins up on-demand
        """
        logger.info(f"🚀 Activating UIP for agent: {agent.agent_id}")

        runtime = self._require_runtime()
        processing_config = self._build_agent_config(agent, config)

        async def pipeline() -> Dict[str, Any]:
            started = datetime.now(timezone.utc)
            user_text = str(input_data.get("user_input", ""))
            tokens = user_text.split()
            await asyncio.sleep(min(len(tokens) * 0.01, 0.5))
            resource_snapshot = runtime.manager.snapshot().to_dict()
            return {
                "success": True,
                "processed": True,
                "agent_id": agent.agent_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "token_count": len(tokens),
                "processing_config": processing_config,
                "resource_state": self._summarise_resources(runtime, resource_snapshot),
                "latency_ms": round((datetime.now(timezone.utc) - started).total_seconds() * 1000, 2),
                "input_digest": {
                    "preview": user_text[:120],
                    "has_more": len(user_text) > 120,
                },
            }

        # The demo runtime runs in a constrained container; using a single CPU slot
        # avoids exhausting the shared lease pool when stress tools fan out requests.
        result = await runtime.scheduler.run_immediate(
            label=f"UIP:{agent.agent_id}",
            coro_factory=pipeline,
            cpu_slots=1,
            memory_mb=96,
            disk_mb=0,
            priority=5,
        )

        logger.info(f"✅ UIP processing complete for agent: {agent.agent_id}")
        return result

    async def route_to_scp(
        self,
        agent: BaseAgent,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route cognitive processing request to SCP"""
        logger.info(f"🧠 Routing to SCP from agent: {agent.agent_id}")
        runtime = self._require_runtime()
        if not self.scp_system:
            raise RuntimeError("SCP transport unavailable")

        response = await self.scp_system.submit_meta_cycle(agent.agent_id, request)
        snapshot = response.pop("resource_snapshot", runtime.manager.snapshot().to_dict())
        response["resource_state"] = self._summarise_resources(runtime, snapshot)
        response.setdefault("connected", True)
        response.setdefault("mode", getattr(self.scp_system, "mode", "unknown"))
        response.setdefault("success", True)
        response.setdefault("processed", True)
        return response

    async def get_sop_status(self, agent: BaseAgent) -> Dict[str, Any]:
        runtime = self._require_runtime()
        latest = runtime.monitor.latest()
        return {
            "healthy": True,
            "agent_id": agent.agent_id,
            "resources": self._summarise_resources(runtime, latest or runtime.manager.snapshot().to_dict()),
            "scheduler": {
                "concurrency": runtime.scheduler.concurrency,
            },
        }

    async def get_scp_status(self, agent: Optional[BaseAgent] = None) -> Dict[str, Any]:
        runtime = self._require_runtime()
        if not self.scp_system:
            return {"connected": False, "mode": self.scp_mode}

        status = await self.scp_system.health()
        snapshot = status.pop("resource_snapshot", runtime.manager.snapshot().to_dict())
        status["resource_state"] = self._summarise_resources(runtime, snapshot)
        if agent:
            status["agent_id"] = agent.agent_id
        return status

    def _build_agent_config(
        self,
        agent: BaseAgent,
        base_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build UIP configuration specific to agent type"""
        config = base_config or {}

        config.update({
            "requesting_agent": {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "capabilities": agent.capabilities,
                "priority": agent.capabilities.priority_level
            }
        })

        return config

    def _require_runtime(self) -> SystemRuntime:
        if not self.runtime:
            raise RuntimeError("Protocol runtime not initialized")
        return self.runtime

    def _summarise_resources(
        self,
        runtime: SystemRuntime,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        utilisation = {
            name: runtime.manager.utilisation(name)
            for name in ("cpu_slots", "memory_mb", "disk_mb")
        }
        return {
            "utilisation": utilisation,
            "load_average": snapshot.get("load_average", {}),
            "memory": snapshot.get("memory", {}),
            "disk_free_mb": snapshot.get("disk_free_mb"),
        }


class AgentRegistry:
    """
    Registry for managing all agents in the system
    """

    def __init__(self, protocol_orchestrator: ProtocolOrchestrator):
        self.protocol_orchestrator = protocol_orchestrator
        self.agents: Dict[str, BaseAgent] = {}
        self.system_agent: Optional[SystemAgent] = None

    async def initialize_system_agent(self, *, enable_autonomy: bool = True) -> SystemAgent:
        """Initialize the primary system agent"""
        if self.system_agent is None:
            self.system_agent = SystemAgent(self.protocol_orchestrator)
            self.system_agent.autonomous_processing_enabled = enable_autonomy
            await self.system_agent.initialize()
            self.agents[self.system_agent.agent_id] = self.system_agent
        else:
            self.system_agent.autonomous_processing_enabled = enable_autonomy

        return self.system_agent

    async def create_user_agent(self, user_id: str) -> UserAgent:
        """Create a new user agent"""
        user_agent = UserAgent(user_id, self.protocol_orchestrator)
        await user_agent.initialize()
        self.agents[user_agent.agent_id] = user_agent

        return user_agent

    def get_agent(self, agent_id: str):
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_system_agent(self):
        """Get the system agent"""
        return self.system_agent


# Global registry instance
agent_registry: Optional[AgentRegistry] = None


async def initialize_agent_system(
    *,
    scp_mode: Optional[str] = None,
    enable_autonomy: bool = False,
) -> AgentRegistry:
    """Initialize the complete LOGOS Agent System"""
    global agent_registry

    logger.info("🚀 Initializing LOGOS Agent System...")

    # Initialize protocol orchestrator
    protocol_orchestrator = ProtocolOrchestrator(scp_mode=scp_mode)
    await protocol_orchestrator.initialize_protocols()

    # Initialize agent registry
    agent_registry = AgentRegistry(protocol_orchestrator)

    # Initialize system agent (this starts autonomous operations)
    system_agent = await agent_registry.initialize_system_agent(enable_autonomy=enable_autonomy)

    # Start system agent primary function
    if enable_autonomy:
        asyncio.create_task(system_agent.execute_primary_function())

    logger.info("✅ LOGOS Agent System initialized successfully")
    return agent_registry


if __name__ == "__main__":
    async def main():
        # Initialize agent system
        registry = await initialize_agent_system(enable_autonomy=True)

        # Create a user agent for demonstration
        user_agent = await registry.create_user_agent("demo_user")

        # Simulate user interaction
        response = await user_agent.process_user_input(
            "What are the necessary and sufficient conditions for conditions to be necessary and sufficient?"
        )

        print("User interaction response:", response)

        # Let system agent run for a bit
        await asyncio.sleep(5)

    asyncio.run(main())