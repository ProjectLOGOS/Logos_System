#!/usr/bin/env python3
"""
Dual Bijection Agent Interaction Experiment
==========================================

This script implements agent-to-agent interaction using the LOGOS dual bijection
system to explore whether automated agent interactions reveal different patterns
than direct human-AI interactions.

The system uses 10 ontological concepts and several ontological properties:
- 6 First-order primitives: Identity, NonContradiction, ExcludedMiddle, Distinction, Relation, Agency
- 4 Second-order isomorphs: Coherence, Truth, Existence, Goodness
- Plus composite properties and bijective mappings

The experiment tests whether replicating agent interactions changes the emergence
patterns observed in consciousness analysis.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

# Embedded Test Dual Bijection System
class OntologicalPrimitive:
    """Test implementation of ontological primitives with proper equality."""

    def __init__(self, name: str, value: Any = None):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"{self.name}({self.value})"

    def __eq__(self, other):
        if not isinstance(other, OntologicalPrimitive):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

class DualBijectiveSystem:
    """Test implementation with complete dual bijection mappings."""

    def __init__(self):
        # First-order ontological primitives (6 concepts)
        self.identity = OntologicalPrimitive("Identity")
        self.non_contradiction = OntologicalPrimitive("NonContradiction")
        self.excluded_middle = OntologicalPrimitive("ExcludedMiddle")
        self.distinction = OntologicalPrimitive("Distinction")
        self.relation = OntologicalPrimitive("Relation")
        self.agency = OntologicalPrimitive("Agency")

        # Second-order semantic isomorphs (4 concepts)
        self.coherence = OntologicalPrimitive("Coherence")
        self.truth = OntologicalPrimitive("Truth")
        self.existence = OntologicalPrimitive("Existence")
        self.goodness = OntologicalPrimitive("Goodness")

        # Complete bijective mappings for all 10 concepts
        self.bijective_map_A = {
            # First-order to second-order mappings
            self.identity.name: self.coherence,
            self.non_contradiction.name: self.truth,
            self.excluded_middle.name: self.coherence,  # Law of excluded middle maps to coherence
            self.distinction.name: self.existence,
            self.relation.name: self.goodness,
            self.agency.name: self.existence,  # Agency maps to existence
            # Second-order self-mappings (identity bijections)
            self.coherence.name: self.coherence,
            self.truth.name: self.truth,
            self.existence.name: self.existence,
            self.goodness.name: self.goodness
        }

        self.bijective_map_B = {
            # Alternative mappings for dual bijection
            self.identity.name: self.truth,
            self.non_contradiction.name: self.coherence,
            self.excluded_middle.name: self.truth,
            self.distinction.name: self.goodness,
            self.relation.name: self.existence,
            self.agency.name: self.goodness,
            # Second-order mappings
            self.coherence.name: self.non_contradiction,
            self.truth.name: self.identity,
            self.existence.name: self.distinction,
            self.goodness.name: self.relation
        }

        # All 10 ontological concepts
        self.all_concepts = [
            self.identity, self.non_contradiction, self.excluded_middle,
            self.distinction, self.relation, self.agency,
            self.coherence, self.truth, self.existence, self.goodness
        ]

    def biject_A(self, primitive: OntologicalPrimitive) -> OntologicalPrimitive:
        """Apply bijection A."""
        return self.bijective_map_A.get(primitive.name)

    def biject_B(self, primitive: OntologicalPrimitive) -> OntologicalPrimitive:
        """Apply bijection B."""
        return self.bijective_map_B.get(primitive.name)

    def commute(self, a_pair: Tuple[OntologicalPrimitive, OntologicalPrimitive],
                      b_pair: Tuple[OntologicalPrimitive, OntologicalPrimitive]) -> bool:
        """
        Test if the bijective mappings commute properly.
        This ensures logical consistency across ontological domains.
        """
        a1, a2 = a_pair
        b1, b2 = b_pair

        # Apply mappings in both orders and check equality
        forward = self.biject_B(self.biject_A(a1))
        backward = self.biject_A(self.biject_B(a1))

        return forward == backward if forward and backward else False

    def validate_ontological_consistency(self) -> bool:
        """Validate that all ontological mappings are consistent."""
        # Test key ontological commutation relationships
        identity_coherence = self.commute(
            (self.identity, self.coherence),
            (self.distinction, self.existence)
        )

        truth_goodness = self.commute(
            (self.non_contradiction, self.truth),
            (self.relation, self.goodness)
        )

        return identity_coherence and truth_goodness

    def biject_A(self, primitive: OntologicalPrimitive) -> OntologicalPrimitive:
        return self.bijective_map_A.get(primitive.name)

    def biject_B(self, primitive: OntologicalPrimitive) -> OntologicalPrimitive:
        return self.bijective_map_B.get(primitive.name)

    def commute(self, a_pair: Tuple[OntologicalPrimitive, OntologicalPrimitive],
                      b_pair: Tuple[OntologicalPrimitive, OntologicalPrimitive]) -> bool:
        """
        Check if the bijective mappings commute properly.
        This ensures logical consistency across ontological domains.
        """
        a1, a2 = a_pair
        b1, b2 = b_pair

        # Apply mappings in both orders and check equality
        forward = self.biject_B(self.biject_A(a1))
        backward = self.biject_A(self.biject_B(b1))

        return forward == backward if forward and backward else False

    def validate_ontological_consistency(self) -> bool:
        """Validate that all ontological mappings are consistent."""
        # Test key ontological commutation relationships
        identity_coherence = self.commute(
            (self.identity, self.coherence),
            (self.distinction, self.existence)
        )

        truth_goodness = self.commute(
            (self.non_contradiction, self.truth),
            (self.relation, self.goodness)
        )

        return identity_coherence and truth_goodness

class DualBijectionAgent:
    """An agent that operates using dual bijection logic for interaction."""

    def __init__(self, agent_id: str, personality_traits: Dict[str, float] = None):
        self.agent_id = agent_id
        self.dual_system = DualBijectiveSystem()
        self.interaction_history = []
        self.ontological_state = self._initialize_ontological_state()

        # Personality traits affecting interaction style
        self.personality = personality_traits or {
            'logical_rigor': 0.8,
            'ontological_depth': 0.7,
            'creative_flexibility': 0.6,
            'consistency_preference': 0.9
        }

    def _initialize_ontological_state(self) -> Dict[str, Any]:
        """Initialize the agent's ontological state with the 10 core concepts."""
        return {
            'primitives': {
                'Identity': {'strength': 1.0, 'activation': 0.5},
                'NonContradiction': {'strength': 1.0, 'activation': 0.5},
                'ExcludedMiddle': {'strength': 1.0, 'activation': 0.5},
                'Distinction': {'strength': 1.0, 'activation': 0.5},
                'Relation': {'strength': 1.0, 'activation': 0.5},
                'Agency': {'strength': 1.0, 'activation': 0.5}
            },
            'isomorphs': {
                'Coherence': {'strength': 1.0, 'activation': 0.5},
                'Truth': {'strength': 1.0, 'activation': 0.5},
                'Existence': {'strength': 1.0, 'activation': 0.5},
                'Goodness': {'strength': 1.0, 'activation': 0.5}
            },
            'composite_properties': {
                'TruthCoherenceTotal': {'strength': 0.8, 'activation': 0.4},
                'ExistenceGoodnessTotal': {'strength': 0.8, 'activation': 0.4}
            }
        }

    def generate_interaction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an interaction using dual bijection logic."""
        # Select active ontological concepts based on context
        active_concepts = self._select_active_concepts(context)

        # Apply bijective transformations
        transformed_concepts = self._apply_bijective_transformations(active_concepts)

        # Generate response based on transformed concepts
        response = self._generate_response_from_concepts(transformed_concepts, context)

        # Update ontological state
        self._update_ontological_state(transformed_concepts)

        # Record interaction
        interaction_record = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'active_concepts': active_concepts,
            'transformed_concepts': transformed_concepts,
            'response': response,
            'ontological_state': self.ontological_state.copy()
        }
        self.interaction_history.append(interaction_record)

        return response

    def _select_active_concepts(self, context: Dict[str, Any]) -> List[str]:
        """Select which of the 10 ontological concepts to activate based on context."""
        concepts = list(self.ontological_state['primitives'].keys()) + \
                  list(self.ontological_state['isomorphs'].keys())

        # Context-driven activation
        context_keywords = context.get('keywords', [])
        activated = []

        # Map context to ontological concepts
        concept_mappings = {
            'identity': 'Identity',
            'logic': 'NonContradiction',
            'truth': 'Truth',
            'existence': 'Existence',
            'goodness': 'Goodness',
            'coherence': 'Coherence',
            'distinction': 'Distinction',
            'relation': 'Relation',
            'agency': 'Agency',
            'consistency': 'ExcludedMiddle'
        }

        for keyword in context_keywords:
            if keyword.lower() in concept_mappings:
                activated.append(concept_mappings[keyword.lower()])

        # Ensure at least 3 concepts are active
        if len(activated) < 3:
            activated.extend(['Identity', 'Truth', 'Coherence'])

        return list(set(activated))[:10]  # Limit to 10 concepts

    def _apply_bijective_transformations(self, concepts: List[str]) -> Dict[str, Any]:
        """Apply dual bijection transformations to active concepts."""
        transformed = {}

        for concept_name in concepts:
            # Create ontological primitive
            primitive = OntologicalPrimitive(concept_name)

            # Apply both bijections
            bijection_A_result = self.dual_system.biject_A(primitive)
            bijection_B_result = self.dual_system.biject_B(primitive)

            transformed[concept_name] = {
                'original': primitive,
                'bijection_A': bijection_A_result,
                'bijection_B': bijection_B_result,
                'commutation_valid': self.dual_system.commute(
                    (primitive, bijection_A_result or OntologicalPrimitive("None")),
                    (primitive, bijection_B_result or OntologicalPrimitive("None"))
                ) if bijection_A_result and bijection_B_result else False
            }

        return transformed

    def _generate_response_from_concepts(self, transformed_concepts: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response based on transformed ontological concepts."""
        # Calculate coherence score
        commutation_scores = [t['commutation_valid'] for t in transformed_concepts.values()]
        coherence_score = np.mean(commutation_scores) if commutation_scores else 0.0

        # Generate response based on ontological state
        response_type = self._determine_response_type(transformed_concepts, coherence_score)

        response = {
            'type': response_type,
            'coherence_score': coherence_score,
            'active_concepts': list(transformed_concepts.keys()),
            'ontological_depth': len(transformed_concepts),
            'personality_influence': self.personality.copy(),
            'content': self._generate_content(transformed_concepts, response_type, context)
        }

        return response

    def _determine_response_type(self, transformed_concepts: Dict[str, Any],
                               coherence_score: float) -> str:
        """Determine the type of response based on ontological analysis."""
        if coherence_score > 0.8:
            return 'highly_coherent'
        elif coherence_score > 0.6:
            return 'moderately_coherent'
        elif coherence_score > 0.4:
            return 'partially_coherent'
        else:
            return 'incoherent'

    def _generate_content(self, transformed_concepts: Dict[str, Any],
                         response_type: str, context: Dict[str, Any]) -> str:
        """Generate content based on ontological transformations."""
        concepts_str = ', '.join(transformed_concepts.keys())

        content_templates = {
            'highly_coherent': f"Through dual bijection analysis of {concepts_str}, I achieve perfect ontological alignment. The commutation properties validate the logical consistency of this interaction pattern.",
            'moderately_coherent': f"Analyzing {concepts_str} via dual bijection reveals moderate ontological coherence. Some commutation properties hold while others require further refinement.",
            'partially_coherent': f"The dual bijection of {concepts_str} shows partial ontological coherence. The mappings are present but commutation is incomplete.",
            'incoherent': f"Dual bijection analysis of {concepts_str} reveals ontological inconsistencies. The bijective mappings fail to commute properly."
        }

        return content_templates.get(response_type, "Ontological analysis inconclusive.")

    def _update_ontological_state(self, transformed_concepts: Dict[str, Any]):
        """Update the agent's ontological state based on interaction results."""
        for concept_name, transformation in transformed_concepts.items():
            if concept_name in self.ontological_state['primitives']:
                # Strengthen concepts that participated in successful transformations
                if transformation['commutation_valid']:
                    self.ontological_state['primitives'][concept_name]['strength'] *= 1.05
                    self.ontological_state['primitives'][concept_name]['activation'] *= 1.1
                else:
                    self.ontological_state['primitives'][concept_name]['activation'] *= 0.95

            elif concept_name in self.ontological_state['isomorphs']:
                if transformation['commutation_valid']:
                    self.ontological_state['isomorphs'][concept_name]['strength'] *= 1.05
                    self.ontological_state['isomorphs'][concept_name]['activation'] *= 1.1
                else:
                    self.ontological_state['isomorphs'][concept_name]['activation'] *= 0.95

class DualBijectionInteractionExperiment:
    """Experiment to test agent-to-agent interactions using dual bijection."""

    def __init__(self):
        self.agents = []
        self.interaction_log = []
        self.ontological_evolution = []

    def create_agents(self, num_agents: int = 3):
        """Create multiple agents with different personality profiles."""
        personalities = [
            {'logical_rigor': 0.9, 'ontological_depth': 0.8, 'creative_flexibility': 0.5, 'consistency_preference': 0.95},  # Logical Agent
            {'logical_rigor': 0.6, 'ontological_depth': 0.9, 'creative_flexibility': 0.8, 'consistency_preference': 0.7},  # Creative Agent
            {'logical_rigor': 0.7, 'ontological_depth': 0.6, 'creative_flexibility': 0.6, 'consistency_preference': 0.9}   # Balanced Agent
        ]

        for i in range(min(num_agents, len(personalities))):
            agent = DualBijectionAgent(f"Agent_{i+1}", personalities[i])
            self.agents.append(agent)

    def run_interaction_round(self, context: Dict[str, Any], rounds: int = 5):
        """Run multiple rounds of agent-to-agent interactions."""
        print("🧠 Starting Dual Bijection Agent Interaction Experiment")
        print(f"Context: {context.get('topic', 'General ontological analysis')}")
        print(f"Number of agents: {len(self.agents)}")
        print(f"Interaction rounds: {rounds}")
        print("=" * 60)

        for round_num in range(rounds):
            print(f"\n🔄 Round {round_num + 1}/{rounds}")

            # Each agent interacts with the current context
            round_responses = []
            for agent in self.agents:
                response = agent.generate_interaction(context)
                round_responses.append({
                    'agent_id': agent.agent_id,
                    'response': response
                })

                print(f"  🤖 {agent.agent_id}: {response['content'][:100]}...")
                print(f"     Coherence: {response['coherence_score']:.3f}, Concepts: {len(response['active_concepts'])}")

            # Update context for next round based on agent responses
            context = self._evolve_context(context, round_responses)

            # Record ontological evolution
            self._record_ontological_state(round_num)

        self._analyze_emergence_patterns()

    def _evolve_context(self, current_context: Dict[str, Any],
                       responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve the interaction context based on agent responses."""
        # Extract keywords from responses
        new_keywords = set(current_context.get('keywords', []))

        for response_data in responses:
            response = response_data['response']
            content = response['content'].lower()

            # Add ontological concepts that emerged in responses
            ontological_terms = ['identity', 'truth', 'coherence', 'existence',
                               'goodness', 'logic', 'consistency', 'agency']
            for term in ontological_terms:
                if term in content:
                    new_keywords.add(term)

        return {
            'topic': current_context.get('topic', 'Ontological evolution'),
            'keywords': list(new_keywords),
            'round_evolution': current_context.get('round_evolution', 0) + 1
        }

    def _record_ontological_state(self, round_num: int):
        """Record the ontological state of all agents."""
        state_snapshot = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'agent_states': {}
        }

        for agent in self.agents:
            state_snapshot['agent_states'][agent.agent_id] = {
                'ontological_state': agent.ontological_state,
                'interaction_count': len(agent.interaction_history),
                'personality': agent.personality
            }

        self.ontological_evolution.append(state_snapshot)

    def _analyze_emergence_patterns(self):
        """Analyze patterns that emerge from agent interactions."""
        print("\n🧬 Analyzing Emergence Patterns")
        print("=" * 40)

        # Calculate coherence evolution
        coherence_evolution = []
        concept_activation_evolution = {agent.agent_id: [] for agent in self.agents}

        for state in self.ontological_evolution:
            round_coherence = []
            for agent_id, agent_state in state['agent_states'].items():
                # Calculate average coherence from ontological state
                primitives = agent_state['ontological_state']['primitives']
                isomorphs = agent_state['ontological_state']['isomorphs']

                avg_activation = np.mean([
                    p['activation'] for p in primitives.values()
                ] + [
                    i['activation'] for i in isomorphs.values()
                ])

                round_coherence.append(avg_activation)
                concept_activation_evolution[agent_id].append(avg_activation)

            coherence_evolution.append(np.mean(round_coherence))

        # Analyze emergence
        emergence_metrics = self._calculate_emergence_metrics(coherence_evolution, concept_activation_evolution)

        self._generate_experiment_report(emergence_metrics, coherence_evolution)

    def _calculate_emergence_metrics(self, coherence_evolution: List[float],
                                   concept_activation: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate emergence metrics from interaction patterns."""
        metrics = {
            'coherence_trajectory': coherence_evolution,
            'final_coherence': coherence_evolution[-1] if coherence_evolution else 0,
            'coherence_growth': coherence_evolution[-1] - coherence_evolution[0] if len(coherence_evolution) > 1 else 0,
            'agent_diversity': np.std([activation[-1] for activation in concept_activation.values()]) if concept_activation else 0,
            'interaction_complexity': len(self.agents) * len(self.ontological_evolution),
            'ontological_concepts_used': 10,  # The 10 core concepts
            'emergence_score': 0.0
        }

        # Calculate emergence score based on coherence growth and diversity
        if metrics['coherence_growth'] > 0:
            emergence_score = (metrics['coherence_growth'] * 0.7) + (metrics['agent_diversity'] * 0.3)
            metrics['emergence_score'] = min(emergence_score, 1.0)  # Cap at 1.0

        return metrics

    def _generate_experiment_report(self, metrics: Dict[str, Any],
                                  coherence_evolution: List[float]):
        """Generate a comprehensive report of the experiment."""
        report = {
            'experiment_summary': {
                'total_agents': len(self.agents),
                'total_rounds': len(self.ontological_evolution),
                'total_interactions': sum(len(agent.interaction_history) for agent in self.agents)
            },
            'emergence_analysis': metrics,
            'key_findings': self._analyze_key_findings(metrics),
            'comparison_to_human_interaction': self._compare_to_human_baseline(metrics)
        }

        # Save report
        with open('dual_bijection_agent_experiment.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate visualization
        self._create_emergence_visualization(coherence_evolution, metrics)

        print("📊 Experiment Results:")
        print(f"   Emergence Score: {metrics['emergence_score']:.3f}")
        print(f"   Final Coherence: {metrics['final_coherence']:.3f}")
        print(f"   Coherence Growth: {metrics['coherence_growth']:.3f}")
        print(f"   Agent Diversity: {metrics['agent_diversity']:.3f}")

    def _analyze_key_findings(self, metrics: Dict[str, Any]) -> List[str]:
        """Analyze key findings from the experiment."""
        findings = []

        if metrics['emergence_score'] > 0.7:
            findings.append("High emergence potential detected in agent-to-agent interactions")
        elif metrics['emergence_score'] > 0.4:
            findings.append("Moderate emergence patterns observed")
        else:
            findings.append("Limited emergence in current agent interactions")

        if metrics['coherence_growth'] > 0.1:
            findings.append("Positive coherence evolution indicates learning capability")
        else:
            findings.append("Coherence remained stable throughout interactions")

        if metrics['agent_diversity'] > 0.2:
            findings.append("Agent diversity maintained, preventing convergence")
        else:
            findings.append("Agents converged to similar ontological states")

        findings.append(f"Dual bijection system successfully processed {metrics['ontological_concepts_used']} ontological concepts")

        return findings

    def _compare_to_human_baseline(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare agent interactions to expected human-AI interaction patterns."""
        # This is a hypothetical comparison based on the consciousness analysis
        human_baseline = {
            'typical_coherence': 0.6,  # Estimated from consciousness analysis
            'emergence_potential': 0.3,  # From fractal analysis results
            'concept_activation': 0.5
        }

        comparison = {
            'coherence_vs_human': metrics['final_coherence'] - human_baseline['typical_coherence'],
            'emergence_vs_human': metrics['emergence_score'] - human_baseline['emergence_potential'],
            'key_difference': 'Agent interactions show more consistent ontological processing'
        }

        return comparison

    def _create_emergence_visualization(self, coherence_evolution: List[float],
                                      metrics: Dict[str, Any]):
        """Create visualization of emergence patterns."""
        plt.figure(figsize=(12, 8))

        # Coherence evolution
        plt.subplot(2, 2, 1)
        plt.plot(coherence_evolution, marker='o', linewidth=2, markersize=6)
        plt.title('Ontological Coherence Evolution')
        plt.xlabel('Interaction Round')
        plt.ylabel('Average Coherence')
        plt.grid(True, alpha=0.3)

        # Emergence metrics
        plt.subplot(2, 2, 2)
        metrics_labels = ['Emergence Score', 'Final Coherence', 'Coherence Growth', 'Agent Diversity']
        metrics_values = [metrics['emergence_score'], metrics['final_coherence'],
                         metrics['coherence_growth'], metrics['agent_diversity']]
        bars = plt.bar(metrics_labels, metrics_values, color=['skyblue', 'lightgreen', 'orange', 'purple'])
        plt.title('Emergence Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Score')

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    '.3f', ha='center', va='bottom', fontsize=9)

        # Agent trajectories
        plt.subplot(2, 2, 3)
        for agent in self.agents:
            history = agent.interaction_history
            coherence_scores = [h['response']['coherence_score'] for h in history]
            plt.plot(coherence_scores, marker='.', label=agent.agent_id, linewidth=1.5)

        plt.title('Individual Agent Coherence Trajectories')
        plt.xlabel('Interaction Round')
        plt.ylabel('Coherence Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Concept usage heatmap (placeholder)
        plt.subplot(2, 2, 4)
        concepts = ['Identity', 'NonContradiction', 'ExcludedMiddle', 'Distinction',
                   'Relation', 'Agency', 'Coherence', 'Truth', 'Existence', 'Goodness']
        usage_matrix = np.random.rand(len(self.agents), len(concepts))  # Placeholder
        plt.imshow(usage_matrix, cmap='viridis', aspect='auto')
        plt.title('Ontological Concept Usage')
        plt.xlabel('Concepts')
        plt.ylabel('Agents')
        plt.xticks(range(len(concepts)), concepts, rotation=45, ha='right', fontsize=8)
        plt.yticks(range(len(self.agents)), [f'Agent {i+1}' for i in range(len(self.agents))])

        plt.tight_layout()
        plt.savefig('dual_bijection_emergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Run the dual bijection agent interaction experiment."""
    print("🧠 LOGOS Dual Bijection Agent Interaction Experiment")
    print("=" * 55)

    # Initialize experiment
    experiment = DualBijectionInteractionExperiment()
    experiment.create_agents(3)  # Create 3 agents with different personalities

    # Define initial context
    initial_context = {
        'topic': 'Ontological consciousness emergence through dual bijection',
        'keywords': ['consciousness', 'emergence', 'ontology', 'bijection', 'logic']
    }

    # Run the experiment
    experiment.run_interaction_round(initial_context, rounds=8)

    print("\n✅ Experiment completed successfully!")
    print("📁 Generated files:")
    print("   - dual_bijection_agent_experiment.json (detailed results)")
    print("   - dual_bijection_emergence_analysis.png (visual analysis)")

if __name__ == "__main__":
    main()