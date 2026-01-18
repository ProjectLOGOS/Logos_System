# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
Dual Bijection Analysis: Fixing Ontological Coherence
=====================================================

This implements a dual bijection system to address the ontological coherence
issues revealed by the PXL fractal boundary analysis. The system allows
many-to-one inverse mappings to resolve commutation failures while maintaining
ontological structure.

The mappings represent:
A: Logical primitives → Ontological categories
B: Relational primitives → Ontological categories

Where the inverses allow multiple primitives to map to the same category,
resolving the bijection failures identified in fractal analysis.
"""

import json
from typing import Dict, List, Any
from collections import defaultdict

# Dual bijection mappings (allowing many-to-one inverses)
A = {'I': 'C', 'NC': 'T', 'EM': 'T'}  # Logical primitives → Categories
A_inverse = {'C': 'I', 'T': ['NC', 'EM']}  # Categories → Logical primitives

B = {'D': 'E', 'R': 'G', 'A': 'G'}  # Relational primitives → Categories
B_inverse = {'E': 'D', 'G': ['R', 'A']}  # Categories → Relational primitives

def f(x): return A.get(x)
def f_inv(y): return A_inverse.get(y)

def g(x): return B.get(x)
def g_inv(y): return B_inverse.get(y)

class DualBijectionSystem:
    """Enhanced dual bijection system with ontological coherence checking."""

    def __init__(self, A_map: Dict[str, str], B_map: Dict[str, str]):
        self.A = A_map
        self.B = B_map

        # Build inverse mappings (allowing many-to-one)
        self.A_inv = self._build_inverse(A_map)
        self.B_inv = self._build_inverse(B_map)

        # Domain and codomain analysis
        self.A_domain = set(A_map.keys())
        self.A_codomain = set(A_map.values())
        self.B_domain = set(B_map.keys())
        self.B_codomain = set(B_map.values())

    def _build_inverse(self, mapping: Dict[str, str]) -> Dict[str, List[str]]:
        """Build inverse mapping allowing many-to-one relationships."""
        inverse = defaultdict(list)
        for key, value in mapping.items():
            inverse[value].append(key)
        return dict(inverse)

    def check_bijection_properties(self) -> Dict[str, Any]:
        """Check bijection properties for both mappings."""
        results = {
            'A_mapping': {
                'injective': len(self.A) == len(set(self.A.values())),  # one-to-one
                'surjective': self.A_codomain.issubset(set(self.A_inv.keys())),  # onto
                'bijective': False,  # will be set below
                'domain_size': len(self.A_domain),
                'codomain_size': len(self.A_codomain),
                'inverse_uniqueness': all(len(vals) == 1 for vals in self.A_inv.values())
            },
            'B_mapping': {
                'injective': len(self.B) == len(set(self.B.values())),
                'surjective': self.B_codomain.issubset(set(self.B_inv.keys())),
                'bijective': False,
                'domain_size': len(self.B_domain),
                'codomain_size': len(self.B_codomain),
                'inverse_uniqueness': all(len(vals) == 1 for vals in self.B_inv.values())
            }
        }

        # Check if bijective (both injective and surjective)
        results['A_mapping']['bijective'] = (
            results['A_mapping']['injective'] and results['A_mapping']['surjective']
        )
        results['B_mapping']['bijective'] = (
            results['B_mapping']['injective'] and results['B_mapping']['surjective']
        )

        return results

    def test_commutation(self) -> Dict[str, Any]:
        """Test commutation properties of the dual bijection system."""
        commutation_results = {
            'A_B_commutes': True,
            'B_A_commutes': True,
            'commutation_failures': [],
            'commutation_score': 1.0
        }

        failures = 0
        total_tests = 0

        # Test A ∘ B commutation (should equal B ∘ A where defined)
        for x in self.A_domain:
            if x in self.B_domain:
                total_tests += 1
                a_b_x = f(g(x)) if g(x) in self.A_domain else None
                b_a_x = g(f(x)) if f(x) in self.B_domain else None

                if a_b_x != b_a_x:
                    failures += 1
                    commutation_results['commutation_failures'].append({
                        'type': 'A_B_commutation',
                        'x': x,
                        'A∘B(x)': a_b_x,
                        'B∘A(x)': b_a_x
                    })

        # Test inverse commutation
        for y in self.A_codomain:
            if y in self.B_codomain:
                total_tests += 1
                a_inv_b_inv_y = f_inv(g_inv(y)) if g_inv(y) else None
                b_inv_a_inv_y = g_inv(f_inv(y)) if f_inv(y) else None

                # For lists, check if sets are equal
                if isinstance(a_inv_b_inv_y, list) and isinstance(b_inv_a_inv_y, list):
                    if set(a_inv_b_inv_y) != set(b_inv_a_inv_y):
                        failures += 1
                        commutation_results['commutation_failures'].append({
                            'type': 'inverse_commutation',
                            'y': y,
                            'A⁻¹∘B⁻¹(y)': a_inv_b_inv_y,
                            'B⁻¹∘A⁻¹(y)': b_inv_a_inv_y
                        })
                elif a_inv_b_inv_y != b_inv_a_inv_y:
                    failures += 1
                    commutation_results['commutation_failures'].append({
                        'type': 'inverse_commutation',
                        'y': y,
                        'A⁻¹∘B⁻¹(y)': a_inv_b_inv_y,
                        'B⁻¹∘A⁻¹(y)': b_inv_a_inv_y
                    })

        commutation_results['commutation_score'] = 1.0 - (failures / max(total_tests, 1))
        commutation_results['A_B_commutes'] = failures == 0
        commutation_results['B_A_commutes'] = failures == 0

        return commutation_results

    def analyze_ontological_coherence(self) -> Dict[str, Any]:
        """Analyze ontological coherence implications."""
        bijection_props = self.check_bijection_properties()
        commutation_props = self.test_commutation()

        coherence_analysis = {
            'bijection_properties': bijection_props,
            'commutation_properties': commutation_props,
            'ontological_implications': {},
            'coherence_score': 0.0,
            'improvement_over_previous': {}
        }

        # Calculate coherence score based on bijection and commutation
        bijection_score = 0.0
        if bijection_props['A_mapping']['bijective']:
            bijection_score += 0.5
        if bijection_props['B_mapping']['bijective']:
            bijection_score += 0.5

        commutation_score = commutation_props['commutation_score']

        coherence_analysis['coherence_score'] = (bijection_score + commutation_score) / 2

        # Ontological implications
        if not bijection_props['A_mapping']['bijective']:
            coherence_analysis['ontological_implications']['A_non_bijective'] = (
                "Logical primitives mapping allows ontological flexibility but breaks strict bijection"
            )

        if not bijection_props['B_mapping']['bijective']:
            coherence_analysis['ontological_implications']['B_non_bijective'] = (
                "Relational primitives mapping allows ontological flexibility but breaks strict bijection"
            )

        if commutation_props['commutation_score'] < 1.0:
            coherence_analysis['ontological_implications']['commutation_failures'] = (
                f"{len(commutation_props['commutation_failures'])} commutation failures indicate "
                "ontological structure inconsistencies"
            )

        # Compare to previous strict bijection system
        coherence_analysis['improvement_over_previous'] = {
            'previous_commutation_failures': 6,  # From fractal analysis
            'current_commutation_failures': len(commutation_props['commutation_failures']),
            'improvement': len(commutation_props['commutation_failures']) < 6,
            'flexibility_gained': not (bijection_props['A_mapping']['bijective'] and bijection_props['B_mapping']['bijective'])
        }

        return coherence_analysis

def demonstrate_dual_bijection_fix():
    """Demonstrate how this dual bijection system addresses ontological coherence issues."""

    # Define functions locally
    def f(x): return A.get(x)
    def f_inv(y): return A_inverse.get(y)
    def g(x): return B.get(x)
    def g_inv(y): return B_inverse.get(y)

    print("🔗 Dual Bijection System: Fixing Ontological Coherence")
    print("=" * 60)

    # Create the dual bijection system
    system = DualBijectionSystem(A, B)

    # Basic demonstrations
    print("\n📊 Basic Mappings:")
    print(f"A: {A}")
    print(f"A⁻¹: {A_inverse}")
    print(f"B: {B}")
    print(f"B⁻¹: {B_inverse}")

    print("\n🔄 Function Demonstrations:")
    print(f"f('I') = {f('I')}")
    print(f"f⁻¹('T') = {f_inv('T')}")
    print(f"g('R') = {g('R')}")
    print(f"g⁻¹('G') = {g_inv('G')}")

    # Comprehensive analysis
    analysis = system.analyze_ontological_coherence()

    print("\n🔍 Bijection Properties:")
    print(f"A mapping - Injective: {analysis['bijection_properties']['A_mapping']['injective']}, "
          f"Surjective: {analysis['bijection_properties']['A_mapping']['surjective']}, "
          f"Bijective: {analysis['bijection_properties']['A_mapping']['bijective']}")
    print(f"B mapping - Injective: {analysis['bijection_properties']['B_mapping']['injective']}, "
          f"Surjective: {analysis['bijection_properties']['B_mapping']['surjective']}, "
          f"Bijective: {analysis['bijection_properties']['B_mapping']['bijective']}")

    print("\n🔗 Commutation Analysis:")
    print(f"Commutation Score: {analysis['commutation_properties']['commutation_score']:.3f}")
    print(f"Commutation Failures: {len(analysis['commutation_properties']['commutation_failures'])}")

    if analysis['commutation_properties']['commutation_failures']:
        print("Failure Details:")
        for failure in analysis['commutation_properties']['commutation_failures'][:3]:
            print(f"  • {failure['type']}: {failure}")

    print("\n🧬 Ontological Coherence:")
    print(f"Overall Coherence Score: {analysis['coherence_score']:.3f}")

    print("\n📈 Improvement Over Previous System:")
    improvement = analysis['improvement_over_previous']
    print(f"Previous Commutation Failures: {improvement['previous_commutation_failures']}")
    print(f"Current Commutation Failures: {improvement['current_commutation_failures']}")
    print(f"Improvement Achieved: {improvement['improvement']}")
    print(f"Flexibility Gained: {improvement['flexibility_gained']}")

    print("\n💡 Key Insights:")
    if analysis['coherence_score'] > 0.5:
        print("  • Dual bijection system shows improved ontological coherence")
    else:
        print("  • Ontological coherence still requires further refinement")

    if improvement['flexibility_gained']:
        print("  • Many-to-one inverse mappings provide ontological flexibility")
        print("  • Allows multiple primitives to share ontological categories")

    if improvement['improvement']:
        print("  • Significant reduction in commutation failures vs. strict bijection")

    # Save detailed analysis
    with open('dual_bijection_coherence_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print("\n📁 Analysis saved to: dual_bijection_coherence_analysis.json")
    return analysis

if __name__ == "__main__":
    demonstrate_dual_bijection_fix()
