# MODULE_META:
#   module_id: SCM-009
#   layer: SEMANTIC_AXIOM
#   role: INVARIANT_CONSTRAINT_AXIOMS
#   phase_origin: PHASE_AXIOMATIZATION
#   description: Invariant constraint axioms defined in Invariant_Constraints.py.
#   contracts: []
#   allowed_imports: []
#   prohibited_behaviors: [IO, NETWORK, TIME, RANDOM]
#   entrypoints: [run]
#   callable_surface: INTERNAL
#   state_mutation: NONE
#   runtime_spine_binding: NONE
#   depends_on_contexts: []
#   invoked_by: [Canonical_System_Bootstrap_Pipeline]

def enforce_invariants(atom):
    if atom.get("mutated_after_verification"):
        raise RuntimeError("Invariant violation detected")
