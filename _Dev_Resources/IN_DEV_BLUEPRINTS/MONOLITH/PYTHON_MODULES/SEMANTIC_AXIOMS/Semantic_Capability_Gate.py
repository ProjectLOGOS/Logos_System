# MODULE_META:
#   module_id: SCM-012
#   layer: SEMANTIC_AXIOM
#   role: SEMANTIC_CAPABILITY_GATING_AXIOMS
#   phase_origin: PHASE_AXIOMATIZATION
#   description: Semantic capability gating axioms defined in Semantic_Capability_Gate.py.
#   contracts: []
#   allowed_imports: []
#   prohibited_behaviors: [IO, NETWORK, TIME, RANDOM]
#   entrypoints: [run]
#   callable_surface: INTERNAL
#   state_mutation: NONE
#   runtime_spine_binding: NONE
#   depends_on_contexts: []
#   invoked_by: [Canonical_System_Bootstrap_Pipeline]

def admit_capability(cap):
    if not cap.get("mapped"):
        raise RuntimeError("Capability not admitted")
