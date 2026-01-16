# MODULE_META:
#   module_id: SCM-008
#   layer: SEMANTIC_AXIOM
#   role: EVIDENCE_CHAIN_AXIOMS
#   phase_origin: PHASE_AXIOMATIZATION
#   description: Evidence chain axioms defined in Evidence_Chain.py.
#   contracts: []
#   allowed_imports: []
#   prohibited_behaviors: [IO, NETWORK, TIME, RANDOM]
#   entrypoints: [run]
#   callable_surface: INTERNAL
#   state_mutation: NONE
#   runtime_spine_binding: NONE
#   depends_on_contexts: []
#   invoked_by: [Canonical_System_Bootstrap_Pipeline]

def verify_evidence_chain(chain):
    if not chain.get("chain_hash"):
        raise RuntimeError("Evidence chain missing or invalid")
