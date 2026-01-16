# MODULE_META:
#   module_id: SCM-011
#   layer: SEMANTIC_AXIOM
#   role: TEMPORAL_SUPERSESSION_AXIOMS
#   phase_origin: PHASE_AXIOMATIZATION
#   description: Temporal supersession axioms defined in Temporal_Supersession.py.
#   contracts: []
#   allowed_imports: []
#   prohibited_behaviors: [IO, NETWORK, TIME, RANDOM]
#   entrypoints: [run]
#   callable_surface: INTERNAL
#   state_mutation: NONE
#   runtime_spine_binding: NONE
#   depends_on_contexts: []
#   invoked_by: [Canonical_System_Bootstrap_Pipeline]

def supersede(old_atom, new_atom):
    old_atom["valid_to"] = "closed"
    new_atom["supersedes"] = old_atom["id"]
