# MODULE_META:
#   module_id: SCM-017
#   layer: SEMANTIC_AXIOM
#   role: UWM_INGESTION_AXIOMS
#   phase_origin: PHASE_AXIOMATIZATION
#   description: UWM ingestion axioms defined in UWM_Ingestion.py.
#   contracts: []
#   allowed_imports: []
#   prohibited_behaviors: [IO, NETWORK, TIME, RANDOM]
#   entrypoints: [run]
#   callable_surface: INTERNAL
#   state_mutation: NONE
#   runtime_spine_binding: NONE
#   depends_on_contexts: []
#   invoked_by: [Canonical_System_Bootstrap_Pipeline]

def ingest(event):
    if not event.get("approved"):
        raise RuntimeError("UWM ingestion rejected")
