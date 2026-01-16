# MODULE_META:
#   module_id: SCM-016
#   layer: SEMANTIC_AXIOM
#   role: READ_ONLY_AGENT_INTERFACE_AXIOMS
#   phase_origin: PHASE_AXIOMATIZATION
#   description: Read-only agent interface axioms defined in Read_Only_Agent_Interface.py.
#   contracts: []
#   allowed_imports: []
#   prohibited_behaviors: [IO, NETWORK, TIME, RANDOM]
#   entrypoints: [run]
#   callable_surface: INTERNAL
#   state_mutation: NONE
#   runtime_spine_binding: NONE
#   depends_on_contexts: []
#   invoked_by: [Canonical_System_Bootstrap_Pipeline]

def assert_read_only(action):
    if action.get("writes_uwm"):
        raise RuntimeError("Agents are read-only by contract")
