# MODULE_META:
#   module_id: ORX-002
#   layer: ORCHESTRATION
#   role: Primary external system entry point and process initializer
#   phase_origin: PHASE_SYSTEM_ENTRY
#   description: Orchestration entry defined in Logos_System_Entry_Point.py.
#   contracts: []
#   allowed_imports: [SEMANTIC_AXIOMS, SEMANTIC_CONTEXTS, ORCHESTRATION]
#   prohibited_behaviors: [RANDOM]
#   entrypoints: [main]
#   callable_surface: APPLICATION
#   state_mutation: GLOBAL
#   runtime_spine_binding: SYSTEM_ENTRY
#   depends_on_contexts: []
#   invoked_by: [OS_CLI]

"""
LOGOS System Entry Point

NOTE:
- This is the sole public entrypoint for system startup.
- All legacy entrypoints are deprecated.
"""

from PYTHON_MODULES.ORCHESTRATION_AND_ENTRYPOINTS.BOOTSTRAP_PIPELINES.Canonical_System_Bootstrap_Pipeline import system_bootstrap


def main(initial_context: dict) -> dict:
    return system_bootstrap(initial_context)
