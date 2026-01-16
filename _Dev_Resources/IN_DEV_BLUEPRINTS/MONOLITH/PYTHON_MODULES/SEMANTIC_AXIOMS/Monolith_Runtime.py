# MODULE_META:
#   module_id: SCM-019
#   layer: SEMANTIC_AXIOM
#   role: MONOLITH_RUNTIME_AXIOMS
#   phase_origin: PHASE_AXIOMATIZATION
#   description: Monolithic runtime axioms defined in Monolith_Runtime.py.
#   contracts: []
#   allowed_imports: []
#   prohibited_behaviors: [IO, NETWORK, TIME, RANDOM]
#   entrypoints: [run]
#   callable_surface: INTERNAL
#   state_mutation: NONE
#   runtime_spine_binding: NONE
#   depends_on_contexts: []
#   invoked_by: [Canonical_System_Bootstrap_Pipeline]

import json
from pathlib import Path
from importlib.machinery import SourceFileLoader
import importlib.util


def load_contract(name):
    """Load a canonical contract by name from governance store."""
    contract_path = Path("_Governance/Canonical_Contracts") / f"{name}_Contract.json"
    return json.loads(contract_path.read_text())


def load_canonical_module(name):
    """Load a monolith module even if the filename is not a valid Python identifier."""
    module_path = Path("Monolith") / f"{name}.py"
    loader = SourceFileLoader(f"Monolith.{name}", str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def invoke(name, context):
    """Invoke a canonical module by name with its contract."""
    if name not in monolith_routing:
        raise KeyError(f"No monolith routing for {name}")
    return monolith_routing[name](context)


monolith_routing = {
    "3PDN_Validator": lambda ctx: load_canonical_module("3PDN_Validator").run(ctx, load_contract("3PDN_Validator")),
    "3PDN_Constraint": lambda ctx: load_canonical_module("3PDN_Constraint").run(ctx, load_contract("3PDN_Constraint")),
    "Hypostatic_ID_Validator": lambda ctx: load_canonical_module("Hypostatic_ID_Validator").run(ctx, load_contract("Hypostatic_ID_Validator")),
    "Runtime_Input_Sanitizer": lambda ctx: load_canonical_module("Runtime_Input_Sanitizer").run(ctx, load_contract("Runtime_Input_Sanitizer")),
    "Runtime_Context_Initializer": lambda ctx: load_canonical_module("Runtime_Context_Initializer").run(ctx, load_contract("Runtime_Context_Initializer")),
    "Runtime_Mode_Controller": lambda ctx: load_canonical_module("Runtime_Mode_Controller").run(ctx, load_contract("Runtime_Mode_Controller")),
    "Trinitarian_Logic_Core": lambda ctx: load_canonical_module("Trinitarian_Logic_Core").run(ctx, load_contract("Trinitarian_Logic_Core")),
    "Trinitarian_Alignment_Core": lambda ctx: load_canonical_module("Trinitarian_Alignment_Core").run(ctx, load_contract("Trinitarian_Alignment_Core")),
    "Agent_Activation_Gate": lambda ctx: load_canonical_module("Agent_Activation_Gate").run(ctx, load_contract("Agent_Activation_Gate")),
    "Global_Bijective_Recursion_Core": lambda ctx: load_canonical_module("Global_Bijective_Recursion_Core").run(ctx, load_contract("Global_Bijective_Recursion_Core")),
    "Necessary_Existence_Core": lambda ctx: load_canonical_module("Necessary_Existence_Core").run(ctx, load_contract("Necessary_Existence_Core")),
}


if __name__ == "__main__":
    print("Monolith runtime entrypoint. Use invoke(<name>, context) for canonical modules.")
