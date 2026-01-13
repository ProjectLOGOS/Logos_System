# Tetraconscious Integration Notes

## Final file paths
- System_Stack/Logos_AGI/Logos_Agent/fractal_geometry.py
- System_Stack/Logos_AGI/Logos_Agent/mesh_tlm_harmonizer.py
- System_Stack/Logos_AGI/memory/uwm_store.py
- System_Stack/Logos_AGI/memory/memory_discovery_loop.py
- System_Stack/Logos_AGI/control/curiosity_halt_trigger.py
- System_Stack/Logos_AGI/ontology/omni_property_map.py

## Import paths (LOGOS / I1 / I2 / I3)
- from System_Stack.Logos_AGI.Logos_Agent.fractal_geometry import FractalTetrahedron
- from System_Stack.Logos_AGI.Logos_Agent.mesh_tlm_harmonizer import MeshTLMHarmonizer
- from System_Stack.Logos_AGI.memory.uwm_store import UWMStore
- from System_Stack.Logos_AGI.memory.memory_discovery_loop import MemoryDiscoveryLoop, I1_DISCOVERY, I2_DISCOVERY, I3_DISCOVERY
- from System_Stack.Logos_AGI.control.curiosity_halt_trigger import AgentControlCenter, LOGOS_CTRL, I1_CTRL, I2_CTRL, I3_CTRL, broadcast_input_to_I2
- from System_Stack.Logos_AGI.ontology.omni_property_map import OMNIPROPERTY_MAP

## How to call
- I2 calls broadcast_input_to_I2() on input; others check halt_signal before acting.
