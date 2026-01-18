# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# INSTALL_STATUS: SEMANTIC_REWRITE
# SOURCE_LEGACY: example_usage.py

"""
SEMANTIC REWRITE

This module has been rewritten for governed integration into the
LOGOS System Rebuild. Its runtime scope and protocol role have been
normalized, but its original logical structure has been preserved.
"""

"""
ARP Nexus Usage Example
=======================

Demonstrates how to use the ARP Nexus for recursive inter-protocol data processing
with C-value fractal data exchange and cycle management.
"""

import asyncio
import sys
from pathlib import Path

# Add the nexus directory to path
sys.path.insert(0, str(Path(__file__).parent))

from arp_nexus import ARPNexus, ReasoningRequest, ReasoningMode


async def example_standard_reasoning():
    """Example of standard reasoning analysis"""

    nexus = ARPNexus()
    await nexus.initialize()

    # Create a reasoning request
    request = ReasoningRequest(
        request_id="example_standard_001",
        reasoning_mode=ReasoningMode.STANDARD_ANALYSIS,
        input_data={
            "query": "What are the ethical implications of AGI development?",
            "context": "AI safety and alignment research",
            "stakeholders": ["humanity", "AI_systems", "researchers"]
        },
        domain_focus=["AnthroPraxis", "Axiopraxis", "EthosPraxis"],
        mathematical_foundations=True
    )

    # Process the request
    result = await nexus.process_reasoning_request(request)



async def example_recursive_refinement():
    """Example of recursive refinement with SCP and Agent coordination"""

    nexus = ARPNexus()
    await nexus.initialize()

    # Create a recursive refinement request
    request = ReasoningRequest(
        request_id="example_recursive_001",
        reasoning_mode=ReasoningMode.RECURSIVE_REFINEMENT,
        input_data={
            "initial_problem": "Design optimal AGI safety measures",
            "current_solution": {
                "safety_layers": ["monitoring", "constraints"],
                "verification_methods": ["formal_proofs", "testing"]
            },
            "improvement_goals": ["robustness", "adaptability", "transparency"]
        },
        recursive_cycles=5,  # Allow up to 5 refinement cycles
        c_value_data={
            "safety_c": complex(0.8, 0.6),      # Safety optimization fractal
            "robustness_c": complex(0.7, 0.7),  # Robustness fractal
            "ethics_c": complex(0.9, 0.4)       # Ethical alignment fractal
        }
    )

    for key, c_val in request.c_value_data.items():

    # Process the recursive request
    result = await nexus.process_reasoning_request(request)


    if result.c_value_evolution:
        for key, c_val in result.c_value_evolution.items():

    # Check cycle status
    cycle_key = f"ARP_SCP_{request.request_id.split('_')[-1]}"
    status = nexus.get_cycle_status(cycle_key)


async def example_deep_ontological():
    """Example of deep ontological analysis"""

    nexus = ARPNexus()
    await nexus.initialize()

    # Create deep ontological request
    request = ReasoningRequest(
        request_id="example_ontological_001",
        reasoning_mode=ReasoningMode.DEEP_ONTOLOGICAL,
        input_data={
            "ontological_question": "What is the fundamental nature of intelligence?",
            "paradigms": ["biological", "artificial", "hybrid"],
            "dimensions": ["existence", "goodness", "truth"]
        },
        domain_focus=["OntoPraxis", "GnosiPraxis", "CosmoPraxis"],
        mathematical_foundations=True,
        formal_verification=True
    )

    # Process the request
    result = await nexus.process_reasoning_request(request)



async def example_cycle_management():
    """Example of cycle limit management and cascade handling"""

    nexus = ARPNexus()

    # Demonstrate cycle limit changes

    # System can adjust limits
    nexus.set_max_cycles(12, "system")

    # Create emergency packet that triggers cascade detection
    emergency_packet = nexus.data_builder.create_exchange_packet(
        source_protocol="ARP",
        target_protocol="SCP",
        data_payload={
            "emergency": "critical_system_failure",
            "threat_level": "existential",
            "cascade_detected": True
        }
    )


    # Approve cascade override
    approved = nexus.approve_cascade_override(emergency_packet.packet_id, True)


    """Run all usage examples"""

    examples = [
        ("Standard Reasoning", example_standard_reasoning),
        ("Recursive Refinement", example_recursive_refinement),
        ("Deep Ontological Analysis", example_deep_ontological),
        ("Cycle Management", example_cycle_management)
    ]

    for example_name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
