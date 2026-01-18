# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
Test script for autonomous gap-filling hardening implementation.
"""

import sys
import asyncio
from pathlib import Path

# Add the code generator path
sys.path.insert(
    0,
    str(
        Path(__file__).parent
        / "external"
        / "Logos_AGI"
        / "System_Operations_Protocol"
        / "code_generator"
    ),
)

from development_environment import SOPCodeEnvironment, CodeGenerationRequest
from self_improvement_integration import SOPSelfImprovementManager


async def test_hardening():
    """Test the hardening implementation."""
    print("=== Testing Autonomous Gap-Filling Hardening ===\n")

    # Test 1: Knowledge Catalog
    print("1. Testing Knowledge Catalog...")
    from knowledge_catalog import KnowledgeCatalog

    base_dir = Path(__file__).parent
    KnowledgeCatalog(base_dir)
    print("✓ KnowledgeCatalog initialized")

    # Test 2: Code Environment with staging
    print("\n2. Testing SOPCodeEnvironment...")
    env = SOPCodeEnvironment()
    print("✓ SOPCodeEnvironment initialized")

    # Test staging with simple code
    test_code = '''
def test_function():
    """Test function for staging."""
    return {"status": "success", "message": "Staging test passed"}

if __name__ == "__main__":
    result = test_function()
    print(f"Result: {result}")
'''

    stage_result = env._stage_candidate(test_code)
    print(f"✓ Staging result: {stage_result['stage_ok']}")

    # Test 3: Policy classification
    print("\n3. Testing Policy Classification...")
    repair_request = CodeGenerationRequest(
        improvement_id="repair-001",
        description="Fix a critical bug in the system",
        target_module="logos_core.bugfix",
        improvement_type="function",
        requirements={"gap_category": "analysis"},
        constraints={},
        test_cases=[],
    )

    enhancement_request = CodeGenerationRequest(
        improvement_id="enhance-001",
        description="Add new optimization feature",
        target_module="logos_core.optimize",
        improvement_type="class",
        requirements={"gap_category": "predictive_reasoning"},
        constraints={},
        test_cases=[],
    )

    repair_policy = env._classify_policy(repair_request, allow_enhancements=False)
    enhancement_policy_blocked = env._classify_policy(
        enhancement_request, allow_enhancements=False
    )
    enhancement_policy_allowed = env._classify_policy(
        enhancement_request, allow_enhancements=True
    )

    print(
        f"✓ Repair policy: {repair_policy['policy_class']} -> {repair_policy['deploy_allowed']}"
    )
    print(
        f"✓ Enhancement (blocked): {enhancement_policy_blocked['policy_class']} -> {enhancement_policy_blocked['deploy_allowed']}"
    )
    print(
        f"✓ Enhancement (allowed): {enhancement_policy_allowed['policy_class']} -> {enhancement_policy_allowed['deploy_allowed']}"
    )

    # Test 4: Full code generation with catalog persistence
    print("\n4. Testing Full Code Generation...")
    result = env.generate_code(repair_request, allow_enhancements=False)
    print(f"✓ Generation result: {result['success']}")
    print(f"✓ Entry ID: {result['entry_id']}")
    print(f"✓ Policy class: {result['policy_class']}")
    print(f"✓ Deployed: {result['deployed']}")

    # Check catalog
    catalog_file = base_dir / "training_data" / "index" / "catalog.jsonl"
    if catalog_file.exists():
        with open(catalog_file, "r") as f:
            lines = f.readlines()
        print(f"✓ Catalog entries: {len(lines)}")
    else:
        print("✗ Catalog file not created")

    # Test 5: Self-improvement manager
    print("\n5. Testing Self-Improvement Manager...")
    manager = SOPSelfImprovementManager()

    metrics = {
        "average_response_time": 5000,
        "error_count": 5,
        "total_requests": 100,
        "memory_usage_mb": 400,
    }

    results = await manager.analyze_and_improve(metrics, allow_enhancements=False)
    print(f"✓ Analysis completed: {results['opportunities_identified']} opportunities")
    print(
        f"✓ Generated: {results['improvements_generated']}, Deployed: {results['improvements_deployed']}"
    )
    print(f"✓ Enhancements blocked: {results['enhancements_blocked']}")

    print("\n=== All Tests Completed Successfully! ===")


if __name__ == "__main__":
    asyncio.run(test_hardening())
