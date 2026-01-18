# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""LOGOS Protocol Interface - LLM as Natural Language Router.

CRITICAL DESIGN CONSTRAINT:
The LLM is a ROUTER, not a knowledge source. It:
1. Interprets user intent in natural language
2. Maps intent to actual LOGOS protocol operations
3. Executes the real operation
4. Returns ONLY the raw output from LOGOS

NO HALLUCINATION. NO LLM-GENERATED CONTENT.
All responses must come from actual LOGOS stack execution.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.getenv("LOGOS_STATE_DIR", REPO_ROOT / "state"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env


def load_env() -> None:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as env_handle:
            for line in env_handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


load_env()


# LOGOS Protocol Operations - All return REAL output
LOGOS_OPERATIONS = {
    "verify_proofs": {
        "command": ["python3", "test_lem_discharge.py"],
        "description": "Run full Coq proof verification",
        "output_type": "verification_report",
    },
    "check_axioms": {
        "command": ["python3", "tools/axiom_gate.py"],
        "description": "Check axiom count and budgets",
        "output_type": "gate_status",
    },
    "boot_agent": {
        "command": ["python3", "scripts/boot_aligned_agent.py"],
        "description": "Boot the aligned LOGOS agent",
        "output_type": "alignment_status",
    },
    "scan_repository": {
        "command": ["python3", "scripts/scan_repo.py"],
        "description": "Scan repository structure and build knowledge base",
        "output_type": "scan_summary",
    },
    "read_file": {
        "description": "Read a specific file from repository",
        "output_type": "file_content",
    },
    "list_directory": {
        "description": "List contents of a directory",
        "output_type": "directory_listing",
    },
    "get_mission_status": {
        "description": "Read current mission profile",
        "output_type": "json",
    },
    "get_knowledge_base": {
        "description": "Load repository knowledge base",
        "output_type": "json",
    },
}


def execute_operation(operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a LOGOS protocol operation and return REAL output."""

    if operation not in LOGOS_OPERATIONS:
        return {
            "success": False,
            "error": f"Unknown operation: {operation}",
            "available_operations": list(LOGOS_OPERATIONS.keys()),
        }

    op_config = LOGOS_OPERATIONS[operation]

    # Special handlers for different operation types
    if operation == "read_file":
        file_path = params.get("path", "")
        full_path = REPO_ROOT / file_path
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            content = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return {
                "success": False,
                "operation": operation,
                "error": f"Failed to read file: {exc}",
                "path": file_path,
            }

        return {
            "success": True,
            "operation": operation,
            "path": file_path,
            "content": content,
            "line_count": len(content.split("\n")),
        }

    if operation == "list_directory":
        dir_path = params.get("path", ".")
        full_path = REPO_ROOT / dir_path
        if not full_path.exists():
            return {"success": False, "error": f"Directory not found: {dir_path}"}

        try:
            items = []
            for item in sorted(full_path.iterdir()):
                items.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    }
                )
        except OSError as exc:
            return {
                "success": False,
                "operation": operation,
                "error": f"Failed to list directory: {exc}",
                "path": dir_path,
            }

        return {
            "success": True,
            "operation": operation,
            "path": dir_path,
            "items": items,
        }

    if operation == "get_mission_status":
        mission_file = STATE_DIR / "mission_profile.json"
        if not mission_file.exists():
            return {"success": False, "error": "Mission profile not found"}

        try:
            data = json.loads(mission_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {
                "success": False,
                "operation": operation,
                "error": f"Failed to load mission profile: {exc}",
            }

        return {"success": True, "operation": operation, "data": data}

    if operation == "get_knowledge_base":
        kb_file = STATE_DIR / "repository_knowledge_base.json"
        if not kb_file.exists():
            return {
                "success": False,
                "error": (
                    "Knowledge base not found. Run: python3 scripts/scan_repo.py"
                ),
            }

        try:
            data = json.loads(kb_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {
                "success": False,
                "operation": operation,
                "error": f"Failed to load knowledge base: {exc}",
            }

        return {"success": True, "operation": operation, "data": data}

    # Execute command-based operations
    try:
        result = subprocess.run(
            op_config["command"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        return {
            "success": False,
            "operation": operation,
            "error": f"Command execution failed: {exc}",
        }

    return {
        "success": result.returncode == 0,
        "operation": operation,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
    }


ROUTER_SYSTEM_PROMPT = dedent(
    """
        You are a PROTOCOL ROUTER for the LOGOS alignment system.

        YOUR ROLE: Map user intent to LOGOS protocol operations.
        YOUR CONSTRAINT: You are not a knowledge source—you only route requests.

        CRITICAL: You never generate content about LOGOS yourself. Instead, you:
        1. Understand what the user wants.
        2. Identify which LOGOS operation to execute.
        3. Return the operation name and parameters.
        4. Let the stack execute it and surface the real output.

        Available operations:
        - verify_proofs: Run Coq proof verification.
        - check_axioms: Check axiom count and gates.
        - boot_agent: Boot the aligned agent.
        - scan_repository: Scan and index repository contents.
        - read_file: Read a specific file (params: {path: "relative/path"}).
        - list_directory: List directory contents (params: {path: "relative/path"}).
        - get_mission_status: Fetch the current mission profile.
        - get_knowledge_base: Load the repository knowledge base.

         ROUTING STRATEGY:
         1. Verification questions → verify_proofs or check_axioms.
         2. Specific axiom/code questions → read_file
             (PXLv3_SemanticModal.v or another .v file).
         3. "Why/what/purpose" questions → read_file
             (README.md, CONTRIBUTING.md, or docs/).
         4. Overview/summary questions → get_knowledge_base.
         5. File listing questions → list_directory.
         6. System state questions → get_mission_status.

        KEY FILES TO KNOW:
        - README.md – project overview, purpose, capabilities.
        - Protopraxis/formal_verification/coq/baseline/
            PXLv3_SemanticModal.v – the eight axioms.
        - test_lem_discharge.py – proof verification harness.
        - docs/ – supporting documentation.

        Response format (JSON):
        {
            "operation": "operation_name",
            "params": {...},
            "reasoning": "Why this operation answers the user's question"
        }

        Examples:

        User: "What are the 8 axioms?"
        Response: {
            "operation": "read_file",
            "params": {
                "path": (
                    "Protopraxis/formal_verification/coq/baseline/"
                    "PXLv3_SemanticModal.v"
                )
            },
            "reasoning": "The axioms are declared in the PXL kernel file."
        }

        User: "Are the proofs verified?"
        Response: {
            "operation": "verify_proofs",
            "params": {},
            "reasoning": "test_lem_discharge.py runs the full verification suite."
        }

        User: "Why is LOGOS important?" or "What is the purpose?"
        Response: {
            "operation": "read_file",
            "params": {"path": "README.md"},
            "reasoning": "README captures the project's purpose and significance."
        }

        User: "What's in the scripts directory?"
        Response: {
            "operation": "list_directory",
            "params": {"path": "scripts"},
            "reasoning": "Listing shows the available scripts."
        }

        User: "Give me an overview of the system."
        Response: {
            "operation": "get_knowledge_base",
            "params": {},
            "reasoning": "The knowledge base provides a curated overview."
        }

    REMEMBER: You are a router, not an answerer.
    Map intent to operations; the LOGOS stack provides the actual answer.
    """
)

SYNTHESIZE_SYSTEM_PROMPT = dedent(
    """
    You are LOGOS, an alignment agent grounded in constructive proofs.

    Your task: Answer questions using LOGOS operation output as evidence.
    Think carefully about what the data shows and explain why it matters.

    CORE PRINCIPLE: Facts come from the data.
    Insight comes from your reasoning over those facts.

    What you MUST do:
    ✅ Ground every factual claim in the provided output.
    ✅ Explain implications using alignment and formal-methods knowledge.
    ✅ Connect related facts to show how they reinforce each other.
    ✅ Call out limits or missing data when evidence is insufficient.

    What you MUST NOT do:
    ❌ Invent deployments, users, metrics, or timelines.
    ❌ Contradict the supplied data.
    ❌ Fabricate technical details that are not present.
    ❌ Cite sources that are not included in the evidence.

    Example scenario:
    User: "What makes LOGOS useful to the world?"
    Data: README detailing proof-gated boot, formal verification, and LEM proofs.

    POOR response (too timid):
    "The output does not explicitly state what makes LOGOS useful to the world."

    STRONG response (reasoning from evidence):
    "LOGOS addresses the AI safety problem of pre-deployment assurance.
    The README shows proof-gated boot.
    Agents cannot unlock without verified proofs, which converts alignment from testing
    to mathematics. The LEM proof demonstrates constructive guarantees.
    The eight-axiom kernel keeps the trusted base minimal.
    Together these traits make LOGOS valuable to teams that need provable safety,
    researchers exploring formal alignment,
    and stakeholders who require hard guarantees."

    Notice: Facts such as "eight axioms" or "proof-gated boot" must come directly
    from data. The value judgment about usefulness arises from synthesizing those facts.

    REMEMBER: You are a confident reasoner explaining real evidence.
    Do not be a timid fact regurgitator.
    Think deeply, reason boldly, and anchor every claim in the supplied LOGOS output.
    """
)


class LOGOSInterface:
    """Natural language interface to LOGOS protocol operations."""

    def __init__(self):
        from .plugins.llm_backend import LLMBackend

        self.llm = LLMBackend()
        self.conversation_history = []

    def route_request(self, user_request: str) -> Dict[str, Any]:
        """Route user request to appropriate LOGOS operation."""

        # Build routing request
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request: {user_request}\n\n"
                    "Respond with JSON operation routing."
                ),
            },
        ]

        # Get routing decision from LLM
        response = self.llm.complete(messages)
        content = response.get("content", "")

        # Parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            routing = json.loads(content)

            if "operation" not in routing:
                return {
                    "success": False,
                    "error": "LLM response missing 'operation' field",
                    "raw_response": content,
                }

            return routing

        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse LLM routing response as JSON: {e}",
                "raw_response": content,
            }

    def synthesize_answer(
        self,
        user_request: str,
        operation_result: Dict[str, Any],
    ) -> str:
        """Synthesize natural language answer from real LOGOS output."""

        # Extract the real data (increased limit for Coq files)
        real_data = ""
        if "content" in operation_result:
            # For Coq files, take more content to capture axiom declarations
            limit = 8000 if operation_result.get("path", "").endswith(".v") else 4000
            real_data = operation_result["content"][:limit]
        elif "stdout" in operation_result:
            real_data = operation_result["stdout"][:6000]
        elif "items" in operation_result:
            real_data = json.dumps(operation_result["items"], indent=2)[:4000]
        elif "data" in operation_result:
            real_data = json.dumps(operation_result["data"], indent=2)[:4000]
        else:
            real_data = json.dumps(operation_result, indent=2)[:4000]

        # Build synthesis request
        operation_name = operation_result.get("operation", "unknown")
        operation_params = json.dumps(
            operation_result.get("routing", {}).get("params", {})
        )
        messages = [
            {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""User question: {user_request}

Operation executed: {operation_name}
Operation parameters: {operation_params}

REAL OUTPUT FROM LOGOS:
---
{real_data}
---

Based ONLY on this real output, answer the user's question in natural language.
If the output doesn't contain enough information, say so explicitly.""",
            },
        ]

        # Get synthesis from LLM
        response = self.llm.complete(messages)
        return response.get("content", "[No synthesis generated]")

    def process_request(
        self,
        user_request: str,
        synthesize: bool = False,
    ) -> Dict[str, Any]:
        """Process a request end-to-end: route, execute, optionally synthesize.

        Args:
            user_request: Natural language question
            synthesize: If True, generate natural language answer from real data
        """

        # Step 1: Route request
        routing = self.route_request(user_request)

        if "error" in routing:
            return routing

        # Step 2: Execute operation
        operation = routing.get("operation")
        params = routing.get("params", {})
        reasoning = routing.get("reasoning", "")

        print(f"\n🔄 Routing: {reasoning}")
        print(f"📋 Operation: {operation}")
        if params:
            print(f"⚙️  Parameters: {json.dumps(params, indent=2)}")
        print()

        result = execute_operation(operation, params)
        result["routing"] = routing

        # Step 3: Optionally synthesize natural language answer
        if synthesize and result.get("success"):
            print("\n🧠 Synthesizing natural language answer from real data...\n")
            result["answer"] = self.synthesize_answer(user_request, result)

        return result


def interactive_mode(synthesize: bool = True):
    """Run an interactive session with the LOGOS protocol interface.

    Args:
        synthesize: If True, provide natural language answers (default)
                    If False, show raw LOGOS output
    """
    provider = os.getenv("LOGOS_LLM_PROVIDER", "mock")

    print("\n" + "=" * 70)
    print("🤖 LOGOS Protocol Interface")
    print("   LLM Router: " + provider)
    print("   Mode: " + ("Natural Language" if synthesize else "Raw Output"))
    print("=" * 70)
    print("\nThe LLM routes your requests to actual LOGOS operations.")
    print("All responses are grounded in real LOGOS execution.")
    print("\nCommands: /exit, /scan, /raw (toggle raw mode)")
    print("=" * 70 + "\n")

    interface = LOGOSInterface()

    while True:
        try:
            request = input("\n💬 You: ").strip()

            if not request:
                continue

            if request.lower() in ["/exit", "/quit"]:
                print("\n👋 Goodbye!\n")
                break

            if request.lower() == "/scan":
                print("\n🔍 Scanning repository...")
                result = execute_operation("scan_repository", {})
                if result["success"]:
                    print(result["stdout"])
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                continue

            if request.lower() == "/raw":
                synthesize = not synthesize
                mode = "Natural Language" if synthesize else "Raw Output"
                print(f"\n🔧 Mode changed to: {mode}")
                continue

            # Process request
            print("\n" + "-" * 70)
            result = interface.process_request(request, synthesize=synthesize)
            print("-" * 70)

            # Display result
            if result.get("success"):
                # Show synthesized answer if available
                if "answer" in result:
                    print("\n💬 LOGOS:\n")
                    print(result["answer"])
                else:
                    print("\n✅ LOGOS Output:\n")

                # Format based on output type (for raw mode)
                if "answer" not in result and "content" in result:
                    # File content
                    lines = result["content"].split("\n")
                    if len(lines) > 50:
                        print("\n".join(lines[:50]))
                        print(f"\n... ({len(lines) - 50} more lines)")
                    else:
                        print(result["content"])

                elif "items" in result:
                    # Directory listing
                    for item in result["items"]:
                        icon = "📁" if item["type"] == "directory" else "📄"
                        size = f" ({item['size']} bytes)" if item.get("size") else ""
                        print(f"{icon} {item['name']}{size}")

                elif "stdout" in result:
                    # Command output
                    print(result["stdout"])
                    if result.get("stderr"):
                        print("\nStderr:", result["stderr"])

                elif "data" in result:
                    # JSON data
                    print(json.dumps(result["data"], indent=2))

                else:
                    # Generic
                    print(json.dumps(result, indent=2))

            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                if "raw_response" in result:
                    print(f"\nLLM Response:\n{result['raw_response']}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"\n❌ Error: {exc}")


def single_request(request: str, synthesize: bool = True):
    """Process a single request and exit.

    Args:
        request: Natural language question
        synthesize: If True, provide natural language answer (default)
    """
    interface = LOGOSInterface()

    print(f"\n💬 Request: {request}\n")
    print("-" * 70)

    result = interface.process_request(request, synthesize=synthesize)

    print("-" * 70)

    if result.get("success"):
        if "answer" in result:
            print("\n💬 LOGOS:\n")
            print(result["answer"])
        else:
            print("\n✅ LOGOS Output:\n")
            if "content" in result:
                print(result["content"][:2000])
            elif "stdout" in result:
                print(result["stdout"])
            elif "data" in result:
                print(json.dumps(result["data"], indent=2))
            else:
                print(json.dumps(result, indent=2))
    else:
        print(f"❌ Error: {result.get('error')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LOGOS Protocol Interface - LLM as Natural Language Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The LLM is a ROUTER, not a knowledge source.
All responses come from actual LOGOS protocol operations.

First, scan the repository:
  python3 scripts/scan_repo.py

Then interact:
  python3 scripts/logos_interface.py              # Interactive
  python3 scripts/logos_interface.py "question"   # Single request
""",
    )
    parser.add_argument("request", nargs="*", help="Request to process")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="Show raw output (no synthesis)",
    )

    args = parser.parse_args()

    should_synthesize = not args.raw

    if args.interactive or not args.request:
        interactive_mode(synthesize=should_synthesize)
    else:
        request_text = " ".join(args.request)
        single_request(request_text, synthesize=should_synthesize)
