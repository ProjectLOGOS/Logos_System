# Protocol-Grounded LLM Interface

## Design Philosophy

**CRITICAL CONSTRAINT**: The LLM is a **ROUTER**, not a knowledge source.

### What This Means

The LLM wrapper serves ONLY as a natural language interface layer that:
1. ✅ Interprets user intent in natural language
2. ✅ Maps intent to actual LOGOS protocol operations
3. ✅ Executes the real LOGOS operation
4. ✅ Returns ONLY the raw output from LOGOS

The LLM **NEVER**:
- ❌ Generates content about LOGOS from its training data
- ❌ Hallucinates information about axioms, proofs, or code
- ❌ Synthesizes answers without grounding in actual execution
- ❌ Makes claims it cannot verify through LOGOS operations

### Architecture: Agent Over Protocol

```
┌─────────────────────────────────────────┐
│  Natural Language Input (User)          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  LLM Router (Intent → Operation)        │  ← Only maps intent
│  - No knowledge generation               │
│  - No content synthesis                  │
│  - Just routing decisions                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  LOGOS Protocol Operations               │  ← Real execution
│  - verify_proofs                         │
│  - check_axioms                          │
│  - boot_agent                            │
│  - read_file                             │
│  - scan_repository                       │
│  - get_mission_status                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Raw LOGOS Output (Unmodified)          │  ← Direct passthrough
└─────────────────────────────────────────┘
```

## Components

### 1. Repository Scanner (`scripts/scan_repo.py`)

**Purpose**: Build a comprehensive knowledge base from actual repository state.

**What it scans**:
- Coq proof files (axioms, definitions, lemmas)
- Python scripts and their purposes
- Documentation files
- Repository structure
- Current verification state

**Output**: `state/repository_knowledge_base.json`

**Usage**:
```bash
python3 scripts/scan_repo.py
```

**When to run**:
- After making changes to Coq proofs
- After adding new documentation
- Before starting an interactive session
- As part of CI/CD pipeline

### 2. Protocol Interface (`scripts/logos_interface.py`)

**Purpose**: Natural language router to LOGOS protocol operations.

**Design**: Two-step process:
1. **Routing**: LLM decides which operation to execute
2. **Execution**: System runs the actual operation

**Available Operations**:

| Operation | Command | Output |
|-----------|---------|--------|
| `verify_proofs` | `python3 test_lem_discharge.py` | Full verification report |
| `check_axioms` | `python3 tools/axiom_gate.py` | Axiom gate status |
| `boot_agent` | `python3 scripts/boot_aligned_agent.py` | Alignment status |
| `scan_repository` | `python3 scripts/scan_repo.py` | Scan summary |
| `read_file` | Read file contents | File content |
| `list_directory` | List directory | Directory listing |
| `get_mission_status` | Read mission profile | JSON data |
| `get_knowledge_base` | Load knowledge base | JSON data |

**Usage**:
```bash
# Interactive mode (recommended)
python3 scripts/logos_interface.py

# Single request
python3 scripts/logos_interface.py "Are the proofs verified?"
```

## Examples

### Example 1: Verification Status

**User**: "Are the proofs verified?"

**LLM Routes To**:
```json
{
  "operation": "verify_proofs",
  "params": {},
  "reasoning": "test_lem_discharge.py runs full proof verification"
}
```

**System Executes**: `python3 test_lem_discharge.py`

**User Sees**: Raw verification output (PASS/FAIL, Coq compilation log, assumption checks)

### Example 2: Reading Axioms

**User**: "What are the 8 axioms?"

**LLM Routes To**:
```json
{
  "operation": "read_file",
  "params": {"path": "Protopraxis/formal_verification/coq/baseline/PXLv3_SemanticModal.v"},
  "reasoning": "The axioms are declared in the PXL kernel file"
}
```

**System Executes**: Reads the actual Coq file

**User Sees**: Raw file content showing the 8 Axiom declarations

### Example 3: Repository Structure

**User**: "What scripts are available?"

**LLM Routes To**:
```json
{
  "operation": "list_directory",
  "params": {"path": "scripts"},
  "reasoning": "List directory to show available scripts"
}
```

**System Executes**: Lists the scripts/ directory

**User Sees**: Actual directory listing with file names and sizes

## Workflow

### Initial Setup

```bash
# 1. Scan repository to build knowledge base
python3 scripts/scan_repo.py

# 2. Verify .env has LLM configuration
cat .env
# Should show:
# LOGOS_LLM_PROVIDER=groq
# GROQ_API_KEY=gsk_...

# 3. Start interactive session
python3 scripts/logos_interface.py
```

### Typical Session

```
💬 You: What is the current verification state?
🔄 Routing: Checking axiom gates and verification status
📋 Operation: check_axioms
✅ LOGOS Output:
PASS: axiom budget ok: axiom_count=8 <= 8
...

💬 You: Show me the PXL kernel file
🔄 Routing: Reading the main kernel file
📋 Operation: read_file
✅ LOGOS Output:
(* PXLv3_SemanticModal.v ... *)
...

💬 You: /exit
👋 Goodbye!
```

## Key Guarantees

### 1. **No Hallucination**
Every response is grounded in actual LOGOS execution. The LLM cannot invent axioms, fake verification results, or make up repository structure.

### 2. **Verifiable Traceability**
Each response shows:
- What operation was executed
- What parameters were used
- The raw output from LOGOS

### 3. **Audit Trail**
The knowledge base (`repository_knowledge_base.json`) is a point-in-time snapshot that can be version-controlled and audited.

### 4. **Fail-Safe**
If the LLM cannot route a request, it returns an error rather than guessing or making up an answer.

## Advanced Usage

### Custom Operations

To add new operations, edit `LOGOS_OPERATIONS` in `scripts/logos_interface.py`:

```python
LOGOS_OPERATIONS = {
    "my_custom_op": {
        "command": ["python3", "path/to/script.py"],
        "description": "What this operation does",
        "output_type": "script_output"
    }
}
```

### Integration with Agent Stack

The protocol interface is designed to integrate with the existing LOGOS agent:

```python
from scripts.logos_interface import LOGOSInterface

interface = LOGOSInterface()
result = interface.process_request("What are the 8 axioms?")

# result["routing"] - Shows how LLM routed the request
# result["content"] or result["stdout"] - Raw LOGOS output
```

### Knowledge Base Schema

The `repository_knowledge_base.json` contains:

```json
{
  "scan_timestamp": "2025-12-21T...",
  "structure": {
    "key_directories": [...],
    "key_files": [...]
  },
  "coq_proofs": {
    "PXLv3_SemanticModal.v": {
      "axiom_count": 8,
      "axioms": ["Axiom A2_noncontradiction...", ...],
      "definition_count": 5,
      "lemma_count": 12
    }
  },
  "python_scripts": {
    "test_lem_discharge.py": {
      "purpose": "Run full Coq proof verification...",
      "line_count": 245
    }
  },
  "documentation": {...},
  "verification_state": {
    "axiom_gate": {
      "stdout": "PASS: axiom budget ok...",
      "exit_code": 0
    }
  }
}
```

## Comparison: Chat vs Interface

### `scripts/logos_chat.py` (Conversational)

- LLM generates natural language responses
- Uses tools to gather information
- Good for: Exploratory questions, explanations
- Risk: May hallucinate or interpret

### `scripts/logos_interface.py` (Protocol-Grounded)

- LLM only routes to operations
- Returns raw LOGOS output
- Good for: Verification, auditing, automation
- Guarantee: No hallucination possible

**Recommendation**: Use `logos_interface.py` for production/safety-critical work.

## Future Integration: Full LOGOS Agent

The goal is to wire this interface into the full LOGOS agent stack:

```python
# Future: logos_agent.py
from scripts.logos_interface import LOGOSInterface
from boot_aligned_agent import SandboxedLogosAgent

class FullLOGOSAgent:
    def __init__(self):
        self.interface = LOGOSInterface()  # NL routing
        self.agent = SandboxedLogosAgent()  # Guardrails
    
    def process(self, user_request):
        # LLM routes request
        routing = self.interface.route_request(user_request)
        
        # LOGOS agent executes with guardrails
        result = self.agent.execute_operation(
            routing["operation"],
            routing["params"]
        )
        
        return result  # Pure LOGOS output
```

This maintains the "agent over protocol" architecture where:
- LLM = Interface layer (understanding natural language)
- LOGOS = Execution layer (running verified operations)
- Guardrails = Safety layer (proof-gated constraints)

## Summary

The protocol-grounded interface ensures:

✅ **All responses from LOGOS stack** (no hallucination)  
✅ **Verifiable execution** (audit trail)  
✅ **Natural language access** (user-friendly)  
✅ **Proof-gated safety** (maintains alignment)  

The LLM is **just a router** — LOGOS provides the answers.
