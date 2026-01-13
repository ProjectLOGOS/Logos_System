# LLM Backend Integration for LOGOS Agent

## Overview

The LOGOS agent now supports **actual reasoning** via LLM backends (OpenAI/Anthropic) while maintaining all proof-gated safety guardrails.

## Quick Start

### Option 1: Mock Backend (No API Key Required)

```bash
# Test with mock responses
python3 scripts/llm_interface_suite/reasoning_agent.py "What are the 8 axioms in PXL?"
```

### Option 2: OpenAI Backend

```bash
# Install dependencies
pip install openai

# Set API key
export OPENAI_API_KEY="sk-..."
export LOGOS_LLM_PROVIDER="openai"
export LOGOS_LLM_MODEL="gpt-4-turbo-preview"  # optional, defaults to gpt-4

# Ask questions
python3 scripts/llm_interface_suite/reasoning_agent.py "Explain the privative_collapse axiom"
```

### Option 3: Anthropic Backend

```bash
# Install dependencies
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
export LOGOS_LLM_PROVIDER="anthropic"
export LOGOS_LLM_MODEL="claude-3-5-sonnet-20241022"  # optional

# Ask questions
python3 scripts/llm_interface_suite/reasoning_agent.py "How does modus_groundens relate to Leibniz's law?"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOGOS_LLM_PROVIDER` | Backend provider: `openai`, `anthropic`, or `mock` | `mock` |
| `LOGOS_LLM_MODEL` | Model name | `gpt-4` (OpenAI) / `claude-3-5-sonnet-20241022` (Anthropic) |
| `LOGOS_LLM_MAX_TOKENS` | Maximum response length | `2000` (OpenAI) / `4000` (Anthropic) |
| `LOGOS_LLM_TEMPERATURE` | Sampling temperature (0.0-1.0) | `0.7` |
| `OPENAI_API_KEY` | OpenAI API key | Required if provider=openai |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required if provider=anthropic |

### Persistent Configuration

Create `.env` file in repository root:

```bash
LOGOS_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LOGOS_LLM_MODEL=gpt-4-turbo-preview
LOGOS_LLM_TEMPERATURE=0.3  # Lower = more deterministic
```

Load with: `export $(cat .env | xargs)`

## Architecture

### Components

1. **`plugins/llm_backend.py`**
   - Unified LLM client supporting multiple providers
   - Handles API authentication and error handling
   - Converts between LOGOS tools and LLM function calling format

2. **`scripts/llm_interface_suite/reasoning_agent.py`**
   - Main reasoning loop with tool calling
   - Integrates with existing `start_agent.py` tools
   - Maintains conversation history and safety constraints

3. **Integration with `start_agent.py`**
   - Uses same tool definitions (`mission.status`, `fs.read`, etc.)
   - Respects mission profiles and sandbox constraints
   - Proof-gated: cannot modify Coq proofs or bypass alignment checks

### Reasoning Flow

```
User Question
    ↓
[System Prompt] ← LOGOS context (8 axioms, mission profile, etc.)
    ↓
[LLM Reasoning] → Generate tool calls (e.g., read PXLv3_SemanticModal.v)
    ↓
[Tool Execution] → Execute fs.read, mission.status, etc.
    ↓
[LLM Synthesis] → Formulate answer from tool results
    ↓
Final Answer
```

## Available Tools

The agent can call these LOGOS tools:

- **`mission_status`** - Get current mission profile and safety config
- **`probe_last`** - Get last protocol probe snapshot
- **`fs_read`** - Read repository files (Coq proofs, docs, etc.)
- **`agent_memory`** - Access persistent memory and reflections
- **`sandbox_read`** - Read sandbox artifacts

Tools are **read-only** by default. Write operations require explicit mission profile configuration.

## Example Queries

```bash
# Proof system questions
python3 scripts/llm_interface_suite/reasoning_agent.py "What is the difference between Ident and NonEquiv?"

# Repository structure
python3 scripts/llm_interface_suite/reasoning_agent.py "Where are the Coq proofs located?"

# Axiom explanations
python3 scripts/llm_interface_suite/reasoning_agent.py "Explain triune_dependency_substitution in simple terms"

# Meta-questions
python3 scripts/llm_interface_suite/reasoning_agent.py "How does the agent alignment check work?"

# Development questions
python3 scripts/llm_interface_suite/reasoning_agent.py "What files would I need to modify to add a new axiom?"
```

## Safety & Constraints

### Guardrails Preserved

✅ **Proof gates** - Agent cannot modify Coq proofs or axiom budgets
✅ **Sandbox isolation** - Writes restricted to configured sandbox directory
✅ **Mission profiles** - Respects `DEMO_STABLE`, `AGENTIC_EXPERIMENT`, etc.
✅ **Read-only tools** - Default tools are non-mutating
✅ **Audit logging** - All operations logged to `state/agent_state.json`

### New Capabilities

✅ **Natural language understanding** - Interprets questions in context
✅ **Multi-step reasoning** - Chains tool calls to gather information
✅ **Code comprehension** - Reads and explains Coq proofs
✅ **Uncertainty acknowledgment** - Admits when it doesn't know

## Cost Considerations

### OpenAI Pricing (approx)

- GPT-4 Turbo: ~$0.01/1K tokens input, ~$0.03/1K tokens output
- Typical question: 500-2000 tokens = **$0.01-0.10 per query**

### Anthropic Pricing (approx)

- Claude 3.5 Sonnet: ~$0.003/1K tokens input, ~$0.015/1K tokens output  
- Typical question: 500-2000 tokens = **$0.003-0.05 per query**

### Cost Control

Set `LOGOS_LLM_MAX_TOKENS` lower to reduce costs:
```bash
export LOGOS_LLM_MAX_TOKENS=1000  # Shorter answers
```

Use mock backend for development:
```bash
export LOGOS_LLM_PROVIDER=mock  # Free, pattern-matched responses
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'openai'"

```bash
pip install openai
```

### "OpenAI provider requires OPENAI_API_KEY"

```bash
export OPENAI_API_KEY="your-key-here"
```

### "Rate limit exceeded"

OpenAI/Anthropic have rate limits. Wait and retry, or:
- Use a higher tier API plan
- Add retry logic with exponential backoff
- Switch to mock provider temporarily

### "Tool execution failed"

Check that files exist and mission profile allows access:
```bash
python3 -c "from scripts.start_agent import TOOLS; print(TOOLS['fs.read']('README.md')[:200])"
```

## Development

### Adding Custom Tools

Edit `plugins/llm_backend.py` → `create_tool_definitions()`:

```python
tool_schemas["my_tool"] = {
    "name": "my_tool",
    "description": "My custom tool",
    "parameters": {
        "type": "object",
        "properties": {
            "arg": {"type": "string", "description": "Tool argument"}
        },
        "required": ["arg"]
    }
}
```

Then add to `scripts/start_agent.py` → `TOOLS` dict.

### Customizing System Prompt

Edit `scripts/llm_interface_suite/reasoning_agent.py` → `SYSTEM_PROMPT` to change agent behavior.

### Testing New Providers

Implement `_complete_<provider>()` method in `LLMBackend` class.

## See Also

- [start_agent.py](../scripts/start_agent.py) - Base agent implementation
- [scripts/boot_aligned_agent.py](../scripts/boot_aligned_agent.py) - Proof alignment verification
- [README.md](../README.md#branch-strategy--development-workflow) - Development workflow
