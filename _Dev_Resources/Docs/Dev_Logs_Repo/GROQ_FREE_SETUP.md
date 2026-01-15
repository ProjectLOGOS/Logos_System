# 🚀 Free LLM Setup with Groq

## Why Groq?

✅ **100% FREE** tier with generous limits
✅ **Lightning fast** inference (~500 tokens/sec)
✅ **Good models**: Llama 3.3 70B, Mixtral 8x7B, Gemma 2 9B
✅ **OpenAI-compatible** API (easy to use)

## Free Tier Limits

- **14,400 requests/day** per model
- **No credit card required**
- Fast inference speeds (much faster than OpenAI/Anthropic)

## Setup (5 minutes)

### Step 1: Get a Free API Key

1. Go to: https://console.groq.com/
2. Sign up with Google/GitHub (free, no credit card)
3. Go to: https://console.groq.com/keys
4. Click "Create API Key"
5. Copy your key (starts with `gsk_...`)

### Step 2: Install Groq Client

```bash
cd /workspaces/pxl_demo_wcoq_proofs
pip install groq
```

### Step 3: Configure Environment

```bash
export GROQ_API_KEY="gsk_..."  # Paste your key
export LOGOS_LLM_PROVIDER="groq"
```

### Step 4: Test It!

```bash
python3 scripts/llm_interface_suite/reasoning_agent.py "What are the 8 axioms in PXL?"
```

## Available Models (All Free!)

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `llama-3.3-70b-versatile` | 70B | ⚡⚡⚡ | General reasoning (default) |
| `llama-3.1-70b-versatile` | 70B | ⚡⚡⚡ | Complex analysis |
| `mixtral-8x7b-32768` | 47B | ⚡⚡⚡⚡ | Long context (32K tokens) |
| `gemma2-9b-it` | 9B | ⚡⚡⚡⚡⚡ | Super fast, lightweight |

### Change Model

```bash
export LOGOS_LLM_MODEL="mixtral-8x7b-32768"  # For long documents
python3 scripts/llm_interface_suite/reasoning_agent.py "Summarize all 8 axioms"
```

## Example Queries

```bash
# Axiom explanations
python3 scripts/llm_interface_suite/reasoning_agent.py "Explain privative_collapse axiom in simple terms"

# Proof questions
python3 scripts/llm_interface_suite/reasoning_agent.py "What's the difference between Ident and NonEquiv?"

# Repository navigation
python3 scripts/llm_interface_suite/reasoning_agent.py "Where are the Coq proofs located?"

# Development guidance
python3 scripts/llm_interface_suite/reasoning_agent.py "How do I add a new axiom?"
```

## Persistent Configuration

Create `.env` file:

```bash
cat > .env << 'EOF'
GROQ_API_KEY=gsk_YOUR_KEY_HERE
LOGOS_LLM_PROVIDER=groq
LOGOS_LLM_MODEL=llama-3.3-70b-versatile
LOGOS_LLM_TEMPERATURE=0.3
EOF

# Load it
export $(cat .env | xargs)
```

## Cost Comparison

| Provider | Cost per Query | Free Tier |
|----------|----------------|-----------|
| **Groq** | **$0.00** | ✅ 14,400 requests/day |
| OpenAI (GPT-4) | $0.01-0.10 | ❌ None |
| Anthropic (Claude) | $0.003-0.05 | ❌ None |

## Performance

Groq is **10-50x faster** than OpenAI/Anthropic:

- OpenAI GPT-4: ~20 tokens/sec
- Anthropic Claude: ~40 tokens/sec
- **Groq Llama 3.3: ~500 tokens/sec** ⚡

## Troubleshooting

### "ModuleNotFoundError: No module named 'groq'"

```bash
pip install groq
```

### "Groq provider requires GROQ_API_KEY"

```bash
# Get key from: https://console.groq.com/keys
export GROQ_API_KEY="gsk_..."
```

### "Rate limit exceeded"

Free tier: 14,400 requests/day per model. If you hit the limit:
- Switch to a different model
- Wait until next day (resets at midnight UTC)
- Or upgrade to paid tier (still very cheap)

## Full Example

```bash
# One-time setup
pip install groq
export GROQ_API_KEY="gsk_YOUR_KEY_HERE"
export LOGOS_LLM_PROVIDER="groq"

# Ask questions
python3 scripts/llm_interface_suite/reasoning_agent.py "What is modus_groundens axiom?"

# With specific model
export LOGOS_LLM_MODEL="mixtral-8x7b-32768"
python3 scripts/llm_interface_suite/reasoning_agent.py "Analyze all 8 axioms for interdependencies"
```

## Next Steps

1. **Get your free API key**: https://console.groq.com/keys
2. **Install**: `pip install groq`
3. **Configure**: `export GROQ_API_KEY="gsk_..."`
4. **Test**: `python3 scripts/llm_interface_suite/reasoning_agent.py "What are the 8 axioms?"`
5. **Enjoy unlimited free reasoning!** 🎉

---

**Recommended**: Groq is the **best option for LOGOS agent** - free, fast, and powerful.
