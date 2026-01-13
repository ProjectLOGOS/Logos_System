# LOGOS Interactive Chat - Quick Start

Talk naturally with your proof-gated AI alignment agent!

## 🚀 Instant Setup (30 seconds)

### 1. Get a Free API Key

Visit [https://console.groq.com](https://console.groq.com) and sign up for free Groq access (no credit card needed).

### 2. Configure

The `.env` file is already set up with your API key:

```bash
# .env file (fill in your key)
LOGOS_LLM_PROVIDER=groq
GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE
```

### 3. Start Chatting!

```bash
# Interactive mode (recommended)
python3 scripts/logos_chat.py

# Or ask a single question
python3 scripts/logos_chat.py "What are the 8 axioms?"
```

## 💬 Interactive Commands

Once in interactive mode:

- Just type your questions naturally
- `/exit` or `/quit` - End the session
- `/clear` - Clear conversation history
- `/help` - Show help

## 📚 Example Questions

```bash
python3 scripts/logos_chat.py
```

Then try:

```
💬 You: What are the 8 axioms in PXL?
💬 You: Explain what modus_groundens means
💬 You: What files are in the Protopraxis/formal_verification folder?
💬 You: How does the alignment system work?
💬 You: What is the current mission status?
```

## 🎯 What Can LOGOS Do?

- **Explain the proof system**: Ask about axioms, theorems, formal verification
- **Read repository files**: Get code summaries, file contents, structure
- **Answer questions**: Natural language Q&A about the codebase
- **Maintain context**: Remembers conversation history for follow-ups
- **Stay grounded**: All answers verified against actual repository state

## 🔧 Advanced Usage

### Single Question Mode

```bash
python3 scripts/logos_chat.py "What are the 8 axioms?"
```

### Debug Mode

```bash
DEBUG=true python3 scripts/logos_chat.py
```

### Change Provider

```bash
# Edit .env file
LOGOS_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

## 🆓 Free Tier Limits (Groq)

- **14,400 requests per day**
- **~500 tokens/second** inference speed
- **No credit card required**
- Model: `llama-3.3-70b-versatile` (70B parameters)

That's plenty for daily development use!

## 🛡️ Safety Features

LOGOS operates with proof-gated constraints:
- ✅ Can read repository files
- ✅ Can explain proofs and code
- ✅ Can answer questions
- ❌ Cannot modify Coq proofs
- ❌ Cannot change axiom budgets
- ❌ Sandbox writes restricted by mission profile

All capabilities are grounded in verified formal foundations.

## 📖 Help

```bash
python3 scripts/logos_chat.py --help
```

---

**Ready?** Just run:

```bash
python3 scripts/logos_chat.py
```

🤖 **LOGOS awaits your questions!**
