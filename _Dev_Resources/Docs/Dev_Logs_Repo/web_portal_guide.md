# LOGOS Web Portal - Quick Start

## ✨ What Changed

The system now provides **natural language answers grounded in real LOGOS data**:

1. **LLM routes** your question → determines what LOGOS operation to run
2. **LOGOS executes** → returns real data (file contents, verification output, etc.)
3. **LLM synthesizes** → explains that real data in natural language
4. **You receive** → natural language answer WITH ZERO HALLUCINATION

## 🚀 Using the Web Portal

### Start the Server

Legacy `scripts/logos_web_server.py` is quarantined; use the canonical GPT server instead (see _reports/logos_agi_audit/60_PHASE_B_OPERATOR_GUIDE.md).

```bash
cd /workspaces/pxl_demo_wcoq_proofs
python3 scripts/llm_interface_suite/logos_gpt_server.py
```

### Open in Browser

Visit: **http://localhost:8080**

### Share the Portal

The HTML file is self-contained. To share:

1. Copy `logos_portal.html` to others
2. They need to run the server with their own API key
3. Or you can host the server and share the URL

## 🎯 Example Interactions

### Before (Raw Mode)
```
User: "What formal system does LOGOS run on?"
LOGOS: [dumps entire PXLv3_SemanticModal.v file - 200+ lines]
```

### After (Synthesis Mode)
```
User: "What formal system does LOGOS run on?"

🤖 LOGOS:
LOGOS runs on PXL (Protopraxic Logic), a formal system implemented 
in Coq. The system has 8 irreducible metaphysical axioms including 
A2_noncontradiction and A7_triune_necessity. The kernel also includes 
semantic modal operators proven as theorems rather than axioms, 
showing the system's constructive foundation.
```

## 🔧 Command Line Usage

### Natural Language Mode (Default)

```bash
python3 scripts/logos_interface.py "How many axioms does the system have?"
```

Output:
```
💬 LOGOS:
The system has 8 axioms. This is indicated by the axiom budget 
check showing axiom_count=8 for both total and semantic axioms.
```

### Raw Data Mode

```bash
python3 scripts/logos_interface.py --raw "How many axioms?"
```

Output:
```
✅ LOGOS Output:
PASS: axiom budget ok: axiom_count=8 <= 8
PASS: semantic axiom budget ok: axiom_count_semantic=8 <= 8
...
```

### Interactive Mode

```bash
python3 scripts/logos_interface.py

💬 You: What are the 8 axioms?
🤖 LOGOS: [natural language explanation]

💬 You: Are the proofs verified?
🤖 LOGOS: [verification status in plain language]

💬 You: /raw
🔧 Mode changed to: Raw Output

💬 You: /exit
```

## 🛡️ Safety Guarantees

### The Hybrid Model

```
┌─────────────────────────┐
│ User Question           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ LLM Routes              │  ← No knowledge generation
│ (Intent → Operation)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ LOGOS Executes          │  ← Real data only
│ (File read, verify, etc)│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ LLM Synthesizes         │  ← Explains real data
│ (Data → Natural Lang)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Natural Language Answer │  ← Grounded, not hallucinated
└─────────────────────────┘
```

### What the LLM CAN Do

✅ Route questions to LOGOS operations  
✅ Explain real data in natural language  
✅ Summarize file contents  
✅ Format verification results  
✅ Answer questions FROM the data  

### What the LLM CANNOT Do

❌ Generate axiom lists from memory  
❌ Make up verification results  
❌ Hallucinate file contents  
❌ Claim things not in the real data  
❌ Invent repository structure  

### Verification

Every answer includes (in verbose mode):
- 🔄 Routing decision (what operation was chosen)
- 📋 Operation name (what LOGOS did)
- ⚙️ Parameters (what was accessed)
- 🧠 Synthesis note (LLM is explaining real data)

## 📡 API Reference

### GET /api/status

Check server status and configuration.

Response:
```json
{
  "status": "online",
  "provider": "groq",
  "mode": "synthesis (natural language from real data)"
}
```

### POST /api/ask

Ask LOGOS a question.

Request:
```json
{
  "question": "What formal system does LOGOS run on?"
}
```

Response:
```json
{
  "success": true,
  "answer": "LOGOS runs on PXL (Protopraxic Logic)...",
  "operation": "read_file",
  "routing": {
    "operation": "read_file",
    "params": {"path": "..."},
    "reasoning": "The formal system is declared in the kernel"
  }
}
```

## 🎨 Portal Features

- **Dark theme** optimized for extended use
- **Real-time status** indicator
- **Example questions** as clickable chips
- **Typing indicator** during processing
- **Smooth animations** for better UX
- **Mobile responsive** design
- **Conversation history** maintained in session

## 🔑 Environment Setup

The server reads from `.env`:

```bash
LOGOS_LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
DEBUG=false
```

## 📊 Comparison Matrix

| Feature | Chat Mode | Raw Mode | Synthesis Mode ✨ |
|---------|-----------|----------|-------------------|
| Natural language | ✅ | ❌ | ✅ |
| Grounded in real data | ⚠️ Sometimes | ✅ Always | ✅ Always |
| Easy to understand | ✅ | ❌ | ✅ |
| Verifiable | ❌ | ✅ | ✅ |
| Shows operation | ❌ | ✅ | ✅ |
| Risk of hallucination | ⚠️ Medium | ❌ None | ❌ None |

**Synthesis Mode** = Best of both worlds!

## 🚢 Deployment

### Local Development

```bash
python3 scripts/llm_interface_suite/logos_gpt_server.py
# Open http://localhost:8080
```

### Production (with ngrok)

```bash
# Install ngrok
# Start server
python3 scripts/llm_interface_suite/logos_gpt_server.py -p 8080

# In another terminal
ngrok http 8080
# Share the ngrok URL
```

### Docker (Future)

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install groq
CMD ["python3", "scripts/llm_interface_suite/logos_gpt_server.py"]
```

## 📖 Next Steps

1. **Test the portal**: Open http://localhost:8080
2. **Try example questions**: Click the chips
3. **Ask your own**: Type freely
4. **Share**: Send `logos_portal.html` + server instructions

The system is ready for natural language interaction while maintaining proof-gated safety! 🎉
