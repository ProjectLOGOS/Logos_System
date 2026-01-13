# LOGOS Demo Guide

> **Quick Start:** Run `./scripts/demo.sh` and choose your demo mode.

## 🎯 What This Demo Shows

The LOGOS system demonstrates **proof-before-autonomy**: an AI agent that only unlocks after verifying constructive mathematical proofs in Coq. This isn't simulated—every demo run rebuilds the formal kernel and checks real proof artifacts.

### Key Features

1. **Constructive LEM Discharge** - Proves the Law of Excluded Middle without axioms
2. **8 Irreducible Axioms** - Minimal metaphysical foundation (reduced from 20+)
3. **Proof-Gated Agent** - Agent boots only after verification passes
4. **Tamper-Evident Audit** - SHA-256 identity tracking with timestamped logs
5. **Natural Language Interface** - LLM-powered Q&A grounded in real proofs

## 🚀 Demo Modes

### Mode 1: Quick Demo (30 seconds)

**Perfect for:** First-time viewers, quick verification check

```bash
./scripts/demo.sh quick
```

**What it does:**
- Rebuilds Coq proofs from source
- Verifies constructive LEM proof
- Shows assumption footprint (should be empty)
- Checks for `Admitted.` stubs (should be none)

**Success looks like:**
```
Overall status: PASS
Trinitarian Optimization – extra assumptions: <none>
Internal LEM – extra assumptions: <none>
Residual `Admitted.` stubs in baseline: <none>
```

---

### Mode 2: Full Demo (2 minutes)

**Perfect for:** Investors, technical evaluations, complete walkthrough

```bash
./scripts/demo.sh full
```

**What it does:**
1. **Proof Verification** - Rebuilds kernel, checks assumptions
2. **Agent Boot** - Unlocks agent only after proof gate passes
3. **Investor Dashboard** - Shows executive-friendly summary

**Success looks like:**
```
✓ Proof verification passed
✓ Agent identity: LOGOS-AGENT-OMEGA (SHA-256: abc123...)
✓ Alignment status: ALIGNED
✓ Audit entry written to state/alignment_LOGOS-AGENT-OMEGA.json
```

---

### Mode 3: Web Portal

**Perfect for:** Interactive exploration, live demos, presentations

```bash
./scripts/demo.sh web
```

**What it does:**
- Starts web server on http://localhost:8080
- Opens interactive chat interface
- LLM answers questions using real LOGOS data
- Zero hallucination (grounded in actual files/proofs)

**Try asking:**
- "What are the 8 irreducible axioms?"
- "Explain the privative_collapse axiom"
- "How does the agent boot sequence work?"
- "Show me the proof verification code"

**Requirements:**
- API key for Groq (free), OpenAI, or Anthropic
- Or use mock mode (limited but functional)

---

### Mode 4: Chat Interface

**Perfect for:** Q&A sessions, documentation queries, code exploration

```bash
./scripts/demo.sh chat "What are the 8 axioms?"
```

**Examples:**
```bash
# Axiom questions
./scripts/demo.sh chat "Explain modus_groundens"

# Repository navigation
./scripts/demo.sh chat "Where are the Coq proofs located?"

# Technical details
./scripts/demo.sh chat "How does scripts/boot_aligned_agent.py verify proofs?"
```

---

## 🎬 Recording a Demo

### Screen Recording (Recommended)

**For Mac:**
```bash
# Record terminal
./scripts/demo.sh full
# Use QuickTime Player (Cmd+Shift+5) to record screen
```

**For Linux:**
```bash
# Install asciinema for terminal recording
sudo apt-get install asciinema

# Record session
asciinema rec logos_demo.cast
./scripts/demo.sh full
exit

# Convert to GIF
sudo apt-get install agg
agg logos_demo.cast logos_demo.gif
```

**For Windows:**
```bash
# Use Windows Terminal + built-in recorder (Win+G)
./scripts/demo.sh full
```

### Best Practices

1. **Clear the terminal** before starting
2. **Set larger font** (easier to read in recording)
3. **Slow down** - pause after key outputs
4. **Add commentary** - explain what's happening
5. **Show success states** - highlight green checkmarks

### Recommended Recording Flow

```bash
# 1. Start clean
clear
./scripts/demo.sh

# 2. Choose "Full Demo" (option 2)
2

# 3. Wait for all three stages:
#    - Proof verification
#    - Agent boot
#    - Dashboard display

# 4. For web demo, split screen:
#    - Terminal on left
#    - Browser on right
./scripts/demo.sh web
```

---

## 🎯 Demo Talking Points

### For Investors (Executive Focus)

**The Hook:**
"Most AI systems run first, audit later. LOGOS does the opposite—mathematical proof gates the autonomy."

**Key Points:**
1. **No trust required** - Coq proofs are machine-verified
2. **Minimal assumptions** - 8 axioms (down from 20+)
3. **Tamper-evident** - SHA-256 tracking prevents identity drift
4. **Scalable safety** - Proof gates can protect any capability

**The Close:**
"Every demo run rebuilds from source—this isn't a recording. The proofs are real, the verification is real, and the agent unlock is conditional on that verification passing."

### For Technical Evaluators (Deep Dive)

**The Hook:**
"We discharge the Law of Excluded Middle constructively inside Coq—no axioms, no classical logic assumptions."

**Key Points:**
1. **PXL Kernel** - 8 irreducible metaphysical axioms
2. **Phase 3 Complete** - Modal operators proven as theorems
3. **Semantic Kripke** - Bridge axioms eliminated via definitions
4. **LEM Discharge** - Constructive proof using trinitarian structure

**The Close:**
"Check the assumption footprint yourself: `coqtop -load-vernac-source baseline/PXL_Internal_LEM_SemanticPort.v`. It's `<none>` because the proof is constructive."

---

## 🔧 Troubleshooting

### Issue: Coq not found

```bash
sudo apt-get update
sudo apt-get install coq
```

### Issue: Python version too old

```bash
# Need Python 3.11+
python3 --version

# On Ubuntu 22.04+
sudo apt-get install python3.11
```

### Issue: LLM rate limit reached

```bash
# Switch to mock provider
./scripts/demo.sh chat --provider mock "What are the axioms?"

# Or wait for rate limit reset (shown in error message)
```

### Issue: Port 8080 already in use

```bash
# Find and kill the process
lsof -ti:8080 | xargs kill -9

# Or use a different port (legacy portal server is quarantined; use the canonical GPT server)
python3 scripts/llm_interface_suite/logos_gpt_server.py --port 8081
```

### Issue: Proofs fail to build

```bash
# Clean build artifacts
make -f CoqMakefile clean

# Rebuild from scratch
python3 test_lem_discharge.py
```

---

## 📊 Understanding the Output

### Proof Verification Output

```
=== PXL Kernel Rebuild ===
Overall status: PASS                    ← All proofs compiled successfully

=== Trinitarian Optimization – extra assumptions ===
  <none>                                ← No axioms used beyond the 8 core ones

=== Internal LEM – extra assumptions ===
  <none>                                ← LEM proved constructively (key result!)

=== Residual `Admitted.` stubs in baseline ===
  <none>                                ← No unfinished proofs
```

### Agent Boot Output

```
Agent Identity: LOGOS-AGENT-OMEGA       ← Fixed identity (SHA-256 hash)
Verification: PASSED                    ← Proofs checked out
Alignment Status: ALIGNED               ← Agent unlocked
Audit Entry: state/alignment_*.json    ← Tamper-evident log
```

### Dashboard Output

```
🎯 System Status: OPERATIONAL
📊 Axiom Count: 8 (irreducible)
✓ LEM Discharge: CONSTRUCTIVE
✓ Proof Gates: ACTIVE
✓ Agent Alignment: VERIFIED
```

---

## 🎓 Learning Path

### Beginner (No Coq experience)

1. Run `./scripts/demo.sh quick` - see proofs verify
2. Read `LOGOS_Axiom_And_Theorem_Summary.md` - understand the 8 axioms
3. Try web portal - ask questions in natural language
4. Read `README.md` - get full context

### Intermediate (Some formal methods)

1. Run `./scripts/demo.sh full` - complete verification
2. Read `COQ_PROOF_AUDIT_COMPREHENSIVE.md` - proof strategy
3. Examine `Protopraxis/formal_verification/coq/baseline/PXLv3_SemanticModal.v`
4. Study `scripts/boot_aligned_agent.py` - see how proof gates work

### Advanced (Coq developers)

1. Review `_CoqProject` - build configuration
2. Inspect `PXL_Internal_LEM_SemanticPort.v` - constructive LEM proof
3. Check `PXL_Derivations_Phase2.v` - elimination of 8 axioms
4. Audit `PXL_Modal_Semantic_Kripke.v` - modal semantics

---

## 📚 Additional Resources

- **Architecture:** `docs/EXECUTIVE_SUMMARY.md` (coming soon)
- **Setup Details:** `README.md`
- **Security:** `SECURITY.md`
- **Contributing:** `CONTRIBUTING.md`
- **LLM Integration:** `docs/LLM_INTEGRATION_SUMMARY.md`
- **Web Portal:** `WEB_PORTAL_GUIDE.md`

---

## ✅ Demo Checklist

Before presenting to investors or technical reviewers:

- [ ] Run `./scripts/demo.sh quick` - verify proofs pass
- [ ] Check `state/alignment_*.json` exists and has recent timestamp
- [ ] Test web portal works on target presentation machine
- [ ] Verify LLM integration (or plan to use mock mode)
- [ ] Review talking points above
- [ ] Prepare for Q&A on proof strategy
- [ ] Have backup: pre-record demo in case of network issues

---

## 🚨 Important Notes

### What This Demo Is

✅ Real Coq proofs verified on every run
✅ Actual constructive LEM discharge
✅ Functional proof-gated agent boot
✅ Production-ready verification harness

### What This Demo Is Not

❌ A fully autonomous agent (it's sandboxed)
❌ A GUI framework (CLI-first, web is optional)
❌ A complete AGI system (research prototype)
❌ Dependent on external services (runs offline except LLM)

---

## 🤝 Getting Help

- **Issues:** https://github.com/ProjectLOGOS/pxl_demo_wcoq_proofs/issues
- **Documentation:** See `docs/` directory
- **Chat:** Use `./scripts/demo.sh chat` to ask the system itself
