# LOGOS-GPT Chat (Phase 16)

A minimal UIP chat loop that keeps LOGOS as the sole authority. The LLM is advisory-only; every execution remains gated by attestation, UIP approval, and dispatch_tool().

## Running in stub mode

```
LOGOS_DEV_BYPASS_OK=1 \
LLM_ADVISOR_STUB_PAYLOAD='{"reply":"Hi","proposals":[{"tool":"mission.status","args":""}]}' \
python3 scripts/llm_interface_suite/logos_gpt_chat.py \
  --enable-llm-advisor \
  --llm-provider stub \
  --llm-model stub \
  --assume-yes \
  --read-only \
  --max-turns 2 \
  --objective-class CHAT \
  --no-require-attestation
```

## Running with OpenAI (advisor only)

```
OPENAI_API_KEY=... python3 scripts/llm_interface_suite/logos_gpt_chat.py \
  --enable-llm-advisor \
  --llm-provider openai \
  --llm-model gpt-4.1-mini \
  --assume-yes \
  --max-turns 5 \
  --objective-class CHAT
```

- The advisor produces a reply draft and optional tool proposals (max 3). Tools execute only via dispatch_tool() after UIP approval.
- High-impact tools (tool_proposal_pipeline, start_agent) always require UIP approval.
- Attestation is required unless explicitly bypassed with `--no-require-attestation` and `LOGOS_DEV_BYPASS_OK=1`.
- Conversation items and tool results are stored in state/scp_state.json UWM; truth annotations default to HEURISTIC unless validated.
- Each turn emits a run ledger under audit/run_ledgers/ for epistemic accounting.

## Running with Anthropic (advisor only)

```
ANTHROPIC_API_KEY=... python3 scripts/llm_interface_suite/logos_gpt_chat.py \
  --enable-llm-advisor \
  --llm-provider anthropic \
  --llm-model claude-3-5-sonnet-20240620 \
  --assume-yes \
  --max-turns 5 \
  --objective-class CHAT
```

## Streaming replies

- Add `--stream` to stream advisor reply text to stdout while proposals are collected at the end of the turn.
- Streaming respects the same gates: the advisor remains non-authoritative, and any tool proposals still flow through UIP approval and dispatch_tool().
- If a streaming SDK is unavailable, the chat falls back to a single-chunk reply; smoke tests SKIP when no provider keys are set.
