# LOGOS-GPT Web (Phase 18)

Minimal web surface for the advisory-only LOGOS-GPT loop. The LLM remains non-authoritative; tools run only through UIP approval and `dispatch_tool()` with attestation.

## Run the backend

```
uvicorn scripts.logos_gpt_server:app --host 0.0.0.0 --port 8000
```

Environment variables:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (optional; stub if unset)
- `LOGOS_DEV_BYPASS_OK=1` to allow attestation bypass in dev
- `LOGOS_LLM_PROVIDER` / `LOGOS_LLM_MODEL` to set defaults (otherwise stub/gpt-4.1-mini)

Endpoints:
- `GET /health` — attestation status
- `POST /api/chat` — non-streaming chat fallback
- `WS /ws/chat` — streaming assistant text; final message includes proposals and approvals
- `POST /api/approve` — submit UIP decision for pending tool proposals
- `GET /api/session/{session_id}` — memory summary + last ledger path

Per-session continuity:
- Each session binds to an in-memory `LogosAgiNexus` (bounded: 30m idle TTL, max 100 sessions, evict oldest).
- Nexus state refreshes plan history in-process after persistence; working-memory objective tags carry a session marker (`SESSION:<prefix>`) to avoid cross-session leakage while sharing the bounded `state/scp_state.json` store.

## Run the UI (static)

The UI lives in `web/logos-gpt-ui/index.html`. Any static server works:

```
python3 -m http.server 8080 --directory web/logos-gpt-ui
```

Then open `http://localhost:8080` and point it at the backend (default `http://localhost:8000`).

## How approvals work

- All non-low-impact tools (and every high-impact tool) require UIP approval.
- Low-impact read-only tools (mission.status, probe.last) may run without approval when not in read-only mode.
- `/ws/chat` streams reply text only. After the stream ends, a final JSON message lists proposed tools and any pending approvals. The browser can call `/api/approve` to approve or reject; each decision is logged to `audit/run_ledgers/`.

## Safety notes

- Keys stay server-side; the browser never sees provider secrets.
- Attestation is enforced unless explicitly bypassed for development with `LOGOS_DEV_BYPASS_OK=1`.
- No new persistence stores: state lives under `state/`, and ledgers are appended under `audit/run_ledgers/`.
