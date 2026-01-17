# RUNTIME SPINE — LOCK-AND-KEY EXECUTION CONTRACT (NORMATIVE)

Status: **AUTHORITATIVE / NON-BYPASSABLE**

This document defines the mandatory execution order, identity issuance,
and access control semantics for the LOGOS Runtime Spine.

No system component may execute outside this ordering.

---

## Mandatory Execution Order (STRICT)
1. Dual-Site Lock-and-Key (LEM admitted)
2. Universal Session ID issuance
3. Logos constructive compile (LEM discharged)
4. Logos Agent ID issuance
5. Agent + Protocol ID binding
6. SOP handoff

Reordering is forbidden.

---

## Identity Semantics

### Universal Session ID
- Issued by dual compile (external PXL Gate + internal Runtime Compiler)
- Required for existence within session
- Issued to all agents and protocols
- Conveys NO authority

### Logos Agent ID
- Issued only after Logos discharges LEM constructively
- Root of agency and orchestration authority

---

## Protocol Binding Rules

### Exclusive Agent-Bound Protocols
- SCP ↔ I1
- MTP ↔ I2
- ARP ↔ I3
Shared access is forbidden.

### Logos Protocol
- Bound exclusively to Logos Agent ID
- Logos-only access

### Shared Substrate Protocols (e.g. CSP)
- Bound to session ID + all authorized agent IDs
- No authority escalation

### SOP (Airlocked)
- Receives session ID + Logos Agent ID only
- No agent or external access
- Meta-system only

---

## Logos Orchestration Override
Logos MAY:
- Route messages
- Manage lifecycle
- Query health

Logos MAY NOT:
- Execute protocol internals
- Perform reasoning
- Act as another agent

---

## Failure Semantics
- Any failure halts execution
- No retries
- No degraded modes

This contract is binding on:
- All rewrites
- All runtime execution
- All future system evolution
