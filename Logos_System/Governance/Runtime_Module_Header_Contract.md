# RUNTIME MODULE HEADER CONTRACT (NORMATIVE)

Status: **AUTHORITATIVE · NON-BYPASSABLE**

This document defines the mandatory header format required for **all**
Python modules in the LOGOS System, including `__init__.py` files.

A module that does not comply with this contract is treated as **non-existent**
for the purposes of execution, rewrite, import, and audit.

---

## 1. Scope

This contract applies to:
- All rewritten modules
- All runtime modules
- All governance modules
- All package initializers (`__init__.py`)
- All future rewrites without exception

---

## 2. Ontological Rule

> **No header = no existence.**

A Python file without a valid LOGOS runtime header:
- MUST NOT be imported
- MUST NOT be executed
- MUST NOT be committed
- MUST NOT be rewritten further

---

## 3. Required Header Fields

Every module header MUST declare:

- FILE
- PATH (must match filesystem path)
- PROJECT
- PHASE
- STEP
- STATUS
- CLASSIFICATION
- GOVERNANCE (referenced contracts)
- ROLE
- ORDERING GUARANTEE
- PROHIBITIONS
- FAILURE SEMANTICS

All fields are mandatory.

---

## 4. Governance Binding

Every header MUST reference the governing execution contracts that apply
to the module. These references are binding.

---

## 5. Enforcement

This contract is enforced by:
- Rewrite-time header injection
- Rewrite-time validation
- Pre-commit hooks
- CI / audit scans

Violations result in immediate failure.

---

## 6. Versioning

This contract may only be changed by:
- Explicit versioned replacement
- Repo commit
- Audit review

Silent modification is forbidden.

---

## 7. Authority

This document supersedes:
- Comments
- READMEs
- Inline documentation
- Developer intent

Only explicit versioned contracts may override it.

