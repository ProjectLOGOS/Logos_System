"""
===============================================================================
FILE: __init__.py
PATH: Logos_System_Rebuild/Runtime_Spine/Lock_And_Key/__init__.py
PROJECT: LOGOS System
PHASE: Phase-D
STEP: 2.3 — Runtime Spine / Lock-and-Key Init
STATUS: GOVERNED — NON-BYPASSABLE

CLASSIFICATION:
- Runtime Spine Core Package Initializer
- Non-Executing Authorization Surface

GOVERNANCE:
- Runtime_Spine_Lock_And_Key_Execution_Contract.md
- Runtime_Module_Header_Contract.md

ROLE:
Expose Lock-and-Key interfaces without introducing side effects or
implicit execution.

ORDERING GUARANTEE:
Imported only after Runtime Spine package initialization; precedes any
Lock-and-Key execution.

PROHIBITIONS:
- No implicit execution on import
- No protocol or agent activation
- No external side effects

FAILURE SEMANTICS:
Any violation of prohibitions is treated as a Lock-and-Key initialization fault.
===============================================================================
"""

from .lock_and_key import LockAndKeyFailure, execute_lock_and_key

__all__ = ["LockAndKeyFailure", "execute_lock_and_key"]
