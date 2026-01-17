"""
===============================================================================
FILE: __init__.py
PATH: Logos_System_Rebuild/System_Entry_Point/__init__.py
PROJECT: LOGOS System
PHASE: Phase-D
STEP: 2.1 — System Entry Point Package Init
STATUS: GOVERNED — NON-BYPASSABLE

CLASSIFICATION:
- Package Initializer
- Non-Executing Bootstrap Surface

GOVERNANCE:
- System_Entry_Point_Execution_Contract.md
- Runtime_Module_Header_Contract.md

ROLE:
Expose the canonical System Entry Point interface without introducing
side effects or implicit execution.

ORDERING GUARANTEE:
Imported only after explicit START_LOGOS handoff; precedes any runtime
spine execution.

PROHIBITIONS:
- No implicit execution on import
- No protocol or agent activation
- No external side effects

FAILURE SEMANTICS:
Any breach of prohibitions is treated as a startup fault.
===============================================================================
"""

from .System_Entry_Point import START_LOGOS, StartupHalt

__all__ = ["START_LOGOS", "StartupHalt"]
