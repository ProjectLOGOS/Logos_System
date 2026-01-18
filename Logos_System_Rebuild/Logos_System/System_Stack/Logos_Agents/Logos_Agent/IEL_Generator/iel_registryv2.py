# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

#!/usr/bin/env python3
# LOGOS SYSTEM FILE HEADER
# LOGOS_HEADER: v1
# updated_utc: 2026-01-15T07:28:00Z
# path: /workspaces/Logos_System/Logos_System/System_Stack/Advanced_Reasoning_Protocol/iel_toolkit/iel_registry.py
# role: dev_tool
# phase: audit_normalize
# origin: ARP_IEL
# intended_bucket: REWRITE_PROMOTE
# side_effects: none
# entrypoints: module
# depends_on: os
# abbreviations: IEL=Internal Emergent Logic
# notes: Header added for semantic audit; filename kept as-is to avoid import breakage.
# END_LOGOS_HEADER

"""

IEL Registry Module



Central registry for managing Internal Extension Libraries (IELs).

Provides domain discovery, loading, and component access capabilities.

"""

import os
from typing import Any, Dict, List, Optional, Type


class IELRegistry:
    """Registry for managing IEL domains and components."""

    def __init__(self):

        self.domains: Dict[str, Dict[str, Any]] = {}

        self.loaded_modules: Dict[str, Any] = {}

        self.iel_path = os.path.dirname(__file__)

    def load_domain(self, domain_name: str) -> bool:
        """Load a specific IEL domain."""

        try:

            domain_path = os.path.join(self.iel_path, domain_name)

            if not os.path.exists(domain_path):

                return False

            # Add domain to domains registry

            self.domains[domain_name] = {
                "path": domain_path,
                "components": [],
                "description": f"IEL domain: {domain_name}",
            }

            return True

        except Exception:

            return False

    def get_domain(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """Get domain information."""

        return self.domains.get(domain_name)

    def get_component(self, full_name: str) -> Optional[Type]:
        """Get a specific component by full name (domain.component)."""

        try:

            parts = full_name.split(".")

            if len(parts) < 2:

                return None

            domain_name = parts[0]

            component_name = parts[1]

            if domain_name not in self.domains:

                return None

            # This is a placeholder - in a real implementation,

            # you would dynamically load the component class

            return None

        except Exception:

            return None

    def list_domains(self) -> List[str]:
        """List all available domains."""

        return list(self.domains.keys())

    def get_domain_description(self, domain_name: str) -> str:
        """Get description of a domain."""

        domain = self.domains.get(domain_name)

        if domain:

            return domain.get("description", f"No description for {domain_name}")

        return f"Domain '{domain_name}' not found"


# Global registry instance

_registry = None


def get_iel_registry() -> IELRegistry:
    """Get the global IEL registry instance."""

    global _registry

    if _registry is None:

        _registry = IELRegistry()

    return _registry
