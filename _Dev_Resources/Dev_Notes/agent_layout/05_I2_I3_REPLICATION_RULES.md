# I2 / I3 Replication Rules

1. **Universal Tree:** Every agent must match the canonical structure defined in `00_UNIVERSAL_AGENT_TREE_TEMPLATE.md`. That means `_core/`, `config/`, `connections/`, `diagnostics/`, and `protocol_operations/<protocol_prefix>/` with the exact subfolders (`adapters`, `analysis`, `runtime`, `transforms`, `cycle`, `integrations`, `tests`).
2. **Protocol Prefix:** Use the protocol’s unique prefix consistently. I1 uses `scp`, so all protocol-specific code resides inside `protocol_operations/scp/`. When refactoring I2 and I3, create `protocol_operations/<their_prefix>/` (e.g., `i2p`, `i3p`, or whatever identifier belongs to that agent) and mirror the same nested folders.
3. **File Naming:** Within each protocol subfolder, filenames should also carry the prefix where practical (e.g., `scp_*` in I1). The same pattern should hold for I2/I3 to make cross-agent diffing obvious.
4. **Shared Utilities:** `config/`, `connections/`, `diagnostics/`, and `_core/` stay protocol-agnostic. If functionality is reused by multiple protocols, promote it into `Shared_Resources/` or `Protocol_Resources/` rather than duplicating under each agent.
5. **Template as Source of Truth:** Treat the I1 layout after this refactor as the authoritative template. Apply identical moves and import-path updates when reorganizing I2 and I3, adjusting only the protocol prefix.
