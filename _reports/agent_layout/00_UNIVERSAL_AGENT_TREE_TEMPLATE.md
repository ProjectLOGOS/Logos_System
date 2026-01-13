# Universal Agent Tree Template

Every Logos agent (I1, I2, I3) must share the exact directory contract below. All paths are relative to `System_Stack/Logos_Agents_Protocol/Logos_Agents/<Agent_Name>`.

```
<Agent_Name>/
├── _core/
├── config/
├── connections/
├── diagnostics/
└── protocol_operations/
    └── <protocol_prefix>/
        ├── adapters/
        ├── analysis/
        ├── runtime/
        ├── transforms/
        ├── cycle/
        ├── integrations/
        └── tests/
```

## Naming Convention

- `<protocol_prefix>` identifies the protocol that agent executes (e.g., `scp` for I1). The prefix must be unique per protocol and appears in **every** protocol-specific filename or folder.
- Adapters, analysis, runtime, transforms, cycle, integrations, and tests folders live under `protocol_operations/<protocol_prefix>/` and should only contain files prefixed with that protocol (e.g., `scp_*` for I1).
- I2 and I3 must mirror this structure, substituting their own protocol prefixes consistently (e.g., `i2p_`, `i3p_`, or whatever identifiers they own).
- Shared utilities (config, connections, diagnostics, _core) remain protocol-agnostic and are never prefixed.
