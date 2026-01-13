# VS Code Crash Recovery - Quick Reference

## 🚨 Environment is Crashing? Run This:

```bash
./scripts/emergency_recovery.sh
```

Then: **Ctrl+Shift+P** → `Developer: Reload Window`

---

## 📊 Check System Health

```bash
./scripts/vscode_health_check.sh
```

---

## 🧹 Remove Heavy Extensions (One-Time Setup)

```bash
./scripts/cleanup_extensions.sh
```

This removes 15+ non-essential extensions that consume memory.

---

## 🔍 What Was Changed

### Settings Optimized ([.vscode/settings.json](.vscode/settings.json))
- ✅ Added file watcher exclusions for logs/, sandbox/, external/
- ✅ Disabled auto-save on typing (saves on focus change)
- ✅ Limited TypeScript memory to 512MB
- ✅ Disabled Git auto-refresh/auto-fetch
- ✅ Reduced Python analysis memory usage
- ✅ Disabled telemetry

### Extension Recommendations Updated ([.vscode/extensions.json](.vscode/extensions.json))
- ✅ Recommends only `rocq-prover.vsrocq`
- ✅ Marked `coqpilot` as unwanted (conflicts)

### New Monitoring Scripts
- `scripts/vscode_health_check.sh` - System diagnostics
- `scripts/cleanup_extensions.sh` - Remove heavy extensions
- `scripts/emergency_recovery.sh` - Quick crash recovery

### Documentation
- [docs/VSCODE_CRASH_PREVENTION.md](docs/VSCODE_CRASH_PREVENTION.md) - Comprehensive guide

---

## 📈 Current Status (After Optimization)

- **Memory**: 66% used (was 79%) - ✅ Improved!
- **Disk**: 75% used - ⚠️ Monitor
- **Extensions**: 37 installed - 🔧 Run cleanup_extensions.sh
- **Extension Hosts**: 9 processes - ⚠️ High but stable

---

## ⚡ Immediate Actions

### 1. Reload Window Now
**Ctrl+Shift+P** → `Developer: Reload Window`

This applies the new settings and can free memory.

### 2. Optional: Clean Up Extensions
```bash
./scripts/cleanup_extensions.sh
```

Removes:
- GitLens (heavy)
- PowerShell (not needed)
- Test adapters (use terminal)
- Formatters (use terminal)
- Voice coding tools
- ChatGPT extension (redundant with Copilot)

Keeps essential:
- rocq-prover.vsrocq
- ms-python.python
- ms-python.vscode-pylance
- github.copilot*

### 3. Monitor for 60 Seconds
After reload, wait 60 seconds. If it crashes:
1. Run `./scripts/emergency_recovery.sh`
2. Reload again
3. If still crashing, run `./scripts/cleanup_extensions.sh`

---

## 🎯 Root Causes Identified

1. **Memory Pressure**: 7.8GB RAM, no swap, 37 extensions
2. **File Watcher Overhead**: Large directories being watched
3. **Extension Conflicts**: Too many language servers/formatters
4. **Heavy Extensions**: GitLens, test adapters, unused tools

---

## 📚 Learn More

Full troubleshooting guide: [VSCODE_CRASH_PREVENTION.md](docs/VSCODE_CRASH_PREVENTION.md)

---

## 🆘 Still Having Issues?

Try in order:

1. **Emergency recovery**:
   ```bash
   ./scripts/emergency_recovery.sh
   ```

2. **Extension cleanup**:
   ```bash
   ./scripts/cleanup_extensions.sh
   ```

3. **Manual disable extensions**:
   - Extensions panel → Disable all except: rocq-prover.vsrocq, ms-python.python, ms-python.vscode-pylance

4. **Nuclear option** (clears all state):
   ```bash
   rm -rf ~/.vscode-remote/data/User/workspaceStorage/*
   rm -rf ~/.vscode-remote/data/User/globalStorage/*
   ```

5. **Rebuild Codespace**:
   - GitHub Codespaces menu → Rebuild Container
   - Takes 5-10 minutes but gives fresh start

---

## ✅ Prevention Best Practices

- Run health check before long sessions
- Close files/terminals when not in use
- Reload window after intensive operations (building, large searches)
- Keep extensions minimal
- Use terminal for formatting/linting instead of on-save
- Monitor memory: `free -h`

---

*Last updated: 2025-12-21 after system analysis*
