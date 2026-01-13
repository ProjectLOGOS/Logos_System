# VS Code Codespace Crash Prevention & Recovery Guide

## Current Status (as of scan)

### ⚠️ Issues Identified:
1. **High Memory Usage**: 79% (6.2GB/7.8GB) - Only 362MB free RAM, no swap
2. **Disk Usage**: 75% (23GB/32GB) - Approaching limit
3. **Heavy Extensions**: 37 extensions installed, many unnecessary for Coq/Python work
4. **Multiple Extension Hosts**: 2+ running consuming ~1.6GB combined

### ✅ Working Correctly:
- Rocq Prover 9.1.0 installed and accessible
- Settings configured with proper paths
- File watchers properly excluding build artifacts
- Good exclusions for search/indexing

---

## Immediate Actions (Do These Now)

### 1. Run Health Check
```bash
./scripts/vscode_health_check.sh
```

### 2. Clean Up Extensions (Recommended)
```bash
# This will remove heavy, non-essential extensions
./scripts/cleanup_extensions.sh
```

Then: **Command Palette** → `Developer: Reload Window`

### 3. Monitor Memory
```bash
# Watch memory in real-time
watch -n 5 'free -h && echo "" && ps aux | grep extensionHost | grep -v grep'
```

---

## Preventive Configuration

### Updated Settings Applied
The following optimizations were added to `.vscode/settings.json`:

- ✅ Added more file watcher exclusions (logs, sandbox, demos)
- ✅ Disabled auto-save on typing (saves on focus change)
- ✅ Limited TypeScript server memory to 512MB
- ✅ Disabled Git auto-refresh/fetch
- ✅ Disabled telemetry and experiments
- ✅ Reduced Python analysis memory footprint
- ✅ Limited terminal scrollback to 1000 lines

### Essential Extensions Only
Recommended to keep:
- `rocq-prover.vsrocq` - Rocq/Coq language server
- `ms-python.python` - Python core
- `ms-python.vscode-pylance` - Python language server
- `github.copilot` / `github.copilot-chat` - AI assistance
- `ms-python.debugpy` - Python debugging

Safe to remove:
- GitLens (heavy Git features - use `git` CLI instead)
- PowerShell (not needed on Linux)
- Test adapters (run tests via terminal)
- Formatters/linters (run on-demand via terminal)
- Voice coding tools
- React generators

---

## If Crashes Continue

### Step 1: Disable Non-Essential Extensions Manually
1. Open Extensions panel
2. Filter by: `@enabled`
3. Disable everything except:
   - Remote/Codespaces core
   - rocq-prover.vsrocq
   - ms-python.python
   - ms-python.vscode-pylance
   - github.copilot (optional but recommended)

### Step 2: Clear Extension State
```bash
# Nuclear option - clears all workspace storage
rm -rf ~/.vscode-remote/data/User/workspaceStorage/*
rm -rf ~/.vscode-remote/data/User/globalStorage/*

# Clean old logs (keeps last 3 sessions)
cd ~/.vscode-remote/data/logs/
ls -t | tail -n +4 | xargs rm -rf
```

Then reload: **Command Palette** → `Developer: Reload Window`

### Step 3: Verify Rocq Extension
```bash
# Check extension is properly installed
code --list-extensions | grep rocq

# Expected output:
# rocq-prover.vsrocq
```

If missing or broken:
```bash
# Reinstall cleanly
code --uninstall-extension rocq-prover.vsrocq
code --install-extension rocq-prover.vsrocq
```

---

## Understanding the Crashes

### Memory Exhaustion Pattern
- Extension hosts load all extensions in Node.js processes
- Each language server (Python, Coq, TypeScript) runs separately
- With 37 extensions + language servers, easily exceeds 6GB
- No swap means instant crash when RAM runs out

### File Watcher Overload
- VS Code watches all files for changes
- Large directories like `external/` (323MB) cause issues
- Each watcher consumes file descriptors and memory
- Solution: Aggressive exclusions in settings (now applied)

### Extension Host Deadlock
- Some extensions conflict or deadlock each other
- Common culprits: Git extensions, test adapters, formatters
- Extension host crashes and restarts every 30-60 seconds
- Solution: Reduce to minimal essential set

---

## Monitoring Commands

### Memory Status
```bash
free -h
```

### Disk Status
```bash
df -h /workspaces
```

### Extension Processes
```bash
ps aux | grep extensionHost | grep -v grep
```

### Recent Crashes
```bash
ls -lt ~/.vscode-remote/data/logs/ | head -10
```

### Health Check (All-in-One)
```bash
./scripts/vscode_health_check.sh
```

---

## Emergency Recovery

If environment is completely broken:

### Option 1: Rebuild Codespace
GitHub Codespaces → `...` menu → Rebuild Container
- Takes 5-10 minutes
- Keeps workspace files
- Resets all VS Code state

### Option 2: Fresh Extension Install
```bash
# Remove all extensions
code --list-extensions | xargs -n1 code --uninstall-extension

# Install only essentials
code --install-extension rocq-prover.vsrocq
code --install-extension ms-python.python
code --install-extension github.copilot
```

---

## Optimal Workflow

### For Coq Development
1. Open only `.v` files you're actively editing
2. Let Rocq extension process one file at a time
3. Close files when done to free memory

### For Python Work
1. Use terminal for linting/formatting:
   ```bash
   python3 -m pylint script.py
   python3 -m black script.py
   ```
2. Disable on-save formatters
3. Run tests via CLI: `python3 test_lem_discharge.py`

### General
- Close unused terminals (each consumes ~50MB)
- Reload window after intensive operations
- Run health check before long sessions
- Monitor memory if you notice slowdowns

---

## Technical Details

### System Specs
- RAM: 7.8GB (no swap)
- Disk: 32GB total
- CPU: Multi-core (shared Codespace)
- OS: Ubuntu 24.04.2 LTS

### Rocq/Coq Setup
- Version: Rocq Prover 9.1.0 (OCaml 5.1.1)
- Install: `/home/codespace/.opam/rocq/bin/`
- Extension: `rocq-prover.vsrocq`

### File Watcher Limits
- System: `ulimit -n` = 1048576 (very high, good)
- VS Code uses subset based on workspace size
- Current exclusions prevent watching ~500MB of files

---

## Questions?

Run health check first:
```bash
./scripts/vscode_health_check.sh
```

Check logs for specific errors:
```bash
tail -100 ~/.vscode-remote/data/logs/*/exthost*/output.log
```
