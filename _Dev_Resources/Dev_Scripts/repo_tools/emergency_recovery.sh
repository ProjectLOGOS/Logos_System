#!/bin/bash
# Quick VS Code Codespace Recovery Script
# Run this if VS Code starts crashing repeatedly

echo "🚨 VS Code Emergency Recovery"
echo ""

# Step 1: Kill any hung Coq processes
echo "1️⃣  Checking for hung Coq processes..."
pkill -9 coqtop 2>/dev/null || true
pkill -9 coqc 2>/dev/null || true
echo "   ✓ Cleared"

# Step 2: Clean workspace storage
echo "2️⃣  Cleaning extension workspace storage..."
rm -rf ~/.vscode-remote/data/User/workspaceStorage/* 2>/dev/null || true
echo "   ✓ Cleared"

# Step 3: Clean old logs (keep last 2)
echo "3️⃣  Cleaning old log sessions..."
cd ~/.vscode-remote/data/logs/ 2>/dev/null || true
ls -t | tail -n +3 | xargs -r rm -rf 2>/dev/null || true
echo "   ✓ Cleared"

# Step 4: Report status
echo ""
echo "📊 Current Status:"
free -h | grep Mem | awk '{print "   Memory: " $3 " / " $2 " (" int($3/$2*100) "%)"}'
df -h /workspaces | tail -1 | awk '{print "   Disk: " $3 " / " $2 " (" $5 ")"}'
ps aux | grep extensionHost | grep -v grep | wc -l | awk '{print "   Extension hosts: " $1}'

echo ""
echo "✅ Emergency recovery complete!"
echo ""
echo "Next steps:"
echo "  1. Reload window: Ctrl+Shift+P → 'Developer: Reload Window'"
echo "  2. If still crashing, run: ./scripts/cleanup_extensions.sh"
echo "  3. Monitor: ./scripts/vscode_health_check.sh"
