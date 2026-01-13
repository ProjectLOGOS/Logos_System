#!/bin/bash
# VS Code Codespace Health Check & Cleanup Script

set -e

echo "=== VS Code Codespace Health Check ==="
echo ""

# 1. Check disk usage
echo "📊 Disk Usage:"
df -h / /tmp /workspaces | grep -v "Filesystem"
echo ""

# 2. Check memory usage
echo "💾 Memory Usage:"
free -h
echo ""

# 3. Check running extension hosts
echo "🔍 Extension Host Processes:"
ps aux | grep -E "extensionHost|node.*vscode" | grep -v grep | wc -l
echo " extension host processes running"
echo ""

# 4. Check for crashed extensions (recent logs)
echo "📝 Recent Extension Host Sessions:"
ls -lt ~/.vscode-remote/data/logs/ 2>/dev/null | head -5 | tail -4
echo ""

# 5. Check Coq/Rocq processes
echo "⚙️  Coq/Rocq Processes:"
ps aux | grep -E "coq|rocq" | grep -v grep | wc -l
echo " Coq processes running"
echo ""

# Optional: Cleanup old logs
if [ "$1" == "--cleanup" ]; then
    echo "🧹 Cleaning up old logs and caches..."
    
    # Remove old log directories (keep last 3)
    cd ~/.vscode-remote/data/logs/
    ls -t | tail -n +4 | xargs -r rm -rf
    
    # Clean workspace storage (can be regenerated)
    rm -rf ~/.vscode-remote/data/User/workspaceStorage/* 2>/dev/null || true
    
    # Clean Coq build artifacts in workspace
    cd /workspaces/pxl_demo_wcoq_proofs
    find . -name "*.vo" -o -name "*.vok" -o -name "*.vos" -o -name "*.aux" -o -name "*.glob" | wc -l
    echo " Coq build artifacts found (excluded from watchers)"
    
    echo "✅ Cleanup complete!"
fi

echo ""
echo "=== Health Check Complete ==="
echo ""
echo "⚠️  Memory is at $(free | grep Mem | awk '{printf "%.0f%%", $3/$2 * 100}'). Consider:"
echo "   - Disabling unused extensions"
echo "   - Closing unused terminal sessions"
echo "   - Reloading window periodically"
