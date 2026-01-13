#!/bin/bash
# Disable heavy/unnecessary extensions to reduce memory pressure

echo "🔧 Disabling non-essential extensions for this Coq/Python project..."
echo ""

# Extensions that are heavy but not critical for this project
DISABLE_LIST=(
    "eamodio.gitlens"                    # GitLens - heavy Git features
    "ms-vscode.powershell"               # PowerShell - not needed in Linux
    "hbenl.vscode-test-explorer"         # Test Explorer UI
    "littlefoxteam.vscode-python-test-adapter"  # Python test adapter
    "ms-vscode.test-adapter-converter"   # Test adapter converter
    "pokey.talon"                        # Talon voice coding
    "pokey.parse-tree"                   # Parse tree viewer
    "andrewmcgoveran.react-component-generator"  # React - not used here
    "genieai.chatgpt-vscode"             # ChatGPT - redundant with Copilot
    "ms-python.black-formatter"          # Black - can format on-demand
    "ms-python.flake8"                   # Flake8 - can run in terminal
    "ms-python.isort"                    # isort - can run in terminal
    "ms-python.pylint"                   # Pylint - can run in terminal
)

for ext in "${DISABLE_LIST[@]}"; do
    # Check if installed
    if code --list-extensions | grep -q "^${ext}$"; then
        echo "  Uninstalling: $ext"
        code --uninstall-extension "$ext" >/dev/null 2>&1 || true
    fi
done

echo ""
echo "✅ Extension cleanup complete!"
echo ""
echo "Essential extensions remaining:"
echo "  - rocq-prover.vsrocq (Rocq/Coq)"
echo "  - ms-python.python (Python core)"
echo "  - ms-python.vscode-pylance (Python language server)"
echo "  - github.copilot* (AI assistance)"
echo ""
echo "🔄 Reload the window now: Ctrl+Shift+P → 'Developer: Reload Window'"
