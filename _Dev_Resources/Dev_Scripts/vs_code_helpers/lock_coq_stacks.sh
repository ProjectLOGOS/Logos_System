#!/usr/bin/env bash
# Guard rails for Coq stack directories. Defaults to lock (read-only).
# To temporarily allow edits, call: ./_Dev_Resources/Dev_Scripts/repo_tools/lock_coq_stacks.sh unlock "edit coq stacks"

set -euo pipefail

LOCK_PATHS=(
  "/workspaces/Logos_System/PXL_Gate"
  "/workspaces/Logos_System/Logos_System/System_Entry_Point/Runtime_Compiler"
)

usage() {
  cat <<'EOF'
Usage:
  lock_coq_stacks.sh lock              # default; make guarded paths read-only
  lock_coq_stacks.sh status            # show current write bits
  lock_coq_stacks.sh unlock "edit coq stacks"  # requires explicit override token
EOF
}

show_status() {
  for p in "${LOCK_PATHS[@]}"; do
    if [[ -e "$p" ]]; then
      perms=$(stat -c "%A" "$p")
      echo "STATUS $perms $p"
    else
      echo "MISSING $p"
    fi
  done
}

lock_paths() {
  for p in "${LOCK_PATHS[@]}"; do
    if [[ -e "$p" ]]; then
      chmod -R a-w "$p"
      echo "LOCKED $p"
    else
      echo "SKIP (missing) $p"
    fi
  done
}

unlock_paths() {
  token=${1:-}
  if [[ "$token" != "edit coq stacks" ]]; then
    echo "Refusing unlock: override token 'edit coq stacks' is required." >&2
    exit 1
  fi
  for p in "${LOCK_PATHS[@]}"; do
    if [[ -e "$p" ]]; then
      chmod -R u+w "$p"
      echo "UNLOCKED (owner write) $p"
    else
      echo "SKIP (missing) $p"
    fi
  done
}

cmd=${1:-lock}
case "$cmd" in
  lock)
    lock_paths
    ;;
  status)
    show_status
    ;;
  unlock)
    unlock_paths "${2:-}"
    ;;
  *)
    usage
    exit 1
    ;;

esac
