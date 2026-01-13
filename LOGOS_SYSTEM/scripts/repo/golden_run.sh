#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INIT_IDENTITY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --init-identity)
      INIT_IDENTITY=1
      shift
      ;;
    *)
      echo "usage: $0 [--init-identity]" >&2
      exit 1
      ;;
  esac
done
cd "$ROOT"

PY="${PYTHON:-"$ROOT/.venv/bin/python"}"
JOBS="${JOBS:-$(nproc)}"
FINGERPRINT="$ROOT/state/golden_run_fingerprint.txt"

echo "[golden-run] initializing submodules"
git submodule update --init --recursive

echo "[golden-run] rebuilding Coq artifacts"
coq_makefile -f _CoqProject -o CoqMakefile
make -f CoqMakefile clean
make -f CoqMakefile -j"$JOBS"

echo "[golden-run] running proof gate"
"$PY" test_lem_discharge.py

echo "[golden-run] computing identity hash"
if [ "$INIT_IDENTITY" -eq 1 ]; then
  "$PY" scripts/compute_identity_hash.py --init-identity
else
  "$PY" scripts/compute_identity_hash.py
fi

echo "[golden-run] capturing fingerprints"
PARENT_SHA=$(git rev-parse HEAD)
LOGOS_SHA=$(git -C external/Logos_AGI rev-parse HEAD)
VSROCQ_SHA=$(git -C external/vsrocq rev-parse HEAD)
COQ_VERSION=$(coqc -v | head -n 1 | tr -s ' ')
PY_VERSION=$("$PY" - <<'PY'
import platform, sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({platform.python_implementation()})")
PY
)

mkdir -p "$ROOT/state"
cat > "$FINGERPRINT" <<EOF
parent_sha=$PARENT_SHA
logos_agi_sha=$LOGOS_SHA
vsrocq_sha=$VSROCQ_SHA
coq_version=$COQ_VERSION
python_version=$PY_VERSION
EOF
cat "$FINGERPRINT"

echo "[golden-run] verifying submodule cleanliness"
git -C external/Logos_AGI checkout -- .
git -C external/vsrocq checkout -- .
git submodule update --recursive

STATUS=$(git status --porcelain |
  grep -v 'state/golden_run_fingerprint.txt' |
  grep -v 'state/IDENTITY_HASH.txt' || true)
if echo "$STATUS" | grep . >/dev/null 2>&1; then
  echo "[golden-run] repo not clean after run" >&2
  echo "$STATUS" >&2
  exit 1
fi

echo "[golden-run] complete"
