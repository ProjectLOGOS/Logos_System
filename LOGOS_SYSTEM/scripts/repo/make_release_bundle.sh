#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAGING_BASE="/tmp"
STAGING_NAME="logos_audit_bundle_${SHA}"
STAGING_DIR="${STAGING_BASE}/${STAGING_NAME}"

echo "[bundle] staging at ${STAGING_DIR}"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

copy_dir() {
  local src="$1" dst="$2"
  mkdir -p "${dst}"
  rsync -a --delete "${src}/" "${dst}/"
}

copy_file() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "${dst}")"
  install -m 644 "${src}" "${dst}"
}

echo "[bundle] copying audit docs"
copy_dir "$ROOT/AUDIT" "$STAGING_DIR/AUDIT"

echo "[bundle] copying scripts and fingerprints"
mkdir -p "$STAGING_DIR/scripts" "$STAGING_DIR/state"
install -m 755 "$ROOT/scripts/golden_run.sh" "$STAGING_DIR/scripts/golden_run.sh"
install -m 755 "$ROOT/scripts/compute_identity_hash.py" "$STAGING_DIR/scripts/compute_identity_hash.py"
copy_file "$ROOT/state/golden_run_fingerprint.txt" "$STAGING_DIR/state/golden_run_fingerprint.txt"
if [ -f "$ROOT/state/IDENTITY_HASH.txt" ]; then
  copy_file "$ROOT/state/IDENTITY_HASH.txt" "$STAGING_DIR/state/IDENTITY_HASH.txt"
fi

echo "[bundle] copying gate harness and project file"
copy_file "$ROOT/test_lem_discharge.py" "$STAGING_DIR/test_lem_discharge.py"
copy_file "$ROOT/_CoqProject" "$STAGING_DIR/_CoqProject"

echo "[bundle] copying Coq baseline sources (excluding build artifacts)"
mkdir -p "$STAGING_DIR/Protopraxis/formal_verification/coq/baseline"
rsync -a \
  --prune-empty-dirs \
  --exclude='*.vo' \
  --exclude='*.vos' \
  --exclude='*.vok' \
  --exclude='*.glob' \
  --exclude='*.aux' \
  --exclude='*.d' \
  "$ROOT/Protopraxis/formal_verification/coq/baseline/" \
  "$STAGING_DIR/Protopraxis/formal_verification/coq/baseline/"

echo "[bundle] recording external submodule pointer"
LOGOS_SHA="$(git -C "$ROOT/external/Logos_AGI" rev-parse HEAD)"
mkdir -p "$STAGING_DIR/external"
printf "Logos_AGI gitlink: %s\n" "$LOGOS_SHA" > "$STAGING_DIR/external/Logos_AGI.GITLINK"

echo "[bundle] writing manifest and checksums"
RELEASE_DIR="$ROOT/release"
mkdir -p "$RELEASE_DIR"

MANIFEST="$RELEASE_DIR/MANIFEST.txt"
SHA_TXT="$RELEASE_DIR/SHA256SUMS.txt"

(cd "$STAGING_DIR" && find . -type f | sort) > "$MANIFEST"
(cd "$STAGING_DIR" && find . -type f | sort | xargs sha256sum) > "$SHA_TXT"

TARBALL="$RELEASE_DIR/logos_audit_bundle_${SHA}.tar.gz"
echo "[bundle] creating tarball ${TARBALL}"
rm -f "$TARBALL"
tar --sort=name --mtime='UTC 2020-01-01' --owner=0 --group=0 --numeric-owner -czf "$TARBALL" -C "$STAGING_BASE" "$STAGING_NAME"

TARBALL_SHA="$RELEASE_DIR/TARBALL_SHA256SUM.txt"
sha256sum "$TARBALL" > "$TARBALL_SHA"

echo "[bundle] done"