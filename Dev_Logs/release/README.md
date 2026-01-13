# Audit Release Bundle

## Build the bundle
- From repo root: `./scripts/make_release_bundle.sh`
- Produces staging tree at `/tmp/logos_audit_bundle_<SHA>` and a tarball at `release/logos_audit_bundle_<SHA>.tar.gz`.

## Verify integrity
- Check file checksums: `sha256sum -c release/SHA256SUMS.txt` (run inside the staging tree after extracting if desired).
- Check tarball checksum: `sha256sum -c release/TARBALL_SHA256SUM.txt`.
- MANIFEST lists all files included: `release/MANIFEST.txt`.

## Re-run proof gate
- After unpacking: from bundle root, run `./scripts/golden_run.sh` to rebuild and verify. Expected PASS mirrors the gate: overall PASS, `<none>` for extra assumptions and admitted stubs.
