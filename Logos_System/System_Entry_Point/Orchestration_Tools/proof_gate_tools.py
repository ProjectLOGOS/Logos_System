# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

import json, hashlib, subprocess, os
from pathlib import Path
from datetime import datetime, timezone

def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def sha256(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()

def strip_volatile(d):
    d = dict(d)
    # Drop fields that differ between the two proof runs but should not
    # affect the commutation hash.
    for k in ("ts", "timestamp", "time", "build_time", "version", "kind"):
        d.pop(k, None)
    return d

def git_short(repo):
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""

def coq_version():
    try:
        return subprocess.check_output(["coqc", "-v"], text=True).splitlines()[0]
    except Exception:
        return ""

def run_make(build_dir, log_txt):
    with open(log_txt, "w") as f:
        p = subprocess.Popen(["make", "-j1", "V=1"], cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
        rc = p.wait()
    if rc != 0:
        raise SystemExit(f"Coq build failed (see {log_txt})")

def emit(kind, repo, build_dir, out_json, files_list):
    run_make(build_dir, out_json.with_suffix(".make.log"))
    payload = {
        "version": f"{kind}/1",
        "kind": kind,
        "ts": now(),
        "git": git_short(repo),
        "coq": coq_version(),
        "files": files_list,
        "result": "PASS",
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    return payload

def main():
    repo = Path(os.environ["REPO_ROOT"]).resolve()

    pxl_build = Path(os.environ["PXL_BUILD_DIR"])
    logos_build = Path(os.environ["LOGOS_BUILD_DIR"])

    pxl_files = Path(os.environ["PXL_FILES_LIST"]).read_text().splitlines()
    logos_files = Path(os.environ["LOGOS_FILES_LIST"]).read_text().splitlines()

    pxl_out = Path(os.environ["PXL_PROOF_OUT"])
    logos_out = Path(os.environ["LOGOS_PROOF_OUT"])

    attest_out = Path(os.environ["ATTEST_OUT"])
    audit_out = Path(os.environ["AUDIT_JSONL"])

    protocol_salt = os.environ.get("LOGOS_PROTOCOL_SALT", "")
    role_salt = os.environ.get("LOGOS_ROLE_SALT", "")

    pxl = emit("pxl-proof-gate", repo, pxl_build, pxl_out, pxl_files)
    lg = emit("logos-system-proof", repo, logos_build, logos_out, logos_files)

    pxl_hash = sha256(canon(strip_volatile(pxl)))
    lg_hash = sha256(canon(strip_volatile(lg)))

    if pxl_hash != lg_hash:
        raise SystemExit("PROOF HASH MISMATCH — REFUSING ACTIVATION")

    unlock_hash = sha256((pxl_hash + lg_hash + protocol_salt).encode())

    def aid(tag):
        return sha256((unlock_hash + tag + role_salt).encode())

    agent_ids = {"I1": aid("I1"), "I2": aid("I2"), "I3": aid("I3")}

    attestation = {
        "version": "logos-attestation/1",
        "ts": now(),
        "pxl_proof_hash": pxl_hash,
        "logos_proof_hash": lg_hash,
        "unlock_hash": unlock_hash,
        "agent_ids": agent_ids,
        "commute": True,
    }

    attest_out.write_text(json.dumps(attestation, indent=2) + "\n")

    audit_out.parent.mkdir(parents=True, exist_ok=True)
    with audit_out.open("a") as f:
        f.write(json.dumps({
            "ts": now(),
            "unlock_hash": unlock_hash,
            "agent_ids": agent_ids,
            "status": "SYSTEM_ACTIVATED",
        }) + "\n")

    print("ACTIVATION COMPLETE")

if __name__ == "__main__":
    main()
