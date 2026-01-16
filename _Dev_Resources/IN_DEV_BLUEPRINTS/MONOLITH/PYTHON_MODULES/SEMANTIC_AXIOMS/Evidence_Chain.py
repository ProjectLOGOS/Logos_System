def verify_evidence_chain(chain):
    if not chain.get("chain_hash"):
        raise RuntimeError("Evidence chain missing or invalid")
