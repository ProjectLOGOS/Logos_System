def enforce_invariants(atom):
    if atom.get("mutated_after_verification"):
        raise RuntimeError("Invariant violation detected")
