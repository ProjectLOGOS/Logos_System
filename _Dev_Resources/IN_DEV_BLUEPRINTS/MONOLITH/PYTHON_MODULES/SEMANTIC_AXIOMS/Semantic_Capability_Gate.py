def admit_capability(cap):
    if not cap.get("mapped"):
        raise RuntimeError("Capability not admitted")
