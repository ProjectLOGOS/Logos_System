def enforce_phase_5(event):
    if event.get("phase5_status") != "PASS":
        raise RuntimeError("Phase 5 validation failed")
