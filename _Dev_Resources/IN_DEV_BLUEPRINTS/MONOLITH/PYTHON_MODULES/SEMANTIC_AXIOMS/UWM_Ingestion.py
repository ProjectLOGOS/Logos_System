def ingest(event):
    if not event.get("approved"):
        raise RuntimeError("UWM ingestion rejected")
