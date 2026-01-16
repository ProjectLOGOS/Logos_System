def assert_read_only(action):
    if action.get("writes_uwm"):
        raise RuntimeError("Agents are read-only by contract")
