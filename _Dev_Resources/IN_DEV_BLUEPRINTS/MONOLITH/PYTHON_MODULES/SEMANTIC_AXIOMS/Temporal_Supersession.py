def supersede(old_atom, new_atom):
    old_atom["valid_to"] = "closed"
    new_atom["supersedes"] = old_atom["id"]
