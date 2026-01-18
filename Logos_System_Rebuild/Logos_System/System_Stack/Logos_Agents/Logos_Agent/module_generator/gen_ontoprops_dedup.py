# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

import shutil
import sys

src, out = sys.argv[1], sys.argv[2]
shutil.copy(src, out)
print(f"[OK] Copied {src} to {out}")
