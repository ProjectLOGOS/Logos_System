# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

import shutil
import sys

src, out = sys.argv[1], sys.argv[2]
shutil.copy(src, out)
print(f"[OK] Copied {src} to {out}")
