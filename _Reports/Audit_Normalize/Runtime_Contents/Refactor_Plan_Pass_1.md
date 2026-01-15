# Refactor Plan — Pass 1 (Audit/Normalize)

- Generated (UTC): 2026-01-14T17:35:37Z

## Input availability
- **orphan_classification**: present=False count=0
- **inspect_decide**: present=False count=0
- **duplicate_files**: present=False count=0
- **duplicate_directories**: present=False count=0
- **semantic_duplicates**: present=False count=0
- **naming_file_violations**: present=False count=0
- **naming_dir_violations**: present=False count=0
- **naming_module_violations**: present=False count=0

## Hard constraints
- fail_closed: True
- coq_files_immutable: True
- coq_documentation_sidecar_only: True
- logos_token_excluded_from_legacy_scans: True
- naming_convention: Title_Case_With_Underscores
- header_policy: {'py_inline_headers_deferred_until': 'Script_Crawl', 'json_yaml_sidecar': True, 'coq_v_sidecar_only': True}

## Next actions
- Define Crawl Phase 1 as read-only inspection + tagging + plan enrichment
- Build Move_Queue_Pass_1 (Dev_Scripts first; exclude Coq and PXL_Gate unless explicitly instructed)
- Build Rename_Queue_Pass_1 from naming violation lists (batched, reversible)
- Build Merge_Queue_Pass_1 from semantic duplicates and user-provided semantic merge candidates
- Run import-resolution reclass pass (stdlib/site-packages vs internal) before using unresolved imports for moves
