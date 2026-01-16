# Family Exclusion Sanity Check

- SMOKE_DIR: `/workspaces/Logos_System/_Dev_Resources/SMOKE_TEST_FAMILIES`
- DEV_DELTAS_DIR: `/workspaces/Logos_System/_Dev_Resources/Dev_Scripts/family_deltas`
- ACTIVE_DIR: `/workspaces/Logos_System/FAMILY_LIST_TO_PROCESS`

## Findings

- No missing/empty/parse-failing delta artifacts detected.

### Smoke bucket contains non-legacy System_Stack paths (treat as HOLD/RE-TRIAGE, not permanent exclusion)
- FAMILY_010 (159 members) → `FAMILY_001_10_010_expanded_deltas.json`
- FAMILY_011 (35 members) → `FAMILY_001_11_011_expanded_deltas.json`
- FAMILY_016 (25 members) → `FAMILY_001_16_016_expanded_deltas.json`
- FAMILY_017 (4623 members) → `FAMILY_001_17_017_expanded_deltas.json`
- FAMILY_019 (412 members) → `FAMILY_001_19_019_expanded_deltas.json`
- FAMILY_020 (94 members) → `FAMILY_001_20_020_expanded_deltas.json`
- FAMILY_021 (759 members) → `FAMILY_001_21_021_expanded_deltas.json`
- FAMILY_022 (262 members) → `FAMILY_001_22_022_expanded_deltas.json`
- FAMILY_023 (25 members) → `FAMILY_001_23_023_expanded_deltas.json`
- FAMILY_025 (260 members) → `FAMILY_001_25_025_expanded_deltas.json`
- FAMILY_026 (129 members) → `FAMILY_001_26_026_expanded_deltas.json`
- FAMILY_027 (253 members) → `FAMILY_001_27_027_expanded_deltas.json`
- FAMILY_029 (10 members) → `FAMILY_001_29_029_expanded_deltas.json`
- FAMILY_030 (1 members) → `FAMILY_001_30_030_expanded_deltas.json`
- FAMILY_031 (5 members) → `FAMILY_001_31_031_expanded_deltas.json`
- FAMILY_033 (60 members) → `FAMILY_001_33_033_expanded_deltas.json`
- FAMILY_039 (54 members) → `FAMILY_001_39_039_expanded_deltas.json`
- FAMILY_040 (2 members) → `FAMILY_001_40_040_expanded_deltas.json`
- FAMILY_042 (22 members) → `FAMILY_001_42_042_expanded_deltas.json`
- FAMILY_043 (4 members) → `FAMILY_001_43_043_expanded_deltas.json`

### Active queue contains TEST/LEGACY markers (investigate)
- FAMILY_009 (2 members) → `FAMILY_001_09_009_expanded_deltas.json`
- FAMILY_015 (23 members) → `FAMILY_001_15_015_expanded_deltas.json`
