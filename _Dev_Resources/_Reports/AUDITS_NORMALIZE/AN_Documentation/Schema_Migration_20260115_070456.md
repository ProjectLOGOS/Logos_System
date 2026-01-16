Audit scripts updated to emit End-to-End schema diagnostics:
- Added shared diagnostic helper (_diagnostic_record.py)
- Added line/char_start/char_end to imports, symbols, dependencies, side effects, naming
- Fail-closed validation on invalid ranges
No behavioral refactors performed.
