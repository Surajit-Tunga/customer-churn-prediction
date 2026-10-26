# Data pipeline guidelines

- Keep source data in `raw/` and derived data in `processed/`.
- Document each dataset's source, licence, and refresh date.
- Validate schema, missing values, duplicates, and target leakage at each step.
- Do not commit sensitive or licensed data unless its terms permit it.
