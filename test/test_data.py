
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Training Dataset": BASE_DIR / "data/processed/train_telco_churn.csv",
    "Testing Dataset": BASE_DIR / "data/processed/test_telco_churn1.csv"
}

report = []

print("\n========== PROCESSED DATASET TESTING ==========\n")

columns_found = {}

for name, path in DATASETS.items():

    print(f"\n--- {name} ---")

    if not path.exists():
        print("FAIL: Dataset file not found:", path)
        continue

    df = pd.read_csv(path)

    print("PASS: Dataset loaded")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    numeric = df.select_dtypes(include=np.number)
    infinite = int(np.isinf(numeric).sum().sum())

    duplicate_columns = int(df.columns.duplicated().sum())

    print("Missing values:", missing)
    print("Duplicate rows:", duplicates)
    print("Infinite values:", infinite)
    print("Duplicate column names:", duplicate_columns)

    if missing == 0:
        print("PASS: Missing value check")
    else:
        print("FAIL: Missing values found")

    if infinite == 0:
        print("PASS: Infinite value check")
    else:
        print("FAIL: Infinite values found")

    if duplicate_columns == 0:
        print("PASS: Column name check")
    else:
        print("FAIL: Duplicate column names")

    if duplicates == 0:
        print("PASS: Duplicate row check")
    else:
        print("REVIEW: Duplicate rows found")

    # Check target column if present
    target = next(
        (c for c in ["Churn Value", "Churn"] if c in df.columns),
        None
    )

    if target:
        print("\nTarget column:", target)
        print(df[target].value_counts(dropna=False))

        values = set(df[target].dropna().unique())

        if values.issubset({0, 1}):
            print("PASS: Target contains binary values")
        else:
            print("REVIEW: Target has unexpected values")
    else:
        print("\nREVIEW: Target column not identified")

    columns_found[name] = list(df.columns)

    report.append(
        f"{name}\n"
        f"Rows: {len(df)}\n"
        f"Columns: {len(df.columns)}\n"
        f"Missing: {missing}\n"
        f"Duplicates: {duplicates}\n"
        f"Infinite: {infinite}\n"
        f"Duplicate column names: {duplicate_columns}\n"
    )

# Compare dataset columns
print("\n========== DATASET COMPARISON ==========")

if len(columns_found) == 2:
    if columns_found["Training Dataset"] == columns_found["Testing Dataset"]:
        print("PASS: Training and testing columns match")
    else:
        print("FAIL: Training and testing columns do not match")
else:
    print("REVIEW: Both datasets were not loaded")

# Save report
report_dir = BASE_DIR / "reports"
report_dir.mkdir(exist_ok=True)

report_path = report_dir / "processed_dataset_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\nReport saved:", report_path)
print("\n========== TESTING FINISHED ==========")















# import pandas as pd
# from pathlib import Path

# # Get project directory
# BASE_DIR = Path(__file__).resolve().parent.parent

# # Correct dataset location
# TEST_PATH = (
#     BASE_DIR / "data" / "processed" / "test_telco_churn1.csv"
# )

# print("=" * 60)
# print("TEST DATASET DIAGNOSTIC")
# print("=" * 60)

# # Check whether the file exists
# print("\nFile location:")
# print(TEST_PATH)

# if not TEST_PATH.exists():
#     raise FileNotFoundError(
#         f"Dataset not found: {TEST_PATH}"
#     )

# # Load the dataset
# test_data = pd.read_csv(TEST_PATH)

# # Print actual dataset dimensions
# print("\nDataset shape:")
# print(test_data.shape)

# print("\nTotal rows:", test_data.shape[0])
# print("Total columns:", test_data.shape[1])

# # Display every column
# print("\nAll column names:")

# for i, column in enumerate(test_data.columns, start=1):
#     print(i, column)

# # Check target column
# print("\nTarget column exists:",
#       "Churn Value" in test_data.columns)

# # Final verification
# assert test_data.shape[1] == 35, (
#     f"Expected 35 columns, found {test_data.shape[1]}"
# )

# print("\nSUCCESS: Test dataset contains 35 columns.")
