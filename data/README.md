# Data Directory Architecture & Governance

This directory serves as the centralized repository for all dataset artifacts used in the IBM Telco Customer Churn Prediction project. To guarantee technical reproducibility, strict data governance, and downstream model reliability, all stored files follow a strict separation between immutable raw inputs and transformed outputs.

---

## 1. Directory Structure

The `data/` directory strictly isolates raw source materials from derived machine learning artifacts:

```text
data/
├── raw/
│   └── raw_telco_churn.csv        # Immutable source dataset (7,043 rows, 33 attributes)
└── processed/
    ├── train_telco_churn.csv      # 80% stratified training partition (5,634 rows, 24 attributes)
    ├── test_telco_churn.csv       # 20% stratified holdout partition (1,409 rows, 24 attributes)
    └── input_encoders.pkl         # Fitted LabelEncoders serialized for model and app deployment
```

* **`data/raw/`**: Contains original, untouched datasets. Files in this directory are read-only and must never be modified directly or overwritten by processing scripts.
* **`data/processed/`**: Contains cleaned, engineered, encoded, and split datasets, along with serialization binaries required for model execution and application runtime.

---

## 2. Raw Datasets (`data/raw/`)

The primary source file represents historical customer account data originating from IBM Cognos Analytics.

### Dataset Metadata: `raw_telco_churn.csv`

| Property | Specification Details |
| :--- | :--- |
| **File Path** | `data/raw/raw_telco_churn.csv` |
| **Source / Provenance** | IBM Cognos Analytics base samples authored by Steven Macko (via Kaggle repository by `yeanzc`) |
| **Licensing Terms** | Kaggle Tagged: "Other (specified in description)" — Approved for non-commercial academic research, educational prototyping, and internal model benchmarking with proper IBM attribution |
| **Refresh Date** | 2026-10-28 |
| **Data Dimensions** | 7,043 observations × 33 variables |
| **Primary Target Variable** | `Churn Value` (Binary: `0` = Retained, `1` = Churned) |
| **Access Control** | Open academic access; verified synthetic data containing zero Personally Identifiable Information (PII) |

---

## 3. Processed Datasets & Artifacts (`data/processed/`)

Derived datasets are generated via the automated data pipeline executed in `notebooks/customer_churn_data_prep.ipynb`.

### Dataset Metadata: `train_telco_churn.csv`

| Property | Specification Details |
| :--- | :--- |
| **File Path** | `data/processed/train_telco_churn.csv` |
| **Source / Lineage** | Stratified 80% partition derived from `data/raw/raw_telco_churn.csv` (`random_state=42`) |
| **Licensing Terms** | Inherited from raw source (Academic / Educational Prototyping) |
| **Refresh Date** | 2026-10-28 |
| **Data Dimensions** | 5,634 observations × 24 variables |
| **Usage Scope** | Baseline model training (Diyasha Roy) |

### Dataset Metadata: `test_telco_churn.csv`

| Property | Specification Details |
| :--- | :--- |
| **File Path** | `data/processed/test_telco_churn.csv` |
| **Source / Lineage** | Stratified 20% holdout partition derived from `data/raw/raw_telco_churn.csv` (`random_state=42`) |
| **Licensing Terms** | Inherited from raw source (Academic / Educational Prototyping) |
| **Refresh Date** | 2026-10-28 |
| **Data Dimensions** | 1,409 observations × 24 variables |
| **Usage Scope** | Independent model evaluation and metrics benchmarking |

### Artifact Metadata: `input_encoders.pkl`

| Property | Specification Details |
| :--- | :--- |
| **File Path** | `data/processed/input_encoders.pkl` |
| **Source / Lineage** | Serialized Scikit-Learn `LabelEncoder` objects fitted strictly on training partition features (`X_train`) |
| **Licensing Terms** | Internal project deliverable |
| **Refresh Date** | 2026-10-28 |
| **Format** | Binary Python Pickle file (`joblib` serialized dictionary) |
| **Usage Scope** | Categorical value transformation in Streamlit frontend (Surajit Tunga — Task APP-01) and model inference |

---

## 4. Pipeline Quality & Validation Protocols

Every dataset progression from `data/raw/` to `data/processed/` must pass four validation gates executed within the data preprocessing pipeline:

### A. Schema & Data Type Validation
* **Data Type Alignment:** Inspected all 33 initial attributes. Continuous financial attributes, such as `Monthly Charges` and `Total Charges`, were verified for numerical typing.
* **Feature Pruning:** Administrative identifiers and static location indicators (`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `CLTV`) were removed to prevent spurious feature correlation and overfitting.

### B. Missing Value Identification & Coercion
* **Whitespace Bug Resolution:** Identified 11 records in `Total Charges` containing whitespace string characters (`" "`) corresponding strictly to new accounts with `Tenure Months == 0`.
* **Standardized Imputation:** Applied string-to-numeric coercion via `pd.to_numeric(df['Total Charges'], errors='coerce')` and imputed the resulting `NaN` values with `0.0`.
* **Null Audit:** Confirmed 0 remaining missing values across all columns prior to modeling.

### C. Duplicate Record Verification
* **Uniqueness Check:** Evaluated raw records across unique customer keys (`CustomerID`).
* **Audit Result:** 0 duplicate rows were detected, confirming complete record integrity across all 7,043 instances.

### D. Target Leakage & Data Isolation Controls
To ensure objective model performance and prevent artificial metric inflation:
1. **Post-Event Feature Isolation:** Removed `Churn Reason` (qualitative survey feedback captured exclusively post-cancellation) and `Churn Score` (an upstream prediction generated by IBM SPSS Modeler).
2. **Encoder Isolation:** `LabelEncoder` instances were fitted **strictly on the training partition (`X_train`)** and subsequently applied to transform the test set (`X_test`). This guarantees that no categorical distribution parameters leak from the test holdout into model training.

---

## 5. Security, Licensing & Compliance Directives

1. **PII and Privacy Compliance:**
   * The IBM Telco dataset utilizes synthetic customer profiles and generated numbers. No real-world customer names, addresses, or financial credentials exist within these files.
2. **Commit Policy for Sensitive or Licensed Data:**
   * **Rule:** Do not commit proprietary, commercial, or sensitive data to this repository unless explicit written permission and license terms permit public hosting.
   * **Scope Boundary:** Only datasets explicitly cleared under open licenses or non-PII educational terms (such as the current IBM Telco sample) may reside in `data/raw/` or `data/processed/`.
   * **Credentials Protection:** API keys, access tokens, and environment variables (e.g., `Kaggle API` keys) must never be committed to git or exposed within notebook cells.
