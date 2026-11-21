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
| **Licensing Terms** | Apache License, Version 2.0 |
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
| **Licensing Terms** | Inherited from Apache License, Version 2.0 |
| **Refresh Date** | 2026-10-28 |
| **Data Dimensions** | 5,634 observations × 36 variables |
| **Usage Scope** | Baseline model training (Diyasha Roy) |

### Dataset Metadata: `test_telco_churn.csv`

| Property | Specification Details |
| :--- | :--- |
| **File Path** | `data/processed/test_telco_churn.csv` |
| **Source / Lineage** | Stratified 20% holdout partition derived from `data/raw/raw_telco_churn.csv` (`random_state=42`) |
| **Licensing Terms** | Inherited from Apache License, Version 2.0 |
| **Refresh Date** | 2026-10-28 |
| **Data Dimensions** | 1,409 observations × 36 variables |
| **Usage Scope** | Independent model evaluation and metrics benchmarking |

### Artifact Metadata: `preprocessing.pkl` 

| Property | Specification Details | 
| :--- | :--- | 
| **File Path** | `data/processed/preprocessing.pkl` | 
| **Source / Lineage** | Serialized Scikit-Learn `ColumnTransformer` pipeline combining `OneHotEncoder(drop='first')` for nominal categorical features and `StandardScaler()` for continuous features, fitted strictly on training partition features (`X_train`) | 
| **Licensing Terms** | Internal project deliverable (Apache 2.0 compliant) | 
| **Refresh Date** | 2026-11-20 | 
| **Format** | Binary Python Pickle file (`joblib` serialized `ColumnTransformer` object) | 
| **Usage Scope** | Unified feature transformation for model inference and Streamlit frontend UI integration (Surajit Tunga — Task APP-01); transforms 22 raw inputs into the 36-column model feature matrix in a single call | 

---

## 4. Pipeline Quality & Validation Protocols

Every dataset progression from `data/raw/` to `data/processed/` must pass four validation gates executed within the data preprocessing pipeline:

### A. Schema & Data Type Validation
* **Data Type Alignment:** Inspected all 33 initial attributes. Continuous financial attributes, such as `Monthly Charges` and `Total Charges`, were verified for numerical typing.
* **Feature Pruning:** Administrative identifiers, static location indicators, and non-predictive attributes (`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `CLTV`, `Gender`, `Phone Service`) were removed to prevent spurious feature correlation, eliminate noise, and enforce privacy standards.
* **Spatial Data Minimization:** High-granularity location indicators (`Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`) were pruned under an explicit data minimization directive. This eliminates re-identification risks and prevents spatial bias in baseline classifiers.

### B. Missing Value Identification & Coercion
* **Whitespace Bug Resolution:** Identified 11 records in `Total Charges` containing whitespace string characters (`" "`) corresponding strictly to new accounts with `Tenure Months == 0`.
* **Standardized Imputation:** Applied string-to-numeric coercion via `pd.to_numeric(df['Total Charges'], errors='coerce')` and imputed the resulting `NaN` values with `0.0`.
* **Null Audit:** Confirmed 0 remaining missing values across all columns prior to modeling.
* **Zero-Tenure Cohort Allocation:** Implemented `get_tenure_cohort(months)` using `months <= 12`, ensuring all 11 new sign-up accounts (`Tenure Months == 0`) are assigned to `Cohort_0_12` with 0 `NaN` errors and continuous boundary coverage.

### C. Duplicate Record Verification
* **Uniqueness Check:** Evaluated raw records across unique customer keys (`CustomerID`).
* **Audit Result:** 0 duplicate rows were detected, confirming complete record integrity across all 7,043 instances.

### D. Target Leakage & Data Isolation Controls
To ensure objective model performance and prevent artificial metric inflation:
1. **Post-Event Feature Isolation:** Removed `Churn Reason` (qualitative survey feedback captured exclusively post-cancellation) and `Churn Score` (an upstream prediction generated by IBM SPSS Modeler).
2. **Preprocessor Isolation & Zero Data Leakage:** The unified `ColumnTransformer` pipeline (incorporating `OneHotEncoder` and `StandardScaler`) was fitted **strictly on the training partition (`X_train`)** and subsequently applied to transform the test holdout (`X_test`). This guarantees that neither categorical category levels nor continuous scaling parameters (mean and variance) leak from the test set into model training.

### E. Feature Engineering & Multicollinearity Validation 
To ensure engineered risk indicators improve model performance without introducing severe multicollinearity to linear models: 
* **Ablation Validation (5-Fold Cross-Validation):** Evaluated baseline Logistic Regression with and without engineered features (`Tenure_Cohort`, `Total_Services`, `Log_Monthly_Tenure_Ratio`). Including engineered features increased ROC-AUC from **0.8577 to 0.8614 (+0.37%)** and Recall from **0.8133 to 0.8234 (+1.01%)**. 
* **Multicollinearity Mitigation (VIF Diagnostic):** Applied a log-1p transformation (`Log_Monthly_Tenure_Ratio = log(1 + Ratio)`), stabilizing its Variance Inflation Factor (VIF) to **4.11** (well below the 5.0 threshold) and `Total_Services` VIF to **2.91**.

---

## 5. Security, Licensing & Compliance Directives

1. **PII and Privacy Compliance:**
   * Formally verified against IBM Cognos Analytics sample dataset documentation: raw materials consist strictly of synthetic customer profiles containing zero real-world names, street addresses, or financial identifiers.
2. **Commit Policy for Sensitive or Licensed Data:**
   * **Rule:** Do not commit proprietary, commercial, or sensitive data to this repository unless explicit written permission and license terms permit public hosting.
   * **Scope Boundary:** Only datasets explicitly cleared under open licenses or non-PII educational terms (such as the current IBM Telco sample) may reside in `data/raw/` or `data/processed/`.
   * **Credentials Protection:** API keys, access tokens, and environment variables (e.g., `Kaggle API` keys) must never be committed to git or exposed within notebook cells.
3. **Licensing & Governance Compliance** 
   * **License Terms:** Governed under the **Apache License, Version 2.0** (`LICENSE.txt` located at root).
   * **Upstream Attribution:** Original raw sample dataset authored by Steven Macko (Copyright 2019 IBM Corporation). 
   * **Commercial Prototyping:** Fully permitted under Section 2 & 3 of Apache 2.0; derivative pipeline scripts and serialized `preprocessing.pkl` artifacts are open for internal application deployment.