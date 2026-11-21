
---

### **PHASE 1: Aryan's**

---
### **PHASE 2: Data Pipeline & Analytical Risk Indicators**

#### **Objectives**
- **Establish a reproducible, leakage-free data pipeline**, process the dataset, and perform **empirically validated feature engineering** to support baseline model training and app integration.

#### **Key Activities**
- **Governance Audit:** Documented IBM Cognos dataset provenance (`Copyright 2019 IBM Corporation`) and verified **Apache 2.0 open-source license** compliance across academic, prototyping, and derivative distribution scopes.
- **Cleaning & Safeguards:** Coerced `Total Charges` to numeric float (imputing 11 blank `" "` values with `0.0`), pruned **15 non-predictive/leakage columns** (9 spatial/administrative identifiers, 2 target-leakage variables `Churn Reason` and `Churn Score`, 2 redundant/upstream metrics `Churn Label` and `CLTV`, and 2 statistically non-predictive features `Gender` and `Phone Service`), and documented **Platt Scaling** probability calibration safeguards.
- **Feature Engineering & Validation:** Derived three subscriber risk metrics—**`Tenure_Cohort`**, **`Total_Services`**, and **`Log_Monthly_Tenure_Ratio`** (log-1p transformed financial burden ratio)—validated via 5-fold cross-validation ablation (+0.37% ROC-AUC, +1.01% Recall) and VIF diagnostics (VIF = 4.11).
- **Partitioning & Unified Serialization:** Created an **80/20 stratified train-test split** (`random_state=42`) and unified `StandardScaler` and `OneHotEncoder(drop='first')` inside a Scikit-Learn **`ColumnTransformer`** fitted strictly on training data.

#### **Owners**
- **Aazeen Neyaz** (Business Strategy & Data Engineering Lead — Task ID: **BIZ-02**).

#### **Deliverables**
- **`LICENSE.txt`** (Apache License 2.0 root repository text).
- **Reproducible Preprocessing Module:** `src/data_preprocessing_and_feature_engineering.ipynb`.
- **Stratified Analytical CSV Splits:** `data/processed/train_telco_churn.csv` (5,634 rows × 35 columns) and `data/processed/test_telco_churn.csv` (1,409 rows × 35 columns).
- **Serialized Unified Preprocessor:** Binary pickle file `data/processed/preprocessing.pkl` containing the fitted training-set `ColumnTransformer` pipeline.

*Note: Detailed report provided at `section_2_data_preprocessing_and_feature_engineering_report.md` in folder: `docs/`.*

---

### **PHASE 3: Diyasha's**

---

### **PHASE 4: Sovan's**

---

### **PHASE 5: Surajit's**

---