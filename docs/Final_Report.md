
---

### **PHASE 1: Aryan's**

---
### **PHASE 2: Data Pipeline & Analytical Risk Indicators**

#### **Objectives**
* **Establish a complete data pipeline**, process the dataset, and perform **feature engineering** to support baseline model training.

---

#### **Key Activities**
* **Governance Audit:** Documented IBM Cognos dataset provenance and verified **Apache 2.0 open-source license** compliance.
* **Cleaning & Safeguards:** Coerced `Total Charges` to numeric float (imputing blank `" "` values with `0.0`), pruned **9 administrative and spatial attributes**, and removed **target-leakage variables** (`Churn Reason` and `Churn Score`).
* **Feature Engineering:** Derived three subscriber risk metrics—**`Tenure_Cohort`**, **`Total_Services`**, and **`Monthly_to_Tenure_Ratio`**—to model non-linear churn propensity.
* **Partitioning & Serialization:** Created an **80/20 stratified train-test split** and serialized `LabelEncoder` objects fitted strictly on training data.

---

#### **Owners**
* **Aazeen Neyaz** (Data Preparation and Risk Analytics — Member ID: **BIZ-02**).

---

#### **Deliverables**
* **`LICENSE.txt`** (Apache License 2.0 root repository text).
* **Reproducible Preprocessing Module:** `data_preprocessing_and_feature_engineering.ipynb`.
* **Stratified Analytical CSV Splits:** `train_telco_churn.csv` (5,634 rows × 24 columns) and `test_telco_churn.csv` (1,409 rows × 24 columns).
* **Serialized Categorical Encoders:** Binary pickle file `input_encoders.pkl` containing fitted training-set `LabelEncoder` objects.

*Note: Detailed report provided at `section_2_data_preprocessing_and_feature_engineering` in folder : `docs/`.*

---

### **PHASE 3: Diyasha's**

---

### **PHASE 4: Sovan's**

---

### **PHASE 5: Surajit's**

---