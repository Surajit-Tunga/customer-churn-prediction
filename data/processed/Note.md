# Processed Data 

## 1. Processed Deliverables & File Index
| Output File / Artifact | Dimensions / Format | Description & Role | Downstream Owner |
| :--- | :--- | :--- |:--- |
| `train_telco_churn.csv` | 5,634 rows x 24 columns | Stratified 80% training partition | Diyasha (Baseline Modelling) |
| `test_telco_churn.csv` | 1,409 rows x 24 columns | Stratified 20% holdout test partition | Diyasha (Model Validation) |
| `input_encoders.pkl` | Binary Pickle File (`joblib`) | Dictionary of fitted `LabelEncoder` objects | Surajit (Streamlit App Integration) |

---

## 2. Preprocessing & Transformation Pipeline Summary

All data preparation and feature transformations were executed reproducibly in Python (`pandas`, `scikit-learn`, `joblib`):

### A. Schema Cleaning & Quality Assurance
* **String Coercion & Imputation:** Converted continuous financial field `Total Charges` from string to numeric float (`pd.to_numeric(errors='coerce')`) and imputed 11 blank entries with `0.0`.
* **Administrative & Leakage Removal:** Pruned 13 non-predictive administrative, geographic, and target-leakage attributes (`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `Churn Label`, `Churn Reason`, `Churn Score`, `CLTV`).

### B. Feature Engineering & Risk Indicators
1. **`Tenure_Cohort`:** Categorical binning of customer lifespan into risk stages (`Cohort_0_12`, `Cohort_13_24`, `Cohort_25_48`, `Cohort_49_72`).
2. **`Total_Services`:** Vectorized sum of active subscribed value-added services (`Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`).
3. **`Monthly_to_Tenure_Ratio`:** Financial burden index computed as `Monthly Charges / (Tenure Months + 1)`.

### C. Partitioning & Leakage-Free Categorical Encoding
* **Stratified Train-Test Split:** Partitioned using
`train_test_split(test_size=0.20, random_state=42, stratify=y)`, preserving the target balance (~26.54% churn) across 5,634 training records and 1,409 testing records.
* **Categorical Encoding Isolation:** `LabelEncoder` instances were fitted **strictly on the training partition (`X_train`)** and then applied to transform the test set (`X_test`), preventing information leakage from holdout samples.
* **Artifact Serialization:** All fitted encoders were serialized into `input_encoders.pkl` using `joblib.dump()` to ensure inference-time transformations in the Streamlit app match training parameters.

---

## 3. Split & Versioning Conventions
* **Random Seed:** `random_state=42` across all split operations.
* **Index Persistence:** CSV files are exported with `index=False` to prevent index column proliferation.
* **Handoff Contract:** The processed dataset schema (23 predictive features + 1 target `Churn Value`) is frozen for model comparison and Streamlit app integration.