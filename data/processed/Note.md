### Processed Data & Pipeline Documentation

#### 1. Processed Deliverables & File Index
| Output File / Artifact | Dimensions / Format | Description & Role | Downstream Owner |
| ------ | ------ | ------ | ------ |
| `train_telco_churn.csv` | 5,634 rows × 35 columns | Stratified 80% training partition (34 feature columns + 1 target `Churn Value`) | Diyasha Roy (Baseline Modelling & Validation) |
| `test_telco_churn.csv` | 1,409 rows × 35 columns | Stratified 20% holdout test partition (34 feature columns + 1 target `Churn Value`) | Diyasha Roy (Independent Benchmark) |
| `preprocessing.pkl` | Binary Pickle File (`joblib`) | Serialized Scikit-Learn `ColumnTransformer` (`OneHotEncoder(drop='first')` + `StandardScaler()`) fitted strictly on `X_train` | Surajit Tunga (Streamlit App Integration) |

--------------------------------------------------------------------------------

#### 2. Preprocessing & Transformation Pipeline Summary
All data preparation and feature transformations were executed reproducibly in Python (`pandas`, `scikit-learn`, `joblib`):

##### A. Schema Cleaning & Data Minimization
*   **String Coercion & Imputation:** Converted continuous financial field `Total Charges` from string to numeric float (`pd.to_numeric(errors='coerce')`) and imputed 11 blank entries (corresponding to new accounts with `Tenure Months == 0`) with `0.0`.
*   **Administrative, Leakage & Noise Removal:** Pruned 15 non-predictive administrative, spatial, target-leakage, and noisy attributes (`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `Churn Label`, `Churn Reason`, `Churn Score`, `CLTV`, `Gender`, `Phone Service`).
    *   *Spatial Data Minimization:* Dropping granular spatial fields (`City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`) eliminates re-identification risks and prevents spatial bias.
    *   *Noise Pruning:* `Gender` (Cramér's V = 0.01, p = 0.487) and `Phone Service` (Cramér's V = 0.01, p = 0.339) were pruned to eliminate non-predictive noise.

##### B. Feature Engineering, Validation & Multicollinearity Control
1.  **`Tenure_Cohort`:** Continuous customer lifespan categorized into behavioral cohorts (`Cohort_0_12`, `Cohort_13_24`, `Cohort_25_48`, `Cohort_49_72`) via inclusive boundary logic (`get_tenure_cohort`). All 11 zero-tenure accounts (`Tenure Months == 0`) are assigned to `Cohort_0_12` (`<= 12` months) with zero `NaN` values.
2.  **`Total_Services`:** Composite count of active subscribed add-on internet services (`Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`).
3.  **`Log_Monthly_Tenure_Ratio`:** Financial burden index computed as `np.log1p(Monthly Charges / (Tenure Months + 1))`. Applying `log1p` eliminates upper-tail skewness (ratio values range 1.6–100+) and stabilizes Variance Inflation Factor (VIF) to **4.11** (well under the 5.0 threshold).
4.  **Add-On Service Level Preservation:** All 6 add-on service columns are encoded with distinct `'No internet service'` dummy levels under `OneHotEncoder(drop='first')`, preserving the empirical low-churn baseline (~7.4%) of non-internet subscribers.
5.  **Empirical Validation:** 5-fold cross-validation ablation confirmed that adding engineered features improves baseline Logistic Regression ROC-AUC from **0.8577 to 0.8614 (+0.37%)** and Recall from **0.8133 to 0.8234 (+1.01%)**.

##### C. Partitioning, Categorical Encoding & Calibration Directives
*   **Stratified Train-Test Split:** Partitioned using `train_test_split(test_size=0.20, random_state=42, stratify=y)`, preserving target balance (~26.54% churn prevalence) across 5,634 training records and 1,409 testing records.
*   **Leakage-Free ColumnTransformer:** Replaced legacy `LabelEncoder` loops with a unified Scikit-Learn `ColumnTransformer` (`OneHotEncoder(drop='first')` + `StandardScaler()`), fitted **strictly on the training partition (`X_train`)** and applied to transform `X_test` via `preprocessor.transform(X_test)`.
*   **Artifact Serialization:** The fitted pipeline is exported to `data/processed/preprocessing.pkl` using `joblib.dump()`. In Surajit's Streamlit frontend (`app/main.py`), user form inputs (22 raw predictor features) are transformed into the **34-column model feature matrix** ($X$) in a single call.
*   **Probability Calibration Directive:** Raw output probabilities from `class_weight='balanced'` classifiers suffer from score distortion and must undergo post-hoc calibration (**Platt Scaling** via `CalibratedClassifierCV(method='sigmoid')`) before risk percentages are displayed in the Streamlit UI.

--------------------------------------------------------------------------------

#### 3. Split & Versioning Conventions
*   **Random Seed:** `random_state=42` enforced across all dataset splitting operations.
*   **Index Persistence:** CSV files are exported with `index=False` to prevent index column proliferation.
*   **Handoff Contract:** The processed dataset schema (22 raw predictor features expanding to **34 encoded predictor features**, plus 1 target `Churn Value` = **35 total columns** in CSV splits) is frozen for model comparison (Diyasha Roy) and Streamlit app integration (Surajit Tunga).
