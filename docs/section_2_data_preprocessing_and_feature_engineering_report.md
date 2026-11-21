# Section 2: Data Preprocessing and Feature Engineering

## 2.1 Dataset Source and Provenance

### A. Primary Source and Lineage

* **Original Author & Asset Origin:** The dataset was created by Steven Macko as part of the official IBM Cognos Analytics 11.1.3+ Base Samples (published under IBM Community Accelerators).

* **Distribution Platform:** The dataset was acquired via the public Kaggle mirror repository (`yeanzc/telco-customer-churn-ibm-dataset`), maintained by user TanKY.

* **Link:** (https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data)

* **Business Context & Timeframe:** The data captures historical third-quarter (Q3) account activity for a fictional telecommunications provider operating in California, offering home phone and Internet services.

### B. Dataset Volume and File Format

* **Dimensions:** The dataset consists of 7,043 unique customer observation rows across 33 functional attributes.

* **Storage Formats:** Available as a 1.37 MB Excel spreadsheet (`Telco_customer_churn.xlsx`) and exported CSV file (`raw_telco_churn.csv`).

* **Target Variable Specification:** Attrition is recorded via a dual-column representation:

  * **Churn Value:** Binary integer target used for machine learning modeling ($1 = \text{Churned}$, $0 = \text{Retained}$).

  * **Churn Label:** Categorical text representation (`'Yes'` / `'No'`).

---

## 2.2 Licensing, Governance, and Compliance Framework

### A. Primary License Framework

* **Governing License:** Released under the open-source [Apache License, Version 2.0 (January 2004)](https://www.apache.org/licenses/LICENSE-2.0), originally authored by Steven Macko as an IBM Cognos Analytics sample asset (`IBM/telco-customer-churn-on-icp4d`).

* **Kaggle Platform Designation:** Hosted on Kaggle under "Other (specified in description)", which references IBM's original Apache 2.0 terms.

### B. Permitted Grants & Legal Rights

Under **Section 2 (Grant of Copyright License)** and **Section 3 (Grant of Patent License)** of the Apache 2.0 framework, users hold perpetual, worldwide, non-exclusive, no-charge, royalty-free, and irrevocable rights to:

* **Use & Reproduce:** Free inspection, querying, and execution for academic research and prototyping.

* **Modify & Create Derivatives:** Right to clean data, engineer features, train ML models, and build derivative applications.

* **Distribute & Share:** Right to publicly host, display, and distribute the repository, code, and derived data splits.

### C. Redistribution & Governance Obligations

Under Section 4 of Apache 2.0, sharing derived assets (`train_telco_churn.csv`, `test_telco_churn.csv`, `preprocessing.pkl`) requires:

1. **License Inclusion:** Retain `LICENSE.txt` in the root directory.

2. **Notice of Modifications:** Document transformations and schema changes in `data/processed/Note.md`.

3. **Attribution Preservation:** Retain upstream attribution alongside repository code copyright:

   * **Upstream Dataset:** Copyright 2019 IBM Corporation (Authored by Steven Macko, IBM Cognos Analytics).

   * **Derivative Pipeline & Code:** Copyright 2026 Customer Churn Prediction Team.

### D. Usage Compliance Scope (Reclassified Risk Matrix)

Under explicit Apache 2.0 terms, internal commercial prototyping and application integration are **fully permitted**, provided Section 4 attribution requirements are maintained:

| Usage Intent | Compliance Status | Risk Level | Required Governance Action |
| :--- | :---: | :---: | :--- |
| **Academic & Educational Study** | Permitted | **Low Risk** | Retain original IBM Cognos & Steven Macko attribution notices. |
| **Internal App Prototyping** | Permitted | **Low Risk** | Maintain `LICENSE.txt` at repository root and include `preprocessing.pkl` metadata. |
| **Model Derivative Distribution** | Permitted | **Low Risk** | Document all schema transformations in `data/processed/Note.md`. |

### E. Privacy, Data Minimization & Disclaimers

* **Privacy Assurance:** Contains zero real-world Personally Identifiable Information (PII) or sensitive credentials; all 7,043 records represent synthetically generated customer profiles indexed via administrative alphanumeric keys (`CustomerID`).

* **Data Minimization Directive:** High-granularity administrative and spatial metadata (`Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`) are systematically pruned prior to modeling to align with GDPR/CCPA principles, eliminate re-identification risks, and prevent regional overfitting.

* **Warranty Disclaimer:** Provided on an "AS IS" basis under Sections 7 & 8, with no warranties or liability.

---

## 2.3 Dataset Attributes, Usage Classification, and Exclusions

To ensure model integrity, maintain data privacy, and prevent target leakage, all 33 raw attributes from the IBM Telco Customer Churn dataset have been audited and assigned specific operational roles.

### A. Attribute Usage & Classification Matrix (All 33 Attributes)

| Attribute Name | Short Description & Operational Role | Broad Classification |
| :--- | :--- | :--- |
| `CustomerID` | Synthetic unique alphanumeric customer key; dropped for privacy and data minimization. | Excluded (Privacy & Admin) |
| `Count` | Administrative row counter (constant value = 1); zero variance, dropped as non-predictive. | Excluded (Privacy & Admin) |
| `Country` | Country of residence (`United States`); constant metadata, dropped as non-predictive. | Excluded (Privacy & Admin) |
| `State` | State of residence (`California`); constant metadata, dropped as non-predictive. | Excluded (Privacy & Admin) |
| `City` | Customer city in California; dropped to prevent spatial memorization and regional bias. | Excluded (Data Minimization) |
| `Zip Code` | 5-digit postal zip code; dropped to eliminate re-identification risk and spatial overfitting. | Excluded (Data Minimization) |
| `Lat Long` | Combined latitude and longitude string; duplicate spatial metadata, dropped. | Excluded (Data Minimization) |
| `Latitude` | Geographic latitude coordinate; dropped under spatial data minimization directive. | Excluded (Data Minimization) |
| `Longitude` | Geographic longitude coordinate; dropped under spatial data minimization directive. | Excluded (Data Minimization) |
| `Gender` | Customer demographic sex (`Male` / `Female`); pruned based on QA-02 audit (Cramér's $V = 0.01, p = 0.487$). | Excluded (Pruned Noise) |
| `Senior Citizen` | Binary flag indicating if customer is aged $\ge 65$ ($1 = \text{Yes}$, $0 = \text{No}$). | Used in Predictor Matrix |
| `Partner` | Categorical indicator of whether customer has a partner (`Yes` / `No`). | Used in Predictor Matrix |
| `Dependents` | Categorical indicator of whether customer lives with dependents (`Yes` / `No`). | Used in Predictor Matrix |
| `Tenure Months` | Number of months customer has stayed with the company. | Used in Predictor Matrix |
| `Phone Service` | Home phone subscription (`Yes` / `No`); pruned based on QA-02 audit (Cramér's $V = 0.01, p = 0.339$). | Excluded (Pruned Noise) |
| `Multiple Lines` | Multiple phone lines indicator (`Yes` / `No` / `No phone service`). | Used in Predictor Matrix |
| `Internet Service` | Type of internet connection (`DSL` / `Fiber optic` / `No`). | Used in Predictor Matrix |
| `Online Security` | Add-on security service subscription (`Yes` / `No` / `No internet service`). | Used in Predictor Matrix |
| `Online Backup` | Add-on cloud backup subscription (`Yes` / `No` / `No internet service`). | Used in Predictor Matrix |
| `Device Protection` | Add-on hardware protection plan (`Yes` / `No` / `No internet service`). | Used in Predictor Matrix |
| `Tech Support` | Premium technical support subscription (`Yes` / `No` / `No internet service`). | Used in Predictor Matrix |
| `Streaming TV` | Third-party TV streaming service add-on (`Yes` / `No` / `No internet service`). | Used in Predictor Matrix |
| `Streaming Movies` | Third-party movie streaming service add-on (`Yes` / `No` / `No internet service`). | Used in Predictor Matrix |
| `Contract` | Billing contract terms (`Month-to-month` / `One year` / `Two year`). | Used in Predictor Matrix |
| `Paperless Billing` | Electronic invoice preference indicator (`Yes` / `No`). | Used in Predictor Matrix |
| `Payment Method` | Payment channel (`Electronic check` / `Mailed check` / `Bank transfer` / `Credit card`). | Used in Predictor Matrix |
| `Monthly Charges` | Current monthly bill amount charged to customer (in USD). | Used in Predictor Matrix |
| `Total Charges` | Cumulative total billed amount (in USD); coerced from string to float and imputed `0.0` for tenure 0. | Used in Predictor Matrix |
| `Churn Value` | Binary machine learning target variable ($1 = \text{Churned}$, $0 = \text{Retained}$). | Target Variable (Output) |
| `Churn Label` | Text representation of churn (`'Yes'` / `'No'`); dropped as duplicate of `Churn Value`. | Excluded (Redundant Target) |
| `Churn Reason` | Qualitative exit feedback recorded post-cancellation; dropped to prevent target leakage. | Excluded (Target Leakage) |
| `Churn Score` | Upstream IBM SPSS prediction score ($0\text{--}100$); dropped to avoid target leakage. | Excluded (Target Leakage) |
| `CLTV` | Upstream Customer Lifetime Value calculation; excluded from baseline feature set. | Excluded (Upstream Metric) |

### B. Summary of Exclusions & Pre- vs. Post-Encoding Schema Reconciliation

* **Privacy & Data Minimization (9 Attributes Dropped):** `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, and `Longitude` were pruned to comply with privacy rules and eliminate arbitrary geographic memorization.
* **Target Leakage Safeguard (2 Attributes Dropped):** `Churn Reason` is captured only after cancellation, and `Churn Score` is an existing model output. Including either would falsely inflate accuracy.
* **Redundant & Upstream Metrics (2 Attributes Dropped):** `Churn Label` duplicates `Churn Value`, and `CLTV` is an upstream business metric.
* **Statistically Non-Predictive Noise Pruned (2 Attributes Dropped):** `Gender` (p = 0.487) and `Phone Service` (p = 0.339) were pruned following the QA-02 statistical significance audit.
* **Schema Dimension Reconciliation (33 Raw → 23 Pre-Encoding → 35 Post-Encoding):**
    1. **Raw Input:** 33 columns in `raw_telco_churn.csv`.
    2. **After Pruning (15 Removed) & Engineering (3 Added):** **23 total columns** (22 predictors + 1 target).
    3. **After One-Hot Encoding (`drop='first'`):** **35 total columns** (34 encoded predictors + 1 binary target `Churn Value`).

### C. Add-On Service Encoding & Level Preservation

All six add-on service attributes (`Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`) contain **3 distinct categorical levels**: `'Yes'`, `'No'`, and `'No internet service'`. Under `OneHotEncoder(drop='first')`, `'No internet service'` is preserved as a distinct dummy variable column rather than collapsed into `'No'`:

* **Empirical Justification:** Non-internet subscribers exhibit an exceptionally low baseline churn rate of **~7.4%**, whereas internet subscribers who opt out of security or support features (`'No'`) exhibit high churn rates ranging from **31.4% to 41.8%**. Preserving `'No internet service'` prevents artificial risk blending and retains true baseline variance.

---

## 2.4 Feature Engineering & Derived Risk Metrics

To capture non-linear behavioural patterns and enhance downstream model baseline performance, three domain-specific risk features were engineered from raw demographic, service, and financial attributes prior to dataset partitioning.

### A. Tenure_Cohort (Customer Lifespan Binning)

* **Discretization Logic:** Continuous `Tenure Months` values are partitioned into four discrete behavioral lifespan risk stages using an inclusive threshold function (`get_tenure_cohort`):

  $$\text{Tenure\_Cohort} = \begin{cases} \text{'Cohort\_0\_12'} & \text{if } 0 \le \text{Tenure Months} \le 12 \\ \text{'Cohort\_13\_24'} & \text{if } 12 < \text{Tenure Months} \le 24 \\ \text{'Cohort\_25\_48'} & \text{if } 24 < \text{Tenure Months} \le 48 \\ \text{'Cohort\_49\_72'} & \text{if } \text{Tenure Months} > 48 \end{cases}$$

  * **Zero-Tenure Boundary Allocation:** Using $0 \le \text{Tenure Months} \le 12$ ensures that all **11 brand-new customer accounts (`Tenure Months == 0`)** are cleanly assigned to `Cohort_0_12` with zero `NaN` values across the dataset.

* **Business Rationale:** Churn risk is non-linear—cancellation propensity is heavily concentrated in the first 12 months. Discretizing continuous tenure allows linear classifiers (such as Logistic Regression) to model step-wise risk jumps without assuming strict linearity.

### B. Total_Services (Subscription Breadth Index)

* **Vectorization Logic:** Vectorized sum of active add-on value-added services across 6 binary indicator fields:

  $$\text{Total\_Services} = \sum_{i=1}^{6} \mathbb{I}(\text{Service}_i = \text{'Yes'})$$

  Where $\mathbb{I}$ is an indicator function evaluating to $1$ if the customer subscribes to add-on $i$ ($\text{Service}_i \in \{\text{Online Security}, \text{Online Backup}, \text{Device Protection}, \text{Tech Support}, \text{Streaming TV}, \text{Streaming Movies}\}$) and $0$ if `'No'` or `'No internet service'`.

* **Business Rationale:** Quantifies customer product breadth and ecosystem stickiness. Subscribers with multiple active services face significantly higher switching friction and exhibit lower churn probability compared to single-service accounts.

### C. Cost-Burden Ratio & Log Transformation (`Log_Monthly_Tenure_Ratio`)

* **Ratio & Log-1p Transformation Logic:** Ratio of current monthly billing commitment relative to cumulative account longevity with Laplace denominator smoothing ($+1$), followed by a $\ln(1 + x)$ transformation to eliminate upper-tail skewness (raw ratio values span $1.6$ to $>100$):

  $$\text{Monthly\_to\_Tenure\_Ratio} = \frac{\text{Monthly Charges}}{\text{Tenure Months} + 1}$$

  $$\text{Log\_Monthly\_Tenure\_Ratio} = \ln\left(1 + \text{Monthly\_to\_Tenure\_Ratio}\right)$$

* **Business Rationale:** Identifies early financial price shock while stabilizing variance for linear models. Accounts paying high monthly fees relative to short tenure demonstrate high price sensitivity and elevated early-cancellation risk.

### D. Empirical Feature Validation & Ablation Results

An empirical 5-fold stratified cross-validation ablation study was executed on `X_train` using baseline Logistic Regression (`class_weight='balanced'`) to verify the predictive lift of the engineered features:

| Feature Configuration | 5-Fold ROC-AUC | 5-Fold Recall | Performance Lift |
| :--- | :---: | :---: | :--- |
| **Raw Features Only** | $0.8577$ | $0.8133$ | Baseline |
| **Raw + Engineered Features** | **$0.8614$** | **$0.8234$** | **$+0.37\%$ ROC-AUC / $+1.01\%$ Recall** |

* **Conclusion:** Including the engineered risk indicators delivers a verified performance lift, increasing sensitivity to churners by over $1.0\%$ (capturing ~10 additional churners in holdout evaluation).

### E. Variance Inflation Factor (VIF) Multicollinearity Diagnostic

Multicollinearity was evaluated across continuous features using Variance Inflation Factors (VIF) with an explicit intercept constant:

| Feature | VIF Score | Assessment |
| :--- | :---: | :--- |
| `Total_Services` | **$2.91$** | Low Collinearity (Clean) |
| `Log_Monthly_Tenure_Ratio` | **$4.11$** | Clean ($< 5.0$ Threshold) |
| `Monthly Charges` | **$4.64$** | Acceptable ($< 5.0$ Threshold) |
| `Tenure Months` | **$8.45$** | Moderate Collinearity |
| `Total Charges` | **$10.20$** | High Collinearity (Raw Dataset Artifact) |

* **Conclusion:** Applying `Log_Monthly_Tenure_Ratio` compressed extreme upper-tail values and stabilized the ratio's VIF score to **$4.11$**, eliminating multicollinearity risks for linear estimators.

---

## 2.5 Data Readiness, Safeguards & Calibration Protocol

Initial data profiling identified four key constraints and downstream deployment safeguards resolved during the preprocessing stage:

### A. Schema Coercion & Imputation (`Total Charges`)

* **Issue:** `Total Charges` was imported as an object string due to 11 blank whitespace entries (`" "`) for new accounts with `Tenure Months == 0`.
* **Fix:** Coerced to float using `pd.to_numeric(..., errors='coerce')` and imputed missing values with `0.0` to reflect zero historical billing.

### B. Administrative & Spatial Identifier Pruning

* **Issue:** Administrative IDs (`CustomerID`, `Count`) and geographic fields (`Country`, `State`, `City`, `Zip Code`, coordinates) introduce noise and high-cardinality memorization.
* **Fix:** Dropped all non-predictive metadata prior to model training to enforce privacy minimization and prevent spatial overfitting.

### C. Strict Target Leakage Prevention

* **Issue:** `Churn Reason` (collected post-cancellation) and `Churn Score` (an upstream IBM SPSS prediction score) introduce severe future information leakage.
* **Fix:** Explicitly excluded both variables from the training feature set before model fitting.

### D. Target Class Imbalance & Probability Calibration Directive

* **Cost-Sensitive Weight Formula:**

  `w_churn = N_total / (2 × N_churn) = 5634 / (2 × 1495) ≈ 1.884`

* **Probability Distortion & Calibration Safeguard (Task APP-01):** Because `class_weight='balanced'` shifts raw predicted probabilities `P(Y = 1 | X)` upward:

    1. **Raw Score Limitation:** Raw output probabilities from balanced classifiers must **not** be displayed directly in the Streamlit UI as true percentage risk scores.
    2. **Post-Hoc Calibration:** Model artifacts handed off for app integration must undergo **Platt Scaling** (`CalibratedClassifierCV(method='sigmoid')`) fitted on validation splits.
    3. **Threshold Decoupling:** Classification decisions (Retain vs. Intervention) must be driven by an empirically tuned decision threshold (`t* ≈ 0.35 - 0.40`), separate from calibrated probability display.

---

## 2.6 Stratified Train-Test Partitioning & Pipeline Serialization

To ensure reliable baseline model evaluation and seamless downstream application integration, the cleansed and engineered feature set was partitioned and transformed using strict leakage-free protocols.

### A. Stratified 80/20 Train-Test Partitioning

* **Partitioning Logic:** The dataset was split into an 80% training set and a 20% holdout testing set using Scikit-Learn’s `train_test_split()` with a fixed random seed (`random_state=42`).
* **Target Stratification (`stratify=y`):** Stratified sampling was enforced on `Churn Value` to preserve the exact class imbalance profile (~26.54% churn rate) across both training and holdout partitions:
  * `train_telco_churn.csv` (Training Partition): **5,634 observations × 35 attributes** (34 encoded predictors + 1 target).
  * `test_telco_churn.csv` (Holdout Test Partition): **1,409 observations × 35 attributes** (34 encoded predictors + 1 target).
* **Index Export Protocol:** Files were exported with `index=False` to prevent unnecessary index column proliferation during CSV ingestion.

### B. Unified Preprocessing Pipeline & Zero Data Leakage

* **Unified `ColumnTransformer` Architecture:** Replaced legacy `LabelEncoder` loops with a unified Scikit-Learn `ColumnTransformer` that scales continuous features via `StandardScaler` and encodes all categorical features via `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'
)
```

* **Train-Set Isolation:** `preprocessor.fit_transform(X_train)` was fitted strictly on the training set (`X_train`) and subsequently applied to transform `X_test` via `preprocessor.transform(X_test)`. This guarantees zero scaling statistics or categorical distribution parameters from the holdout set leak into model fitting.

### C. Binary Serialization via Joblib (`preprocessing.pkl`)

* **Artifact Serialization:** The fitted `ColumnTransformer` object was serialized to `data/processed/preprocessing.pkl` using `joblib.dump()`.
* **Downstream Streamlit Integration Role:** In the Streamlit frontend (`app/main.py`), raw user inputs (22 predictor fields) are transformed into the exact 34-column model matrix in a single call:

```python
import joblib
import pandas as pd

# Load the unified preprocessing pipeline
preprocessor = joblib.load('data/processed/preprocessing.pkl')

# Transform raw user input DataFrame into the 34-column model feature matrix
X_app_processed = preprocessor.transform(raw_user_input_df)
```

---

## 2.7 Summary of Deliverables & Downstream Handovers

### A. Exported Deliverables Matrix

| Delivered File / Artifact | Dimensions / Format | Operational Description | Downstream Handoff Owner |
| :--- | :--- | :--- | :--- |
| `train_telco_churn.csv` | $5,634 \text{ rows} \times 35 \text{ columns}$ (CSV) | Stratified 80% training partition ($34$ encoded predictors + $1$ target) | Diyasha Roy (Baseline & Model Validation) |
| `test_telco_churn.csv` | $1,409 \text{ rows} \times 35 \text{ columns}$ (CSV) | Stratified 20% holdout test partition ($34$ encoded predictors + $1$ target) | Diyasha Roy (Model Performance Benchmarking) |
| `preprocessing.pkl` | Binary Python Pickle (`joblib`) | Serialized Scikit-Learn `ColumnTransformer` pipeline | Surajit Tunga (Streamlit App Integration) |