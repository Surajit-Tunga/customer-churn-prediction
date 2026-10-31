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

  * **Churn Value:** Binary integer target used for machine learning modeling (1 = Churned, 0 = Retained).

  * **Churn Label:** Categorical text representation ('Yes' / 'No').

---

## 2.2 Licensing, Governance, and Compliance Framework

### A. Primary License Framework

* **Governing License:** Released under the open-source [Apache License, Version 2.0 (January 2004)](https://www.apache.org/licenses/LICENSE-2.0), originally authored by Steven Macko as an IBM Cognos Analytics sample asset (`IBM/telco-customer-churn-on-icp4d`).

* **Kaggle Platform Designation:** Hosted on Kaggle under "Other (specified in description)", which references IBM's original Apache 2.0 terms.

### B. Permitted Grants & Legal Rights

Under Sections 2 and 3 of the Apache 2.0 framework, users hold perpetual, royalty-free, worldwide rights to:

* **Use & Reproduce:** Free inspection, querying, and execution for academic research and prototyping.

* **Modify & Create Derivatives:** Right to clean data, engineer features, train ML models, and build applications.

* **Distribute & Share:** Right to publicly host and distribute the repository, code, and derived data splits.

### C. Redistribution & Governance Obligations

Under Section 4 of Apache 2.0, sharing derived assets (`train_telco_churn.csv`, `test_telco_churn.csv`, `input_encoders.pkl`) requires:

1. **License Inclusion:** Retain `LICENSE.txt` in the root directory.

2. **Notice of Modifications:** Document transformations and schema changes in `data/processed/Note.md`.

3. **Attribution Preservation:** Retain upstream attribution alongside repository code copyright:

   * Upstream Dataset: Copyright 2026 IBM Corporation

   * Pipeline & Code: Copyright 2026 Customer Churn Prediction Team

### D.Usage Compliance Scope

| Usage Intent | Status | Risk Level | Required Action | 
| ----- | ----- | ----- | ----- | 
| **Education & Research** | Permitted | Low | Retain IBM attribution & `LICENSE.txt`. | 
| **App Prototyping** | Permitted | Low | Keep code & `input_encoders.pkl` under Apache 2.0. | 
| **Commercial Rollout** | Restricted | High | Conduct formal legal audit of IBM enterprise terms. | 

### E. Privacy, Data Minimization & Disclaimers

* **Privacy Assurance:** Contains zero real-world PII; uses synthetic customer IDs (`CustomerID`).

* **Data Minimization:** Administrative and geographic metadata are pruned prior to modeling.

* **Warranty Disclaimer:** Provided on an "AS IS" basis under Sections 7 & 8, with no warranties or liability.

## 2.3 Dataset Attributes, Usage Classification, and Exclusions

To ensure model integrity, maintain data privacy, and prevent target leakage, all 33 raw attributes from the IBM Telco Customer Churn dataset have been audited and assigned specific operational roles.

### A. Attribute Usage & Classification Matrix (All 33 Attributes)

| Attribute Name | Short Description & Operational Role | Broad Classification | 
| ----- | ----- | ----- | 
| `CustomerID` | Synthetic unique customer alphanumeric ID; dropped for privacy and data minimization. | Excluded (Privacy & Admin) | 
| `Count` | Administrative row counter (constant value = 1); dropped as non-predictive. | Excluded (Privacy & Admin) | 
| `Country` | Country of residence (United States); dropped as non-predictive static location. | Excluded (Privacy & Admin) | 
| `State` | State of residence (California); dropped as non-predictive static location. | Excluded (Privacy & Admin) | 
| `City` | Customer city in California; dropped to prevent spatial overfitting and enforce data minimization. | Excluded (Privacy & Admin) | 
| `Zip Code` | Postal zip code; dropped to avoid spatial memorization and privacy risk. | Excluded (Privacy & Admin) | 
| `Lat Long` | Combined latitude and longitude string; dropped as non-predictive spatial metadata. | Excluded (Privacy & Admin) | 
| `Latitude` | Geographic latitude coordinate; dropped for spatial data minimization. | Excluded (Privacy & Admin) | 
| `Longitude` | Geographic longitude coordinate; dropped for spatial data minimization. | Excluded (Privacy & Admin) | 
| `Gender` | Customer demographic gender (Male / Female). | Used for Model Training | 
| `Senior Citizen` | Binary flag indicating if customer is aged 65 or older (1 = Yes, 0 = No). | Used for Model Training | 
| `Partner` | Categorical indicator of whether customer has a partner (Yes / No). | Used for Model Training | 
| `Dependents` | Categorical indicator of whether customer lives with dependents (Yes / No). | Used for Model Training | 
| `Tenure Months` | Number of months customer has stayed with the company. | Used for Model Training | 
| `Phone Service` | Indicates whether customer subscribes to home phone service (Yes / No). | Used for Model Training | 
| `Multiple Lines` | Indicates if customer has multiple phone lines (Yes / No / No phone service). | Used for Model Training | 
| `Internet Service` | Type of internet connection (DSL / Fiber optic / No). | Used for Model Training | 
| `Online Security` | Add-on security service subscription (Yes / No / No internet service). | Used for Model Training | 
| `Online Backup` | Add-on cloud backup subscription (Yes / No / No internet service). | Used for Model Training | 
| `Device Protection` | Add-on hardware protection plan (Yes / No / No internet service). | Used for Model Training | 
| `Tech Support` | Premium technical support subscription (Yes / No / No internet service). | Used for Model Training | 
| `Streaming TV` | Third-party TV streaming service add-on (Yes / No / No internet service). | Used for Model Training | 
| `Streaming Movies` | Third-party movie streaming service add-on (Yes / No / No internet service). | Used for Model Training | 
| `Contract` | Billing contract terms (Month-to-month / One year / Two year). | Used for Model Training | 
| `Paperless Billing` | Indicates paperless billing adoption (Yes / No). | Used for Model Training | 
| `Payment Method` | Payment channel (Electronic check / Mailed check / Bank transfer / Credit card). | Used for Model Training | 
| `Monthly Charge` | Current monthly bill amount charged to customer (in USD). | Used for Model Training | 
| `Total Charges` | Cumulative total billed amount (in USD); coerced from string to float. | Used for Model Training | 
| `Churn Value` | Binary machine learning target variable (1 = Churned, 0 = Retained). | Target Variable (Output) | 
| `Churn Label` | Text representation of churn (Yes / No); dropped as duplicate of Churn Value. | Excluded (Redundant Target) | 
| `Churn Reason` | Qualitative feedback recorded post-cancellation; dropped to prevent target leakage. | Excluded (Target Leakage) | 
| `Churn Score` | Upstream IBM SPSS prediction score (0–100); dropped to avoid target leakage. | Excluded (Target Leakage) | 
| `CLTV` | Upstream Customer Lifetime Value calculation; excluded from baseline feature set. | Excluded (Upstream Metric) | 

### B. Summary of Exclusions & Selection Rationale

* **Privacy & Data Minimization (9 Attributes Dropped):** `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, and `Longitude` were pruned to comply with privacy rules and eliminate arbitrary geographic memorization.

* **Target Leakage Safeguard (2 Attributes Dropped):** `Churn Reason` is captured only after cancellation, and `Churn Score` is an existing model output. Including either would falsely inflate accuracy.

* **Redundant & Upstream Metrics (2 Attributes Dropped):** `Churn Label` duplicates `Churn Value`, and `CLTV` is an upstream business metric.

* **Retained Predictor Count Verification:** The final predictor matrix contains 23 total features (11 raw demographics/billing + 9 add-on services + 3 engineered risk features) plus 1 target variable (`Churn Value`), matching the output CSV dimensions ($5,634 \times 24$ and $1,409 \times 24$).

## 2.4 Feature Engineering & Derived Risk Metrics

To capture non-linear behavioural patterns and enhance downstream model baseline performance, three domain-specific risk features were engineered from raw demographic, service, and financial attributes prior to categorical encoding.

### A. Tenure_Cohort (Customer Lifespan Binning)

* **Discretization Logic:** Continuous `Tenure Months` values are binned into four discrete lifespan risk stages:

  * `Cohort_0_12` ($0 < \text{Tenure} \le 12$ months): High-risk onboarding and evaluation phase.

  * `Cohort_13_24` ($13 \le \text{Tenure} \le 24$ months): Early loyalty transition phase.

  * `Cohort_25_48` ($25 \le \text{Tenure} \le 48$ months): Mid-term established subscriber phase.

  * `Cohort_49_72` ($49 \le \text{Tenure} \le 72+$ months): Long-term loyal account phase.

* **Business Rationale:** Churn risk is non-linear—cancellation propensity is heavily concentrated in the first 12 months. Discretizing continuous tenure allows linear classifiers (such as Logistic Regression) to model step-wise risk jumps without assuming strict linearity.

### B. Total_Services (Subscription Breadth Index)

* **Vectorization Logic:** Vectorized sum of active add-on value-added services across 6 binary indicator fields:

  $$\text{Total Services} = \sum_{i=1}^{6} \text{Service}_i$$

  Where $\text{Service}_i \in \{\text{Online Security}, \text{Online Backup}, \text{Device Protection}, \text{Tech Support}, \text{Streaming TV}, \text{Streaming Movies}\}$.

  <sup>**Mapping:** For each $\text{Service}_i$, assign $1$ if the value is 'Yes', and $0$ if the value is 'No' or 'No internet service'.</sup>

* **Business Rationale:** Quantifies customer product breadth and ecosystem stickiness. Subscribers with multiple active services face significantly higher switching friction and exhibit lower churn probability compared to single-service accounts.

### C. Monthly_to_Tenure_Ratio (Financial Burden / Price Sensitivity Index)

* **Ratio Logic:** Ratio of current monthly billing commitment relative to cumulative account longevity:

  $$\text{Monthly_to_Tenure_Ratio} = \frac{\text{Monthly Charges}}{\text{Tenure Months} + 1}$$

  *(Note: A Laplace-style $+1$ smoothing factor is added to the denominator to eliminate division-by-zero errors for brand-new accounts where $\text{Tenure Months} = 0$).*

* **Business Rationale:** Identifies early financial price shock. Accounts paying high monthly fees relative to short tenure demonstrate high price sensitivity and elevated early-cancellation risk.

## 2.5 Data Readiness and Safeguards

Initial data profiling identified four key constraints that were resolved during the preprocessing stage:

### A. Schema Coercion & Imputation (Total Charges)

* **Issue:** `Total Charges` was imported as an object string due to 11 blank whitespace entries (`" "`) for new accounts with `Tenure Months == 0`.

* **Fix:** Coerced to float using `pd.to_numeric(..., errors='coerce')` and imputed missing values with `0.0` to reflect zero historical billing.

### B. Administrative & Spatial Identifier Pruning

* **Issue:** Administrative IDs (`CustomerID`, `Count`) and geographic fields (`Country`, `State`, `City`, `Zip Code`, coordinates) introduce noise and high-cardinality memorization.

* **Fix:** Dropped all non-predictive metadata prior to model training to enforce privacy minimization and prevent spatial overfitting.

### C. Strict Target Leakage Prevention

* **Issue:** `Churn Reason` (collected post-cancellation) and `Churn Score` (an upstream IBM SPSS prediction score) introduce severe future information leakage.

* **Fix:** Explicitly excluded both variables from the training feature set before model fitting.

### D. Target Class Imbalance & Modelling Directives

* **Profile:** `Churn Value` exhibits a ~26.54% churn rate (1,869 churned vs. 5,174 retained), setting a baseline majority-class accuracy benchmark of 73.46%.

* **Directives:** Downstream models should use `class_weight='balanced'` or apply training-fold SMOTE to prioritize recall and avoid majority-class bias.

## 2.6 Stratified Train-Test Partitioning & Encoder Serialization

To ensure reliable baseline model evaluation and seamless downstream application integration, the cleansed and engineered feature set was partitioned and encoded using strict leakage-free protocols.

### A. Stratified 80/20 Train-Test Partitioning

* **Partitioning Logic:** The dataset was split into an 80% training set and a 20% holdout testing set using Scikit-Learn’s `train_test_split()` with a fixed random seed (`random_state=42`).

* **Target Stratification (`stratify=y`):** Stratified sampling was enforced on `Churn Value` to preserve the exact class imbalance profile (~26.54% churn rate) across both training and holdout partitions:

  * `train_telco_churn.csv` (Training Partition): 5,634 observations × 24 attributes (23 predictors + 1 target).

  * `test_telco_churn.csv` (Holdout Test Partition): 1,409 observations × 24 attributes (23 predictors + 1 target).

* **Index Export Protocol:** Files were exported with `index=False` to prevent unnecessary index column proliferation during CSV ingestion.

### B. Strict Categorical Encoding Isolation

* **Leakage Safeguard:** Binary nominal features (`Gender`, `Partner`, `Dependents`, `Phone Service`, `Paperless Billing`) and multi-class categorical attributes (`Multiple Lines`, `Internet Service`, `Contract`, `Payment Method`, `Tenure_Cohort`) were encoded using Scikit-Learn’s `LabelEncoder`.

* **Train-Set Isolation:** `LabelEncoder` instances were fitted strictly on the training set (`X_train`) and subsequently applied to transform `X_test`. This guarantees that no categorical distribution parameters or unseen class statistics from the holdout set leak into model fitting.

### C. Binary Serialization via Joblib (`input_encoders.pkl`)

* **Artifact Serialization:** All fitted `LabelEncoder` objects were compiled into a dictionary and serialized as a single binary file, `input_encoders.pkl`, using `joblib.dump()`.

* **Downstream Integration Role:** This artifact enables the Streamlit frontend web application to apply exact training-time integer transformations to real-time user inputs during inference.

## 2.7 Summary of Deliverables & Downstream Handovers

| Delivered File / Artifact | Dimensions / Format | Operational Description | Downstream Handoff Owner | 
| ----- | ----- | ----- | ----- | 
| `train_telco_churn.csv` | 5,634 rows × 24 columns (CSV) | Stratified 80% training dataset split | Diyasha Roy (Baseline & Model Validation) | 
| `test_telco_churn.csv` | 1,409 rows × 24 columns (CSV) | Stratified 20% holdout testing split | Diyasha Roy (Model Performance Benchmarking) | 
| `input_encoders.pkl` | Binary Python Pickle (joblib) | Serialized dictionary of fitted `LabelEncoder` objects | Surajit Tunga (Streamlit App Integration) | 
