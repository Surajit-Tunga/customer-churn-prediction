# Customer Churn Prediction

A machine-learning project that analyzes customer account data, compares churn
prediction models, and provides an interactive Streamlit application for
estimating an individual customer's churn probability.

This project was developed as part of the **IDEAS Technology Innovation Hub
(TIH), ISI Kolkata internship**. It demonstrates an end-to-end data science
workflow, from data preparation and model evaluation to an application
prototype.

## The problem

Customer churn occurs when a customer stops using a service. If a business can
identify customers who may be at risk of leaving, it can investigate their
needs and consider appropriate retention actions. Churn prediction is
challenging because customer behavior depends on a mix of account tenure,
services, contract terms, and billing choices; a useful model must also avoid
using information that would not be available before churn.

## The solution

The project uses the IBM Telco Customer Churn sample dataset to build and
compare two supervised classification approaches:

- **Champion:** class-balanced Logistic Regression
- **Challenger:** XGBoost

The workflow cleans and engineers customer features, creates stratified
training and holdout datasets, fits a reusable preprocessing transformer on
the training data, and evaluates both models on held-out records. The model
artifacts are calibrated with sigmoid (Platt) scaling. A Streamlit app uses the
same preprocessing artifact and lets a user choose either model for a
single-customer prediction.

This is an analytical prototype, not an automated retention system. A
prediction is an estimate based on historical sample data; it is not a
guarantee of future behavior or a substitute for human review.

## What the project does

### Data and modeling

- Reads the raw Telco churn data from `data/raw/`.
- Converts blank `Total Charges` values to numeric values and handles the
  zero-tenure records.
- Removes identifiers, location fields, post-churn/leakage fields, and
  non-predictive fields from model training.
- Derives `Tenure_Cohort`, `Total_Services`, and
  `Log_Monthly_Tenure_Ratio`.
- Splits the data into stratified 80% training and 20% holdout sets using a
  fixed random seed.
- Fits a `ColumnTransformer` on training data only. It scales numeric features
  and one-hot encodes categorical features.
- Compares the champion and challenger using cross-validation and holdout
  metrics, including ROC-AUC, PR-AUC, Brier score, recall, precision, F1, and
  accuracy.
- Calibrates model probabilities with sigmoid calibration.

The preprocessor accepts 21 raw or engineered fields and produces the
34-column model feature matrix. Processed CSV files contain those 34 features
plus the `Churn Value` target.

### Streamlit application

- Collects a customer's account, tenure, services, contract, billing, and
  charge details through an interactive form.
- Lets the user select the Champion or Challenger model.
- Validates charge ranges and related input combinations.
- Displays estimated churn probability and an application-side risk band:
  Low below 35%, Medium from 35% to below 60%, and High at 60% or above.
- Uses the same fitted preprocessor as the model-training workflow.

The app currently supports one customer at a time. Batch scoring, prediction
history, model explanations, and hosted deployment are not included.

## Repository structure

```text
.
├── app/
│   ├── app.py                         # Streamlit prediction application
│   └── components/
│       └── __init__.py
├── data/
│   ├── raw/
│   │   └── raw_telco_churn.csv        # Source Telco customer data
│   └── processed/
│       ├── train_telco_churn.csv      # Transformed training split
│       ├── test_telco_churn.csv       # Transformed holdout split
│       ├── preprocessing.pkl          # Fitted preprocessing transformer
│       └── preprocessor.pkl           # Compatibility copy
├── docs/
│   ├── Final_Report.md
│   ├── data_quality_report.md
│   ├── section_2_data_preprocessing_and_feature_engineering_report.md
│   └── champion_challenger_evaluation.json
├── models/
│   ├── champion_logistic.joblib
│   ├── challenger_boosting.joblib
│   └── model_metadata.json
├── notebooks/                         # Project workspace notes
├── src/
│   ├── data_preprocessing_and_feature_engineering.ipynb
│   ├── train.py                        # Preprocess and train both models
│   ├── evaluate.py                     # Evaluate models on the holdout set
│   └── __init__.py
├── tests/
│   ├── README.md
│   ├── test_data.py
│   └── test_model.py
├── .streamlit/
│   └── config.toml
├── application_report.md
├── modelupdate.md
├── CONTRIBUTING.md
├── LICENSE.txt
├── requirements.txt
└── README.md
```

## Setup

Python 3.12 is the tested runtime. From the repository root, create and
activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

Trained models and the preprocessing artifact are included in the repository,
so the app can be started directly after installing dependencies:

```bash
python -m streamlit run app/app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in
your browser.

## Retrain and evaluate

To regenerate the preprocessor, processed splits, and model artifacts from the
raw dataset, run:

```bash
python src/train.py
python src/evaluate.py
```

Training writes the artifacts to `data/processed/` and `models/`. Evaluation
writes holdout metrics and the champion recommendation to
`docs/champion_challenger_evaluation.json`.

## Run tests

Run the test suite from the repository root:

```bash
python -m pytest -q
```

Tests cover source data checks, model and preprocessing integration, input
validation, prediction output, and Streamlit form behavior.

## Data, governance, and limitations

The project uses the IBM Telco Customer Churn sample data. See the data
documentation and reports for provenance, licensing notes, cleaning
decisions, and feature-engineering details. Do not add sensitive or unlicensed
customer data to the repository.

The model was developed and evaluated on this sample dataset. Performance on
other populations or future data may differ. The application is not currently
deployed and has no public URL.
---
