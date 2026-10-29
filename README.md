# Customer Churn Prediction

A project scaffold for exploring customer churn data, building and evaluating
baseline prediction models, and presenting churn risk through a Streamlit
dashboard.

> **Project status:** This repository is an early scaffold. The Streamlit page
is a small, illustrative rule-based demo; the `src/` pipeline modules contain
no model implementation yet, and no dataset is included. Consequently, a
baseline model cannot currently be trained or reproduced.

## Project structure

```text
.
├── app/
│   ├── app.py                 # Streamlit dashboard entry point
│   └── components/            # Reusable dashboard components
├── data/
│   ├── raw/                   # Original source data (not included)
│   └── processed/             # Derived datasets
├── docs/                      # Project documentation
├── models/                    # Model artifacts (not included)
├── notebooks/                 # Exploratory analysis
├── src/
│   ├── data_preprocessing.py  # Data validation and cleaning
│   ├── feature_engineering.py # Feature transformations and selection
│   ├── train.py               # Model training and baseline comparison
│   └── evaluate.py            # Evaluation metrics and reporting
├── tests/                     # Data and model tests
├── CONTRIBUTING.md
└── requirements.txt           # Python app dependency
```

## Requirements and installation

Use Python 3 and `pip`. From the repository root, create and activate a virtual
environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The app dependency is listed in `requirements.txt` and will be installed with
the command above.

## Run the Streamlit app

From the repository root, run:

```bash
python -m streamlit run app/app.py
```

The app accepts a few customer details and displays a rule-based demo risk
score. This score is not a trained or validated prediction and should not be
used to make customer decisions.

## Baseline model: current status and reproduction

There is not yet a reproducible baseline model in this repository. The training,
preprocessing, feature engineering, and evaluation modules are placeholders,
and no source dataset or trained artifact is checked in. No model type,
features, split strategy, random seed, or evaluation results have been defined,
so there are no baseline metrics to report.

Once those pieces are implemented, the intended reproduction workflow is:

1. Obtain the project dataset and record its source, licence, and acquisition
   date. Place permitted source data in `data/raw/`; do not commit sensitive or
   unlicensed data.
2. Record the dataset schema and target column, and implement validation and
   preprocessing in `src/data_preprocessing.py`.
3. Define and document the baseline features and model in `src/` and
   `src/train.py`, including the train/test split strategy and random seed.
4. Add all runtime and test dependencies to `requirements.txt`.
5. Implement evaluation in `src/evaluate.py` and report the chosen metrics.
6. Add tests and save model artifacts, if needed, under `models/`, together
   with the training-data and feature-set details.

The natural command-line entry point is expected to be `python -m src.train`
once the training module has been implemented. At present, running it does not
train a model or produce metrics. Update this section with the exact dataset
preparation and training commands when the pipeline is ready.

## Tests

The test files are also scaffolds at present. Once test dependencies and test
cases are implemented, run them from the repository root with:

```bash
python -m pytest
```

## Data and privacy

Keep raw source data separate from derived data. Before adding a dataset,
document its source and licence and verify that repository storage is allowed.
Do not commit personally identifiable or otherwise sensitive customer data.
