# Customer Churn Prediction

This repository contains an IBM Telco Customer Churn analysis project covering
preprocessing, feature engineering, model training, evaluation, and a Streamlit
application for interactive churn-risk predictions.

## Repository structure

```text
.
├── app/
│   ├── app.py                 # Streamlit ML prediction application
│   └── components/           # Reusable UI components
├── data/
│   ├── raw/                  # Original source dataset
│   │   └── raw_telco_churn.csv
│   └── processed/            # Cleaned and split datasets
│       ├── train_telco_churn.csv
│       ├── test_telco_churn.csv
│       └── input_encoders.pkl
├── docs/
│   ├── Final_Report.md       # Project summary report
│   ├── data_quality_report.md # Data-quality and churn analysis report
│   ├── section_2_data_preprocessing_and_feature_engineering_report.md
│   └── champion_challenger_evaluation.json
├── models/                   # Trained model artifacts
├── notebooks/                # Exploratory notebooks
├── src/
│   ├── data_preprocessing_and_feature_engineering.ipynb
│   ├── train.py              # Champion/challenger model training
│   ├── evaluate.py           # Holdout evaluation metrics
│   └── __init__.py
├── test/
│   ├── README.md             # Testing notes
│   ├── test_data.py           # Data QC checks
│   ├── test_telco_data.py     # Telco validation suite
│   └── test_model.py          # Model-level validation checks
├── CONTRIBUTING.md
├── LICENSE.txt
├── README.md
├── requirements.txt
└── .streamlit/
```

## Project overview

The workflow is organized around the IBM Telco dataset and includes:

- data validation and cleaning in the preprocessing pipeline
- engineered churn-risk indicators and train/test split creation
- model comparison using champion and challenger approaches
- reproducible evaluation metrics and project reports
- a Streamlit app connected to the saved Champion and Challenger model artifacts

## Setup

Use Python 3 and install dependencies from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

## Run the app

From the repository root, install the dependencies and start Streamlit:

```bash
python -m streamlit run app/app.py
```

The app loads the saved model artifacts and feature schema from `models/`. It
provides a customer input form, lets you select the Champion or Challenger
model, computes the required engineered features, and displays the model's
estimated churn probability and a risk category. The risk categories are
application-side bands: Low below 35%, Medium from 35% to below 60%, and High
at 60% or above.

The prediction is an estimate from the selected trained model, not a guarantee
of customer behavior or a substitute for human review. The app has been
verified locally with both model choices. It is not currently deployed, so
there is no public application URL.

### Current application status

- **Available:** Single-customer interactive prediction using the saved
  Champion or Challenger artifact.
- **Not yet included:** Batch CSV predictions, prediction history, and
  model-explanation features.
- **Not yet complete for launch:** Deployment setup and hosted-environment
  verification.

The application depends on a stable model integration contract. The model
owner is responsible for maintaining the artifacts, feature schema and
encodings, dependencies, and prediction-score meaning. Changes to that contract
may require corresponding application updates.

## Train and evaluate the model

From the repository root:

```bash
python src/train.py
python src/evaluate.py
```

The training script creates model artifacts in `models/`, and the evaluation
script writes metrics to [docs/champion_challenger_evaluation.json](docs/champion_challenger_evaluation.json).

## Relevant documentation

- [docs/Final_Report.md](docs/Final_Report.md)
- [docs/data_quality_report.md](docs/data_quality_report.md)
- [docs/section_2_data_preprocessing_and_feature_engineering_report.md](docs/section_2_data_preprocessing_and_feature_engineering_report.md)
- [data/README.md](data/README.md)
- [test/README.md](test/README.md)

## Data and governance

- Raw dataset files are stored under [data/raw](data/raw)
- Processed datasets and encoder artifacts are stored under [data/processed](data/processed)
- Keep the raw data immutable and document source, licensing, and usage before
  committing new datasets
- Do not store sensitive or unlicensed customer data in the repository

---