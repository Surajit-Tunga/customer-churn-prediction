# Contribution guide

This guide records the Autumn Internship 2026 team responsibilities for the
Customer Churn Prediction & Retention Decision Support project. File ownership
means the person responsible for driving and maintaining that work; it does not
prevent teammates from proposing changes, reviewing, or helping with it.

## Team roles and file ownership

| Teammate | Folders and files they lead | Main responsibility |
|---|---|---|
| **Aryan Kashyap** | `docs/project_charter.md`; `docs/data_dictionary.md` (business interpretation); `docs/customer_risk_framework.md` (planned); business sections of the final report in `docs/` | Define the business problem, churn use cases, feature interpretation, and retention actions. |
| **Aazeen Neyaz** | `data/` (especially `data/raw/` and `data/processed/` guidance); `src/data_preprocessing.py`; `src/feature_engineering.py`; `docs/data_quality_report.md` (planned) | Acquire and document data; build reproducible preprocessing and risk indicators; check data quality and leakage. |
| **Diyasha Roy** | `notebooks/`; `src/train.py`; `src/evaluate.py`; `models/Note.md`; `docs/model_validation_report.md` (planned) | Lead EDA, model training and comparison, evaluation, threshold analysis, and model selection. |
| **Sovan Ray** | `tests/` (including `tests/test_data.py` and `tests/test_model.py`); `docs/test_evidence.md` (planned) | Own independent data, model, and application checks; maintain test evidence and release findings. |
| **Surajit Tunga** | `app/`; `.streamlit/`; `README.md`; `requirements.txt`; repository structure and integration | Build and integrate the Streamlit application; maintain setup instructions and dependencies. |

Ownership identifies the lead, not an exclusive right to edit. Coordinate
changes that affect another teammate's area with that owner. Items marked
**planned** should be created when that work starts.

### Aryan Kashyap — Business problem and analytical design

**Primary ownership**

- `docs/project_charter.md` — business problem, intended users, prediction
  target, supported decisions, use cases, and success criteria.
- `docs/data_dictionary.md` — review feature definitions and business
  interpretations with the data and modelling owners.
- `docs/customer_risk_framework.md` — customer-risk prioritisation and proposed
  retention action framework (create this when that work begins).
- Business interpretation and commercial recommendations in the final report
  under `docs/`.

**Responsibilities and deliverables**

- Frame the problem as a practical decision-support use case.
- Agree with the team on what counts as churn and when a prediction would be
  made; document assumptions and label timing.
- Explain important features in business terms and identify appropriate
  customer-risk use cases.
- Keep recommendations proportionate to the evidence; distinguish model
  findings from proposed business actions.

### Aazeen Neyaz — Data preparation and risk analytics

**Primary ownership**

- `data/raw/Note.md` and `data/Note.md` — dataset source, licence, acquisition,
  access, and data-handling guidance.
- `src/data_preprocessing.py` — reproducible cleaning, validation, and leakage
  checks.
- `src/feature_engineering.py` — risk indicators and feature transformations,
  coordinated with the modelling owner.
- `docs/data_quality_report.md` — data quality findings, preprocessing
  decisions, and limitations (create this when the analysis begins).
- `data/processed/Note.md` — processed data and split versioning conventions.

**Responsibilities and deliverables**

- Acquire and document a dataset that the team is permitted to use.
- Investigate data types, missing values, duplicates, class balance, unusual
  values, and potential target leakage.
- Make preprocessing reproducible and document how each transformation treats
  invalid or missing values.
- Develop supplementary risk indicators and state their limitations.
- Do not commit restricted raw data, credentials, or personally identifiable
  information.


### Diyasha Roy — Statistical analysis and model validation

**Primary ownership**

- `notebooks/` — EDA and experiment notebooks, including recorded dataset and
  split details.
- `src/train.py` — baseline training and feasible model comparisons.
- `src/evaluate.py` — reproducible metrics, threshold analysis, and evaluation
  outputs.
- `models/Note.md` — model artifact, feature set, and versioning documentation.
- `docs/model_validation_report.md` — model comparison, error analysis, and
  model-selection rationale (create this when the analysis begins).

**Responsibilities and deliverables**

- Establish a baseline and compare at least one alternative when feasible.
- Define a defensible train/validation/test strategy with the data owner; keep
  test data out of training and model selection.
- Evaluate relevant metrics, class imbalance, false positives, false
  negatives, and decision thresholds.
- Explain which model best supports the stated use case and why; record
  limitations and reproducible results.
- Coordinate the saved model interface with the application and test owners
  before integration.


### Sovan Ray — Data quality, testing, and monitoring

**Primary ownership**

- `tests/test_data.py` — input schema, data quality, and preprocessing tests.
- `tests/test_model.py` — model input/output and metric reproducibility tests.
- Dashboard-related tests under `tests/` — agree the test module name with the
  application owner when adding them.
- `docs/test_evidence.md` — test execution evidence and release checks (create
  this when testing begins).
- Defect and test-evidence register — keep issues, owners, reproduction steps,
  status, and verification evidence current in the agreed team tracker or
  `docs/test_evidence.md`.

**Responsibilities and deliverables**

- Independently check data types, value ranges, unusual records, and
  preprocessing consistency.
- Review selected features and analytical assumptions without replacing the
  responsible owner's decision.
- Test data inputs, model outputs, and important dashboard scenarios,
  including invalid inputs and understandable error handling.
- Verify reported metrics against the saved/reproducible workflow.
- Report the command used, outcome, and any known limitation for release checks.

### Surajit Tunga — Application engineering and integration

**Primary ownership**

- `app/` — Streamlit dashboard, reusable components, input validation, and
  visual explanations.
- `.streamlit/` — safe application configuration; keep secrets out of source
  control.
- `README.md` — setup, run instructions, and project overview.
- `requirements.txt` — application and project dependencies.
- Repository structure and integration workflow, in coordination with all
  owners.

**Responsibilities and deliverables**

- Integrate only the agreed data and model interfaces; coordinate changes to
  shared interfaces with their owners.
- Provide useful validation messages and clear behavior when expected data or
  model artifacts are unavailable.
- Keep setup reproducible and document the commands another intern needs to
  install, run, and explain the project.
- Coordinate integration and record integration-test evidence with the test
  owner.

## Contribution workflow

### Do

- Check the role ownership above and coordinate changes that affect another
  teammate's work.
- Create a branch for your task and keep each change focused.
- Run relevant tests or checks, and update documentation when needed.
- Ask a teammate to review your work and review someone else's contribution.

### Avoid

- Committing directly to the main branch or including unrelated changes.
- Committing secrets, credentials, private customer information, or data that
  the project is not permitted to share.
- Changing shared data, model, or application interfaces without discussing it
  with the affected owners.
- Reporting results that have not been checked or cannot be reproduced.

### Commits

Make small commits with clear messages describing the change, for example:

```text
Add missing-value checks to preprocessing
Document churn target and intended use
Test invalid dashboard input
```

### Pull requests

Open a pull request against the main branch. Describe what changed and why,
list relevant tests or checks and their results, and note any limitations or
follow-up work. Include screenshots for visible dashboard changes when useful.
Request at least one teammate's review, address feedback, and make sure
relevant checks pass before merging.

---