# Model Update Report

## Update Status

**Model update completed:** Yes.

The champion and challenger models were regenerated on 22 November 2026 using the remediated preprocessing pipeline and the processed 80/20 train-test data.

## Updated Artifacts

- `data/processed/train_telco_churn.csv`: 5,634 rows, 35 columns (34 transformed features plus `Churn Value`)
- `data/processed/test_telco_churn.csv`: 1,409 rows, 35 columns (34 transformed features plus `Churn Value`)
- `data/processed/preprocessing.pkl`: standardized `ColumnTransformer`
- `data/processed/preprocessor.pkl`: compatibility copy of the standardized preprocessor
- `models/champion_logistic.joblib`: retrained and sigmoid-calibrated Logistic Regression
- `models/champion_logistic.joblib`: logistic champion artifact
- `models/challenger_boosting.joblib`: retrained and sigmoid-calibrated XGBoost
- `models/model_metadata.json`: updated 34-feature model schema and calibration metadata
- `docs/champion_challenger_evaluation.json`: updated holdout metrics

## Preprocessing Changes

- Replaced manual model preparation with the fitted 21-input `ColumnTransformer`.
- Applied `StandardScaler` to five numeric features.
- Applied `OneHotEncoder(drop='first', handle_unknown='ignore')` to 15 categorical features.
- Added `Log_Monthly_Tenure_Ratio` with `log1p`.
- Preserved `Monthly_to_Tenure_Ratio` as the remainder feature.
- Added inclusive tenure cohorts for zero-tenure and boundary values.
- Preserved `No internet service` as a distinct add-on-service category.
- Used `random_state=42` and a stratified 80/20 split.

## Model Calibration

Both final model artifacts use post-hoc Platt scaling through `CalibratedClassifierCV(method='sigmoid')`.

Raw balanced-class probabilities are no longer used directly as final model artifacts. This improves probability calibration and the displayed probability's interpretation.

## Final Holdout Metrics

| Model | ROC-AUC | PR-AUC | Brier Score | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Champion (Logistic Regression) | 0.854879 | 0.672274 | 0.132529 | 0.577540 | 0.672897 | 0.621583 | 0.813343 |
| Challenger (XGBoost) | 0.854345 | 0.667777 | 0.133995 | 0.585561 | 0.646018 | 0.614306 | 0.804826 |

**Production recommendation:** Champion (Logistic Regression).

The challenger has slightly higher recall, but the champion has better ROC-AUC, PR-AUC, Brier score, precision, F1, and accuracy on the held-out test partition.

## Validation

- Train split: 5,634 rows, 35 columns, 0 missing values
- Test split: 1,409 rows, 35 columns, 0 missing values
- Target balance: approximately 73.46% retained and 26.54% churned
- Preprocessor input: 21 raw/engineered fields
- Preprocessor output: 34 model features
- Test suite: 26 tests passed
