# Application Update Report

## Update Status

**Application update completed:** Yes.

The Streamlit application has been updated to use the remediated preprocessing and calibrated model artifacts.

## Changes Made

- Added loading for `data/processed/preprocessing.pkl`.
- Replaced the application's manual LabelEncoder-style numeric mappings with `preprocessor.transform(raw_df)`.
- Added `Log_Monthly_Tenure_Ratio` calculation using `log1p`.
- Added the 21-field preprocessor input contract to `model_metadata.json`.
- Updated model input generation to produce 34 transformed features.
- Added a startup check that requires the preprocessor artifact before the app can run.
- Removed Gender from the interface because it is excluded from the model.
- Kept Phone Service and Multiple Lines displayed for compatibility, while excluding both from the model contract.
- Preserved `No internet service` handling for digital add-on fields.
- Retained the application risk bands: Low below 35%, Medium from 35% to below 60%, and High at 60% or above.
- The selected champion or challenger now receives calibrated `predict_proba` values.

## Model Integration Contract

The app collects 21 preprocessor inputs and transforms them into 34 model columns:

1. `data/processed/preprocessing.pkl` transforms the raw/engineered customer record.
2. The app passes the 34-column result to the selected calibrated model artifact.
3. `predict_proba` provides the calibrated churn probability.
4. The existing UI converts the probability into Low, Medium, or High risk.

## Files Updated

- `app/app.py`
- `models/model_metadata.json`
- `tests/test_data.py`
- `tests/test_model.py`

## Application Verification

- Streamlit form workflow remains functional.
- Champion and challenger loading remains functional.
- Zero-tenure and zero-total-charge input is handled without crashing.
- Invalid cross-field inputs are rejected.
- Five raw-data parity rows match app transformations and direct model transformations.
- All 26 repository tests pass under Python 3.12 with scikit-learn 1.7.2.

## Run Instructions

```powershell
py -3.12 src/train.py
py -3.12 -m streamlit run app/app.py
```
