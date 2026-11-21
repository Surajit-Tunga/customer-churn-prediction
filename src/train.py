# Diyasha Roy 5/11/26
"""Model training and baseline comparison."""
"""
src/train.py
Champion-Challenger Training Pipeline:
- Champion: L2-penalized Logistic Regression (Interpretable GLM)
- Challenger: Gradient Boosting / XGBoost Classifier (Non-linear Ensemble)
"""

import json
import logging
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Try importing XGBoost; fall back to scikit-learn's native HistGradientBoosting if not installed
try:
    from xgboost import XGBClassifier
    BOOSTING_TYPE = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    BOOSTING_TYPE = "HistGradientBoosting"
    logger.warning("XGBoost is unavailable; falling back to HistGradientBoosting.")

# Directory configuration matching project architecture
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "raw_telco_churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Churn Value"

NUMERICAL_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Monthly_to_Tenure_Ratio"
]

# Non-informative features screened out during hypothesis testing
PRUNED_COLS = ["Phone Service", "Gender", "Multiple Lines"]


FEATURE_SCHEMA = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "Tenure_Cohort",
    "Total_Services",
    "Monthly_to_Tenure_Ratio",
]


def prepare_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the same 19 ordered inputs used by the Streamlit form."""
    source = df.drop(columns=PRUNED_COLS, errors="ignore").copy()
    tenure = pd.to_numeric(source["Tenure Months"]).astype(int)
    monthly_charges = pd.to_numeric(source["Monthly Charges"])
    total_charges = pd.to_numeric(source["Total Charges"], errors="coerce").fillna(0.0)
    internet_service = source["Internet Service"].astype(str)

    service_columns = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    services = source[service_columns].astype(str).copy()
    services.loc[internet_service == "No", service_columns] = "No internet service"

    encoded = pd.DataFrame(index=source.index)
    for column in ["Senior Citizen", "Partner", "Dependents", "Paperless Billing"]:
        encoded[column] = source[column].map({"No": 0, "Yes": 1})
    encoded["Tenure Months"] = tenure
    encoded["Internet Service"] = internet_service.map({"DSL": 0, "Fiber optic": 1, "No": 2})
    service_map = {"No": 0, "No internet service": 1, "Yes": 2}
    for column in service_columns:
        encoded[column] = services[column].map(service_map)
    encoded["Contract"] = source["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    encoded["Payment Method"] = source["Payment Method"].map(
        {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3,
        }
    )
    encoded["Monthly Charges"] = monthly_charges
    encoded["Total Charges"] = total_charges
    encoded["Tenure_Cohort"] = pd.cut(
        tenure,
        bins=[-1, 12, 24, 48, 72],
        labels=["Cohort_0_12", "Cohort_13_24", "Cohort_25_48", "Cohort_49_72"],
    ).map(
        {
            "Cohort_0_12": 0,
            "Cohort_13_24": 1,
            "Cohort_25_48": 2,
            "Cohort_49_72": 3,
        }
    )
    encoded["Total_Services"] = (services == "Yes").sum(axis=1)
    encoded["Monthly_to_Tenure_Ratio"] = monthly_charges / (tenure + 1)
    return encoded[FEATURE_SCHEMA]


def load_dataset():
    """Load raw customer records and build a split using the app's input schema."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw training data not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    X = prepare_training_features(df)
    y = df[TARGET_COL].astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Loaded raw customer data with the app's 19-feature schema.")
    return X_train, y_train


def build_preprocessor(feature_names):
    """Encapsulates scaling for numericals and passthrough for categorical features."""
    num_cols = [c for c in NUMERICAL_COLS if c in feature_names]
    cat_cols = [c for c in feature_names if c not in num_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", "passthrough", cat_cols)
        ]
    )


def build_champion_pipeline(preprocessor):
    """Champion Model: Balanced Logistic Regression."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ))
    ])


def build_challenger_pipeline(preprocessor, scale_pos_weight: float):
    """Challenger Model: Gradient Boosting / XGBoost."""
    if BOOSTING_TYPE == "XGBoost":
        estimator = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    else:
        estimator = XGBClassifier(
            max_iter=150,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])


def main():
    logger.info("Initializing Model Training Workflow...")
    X_train, y_train = load_dataset()

    # Exact class weight: N / (2 * N_minority)
    n_0 = (y_train == 0).sum()
    n_1 = (y_train == 1).sum()
    scale_pos_weight = float(n_0 / n_1)
    logger.info(f"Training set: N={len(y_train)}, Class 0={n_0}, Class 1={n_1} (Ratio: {scale_pos_weight:.2f})")

    preprocessor = build_preprocessor(list(X_train.columns))

    champion = build_champion_pipeline(preprocessor)
    challenger = build_challenger_pipeline(preprocessor, scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["roc_auc", "average_precision", "recall", "f1"]

    logger.info("Cross-validating Champion (Logistic Regression)...")
    cv_champ = cross_validate(champion, X_train, y_train, cv=cv, scoring=scoring)

    logger.info(f"Cross-validating Challenger ({BOOSTING_TYPE})...")
    cv_chall = cross_validate(challenger, X_train, y_train, cv=cv, scoring=scoring)

    print("\n" + "=" * 65)
    print("5-FOLD STRATIFIED CV BENCHMARK (TRAIN SET)")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Champion (Logistic)':<18} | {'Challenger (' + BOOSTING_TYPE + ')':<20}")
    print("-" * 65)
    for m in scoring:
        champ_mean = cv_champ[f"test_{m}"].mean()
        chall_mean = cv_chall[f"test_{m}"].mean()
        print(f"{m:<20} | {champ_mean:<18.4f} | {chall_mean:<20.4f}")
    print("=" * 65 + "\n")

    # Fit both models on full training data
    champion.fit(X_train, y_train)
    challenger.fit(X_train, y_train)

    # Export serialized pipelines
    champ_path = MODEL_DIR / "champion_logistic.joblib"
    chall_path = MODEL_DIR / "challenger_boosting.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"

    joblib.dump(champion, champ_path)
    joblib.dump(challenger, chall_path)

    metadata = {
        "champion": {
            "algorithm": "Logistic Regression (L2 Balanced)",
            "cv_roc_auc": float(cv_champ["test_roc_auc"].mean()),
            "cv_pr_auc": float(cv_champ["test_average_precision"].mean()),
            "artifact": str(champ_path.name)
        },
        "challenger": {
            "algorithm": BOOSTING_TYPE,
            "cv_roc_auc": float(cv_chall["test_roc_auc"].mean()),
            "cv_pr_auc": float(cv_chall["test_average_precision"].mean()),
            "artifact": str(chall_path.name)
        },
        "feature_schema": list(X_train.columns),
        "target": TARGET_COL
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Artifacts exported to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
    
    