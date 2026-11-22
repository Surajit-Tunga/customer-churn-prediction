"""Reproducible champion/challenger training pipeline."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    BOOSTING_TYPE = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    XGBClassifier = HistGradientBoostingClassifier
    BOOSTING_TYPE = "HistGradientBoosting"
    logger.warning("XGBoost is unavailable; falling back to HistGradientBoosting.")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "raw_telco_churn.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
TARGET_COL = "Churn Value"

PRUNED_COLS = ["Gender", "Phone Service"]

NUMERICAL_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Total_Services",
    "Log_Monthly_Tenure_Ratio",
]

CATEGORICAL_COLS = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Multiple Lines",
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
    "Tenure_Cohort",
]

PREPROCESSOR_INPUT_COLS = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Multiple Lines",
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
    "Log_Monthly_Tenure_Ratio",
]


def get_tenure_cohort(tenure_months: int) -> str:
    if tenure_months <= 12:
        return "Cohort_0_12"
    if tenure_months <= 24:
        return "Cohort_13_24"
    if tenure_months <= 48:
        return "Cohort_25_48"
    return "Cohort_49_72"


def prepare_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    source = df.drop(columns=PRUNED_COLS, errors="ignore").copy()
    tenure = pd.to_numeric(source["Tenure Months"]).astype(int)
    monthly = pd.to_numeric(source["Monthly Charges"], errors="coerce")
    total = pd.to_numeric(source["Total Charges"], errors="coerce").fillna(0.0)

    internet = source["Internet Service"].astype(str)
    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    services = source[service_cols].astype(str).copy()
    services.loc[internet == "No", service_cols] = "No internet service"

    ratio = monthly / (tenure + 1)
    prepared = pd.DataFrame(index=source.index)
    prepared["Senior Citizen"] = source["Senior Citizen"]
    prepared["Partner"] = source["Partner"]
    prepared["Dependents"] = source["Dependents"]
    prepared["Tenure Months"] = tenure
    prepared["Multiple Lines"] = source["Multiple Lines"]
    prepared["Internet Service"] = internet
    for column in service_cols:
        prepared[column] = services[column]
    prepared["Contract"] = source["Contract"]
    prepared["Paperless Billing"] = source["Paperless Billing"]
    prepared["Payment Method"] = source["Payment Method"]
    prepared["Monthly Charges"] = monthly
    prepared["Total Charges"] = total
    prepared["Tenure_Cohort"] = tenure.map(get_tenure_cohort)
    prepared["Total_Services"] = (services == "Yes").sum(axis=1).astype(int)
    prepared["Monthly_to_Tenure_Ratio"] = ratio
    prepared["Log_Monthly_Tenure_Ratio"] = np.log1p(ratio)
    return prepared[PREPROCESSOR_INPUT_COLS]


def prepare_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible helper used by tests and parity checks."""
    return prepare_raw_features(df)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=True,
    )


def build_champion_pipeline():
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )


def build_challenger_pipeline(scale_pos_weight: float):
    if BOOSTING_TYPE == "XGBoost":
        return XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    return XGBClassifier(
        max_iter=150,
        max_depth=4,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )


def evaluate_candidate(name: str, model, X_test, y_test, threshold: float = 0.5):
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return {
        "model_name": name,
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "brier_score": float(brier_score_loss(y_test, probability)),
        "recall_sensitivity": float(recall_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "f1_score": float(f1_score(y_test, prediction)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
    }


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw training data not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    y = df[TARGET_COL].astype(int)
    X = prepare_raw_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_train_processed = pd.DataFrame(
        X_train_processed, columns=feature_names, index=X_train.index
    )
    X_test_processed = pd.DataFrame(
        X_test_processed, columns=feature_names, index=X_test.index
    )

    train_output = X_train_processed.copy()
    train_output[TARGET_COL] = y_train
    test_output = X_test_processed.copy()
    test_output[TARGET_COL] = y_test
    train_output.to_csv(PROCESSED_DIR / "train_telco_churn.csv", index=False)
    test_output.to_csv(PROCESSED_DIR / "test_telco_churn.csv", index=False)

    preprocessor_path = PROCESSED_DIR / "preprocessing.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(preprocessor, PROCESSED_DIR / "preprocessor.pkl")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    champion_base = build_champion_pipeline()
    challenger_base = build_challenger_pipeline(scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["roc_auc", "average_precision", "recall", "f1"]
    cv_champ = cross_validate(champion_base, X_train_processed, y_train, cv=cv, scoring=scoring)

    champion = CalibratedClassifierCV(
        estimator=champion_base,
        method="sigmoid",
        cv=cv,
    )
    champion.fit(X_train_processed, y_train)

    cv_chall = cross_validate(
        challenger_base,
        X_train_processed,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    challenger = CalibratedClassifierCV(
        estimator=challenger_base,
        method="sigmoid",
        cv=cv,
    )
    challenger.fit(X_train_processed, y_train)

    joblib.dump(champion, MODEL_DIR / "champion_logistic.joblib")
    joblib.dump(challenger, MODEL_DIR / "challenger_boosting.joblib")

    champion_metrics = evaluate_candidate(
        "Champion (Logistic Regression)",
        champion,
        X_test_processed,
        y_test,
    )
    challenger_metrics = evaluate_candidate(
        f"Challenger ({BOOSTING_TYPE})",
        challenger,
        X_test_processed,
        y_test,
    )
    recommendation = (
        "Champion (Logistic Regression)"
        if champion_metrics["roc_auc"] >= challenger_metrics["roc_auc"]
        else f"Challenger ({BOOSTING_TYPE})"
    )
    evaluation = {
        "champion": champion_metrics,
        "challenger": challenger_metrics,
        "recommendation": recommendation,
    }
    with open(DOCS_DIR / "champion_challenger_evaluation.json", "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=4)

    metadata = {
        "champion": {
            "algorithm": "Logistic Regression (L2 Balanced, Calibrated)",
            "cv_roc_auc": float(cv_champ["test_roc_auc"].mean()),
            "cv_pr_auc": float(cv_champ["test_average_precision"].mean()),
            "artifact": "champion_logistic.joblib",
        },
        "challenger": {
            "algorithm": f"{BOOSTING_TYPE} (Calibrated)",
            "cv_roc_auc": float(cv_chall["test_roc_auc"].mean()),
            "cv_pr_auc": float(cv_chall["test_average_precision"].mean()),
            "artifact": "challenger_boosting.joblib",
        },
        "feature_schema": feature_names,
        "preprocessor_input_schema": PREPROCESSOR_INPUT_COLS,
        "target": TARGET_COL,
        "calibration": "Platt scaling (sigmoid)",
        "decision_threshold": 0.60,
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    print("Updated 34-column processed train/test splits")
    print("Updated data/processed/preprocessing.pkl")
    print("Updated champion and challenger model artifacts")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
