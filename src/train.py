# Diyasha Roy 5/11/26
"""Model training and baseline comparison."""
"""
src/train.py
Champion-Challenger Training Pipeline:
- Champion: L2-penalized Logistic Regression (Interpretable GLM)
- Challenger: Gradient Boosting / XGBoost Classifier (Non-linear Ensemble)
"""

import os
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Try importing XGBoost; fall back to scikit-learn's native HistGradientBoosting if not installed
try:
    from xgboost import XGBClassifier
    BOOSTING_TYPE = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    BOOSTING_TYPE = "HistGradientBoosting"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Directory configuration matching project architecture
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
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


def load_dataset():
    """Loads processed train dataset, with automatic fallback to root CSV."""
    train_path = PROCESSED_DATA_DIR / "train_telco_churn.csv"
    root_csv = BASE_DIR / "telco churn .csv"
    
    if train_path.exists():
        logger.info(f"Loading existing processed train data: {train_path}")
        df = pd.read_csv(train_path)
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL].astype(int)
    elif root_csv.exists():
        logger.info(f"Generating stratified 80/20 train-test split from {root_csv}...")
        df = pd.read_csv(root_csv)
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        train_df.to_csv(PROCESSED_DATA_DIR / "train_telco_churn.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test_telco_churn.csv", index=False)
        logger.info("Saved train_telco_churn.csv and test_telco_churn.csv to data/processed/")
        X, y = X_train, y_train
    else:
        raise FileNotFoundError("Could not locate train_telco_churn.csv or telco churn .csv")

    # Prune non-informative columns if present
    X = X.drop(columns=[c for c in PRUNED_COLS if c in X.columns], errors="ignore")
    return X, y


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
    
    