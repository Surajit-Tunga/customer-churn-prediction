# Diyasha Roy 5/11/26
"""Model evaluation metrics and reporting."""
"""
src/evaluate.py
Statistical Holdout Evaluation: Champion vs. Challenger
Computes ROC-AUC, PR-AUC, F1, Brier Score, and Cost-Weighted Sensitivity.
"""

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test_telco_churn.csv"
MODEL_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Churn Value"


def evaluate_candidate(name: str, model, X_test, y_test, threshold: float = 0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "model_name": name,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "brier_score": float(brier_score_loss(y_test, y_proba)),
        "recall_sensitivity": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }


def main():
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data missing: {TEST_DATA_PATH}")

    champ_path = MODEL_DIR / "champion_logistic.joblib"
    chall_path = MODEL_DIR / "challenger_boosting.joblib"

    if not champ_path.exists() or not chall_path.exists():
        raise FileNotFoundError("Model artifacts missing. Run `python -m src.train` first.")

    champion = joblib.load(champ_path)
    challenger = joblib.load(chall_path)

    test_df = pd.read_csv(TEST_DATA_PATH)
    y_test = test_df[TARGET_COL].astype(int)
    X_test = test_df.drop(columns=[TARGET_COL])

    # Align with model input schema
    metadata_path = MODEL_DIR / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            features = json.load(f)["feature_schema"]
        X_test = X_test[[c for c in features if c in X_test.columns]]

    logger.info(f"Evaluating models on {len(test_df)} holdout records...")
    res_champ = evaluate_candidate("Champion (Logistic Regression)", champion, X_test, y_test)
    res_chall = evaluate_candidate("Challenger (Gradient Boosting)", challenger, X_test, y_test)

    # Print Side-by-Side Comparison
    print("\n" + "=" * 65)
    print(f"{'HOLDOUT EVALUATION METRIC':<25} | {'CHAMPION':<16} | {'CHALLENGER':<16}")
    print("=" * 65)
    metrics_to_print = [
        ("ROC-AUC (Discrimination)", "roc_auc"),
        ("PR-AUC (Imbalance)", "pr_auc"),
        ("Recall (Churn Capture)", "recall_sensitivity"),
        ("Precision", "precision"),
        ("F1-Score", "f1_score"),
        ("Brier Score (Calibration)", "brier_score"),
    ]
    for label, key in metrics_to_print:
        print(f"{label:<25} | {res_champ[key]:<16.4f} | {res_chall[key]:<16.4f}")
    print("=" * 65)

    # Determine Production Winner
    winner = "Champion (Logistic Regression)" if res_champ["roc_auc"] >= res_chall["roc_auc"] else "Challenger (Gradient Boosting)"
    print(f"\n=> Production Promotion Recommendation: {winner.upper()}")

    # Save to docs/
    report_path = DOCS_DIR / "champion_challenger_evaluation.json"
    with open(report_path, "w") as f:
        json.dump({"champion": res_champ, "challenger": res_chall, "recommendation": winner}, f, indent=4)
    logger.info(f"Evaluation report written to: {report_path}")


if __name__ == "__main__":
    main()

    