# Diyasha Roy - revised after the project review of 12 Nov
"""Model training and baseline comparison.

Champion-Challenger training pipeline:
- Champion:   L2-penalised Logistic Regression (interpretable GLM)
- Challenger: Gradient Boosting / XGBoost classifier (non-linear ensemble)

Changes made after the review:
1. One-hot encoding inside the pipeline. Aazeen's files store categories as
   integer codes (LabelEncoder), which have no real order. The codes are now
   one-hot encoded for BOTH models, so Logistic Regression no longer reads
   "Electronic check = 2" as larger than "Mailed check = 1", and the comparison
   with the boosting model is fair. The app still sends the same integer codes.
2. Hyperparameter tuning with a small, stated search space (GridSearchCV,
   5-fold stratified CV, training data only).
3. Cross-validation results are REPORTED: mean and spread for ROC-AUC, PR-AUC,
   recall, F1 and Brier score, plus the paired fold-by-fold difference.
4. The model is SELECTED on cross-validation evidence, not on the holdout.
   The test file is not read in this script; src/evaluate.py touches it once.
   Rule: the Challenger is promoted only if its average gain is larger than the
   fold-to-fold spread of that gain (on ROC-AUC and PR-AUC). Otherwise the
   simpler Champion is kept.
5. Probability calibration (Platt scaling / sigmoid) is fitted on training
   data and saved next to each model, because class weighting makes raw
   probabilities too high.

Run from the repository root:  python -m src.train
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Try importing XGBoost; fall back to scikit-learn's HistGradientBoosting.
try:
    from xgboost import XGBClassifier

    BOOSTING_TYPE = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    BOOSTING_TYPE = "HistGradientBoosting"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Directory configuration matching project architecture
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TUNING_SEED = 42  # folds used to choose hyperparameters
COMPARISON_SEED = 7  # different folds used to compare the two tuned models
N_SPLITS = 5

TARGET_COL = "Churn Value"

# Truly numeric columns. Total_Services is a count (0 to 6), so it is scaled,
# not one-hot encoded. Every other column is an integer category code.
NUMERICAL_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Total_Services",
    "Monthly_to_Tenure_Ratio",
]

# Non-informative features screened out during hypothesis testing (QA-02)
PRUNED_COLS = ["Phone Service", "Gender", "Multiple Lines"]

# Search spaces (stated so the tuning can be reproduced and reported)
CHAMPION_GRID = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
if BOOSTING_TYPE == "XGBoost":
    CHALLENGER_GRID = {
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.05, 0.10],
        "classifier__n_estimators": [100, 200],
    }
else:
    CHALLENGER_GRID = {
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.05, 0.10],
        "classifier__max_iter": [100, 200],
    }


def load_dataset():
    """Loads the processed TRAIN dataset only (the test file is not touched here)."""
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
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(
            PROCESSED_DATA_DIR / "test_telco_churn.csv", index=False
        )
        X, y = X_train, y_train
    else:
        raise FileNotFoundError("Could not locate train_telco_churn.csv or the root CSV")

    # Prune non-informative columns if present
    X = X.drop(columns=[c for c in PRUNED_COLS if c in X.columns], errors="ignore")
    return X, y


def make_one_hot_encoder():
    """One-hot encoder that works on both newer and older scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn older than 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(feature_names):
    """Scales numeric columns and one-hot encodes the integer category codes."""
    num_cols = [c for c in NUMERICAL_COLS if c in feature_names]
    cat_cols = [c for c in feature_names if c not in num_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", make_one_hot_encoder(), cat_cols),
        ]
    )


def build_champion_pipeline(preprocessor):
    """Champion Model: Balanced Logistic Regression."""
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    # default penalty is L2; left unset so it works on every scikit-learn version
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_challenger_pipeline(preprocessor, scale_pos_weight: float):
    """Challenger Model: Gradient Boosting / XGBoost."""
    if BOOSTING_TYPE == "XGBoost":
        estimator = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        estimator = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def tune(name, pipeline, grid, X, y):
    """Grid search on the training data only. Returns the best fitted pipeline."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=TUNING_SEED)
    search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=cv, n_jobs=1)
    logger.info(f"Tuning {name} over {grid} ...")
    search.fit(X, y)
    best = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    logger.info(f"{name}: best parameters {best} (CV PR-AUC {search.best_score_:.4f})")
    return search.best_estimator_, best


SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "f1": "f1",
    "brier": "neg_brier_score",
}


def cv_scores(model, X, y):
    """Per-fold scores on folds that are identical for both models (paired)."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=COMPARISON_SEED)
    raw = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    folds = {m: np.asarray(raw[f"test_{m}"]) for m in SCORING}
    folds["brier"] = -folds["brier"]  # report as a normal Brier score (lower is better)
    return folds


def summarise(folds):
    return {m: {"mean": float(v.mean()), "std": float(v.std(ddof=1))} for m, v in folds.items()}


def select_model(champ_folds, chall_folds):
    """Selection rule on CV evidence only (the holdout is NOT used)."""
    result = {}
    promote = True
    for m in ["roc_auc", "pr_auc"]:
        diff = chall_folds[m] - champ_folds[m]
        mean, sd = float(diff.mean()), float(diff.std(ddof=1))
        result[m] = {"mean_gain": mean, "std_of_gain": sd, "folds_won": int((diff > 0).sum())}
        promote = promote and (mean > sd)
    result["rule"] = (
        "Promote the Challenger only if its mean paired gain is larger than the "
        "fold-to-fold standard deviation of the gain on BOTH ROC-AUC and PR-AUC; "
        "otherwise keep the simpler, interpretable Champion."
    )
    result["selected"] = "challenger" if promote else "champion"
    return result


def calibrate(model, X, y):
    """Platt scaling (sigmoid), fitted with internal cross-validation on training data."""
    try:
        cal = CalibratedClassifierCV(estimator=clone(model), method="sigmoid", cv=N_SPLITS)
    except TypeError:  # scikit-learn older than 1.2
        cal = CalibratedClassifierCV(base_estimator=clone(model), method="sigmoid", cv=N_SPLITS)
    return cal.fit(X, y)


def print_cv_table(champ_s, chall_s):
    print("\n" + "=" * 78)
    print(f"{N_SPLITS}-FOLD STRATIFIED CV ON THE TRAINING SET (mean +/- std over folds)")
    print("=" * 78)
    print(f"{'Metric':<12} | {'Champion (Logistic)':<24} | {'Challenger (' + BOOSTING_TYPE + ')':<30}")
    print("-" * 78)
    for m in SCORING:
        a, b = champ_s[m], chall_s[m]
        print(f"{m:<12} | {a['mean']:.4f} +/- {a['std']:.4f}      | {b['mean']:.4f} +/- {b['std']:.4f}")
    print("=" * 78)


def main():
    logger.info("Initializing Model Training Workflow...")
    X_train, y_train = load_dataset()

    # Exact class weight: N_negative / N_positive
    n_0 = int((y_train == 0).sum())
    n_1 = int((y_train == 1).sum())
    scale_pos_weight = float(n_0 / n_1)
    logger.info(
        f"Training set: N={len(y_train)}, Class 0={n_0}, Class 1={n_1} "
        f"(Ratio: {scale_pos_weight:.2f}); features used: {X_train.shape[1]}"
    )

    preprocessor = build_preprocessor(list(X_train.columns))
    champion, champ_params = tune(
        "Champion", build_champion_pipeline(preprocessor), CHAMPION_GRID, X_train, y_train
    )
    challenger, chall_params = tune(
        f"Challenger ({BOOSTING_TYPE})",
        build_challenger_pipeline(build_preprocessor(list(X_train.columns)), scale_pos_weight),
        CHALLENGER_GRID,
        X_train,
        y_train,
    )

    # Compare the two tuned models on identical folds (paired).
    logger.info("Cross-validating both tuned models on identical folds ...")
    champ_folds = cv_scores(champion, X_train, y_train)
    chall_folds = cv_scores(challenger, X_train, y_train)
    champ_s, chall_s = summarise(champ_folds), summarise(chall_folds)
    print_cv_table(champ_s, chall_s)

    selection = select_model(champ_folds, chall_folds)
    print(f"\nPaired gain of Challenger over Champion (per fold): "
          f"ROC-AUC {selection['roc_auc']['mean_gain']:+.4f} (spread {selection['roc_auc']['std_of_gain']:.4f}), "
          f"PR-AUC {selection['pr_auc']['mean_gain']:+.4f} (spread {selection['pr_auc']['std_of_gain']:.4f})")
    print(f"Selected on CV evidence: {selection['selected'].upper()}\n")

    # Fit on the full training data
    champion.fit(X_train, y_train)
    challenger.fit(X_train, y_train)

    # Calibrated copies (Platt scaling)
    logger.info("Fitting probability calibration (sigmoid) on training data ...")
    champion_cal = calibrate(champion, X_train, y_train)
    challenger_cal = calibrate(challenger, X_train, y_train)

    # Export serialized pipelines
    champ_path = MODEL_DIR / "champion_logistic.joblib"
    chall_path = MODEL_DIR / "challenger_boosting.joblib"
    champ_cal_path = MODEL_DIR / "champion_logistic_calibrated.joblib"
    chall_cal_path = MODEL_DIR / "challenger_boosting_calibrated.joblib"
    joblib.dump(champion, champ_path)
    joblib.dump(challenger, chall_path)
    joblib.dump(champion_cal, champ_cal_path)
    joblib.dump(challenger_cal, chall_cal_path)

    metadata = {
        "champion": {
            "algorithm": "Logistic Regression (L2 Balanced)",
            "cv_roc_auc": champ_s["roc_auc"]["mean"],
            "cv_roc_auc_std": champ_s["roc_auc"]["std"],
            "cv_pr_auc": champ_s["pr_auc"]["mean"],
            "best_params": champ_params,
            "artifact": champ_path.name,
            "calibrated_artifact": champ_cal_path.name,
        },
        "challenger": {
            "algorithm": BOOSTING_TYPE,
            "cv_roc_auc": chall_s["roc_auc"]["mean"],
            "cv_roc_auc_std": chall_s["roc_auc"]["std"],
            "cv_pr_auc": chall_s["pr_auc"]["mean"],
            "best_params": chall_params,
            "artifact": chall_path.name,
            "calibrated_artifact": chall_cal_path.name,
        },
        "feature_schema": list(X_train.columns),
        "target": TARGET_COL,
        "selected_model": selection["selected"],
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Evidence file stored in docs/ so the decision can be reviewed and reproduced
    report = {
        "cv_folds": N_SPLITS,
        "tuning_seed": TUNING_SEED,
        "comparison_seed": COMPARISON_SEED,
        "search_space": {"champion": CHAMPION_GRID, f"challenger_{BOOSTING_TYPE}": CHALLENGER_GRID},
        "best_params": {"champion": champ_params, "challenger": chall_params},
        "cv_summary": {"champion": champ_s, "challenger": chall_s},
        "paired_gain_challenger_minus_champion": {
            k: v for k, v in selection.items() if k in ("roc_auc", "pr_auc")
        },
        "selection_rule": selection["rule"],
        "selected_model": selection["selected"],
        "note": (
            "Tuning and comparison use training data only. Tuned scores are slightly "
            "optimistic; the holdout (src/evaluate.py) is the unbiased check."
        ),
    }
    with open(DOCS_DIR / "cv_comparison.json", "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved models to {MODEL_DIR} and CV evidence to {DOCS_DIR / 'cv_comparison.json'}")


if __name__ == "__main__":
    main()
