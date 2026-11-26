
"""
parity_check.py
T11: Updated Champion vs Challenger Prediction Comparison

Compatible with the updated 19-feature model schema.

Run from the project root:
    python -W ignore parity_check.py

Or specify another app file:
    python -W ignore parity_check.py app/app.py

Output:
    parity_report.txt
"""

import importlib.util
import io
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parent

APP_FILE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else BASE / "app" / "app.py"
)

if not APP_FILE.is_absolute():
    APP_FILE = BASE / APP_FILE

TEST_CSV = BASE / "data" / "processed" / "test_telco_churn.csv"
MODEL_DIR = BASE / "models"

TOLERANCE = 0.01

out = io.StringIO()


def say(*args):
    line = " ".join(str(x) for x in args)
    print(line)
    out.write(line + "\n")


# --------------------------------------------------
# 2. LOAD UPDATED APP
# --------------------------------------------------

logging.disable(logging.CRITICAL)

if not APP_FILE.exists():
    sys.exit(f"App file not found: {APP_FILE}")

if not TEST_CSV.exists():
    sys.exit(f"Test dataset not found: {TEST_CSV}")

spec = importlib.util.spec_from_file_location(
    "app_under_test", APP_FILE
)

app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

meta = app.load_model_metadata()

schema = meta["feature_schema"]
target_col = meta["target"]

test = pd.read_csv(TEST_CSV)

missing = [c for c in schema if c not in test.columns]

if missing:
    sys.exit(
        "Missing model features in processed test data:\n"
        + "\n".join(missing)
    )

if target_col not in test.columns:
    sys.exit(f"Target column missing: {target_col}")


# --------------------------------------------------
# 3. LOAD UPDATED MODELS
# --------------------------------------------------

champion = joblib.load(
    MODEL_DIR / meta["champion"]["artifact"]
)

challenger = joblib.load(
    MODEL_DIR / meta["challenger"]["artifact"]
)

X_test = test[schema].copy()

test["p_champ"] = champion.predict_proba(X_test)[:, 1]

test["p_chall"] = challenger.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 4. UPDATED ENCODING MAPS
# --------------------------------------------------

BOOL_MAP = app.BOOL_MAP

INTERNET_SERVICE_MAP = app.INTERNET_SERVICE_MAP

SERVICE_STATUS_MAP = app.SERVICE_STATUS_MAP

CONTRACT_MAP = app.CONTRACT_MAP

PAYMENT_METHOD_MAP = app.PAYMENT_METHOD_MAP

TENURE_COHORT_MAP = app.TENURE_COHORT_MAP


def inverse_map(mapping):
    return {int(v): k for k, v in mapping.items()}


def decode_value(row, column, mapping):
    value = row[column]

    if pd.isna(value):
        raise ValueError(
            f"Missing value in column: {column}"
        )

    # Handle encoded numerical categories.
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)

        if numeric.is_integer():
            value = int(numeric)

    reverse = inverse_map(mapping)

    if value not in reverse:
        raise ValueError(
            f"Unknown encoded value {value} in {column}"
        )

    return reverse[value]


# --------------------------------------------------
# 5. DECODE PROCESSED CUSTOMER INTO APP INPUTS
# --------------------------------------------------

def decode(row):

    tenure = int(row["Tenure Months"])

    data = {

        # These three features were pruned in train.py.
        # They are still required by the Streamlit form.
        "gender": "Female",
        "phone_service": "No",
        "multiple_lines": "No phone service",

        "senior_citizen": decode_value(
            row, "Senior Citizen", BOOL_MAP
        ),

        "partner": decode_value(
            row, "Partner", BOOL_MAP
        ),

        "dependents": decode_value(
            row, "Dependents", BOOL_MAP
        ),

        "tenure_months": tenure,

        "internet_service": decode_value(
            row, "Internet Service",
            INTERNET_SERVICE_MAP
        ),

        "online_security": decode_value(
            row, "Online Security",
            SERVICE_STATUS_MAP
        ),

        "online_backup": decode_value(
            row, "Online Backup",
            SERVICE_STATUS_MAP
        ),

        "device_protection": decode_value(
            row, "Device Protection",
            SERVICE_STATUS_MAP
        ),

        "tech_support": decode_value(
            row, "Tech Support",
            SERVICE_STATUS_MAP
        ),

        "streaming_tv": decode_value(
            row, "Streaming TV",
            SERVICE_STATUS_MAP
        ),

        "streaming_movies": decode_value(
            row, "Streaming Movies",
            SERVICE_STATUS_MAP
        ),

        "contract": decode_value(
            row, "Contract", CONTRACT_MAP
        ),

        "paperless_billing": decode_value(
            row, "Paperless Billing", BOOL_MAP
        ),

        "payment_method": decode_value(
            row, "Payment Method",
            PAYMENT_METHOD_MAP
        ),

        "monthly_charges": round(
            float(row["Monthly Charges"]), 2
        ),

        "total_charges": round(
            float(row["Total Charges"]), 2
        ),
    }

    return data


# --------------------------------------------------
# 6. APP INPUT LABELS
# --------------------------------------------------

LABELS = [

    ("gender", "Gender"),
    ("senior_citizen", "Senior Citizen"),
    ("partner", "Partner"),
    ("dependents", "Dependents"),
    ("tenure_months", "Tenure Months"),
    ("phone_service", "Phone Service"),
    ("multiple_lines", "Multiple Lines"),
    ("internet_service", "Internet Service"),
    ("online_security", "Online Security"),
    ("online_backup", "Online Backup"),
    ("device_protection", "Device Protection"),
    ("tech_support", "Tech Support"),
    ("streaming_tv", "Streaming TV"),
    ("streaming_movies", "Streaming Movies"),
    ("contract", "Contract"),
    ("paperless_billing", "Paperless Billing"),
    ("payment_method", "Payment Method"),
    ("monthly_charges", "Monthly Charges"),
    ("total_charges", "Total Charges"),

]


# --------------------------------------------------
# 7. SELECT FIVE CUSTOMERS
# --------------------------------------------------

order = test["p_champ"].rank(method="first")

targets = np.quantile(
    order,
    [0.0, 0.25, 0.50, 0.75, 1.0]
)

rows = [
    (order - target).abs().idxmin()
    for target in targets
]

# Avoid duplicate customers if risk ranks are tied.
rows = list(dict.fromkeys(rows))


# --------------------------------------------------
# 8. REPORT HEADER
# --------------------------------------------------

say("=" * 70)
say("T11 PREDICTION COMPARISON")
say(f"App file tested : {APP_FILE}")
say(
    f"Test data       : {TEST_CSV.name} "
    f"({len(test)} customers), tolerance {TOLERANCE}"
)
say(f"Target column   : {target_col}")
say(f"Model features  : {len(schema)}")
say("=" * 70)


# --------------------------------------------------
# 9. CUSTOMER-BY-CUSTOMER PARITY TEST
# --------------------------------------------------

passed = 0
summary = []

for n, idx in enumerate(rows, start=1):

    row = test.loc[idx]

    data = decode(row)

    direct = {

        "Champion": float(row["p_champ"]),

        "Challenger": float(row["p_chall"]),

    }

    say("")
    say(f"CUSTOMER {n}   (row {idx} of the test file)")

    say("  Enter these values in the app:")

    for key, label in LABELS:

        say(f"    {label:<18}: {data[key]}")

    # Actual target from the updated processed dataset.
    actual = int(row[target_col])

    say(f"  Actual {target_col:<14}: {actual}")

    # --------------------------------------------------
    # APP PREDICTIONS
    # --------------------------------------------------

    app_p = {}
    error = None

    for name in ("Champion", "Challenger"):

        try:

            app_p[name] = float(
                app.predict_churn(name, data)
            )

        except Exception as exc:

            error = f"{type(exc).__name__}: {exc}"
            break

    if error:

        say(f"  App code path    : FAILED -> {error}")
        say("  RESULT           : FAIL")

        summary.append((n, "FAIL", None))

        continue

    # --------------------------------------------------
    # FEATURE COMPARISON
    # --------------------------------------------------

    rec = app.prepare_customer_record(data)

    expected = row[schema].astype(float).to_numpy()

    actual_features = (
        rec[schema].iloc[0].astype(float).to_numpy()
    )

    feature_differences = np.abs(
        actual_features - expected
    )

    feat_ok = np.allclose(
        actual_features,
        expected,
        atol=1e-8,
        rtol=0
    )

    diff = max(
        abs(app_p[name] - direct[name])
        for name in direct
    )

    ok = feat_ok and diff <= TOLERANCE

    passed += int(ok)

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    say(
        f"  Model direct     : "
        f"Champion {direct['Champion'] * 100:5.1f}% | "
        f"Challenger {direct['Challenger'] * 100:5.1f}%"
    )

    say(
        f"  App code path    : "
        f"Champion {app_p['Champion'] * 100:5.1f}% | "
        f"Challenger {app_p['Challenger'] * 100:5.1f}%"
    )

    say(
        f"  App risk label   : "
        f"Champion {app.format_risk(app_p['Champion'])} | "
        f"Challenger {app.format_risk(app_p['Challenger'])}"
    )

    say(
        f"  Features equal   : "
        f"{'PASS' if feat_ok else 'FAIL'} "
        f"(app encoding vs processed row, {len(schema)} features)"
    )

    say(f"  Largest difference: {diff:.4f}")

    say(
        f"  RESULT           : "
        f"{'PASS' if ok else 'FAIL'}"
    )

    summary.append(
        (n, "PASS" if ok else "FAIL", diff)
    )


# --------------------------------------------------
# 10. OVERALL RESULT
# --------------------------------------------------

say("")
say("=" * 70)

say(
    f"T11 OVERALL: {passed} of {len(rows)} "
    "customers PASS"
)

say(
    "Next: use the displayed customer inputs in the Streamlit app "
    "and check that the Churn probability matches the "
    "'Model direct' value for the selected model."
)

say(
    "Note: Gender, Phone Service and Multiple Lines are not "
    "used by the updated 19-feature model."
)

say("=" * 70)


# --------------------------------------------------
# 11. SAVE REPORT
# --------------------------------------------------

report_path = BASE / "parity_report.txt"

report_path.write_text(
    out.getvalue(),
    encoding="utf-8"
)

print("\nSaved to parity_report.txt")
