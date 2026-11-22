"""Streamlit app for scoring customer churn using the trained model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
PREPROCESSOR_PATH = BASE_DIR / "data" / "processed" / "preprocessing.pkl"

HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.35
MODEL_SETUP_INSTRUCTION = "Run `python src/train.py` before `streamlit run app/app.py`."

BOOL_MAP = {"No": 0, "Yes": 1}
MULTIPLE_LINES_MAP = {"No": 0, "No phone service": 1, "Yes": 2}
INTERNET_SERVICE_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}
SERVICE_STATUS_MAP = {"No": 0, "No internet service": 1, "Yes": 2}
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_METHOD_MAP = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}
TENURE_COHORT_MAP = {
    "Cohort_0_12": 0,
    "Cohort_13_24": 1,
    "Cohort_25_48": 2,
    "Cohort_49_72": 3,
}


@st.cache_data
def load_model_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    metadata = load_model_metadata()
    artifact_name = metadata[model_name.lower()]["artifact"]
    model_path = MODEL_DIR / artifact_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_path}")
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


def get_tenure_cohort(tenure_months: int) -> str:
    if tenure_months <= 12:
        return "Cohort_0_12"
    if tenure_months <= 24:
        return "Cohort_13_24"
    if tenure_months <= 48:
        return "Cohort_25_48"
    return "Cohort_49_72"


def encode_categorical_value(column_name: str, value):
    if column_name in {"Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"}:
        return BOOL_MAP[str(value)]
    if column_name == "Gender":
        return {"Female": 0, "Male": 1}[str(value)]
    if column_name == "Multiple Lines":
        return MULTIPLE_LINES_MAP[str(value)]
    if column_name == "Internet Service":
        return INTERNET_SERVICE_MAP[str(value)]
    if column_name in {
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    }:
        return SERVICE_STATUS_MAP[str(value)]
    if column_name == "Contract":
        return CONTRACT_MAP[str(value)]
    if column_name == "Payment Method":
        return PAYMENT_METHOD_MAP[str(value)]
    if column_name == "Tenure_Cohort":
        return TENURE_COHORT_MAP[str(value)]
    raise ValueError(f"No encoding rule found for {column_name}")


def prepare_customer_record(data: dict) -> pd.DataFrame:
    metadata = load_model_metadata()
    tenure_months = int(data["tenure_months"])
    online_security = data["online_security"]
    online_backup = data["online_backup"]
    device_protection = data["device_protection"]
    tech_support = data["tech_support"]
    streaming_tv = data["streaming_tv"]
    streaming_movies = data["streaming_movies"]
    monthly_charges = float(data["monthly_charges"])

    if data["internet_service"] == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

    raw_record = {
        "Senior Citizen": data["senior_citizen"],
        "Partner": data["partner"],
        "Dependents": data["dependents"],
        "Tenure Months": tenure_months,
        "Multiple Lines": data["multiple_lines"],
        "Internet Service": data["internet_service"],
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": data["contract"],
        "Paperless Billing": data["paperless_billing"],
        "Payment Method": data["payment_method"],
        "Monthly Charges": monthly_charges,
        "Total Charges": float(data["total_charges"]),
        "Tenure_Cohort": get_tenure_cohort(tenure_months),
        "Total_Services": sum(
            1
            for service in [
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
            ]
            if service == "Yes"
        ),
        "Monthly_to_Tenure_Ratio": monthly_charges / (tenure_months + 1),
        "Log_Monthly_Tenure_Ratio": math.log1p(
            monthly_charges / (tenure_months + 1)
        ),
    }
    input_schema = metadata.get("preprocessor_input_schema")
    if input_schema is None:
        input_schema = [
            "Senior Citizen", "Partner", "Dependents", "Tenure Months",
            "Multiple Lines", "Internet Service", "Online Security",
            "Online Backup", "Device Protection", "Tech Support",
            "Streaming TV", "Streaming Movies", "Contract",
            "Paperless Billing", "Payment Method", "Monthly Charges",
            "Total Charges", "Tenure_Cohort", "Total_Services",
            "Monthly_to_Tenure_Ratio", "Log_Monthly_Tenure_Ratio",
        ]
    raw_df = pd.DataFrame([[raw_record[col] for col in input_schema]], columns=input_schema)
    preprocessor = load_preprocessor()
    transformed = preprocessor.transform(raw_df)
    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
    )


def predict_churn(model_name: str, data: dict):
    model = load_model(model_name)
    record = prepare_customer_record(data)
    probability = float(model.predict_proba(record)[0, 1])
    return probability


def format_risk(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High risk"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium risk"
    return "Low risk"


def risk_message(probability: float) -> str:
    risk_label = format_risk(probability)
    if risk_label == "High risk":
        return "High risk: this customer may need a retention intervention."
    if risk_label == "Medium risk":
        return "Medium risk: consider reviewing this customer's retention needs."
    return "Low risk: this customer appears relatively stable based on the current model."


def validate_customer_inputs(
    tenure_months, monthly_charges, total_charges, phone_service, multiple_lines
) -> str | None:
    try:
        tenure = int(tenure_months)
        monthly = float(monthly_charges)
        total = float(total_charges)
    except (TypeError, ValueError):
        return "Enter valid values for tenure and charges."

    if not math.isfinite(monthly) or not 0 <= monthly <= 120:
        return "Monthly Charges must be between $0 and $120."
    if not math.isfinite(total) or not 0 <= total <= 9000:
        return "Total Charges must be between $0 and $9,000."
    if (tenure == 0) != (total == 0):
        return "Tenure and Total Charges must both be zero, or both be greater than zero."
    if phone_service == "No" and multiple_lines == "Yes":
        return "Multiple Lines cannot be Yes when Phone Service is No."
    return None


def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
    st.title("Customer Churn Prediction")
    st.caption("Predict churn risk using the trained champion or challenger model.")

    if (
        not MODEL_DIR.is_dir()
        or not METADATA_PATH.is_file()
        or not PREPROCESSOR_PATH.is_file()
    ):
        st.error(f"Model files are missing. {MODEL_SETUP_INSTRUCTION}")
        st.stop()

    model_name = st.sidebar.selectbox("Select model", ["Champion", "Challenger"], index=0)

    metadata = load_model_metadata()
    st.sidebar.markdown("### Model details")
    st.sidebar.write(f"**Algorithm:** {metadata[model_name.lower()]['algorithm']}")
    st.sidebar.write(f"**Target:** {metadata['target']}")
    st.sidebar.write(f"**Features:** {len(metadata['feature_schema'])}")

    try:
        load_model(model_name)
        load_preprocessor()
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"Model files could not be loaded: {exc}")
        st.stop()

    with st.form("customer_form"):
        st.subheader("Customer profile")
        col1, col2 = st.columns(2)

        with col1:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure_months = st.slider("Tenure Months", min_value=0, max_value=72, value=12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            st.caption("not used by the model")
            multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
            st.caption("not used by the model")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        with col2:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=70.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=500.0, step=1.0)

        validation_message = validate_customer_inputs(
            tenure_months, monthly_charges, total_charges, phone_service, multiple_lines
        )
        if validation_message:
            st.error(validation_message)
        submitted = st.form_submit_button("Predict churn risk", disabled=bool(validation_message))

    if submitted:
        data = {
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
        }

        try:
            probability = predict_churn(model_name, data)
            churn_probability = probability * 100
            risk_label = format_risk(probability)

            st.subheader("Prediction result")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Churn probability", f"{churn_probability:.1f}%")
            col_b.metric("Risk category", risk_label)
            col_c.metric("Model", model_name)

            st.progress(probability)

            message = risk_message(probability)
            if risk_label != "Low risk":
                st.warning(message)
            else:
                st.success(message)

            if internet_service == "No":
                st.info("Since the customer has no internet service, the digital service add-ons are automatically treated as 'No internet service'.")

        except Exception as exc:  # pragma: no cover - UI fallback
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
