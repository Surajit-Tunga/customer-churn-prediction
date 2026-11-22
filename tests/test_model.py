"""Model, Streamlit workflow, and release-critical regression tests."""

import logging
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import app as churn_app
from src import train


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "raw_telco_churn.csv"
PARITY_ROW_POSITIONS = [869, 1176, 1283, 1327, 1090]
FEATURES = [
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


def customer(**overrides):
	data = {
		"senior_citizen": "No",
		"partner": "No",
		"dependents": "No",
		"tenure_months": 12,
		"phone_service": "Yes",
		"multiple_lines": "No",
		"internet_service": "DSL",
		"online_security": "No",
		"online_backup": "No",
		"device_protection": "No",
		"tech_support": "No",
		"streaming_tv": "No",
		"streaming_movies": "No",
		"contract": "Month-to-month",
		"paperless_billing": "Yes",
		"payment_method": "Electronic check",
		"monthly_charges": 70.0,
		"total_charges": 840.0,
	}
	return data | overrides


def run_app_script():
	from app.app import main

	main()


def app_test():
	return AppTest.from_function(run_app_script).run()


def raw_row_to_customer(row):
	total_charges = pd.to_numeric(pd.Series([row["Total Charges"]]), errors="coerce").fillna(0).iloc[0]
	return {
		"senior_citizen": row["Senior Citizen"],
		"partner": row["Partner"],
		"dependents": row["Dependents"],
		"tenure_months": row["Tenure Months"],
		"phone_service": row["Phone Service"],
		"multiple_lines": row["Multiple Lines"],
		"internet_service": row["Internet Service"],
		"online_security": row["Online Security"],
		"online_backup": row["Online Backup"],
		"device_protection": row["Device Protection"],
		"tech_support": row["Tech Support"],
		"streaming_tv": row["Streaming TV"],
		"streaming_movies": row["Streaming Movies"],
		"contract": row["Contract"],
		"paperless_billing": row["Paperless Billing"],
		"payment_method": row["Payment Method"],
		"monthly_charges": row["Monthly Charges"],
		"total_charges": total_charges,
	}


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
	churn_app.load_model_metadata.clear()
	churn_app.load_model.clear()
	yield
	churn_app.load_model_metadata.clear()
	churn_app.load_model.clear()


def test_t02_python312_dependencies_are_importable():
	import sklearn
	import streamlit
	import xgboost

	assert sys.version_info[:2] == (3, 12)
	assert sklearn.__version__ == "1.7.2"
	assert streamlit.__version__
	assert xgboost.__version__ == "3.1.0"


@pytest.mark.parametrize("model_dir_exists", [False, True])
def test_t03_missing_models_show_setup_instructions_without_traceback(
	monkeypatch, tmp_path, model_dir_exists
):
	model_dir = tmp_path / "models"
	if model_dir_exists:
		model_dir.mkdir()
	monkeypatch.setattr(churn_app, "MODEL_DIR", model_dir)
	monkeypatch.setattr(churn_app, "METADATA_PATH", model_dir / "model_metadata.json")

	result = app_test()

	assert not result.exception
	assert any("python src/train.py" in error.value for error in result.error)


def test_t04_form_loads_and_saved_models_are_estimators():
	result = app_test()

	assert not result.exception
	for model_name in ("Champion", "Challenger"):
		model = churn_app.load_model(model_name)
		assert callable(model.predict_proba)


def test_selected_model_and_preprocessor_load_before_prediction(monkeypatch):
	loaded = []
	monkeypatch.setattr(churn_app, "load_model", lambda name: loaded.append(("model", name)))
	monkeypatch.setattr(churn_app, "load_preprocessor", lambda: loaded.append(("preprocessor", None)))

	result = app_test()

	assert not result.exception
	assert ("model", "Champion") in loaded
	assert ("preprocessor", None) in loaded
	assert not any(metric.label == "Churn probability" for metric in result.metric)


def test_t05_app_collects_or_derives_all_19_model_inputs():
	with open(churn_app.METADATA_PATH, encoding="utf-8") as metadata_file:
		import json

		metadata = json.load(metadata_file)
		feature_schema = metadata["feature_schema"]
	encoded = churn_app.prepare_customer_record(customer())

	assert len(feature_schema) == 34
	assert list(encoded.columns) == feature_schema
	assert len(metadata["preprocessor_input_schema"]) == 21
	assert not {"Gender", "Phone Service", "Multiple Lines"}.intersection(encoded.columns)


def test_t06_high_risk_customer_probability_is_at_least_point_60():
	high_risk_customer = customer(
		tenure_months=1,
		monthly_charges=70.7,
		total_charges=70.7,
		internet_service="Fiber optic",
		contract="Month-to-month",
		payment_method="Electronic check",
	)

	probability = churn_app.predict_churn("Champion", high_risk_customer)

	assert probability >= churn_app.HIGH_RISK_THRESHOLD


def test_t07_low_risk_customer_probability_is_below_point_35():
	low_risk_customer = customer(
		tenure_months=72,
		monthly_charges=70.0,
		total_charges=5000.0,
		contract="Two year",
		payment_method="Credit card (automatic)",
	)

	probability = churn_app.predict_churn("Champion", low_risk_customer)

	assert probability < churn_app.MEDIUM_RISK_THRESHOLD


def test_t08_zero_tenure_and_zero_total_charges_predict_without_crashing():
	probability = churn_app.predict_churn(
		"Champion", customer(tenure_months=0, monthly_charges=0.0, total_charges=0.0)
	)

	assert 0.0 <= probability <= 1.0


@pytest.mark.parametrize(
	("updates", "message"),
	[
		({"monthly_charges": None}, "valid values"),
		({"monthly_charges": -1}, "Monthly Charges"),
		({"total_charges": 9001}, "Total Charges"),
		({"tenure_months": 0, "total_charges": 500}, "both be zero"),
		({"phone_service": "No", "multiple_lines": "Yes"}, "Multiple Lines"),
	],
)
def test_t09_invalid_values_are_rejected_with_validation_message(updates, message):
	values = customer() | updates

	validation_message = churn_app.validate_customer_inputs(
		values["tenure_months"],
		values["monthly_charges"],
		values["total_charges"],
		values["phone_service"],
		values["multiple_lines"],
	)

	assert validation_message is not None
	assert message in validation_message


def test_t09_invalid_form_shows_error_and_disables_predict():
	result = app_test()
	next(widget for widget in result.slider if widget.label == "Tenure Months").set_value(0)
	next(widget for widget in result.number_input if widget.label == "Total Charges ($)").set_value(500)
	next(widget for widget in result.button if widget.label == "Predict churn risk").click()
	result.run()

	assert not result.exception
	assert any("both be zero" in error.value for error in result.error)
	assert next(widget for widget in result.button if widget.label == "Predict churn risk").disabled


@pytest.mark.parametrize(
	("probability", "label"),
	[(0.50, "Medium risk"), (0.55, "Medium risk"), (0.60, "High risk")],
)
def test_t10_output_probability_label_model_and_warning_agree(monkeypatch, probability, label):
	monkeypatch.setattr(churn_app, "predict_churn", lambda model_name, data: probability)
	result = app_test()
	next(widget for widget in result.button if widget.label == "Predict churn risk").click()
	result.run()

	metrics = {metric.label: metric.value for metric in result.metric}
	assert metrics["Churn probability"] == f"{probability * 100:.1f}%"
	assert metrics["Risk category"] == label
	assert metrics["Model"] == "Champion"
	assert any(label in warning.value for warning in result.warning)


def test_t11_app_and_models_match_on_five_requested_rows():
	raw_data = pd.read_csv(RAW_DATA_PATH)
	selected_rows = raw_data.iloc[PARITY_ROW_POSITIONS]
	raw_training_features = train.prepare_training_features(selected_rows).reset_index(drop=True)
	preprocessor = churn_app.load_preprocessor()
	training_features = pd.DataFrame(
		preprocessor.transform(raw_training_features),
		columns=preprocessor.get_feature_names_out(),
	)
	app_features = pd.concat(
		[churn_app.prepare_customer_record(raw_row_to_customer(row)) for _, row in selected_rows.iterrows()],
		ignore_index=True,
	)

	np.testing.assert_allclose(training_features.to_numpy(dtype=float), app_features.to_numpy(dtype=float))
	assert training_features.shape == (5, 34)
	for model_name in ("Champion", "Challenger"):
		model = churn_app.load_model(model_name)
		direct_probabilities = model.predict_proba(app_features)[:, 1]
		app_probabilities = [
			churn_app.predict_churn(model_name, raw_row_to_customer(row))
			for _, row in selected_rows.iterrows()
		]
		differences = np.abs(direct_probabilities - app_probabilities)
		assert float(differences.max()) < 0.01


def test_d03_hist_gradient_fallback_logs_visible_warning(monkeypatch, caplog):
	monkeypatch.setitem(sys.modules, "xgboost", None)
	spec = importlib.util.spec_from_file_location("train_fallback_test", ROOT / "src" / "train.py")
	fallback_module = importlib.util.module_from_spec(spec)
	with caplog.at_level(logging.WARNING):
		spec.loader.exec_module(fallback_module)

	assert fallback_module.BOOSTING_TYPE == "HistGradientBoosting"
	assert "falling back to HistGradientBoosting" in caplog.text


def test_d05_load_model_returns_loaded_estimator(monkeypatch, tmp_path):
	artifact = tmp_path / "model.joblib"
	artifact.touch()
	expected_model = object()
	monkeypatch.setattr(churn_app, "MODEL_DIR", tmp_path)
	monkeypatch.setattr(churn_app, "load_model_metadata", lambda: {"champion": {"artifact": artifact.name}})
	monkeypatch.setattr(churn_app.joblib, "load", lambda path: expected_model)

	assert churn_app.load_model("Champion") is expected_model


def test_d06_labels_and_messages_use_the_same_risk_band():
	for probability, label in ((0.50, "Medium risk"), (0.55, "Medium risk"), (0.60, "High risk")):
		assert churn_app.format_risk(probability) == label
		assert churn_app.risk_message(probability).startswith(label)


def test_d07_bounds_and_cross_field_rules_are_explicit():
	assert churn_app.validate_customer_inputs(12, 120, 9000, "Yes", "No") is None
	assert "Monthly Charges" in churn_app.validate_customer_inputs(12, 120.01, 100, "Yes", "No")
	assert "Total Charges" in churn_app.validate_customer_inputs(12, 70, 9000.01, "Yes", "No")
	assert "both be zero" in churn_app.validate_customer_inputs(0, 70, 1, "Yes", "No")
	assert "Multiple Lines" in churn_app.validate_customer_inputs(12, 70, 100, "No", "Yes")


def test_d08_excluded_inputs_are_retained_but_not_in_model_schema():
	result = app_test()

	for label in ("Phone Service", "Multiple Lines"):
		assert any(widget.label == label for widget in result.selectbox)
	assert not any(widget.label == "Gender" for widget in result.selectbox)
	assert sum(caption.value == "not used by the model" for caption in result.caption) == 2
