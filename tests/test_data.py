"""Tests for the source customer data used by the training pipeline."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "raw_telco_churn.csv"


def test_source_data_has_training_columns_and_unique_customers():
	data = pd.read_csv(RAW_DATA_PATH)

	assert data["CustomerID"].is_unique
	assert {"Tenure Months", "Monthly Charges", "Total Charges", "Churn Value"} <= set(data.columns)


def test_zero_tenure_rows_have_no_total_charges():
	data = pd.read_csv(RAW_DATA_PATH)
	total_charges = pd.to_numeric(data["Total Charges"], errors="coerce").fillna(0)

	assert (total_charges[data["Tenure Months"] == 0] == 0).all()
