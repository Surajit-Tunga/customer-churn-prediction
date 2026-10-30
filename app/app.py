"""A small, illustrative customer churn risk demo."""

import streamlit as st


st.set_page_config(page_title="Customer Churn Demo", page_icon="📊")
st.title("Customer Churn Risk Demo")
st.caption(
    "A simple rule-based example. This is not a trained model or a real "
    "churn probability."
)

st.header("Customer details")
tenure = st.slider("Time with the company (months)", min_value=0, max_value=120, value=12)
monthly_charges = st.number_input(
    "Monthly charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=5.0
)
contract = st.selectbox(
    "Contract type",
    ["Month-to-month", "One year", "Two years"],
)
has_support = st.checkbox("Customer has tech support", value=True)

score = 0
reasons = []

if contract == "Month-to-month":
    score += 35
    reasons.append("Month-to-month contract")
if tenure < 12:
    score += 25
    reasons.append("Less than one year of tenure")
if monthly_charges >= 100:
    score += 20
    reasons.append("Monthly charges of $100 or more")
if not has_support:
    score += 20
    reasons.append("No tech support")

score = min(score, 100)
if score >= 60:
    risk = "High"
elif score >= 30:
    risk = "Medium"
else:
    risk = "Low"

st.header("Demo result")
col1, col2 = st.columns(2)
col1.metric("Illustrative risk score", f"{score}/100")
col2.metric("Risk category", risk)

if reasons:
    st.write("Factors contributing to this score:")
    for reason in reasons:
        st.write(f"- {reason}")
else:
    st.write("None of the demo risk factors were selected.")

st.info(
    "The score uses fixed example rules for demonstration only. It is not "
    "trained, validated, or suitable for customer decisions."
)
