"""
Telco churn data tester.

Usage:
    python test_telco_data.py raw_telco_churn.csv

Prints a PASS/FAIL data quality report, then statistical tests of churn drivers.
Also saves the results to test_report.txt next to the script.
"""
import sys
import io
import numpy as np
import pandas as pd
from scipy import stats

if len(sys.argv) < 2:
    sys.exit("Usage: python test_telco_data.py <path-to-csv>")

path = sys.argv[1]
df = pd.read_csv(path)
out = io.StringIO()


def say(text=""):
    print(text)
    out.write(text + "\n")


def check(name, passed, detail=""):
    say(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))


say("=" * 60)
say(f"FILE: {path}")
say(f"Rows: {len(df)} | Columns: {df.shape[1]}")
say("=" * 60)

# ---------------------------------------------------------------
# 1. DATA QUALITY
# ---------------------------------------------------------------
say("\n1. DATA QUALITY CHECKS\n")

check("No duplicate rows", df.duplicated().sum() == 0, f"{df.duplicated().sum()} duplicates")
check("CustomerID unique", df["CustomerID"].is_unique)
check("CustomerID format NNNN-XXXXX",
      df["CustomerID"].str.match(r"^\d{4}-[A-Z]{5}$").all())

missing = df.isna().sum()
missing = missing[missing > 0]
say(f"[INFO] Columns with missing values: {missing.to_dict() or 'none'}")

# Total Charges stored as text?
tc = pd.to_numeric(df["Total Charges"], errors="coerce")
bad_tc = int(tc.isna().sum())
check("Total Charges is fully numeric", bad_tc == 0,
      f"{bad_tc} non-numeric values (blank/space)")
if bad_tc:
    sub = df[tc.isna()]
    say(f"       All have Tenure = 0? {bool((sub['Tenure Months'] == 0).all())}")
str_cols = df.select_dtypes(include=["object", "string"]).columns
ws = sum((df[c].dropna().astype(str) != df[c].dropna().astype(str).str.strip()).sum()
         for c in str_cols)
check("No leading/trailing whitespace in text cells", ws == 0, f"{ws} cells")
df["Total Charges"] = tc  # use the numeric version for all later tests

check("Churn Label matches Churn Value",
      ((df["Churn Label"] == "Yes").astype(int) == df["Churn Value"]).all())
check("Churn Reason only for churned customers",
      not (df["Churn Reason"].notna() & (df["Churn Value"] == 0)).any())
check("Every churned customer has a reason",
      not (df["Churn Reason"].isna() & (df["Churn Value"] == 1)).any())

check("'No phone service' consistent with Multiple Lines",
      not ((df["Phone Service"] == "No") & (df["Multiple Lines"] != "No phone service")).any())
addons = ["Online Security", "Online Backup", "Device Protection",
          "Tech Support", "Streaming TV", "Streaming Movies"]
check("'No internet service' consistent across add-ons",
      not ((df["Internet Service"] == "No") & (df[addons] != "No internet service").any(axis=1)).any())

check("Latitude/Longitude inside California",
      df["Latitude"].between(32, 42.1).all() and df["Longitude"].between(-125, -114).all())
check("Monthly Charges within 0-200", df["Monthly Charges"].between(0, 200).all())
check("Tenure between 0 and 100 months", df["Tenure Months"].between(0, 100).all())

est = df["Tenure Months"] * df["Monthly Charges"]
ratio = df["Total Charges"] / est.replace(0, np.nan)
odd = df[(ratio < 0.5) | (ratio > 1.5)]
check("Total Charges close to tenure x monthly (0.5x-1.5x)", len(odd) == 0,
      f"{len(odd)} rows: {odd['CustomerID'].tolist()}")

const = [c for c in df.columns if df[c].nunique() == 1]
say(f"[INFO] Constant (useless) columns: {const}")

for c in ["Monthly Charges", "Total Charges", "Tenure Months", "CLTV"]:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    say(f"[INFO] IQR outliers in {c}: {n}")

churn_rate = df["Churn Value"].mean() * 100
say(f"[INFO] Churn rate: {churn_rate:.1f}%")

# ---------------------------------------------------------------
# 2. STATISTICAL TESTS
# ---------------------------------------------------------------
say("\n2. CHI-SQUARE TESTS (categorical vs churn)\n")
y = df["Churn Value"]
cats = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
        "Multiple Lines", "Internet Service"] + addons + \
       ["Contract", "Paperless Billing", "Payment Method"]
rows = []
for c in cats:
    ct = pd.crosstab(df[c], y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    rows.append((c, round(chi2, 1), dof, p, round(v, 3), "yes" if p < 0.05 else "no"))
res = pd.DataFrame(rows, columns=["variable", "chi2", "dof", "p_value", "cramers_v", "significant"])
say(res.sort_values("cramers_v", ascending=False).to_string(index=False))
say("\nCramer's V guide: 0.1 small, 0.3 medium, 0.5 large")

say("\n3. NUMERIC TESTS (churned vs retained)\n")
rows = []
for c in ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]:
    a = df.loc[y == 1, c].dropna()
    b = df.loc[y == 0, c].dropna()
    _, p = stats.mannwhitneyu(a, b)
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    rows.append((c, round(a.mean(), 1), round(b.mean(), 1), round(d, 2), p))
say(pd.DataFrame(rows, columns=["variable", "churned_mean", "retained_mean",
                                "cohens_d", "mannwhitney_p"]).to_string(index=False))
say("\nCohen's d guide: 0.2 small, 0.5 medium, 0.8 large")

say("\n4. CHURN RATE BY KEY GROUPS\n")
for c in ["Contract", "Internet Service", "Payment Method"]:
    say(f"{c}: {(df.groupby(c)['Churn Value'].mean() * 100).round(1).to_dict()}")

if "Churn Score" in df.columns:
    r = stats.spearmanr(df["Churn Score"], y)[0]
    say(f"\n[WARNING] Churn Score correlates {r:.2f} with churn: likely leakage, "
        "exclude it from model features.")

with open("test_report.txt", "w", encoding="utf-8") as f:
    f.write(out.getvalue())
say("\nResults saved to test_report.txt")
