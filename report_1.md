============================================================
FILE: raw_telco_churn.csv
Rows: 7043 | Columns: 33
============================================================

1. DATA QUALITY CHECKS

[PASS] No duplicate rows -> 0 duplicates
[PASS] CustomerID unique
[PASS] CustomerID format NNNN-XXXXX
[INFO] Columns with missing values: {'Churn Reason': 5174}
[FAIL] Total Charges is fully numeric -> 11 non-numeric values (blank/space)
       All have Tenure = 0? True
[FAIL] No leading/trailing whitespace in text cells -> 11 cells
[PASS] Churn Label matches Churn Value
[PASS] Churn Reason only for churned customers
[PASS] Every churned customer has a reason
[PASS] 'No phone service' consistent with Multiple Lines
[PASS] 'No internet service' consistent across add-ons
[PASS] Latitude/Longitude inside California
[PASS] Monthly Charges within 0-200
[PASS] Tenure between 0 and 100 months
[FAIL] Total Charges close to tenure x monthly (0.5x-1.5x) -> 2 rows: ['0323-XWWTN', '2308-STERM']
[INFO] Constant (useless) columns: ['Count', 'Country', 'State']
[INFO] IQR outliers in Monthly Charges: 0
[INFO] IQR outliers in Total Charges: 0
[INFO] IQR outliers in Tenure Months: 0
[INFO] IQR outliers in CLTV: 0
[INFO] Churn rate: 26.5%

2. CHI-SQUARE TESTS (categorical vs churn)

         variable   chi2  dof       p_value  cramers_v significant
         Contract 1184.6    2 5.863038e-258      0.410         yes
  Online Security  850.0    2 2.661150e-185      0.347         yes
     Tech Support  828.2    2 1.443084e-180      0.343         yes
 Internet Service  732.3    2 9.571788e-160      0.322         yes
   Payment Method  648.1    3 3.682355e-140      0.303         yes
    Online Backup  601.8    2 2.079759e-131      0.292         yes
Device Protection  558.4    2 5.505219e-122      0.282         yes
       Dependents  433.7    1  2.500972e-96      0.248         yes
 Streaming Movies  375.7    2  2.667757e-82      0.231         yes
     Streaming TV  374.2    2  5.528994e-82      0.231         yes
Paperless Billing  258.3    1  4.073355e-58      0.191         yes
   Senior Citizen  159.4    1  1.510067e-36      0.150         yes
          Partner  158.7    1  2.139911e-36      0.150         yes
   Multiple Lines   11.3    2  3.464383e-03      0.040         yes
    Phone Service    0.9    1  3.387825e-01      0.011          no
           Gender    0.5    1  4.865787e-01      0.008          no

Cramer's V guide: 0.1 small, 0.3 medium, 0.5 large

3. NUMERIC TESTS (churned vs retained)

       variable  churned_mean  retained_mean  cohens_d  mannwhitney_p
  Tenure Months          18.0           37.6     -0.89  2.419636e-208
Monthly Charges          74.4           61.3      0.47   3.311628e-54
  Total Charges        1531.8         2555.3     -0.48   1.995985e-84
           CLTV        4149.4         4490.9     -0.29   3.244484e-25

Cohen's d guide: 0.2 small, 0.5 medium, 0.8 large

4. CHURN RATE BY KEY GROUPS

Contract: {'Month-to-month': 42.7, 'One year': 11.3, 'Two year': 2.8}
Internet Service: {'DSL': 19.0, 'Fiber optic': 41.9, 'No': 7.4}
Payment Method: {'Bank transfer (automatic)': 16.7, 'Credit card (automatic)': 15.2, 'Electronic check': 45.3, 'Mailed check': 19.1}

[WARNING] Churn Score correlates 0.68 with churn: likely leakage, exclude it from model features.
