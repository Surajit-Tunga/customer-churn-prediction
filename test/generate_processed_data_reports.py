import os
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Run this script from the main project folder.
DATA_DIR = os.path.join("data", "processed")
OUTPUT_DIR = "test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = [
    ("Training Processed Dataset", "train_telco_churn.csv",
     "train_processed_data_test_report.pdf"),
    ("Testing Processed Dataset", "test_telco_churn.csv",
     "test_processed_data_test_report.pdf"),
]

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitleX", parent=styles["Title"],
                          fontSize=17, leading=21, alignment=1, spaceAfter=8))
styles.add(ParagraphStyle(name="SectionX", parent=styles["Heading2"],
                          fontSize=11, leading=14, textColor=colors.HexColor("#17365D"),
                          spaceBefore=9, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"],
                          fontSize=8.5, leading=11, spaceAfter=4))

def fmt(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)

def make_table(rows, widths, font_size=7):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table

def create_report(title, csv_path, pdf_path):
    df = pd.read_csv(csv_path)
    row_count, col_count = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    target = "Churn Value" if "Churn Value" in df.columns else None

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_column_names = int(df.columns.duplicated().sum())
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    infinite_values = 0
    if numeric_cols:
        infinite_values = int(np.isinf(df[numeric_cols].to_numpy(dtype=float)).sum())

    feature_cols = [c for c in df.columns if c != target]
    duplicate_feature_rows = int(df.duplicated(subset=feature_cols).sum()) if feature_cols else 0
    conflicting_label_groups = 0
    if target:
        grouped = df.groupby(feature_cols, dropna=False)[target].nunique(dropna=False)
        conflicting_label_groups = int((grouped > 1).sum())

    story = [
        Paragraph(title, styles["ReportTitleX"]),
        Paragraph(f"Source file: {os.path.basename(csv_path)}", styles["BodyX"]),
        Paragraph("1. Dataset overview", styles["SectionX"]),
        make_table([
            ["Check", "Result"],
            ["Rows", fmt(row_count)],
            ["Columns", fmt(col_count)],
            ["Numeric columns", fmt(len(numeric_cols))],
            ["Non-numeric columns", fmt(len(non_numeric_cols))],
            ["Target column", target or "Not found"],
            ["Missing cells", fmt(missing_cells)],
            ["Completely duplicated rows", fmt(duplicate_rows)],
            ["Duplicate column names", fmt(duplicate_column_names)],
            ["Infinite numeric values", fmt(infinite_values)],
            ["Constant columns", ", ".join(constant_cols) if constant_cols else "None"],
            ["Repeated feature rows (excluding target)", fmt(duplicate_feature_rows)],
            ["Identical feature groups with conflicting labels", fmt(conflicting_label_groups)],
        ], [85*mm, 90*mm], 8),
        Paragraph("2. Data-quality test results", styles["SectionX"]),
    ]

    tests = [
        ["Test", "Result", "Meaning"],
        ["CSV file can be opened", "PASS", "Pandas successfully loaded the CSV."],
        ["Target column", "PASS" if target else "REVIEW",
         "Churn Value was found." if target else "Churn Value was not found."],
        ["Missing values", "PASS" if missing_cells == 0 else "REVIEW",
         "No missing cells." if missing_cells == 0 else f"{missing_cells} missing cells found."],
        ["Full duplicate rows", "PASS" if duplicate_rows == 0 else "REVIEW",
         "No fully duplicated rows." if duplicate_rows == 0 else f"{duplicate_rows} repeated rows found; investigate before deleting."],
        ["Duplicate column names", "PASS" if duplicate_column_names == 0 else "FAIL",
         "Column names are unique." if duplicate_column_names == 0 else "Duplicate names can cause selection problems."],
        ["Infinite numeric values", "PASS" if infinite_values == 0 else "FAIL",
         "No infinite values." if infinite_values == 0 else f"{infinite_values} infinite values found."],
        ["Constant columns", "PASS" if not constant_cols else "REVIEW",
         "No constant columns." if not constant_cols else "Review: " + ", ".join(constant_cols)],
        ["Repeated feature rows", "PASS" if duplicate_feature_rows == 0 else "REVIEW",
         "No repeated feature rows." if duplicate_feature_rows == 0 else f"{duplicate_feature_rows} repeated feature rows; may represent different customers with identical attributes."],
        ["Conflicting labels for same features", "PASS" if conflicting_label_groups == 0 else "REVIEW",
         "No conflicting feature groups." if conflicting_label_groups == 0 else f"{conflicting_label_groups} feature groups have different target labels."],
    ]
    story.append(make_table(tests, [48*mm, 20*mm, 107*mm], 6.7))
    story.append(Paragraph(
        "Note: REVIEW does not automatically mean the dataset is incorrect. Investigate flagged records using original customer IDs where available. Do not delete records automatically.",
        styles["BodyX"]))

    if target:
        story += [
            Paragraph("3. Target distribution", styles["SectionX"]),
            make_table(
                [["Churn Value", "Number of rows", "Percentage"]] +
                [[fmt(value), fmt(count), f"{count/row_count*100:.2f}%"]
                 for value, count in df[target].value_counts(dropna=False).sort_index().items()],
                [55*mm, 55*mm, 55*mm], 8)
        ]

    story += [
        Paragraph("4. Column-by-column inventory", styles["SectionX"]),
        Paragraph("Data type, missing values, unique values and numeric range for every column.", styles["BodyX"])
    ]
    inventory = [["Column", "Data type", "Missing", "Unique", "Minimum", "Maximum"]]
    for col in df.columns:
        series = df[col]
        minimum = fmt(series.min()) if pd.api.types.is_numeric_dtype(series) and series.notna().any() else "—"
        maximum = fmt(series.max()) if pd.api.types.is_numeric_dtype(series) and series.notna().any() else "—"
        inventory.append([col, str(series.dtype), fmt(series.isna().sum()),
                          fmt(series.nunique()), minimum, maximum])
    story.append(make_table(inventory, [47*mm, 23*mm, 18*mm, 18*mm, 30*mm, 30*mm], 6))

    story += [
        Paragraph("5. Numeric feature statistics", styles["SectionX"]),
        Paragraph("Statistics describe values as stored in the processed CSV. Standardised values are not original business units.", styles["BodyX"])
    ]
    stats = [["Column", "Mean", "Std dev", "Minimum", "Median", "Maximum"]]
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s):
            stats.append([col, fmt(s.mean()), fmt(s.std()), fmt(s.min()),
                          fmt(s.median()), fmt(s.max())])
    if len(stats) == 1:
        stats.append(["No numeric columns", "—", "—", "—", "—", "—"])
    story.append(make_table(stats, [55*mm, 24*mm, 24*mm, 24*mm, 24*mm, 24*mm], 6))

    story += [
        Paragraph("6. Binary / encoded feature checks", styles["SectionX"]),
        Paragraph("Numeric columns with no more than two distinct non-missing values are listed below.", styles["BodyX"])
    ]
    binary_cols = [c for c in numeric_cols if df[c].dropna().nunique() <= 2]
    binary = [["Column", "Count of 0", "Count of 1", "Other / missing"]]
    for col in binary_cols:
        zero = int((df[col] == 0).sum())
        one = int((df[col] == 1).sum())
        binary.append([col, fmt(zero), fmt(one), fmt(row_count-zero-one)])
    if len(binary) == 1:
        binary.append(["No binary columns detected", "—", "—", "—"])
    story.append(make_table(binary, [85*mm, 30*mm, 30*mm, 30*mm], 6.5))

    story += [
        Paragraph("7. Outlier screening (IQR method)", styles["SectionX"]),
        Paragraph("Potential outliers are values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR. These are flags for review, not automatic errors.", styles["BodyX"])
    ]
    outliers = [["Column", "Q1", "Q3", "Lower fence", "Upper fence", "Flagged rows"]]
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s):
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
            count = int(((s < low) | (s > high)).sum())
            if count:
                outliers.append([col, fmt(q1), fmt(q3), fmt(low), fmt(high), fmt(count)])
    if len(outliers) == 1:
        story.append(Paragraph("No values outside the IQR fences were detected.", styles["BodyX"]))
    else:
        story.append(make_table(outliers, [45*mm, 24*mm, 24*mm, 27*mm, 27*mm, 28*mm], 6))

    story += [
        Paragraph("8. Important limitations", styles["SectionX"]),
        Paragraph("• This report checks the CSV dataset only; it does not test model accuracy or prediction quality.", styles["BodyX"]),
        Paragraph("• It cannot prove that preprocessing was fitted only on training data. Check the preprocessing code to confirm scalers and encoders are fitted on training data and then applied to test data.", styles["BodyX"]),
        Paragraph("• If customer IDs are absent from processed files, this report cannot prove customer-level separation between train and test.", styles["BodyX"]),
        Paragraph("• Duplicate feature vectors can occur for different customers. Investigate before removing them.", styles["BodyX"]),
        Paragraph("9. Recommended next steps", styles["SectionX"]),
        Paragraph("1. Confirm that train and test feature columns match and are in the same order.", styles["BodyX"]),
        Paragraph("2. Confirm that the target column is not included among model input features.", styles["BodyX"]),
        Paragraph("3. Confirm that preprocessing was fitted on training data only.", styles["BodyX"]),
        Paragraph("4. Investigate duplicate and conflicting-feature findings using source records.", styles["BodyX"]),
        Paragraph("5. Run a separate model evaluation on the hold-out test dataset using metrics such as precision, recall, F1-score, ROC-AUC and a confusion matrix.", styles["BodyX"]),
        Paragraph("10. Conclusion", styles["SectionX"]),
        Paragraph("The CSV was successfully loaded and checked for structure, missing values, duplicates, numeric issues, target distribution, feature statistics and potential outliers. Review the test outcomes above before accepting the dataset. This is a data-validation report, not proof that the data or model is error-free.", styles["BodyX"])
    ]

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=15*mm,
                            leftMargin=15*mm, topMargin=14*mm, bottomMargin=15*mm)
    doc.build(story)

def main():
    for title, filename, pdf_name in DATASETS:
        csv_path = os.path.join(DATA_DIR, filename)
        pdf_path = os.path.join(OUTPUT_DIR, pdf_name)
        if not os.path.isfile(csv_path):
            print(f"ERROR: Could not find {csv_path}")
            print("Check that you are running this script from the main project folder.")
            continue
        try:
            create_report(title, csv_path, pdf_path)
            print(f"SUCCESS: Created {pdf_path}")
        except Exception as error:
            print(f"ERROR while processing {csv_path}: {error}")

if __name__ == "__main__":
    main()
