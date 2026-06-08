from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ProjectB_Scent_Duration_Dataset.xlsx"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def add_title_page(pdf, df, metadata, regression):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.88, "Scent Duration Prediction Report", fontsize=26, weight="bold", color="#143642")
    fig.text(0.06, 0.82, "IFF fragrance longevity modeling summary", fontsize=14, color="#4b5563")

    best = regression.iloc[0]
    lines = [
        f"Dataset rows: {df.shape[0]:,}",
        f"Dataset columns: {df.shape[1]:,}",
        f"Best regression model: {metadata.get('best_model_name', best['model'])}",
        f"Best test RMSE: {best['test_rmse']:.3f} hours",
        f"Best test R2: {best['test_r2']:.3f}",
    ]
    y = 0.68
    for line in lines:
        fig.text(0.08, y, line, fontsize=14, color="#1f2937")
        y -= 0.06

    fig.text(
        0.06,
        0.12,
        "Generated from ProjectB_Scent_Duration_Dataset.xlsx. "
        "See the executed notebook for full charts, SHAP plots, and modeling details.",
        fontsize=10,
        color="#6b7280",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf, title, dataframe, columns=None, max_rows=12):
    data = dataframe.copy()
    if columns:
        data = data[columns]
    data = data.head(max_rows)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=18, weight="bold", color="#143642", pad=18)

    display_data = data.copy()
    for col in display_data.select_dtypes(include="number").columns:
        display_data[col] = display_data[col].map(lambda value: f"{value:.3f}")

    table = ax.table(
        cellText=display_data.values,
        colLabels=display_data.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#143642")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f7f7")
        cell.set_edgecolor("#d9e2ec")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_insights_page(pdf, insights):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.9, "Key Business Insights", fontsize=20, weight="bold", color="#143642")

    y = 0.78
    for idx, insight in enumerate(insights[:5], start=1):
        fig.text(0.08, y, f"{idx}.", fontsize=13, weight="bold", color="#287271")
        fig.text(0.13, y, insight, fontsize=11, color="#1f2937", wrap=True)
        y -= 0.14

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)
    metadata = joblib.load(ROOT / "model_metadata.pkl")
    regression = pd.read_csv(OUTPUT_DIR / "regression_model_performance.csv")
    classification = pd.read_csv(OUTPUT_DIR / "classification_model_performance.csv")
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
    ingredients = pd.read_csv(OUTPUT_DIR / "ingredient_longevity_scores.csv")
    insights_path = REPORT_DIR / "business_insights.json"
    insights = json.loads(insights_path.read_text(encoding="utf-8")) if insights_path.exists() else []

    pdf_path = REPORT_DIR / "scent_duration_report.pdf"
    with PdfPages(pdf_path) as pdf:
        add_title_page(pdf, df, metadata, regression)
        add_table_page(
            pdf,
            "Regression Model Performance",
            regression,
            columns=["model", "test_rmse", "test_mae", "test_r2", "cv_rmse_mean", "cv_r2_mean"],
            max_rows=8,
        )
        add_table_page(
            pdf,
            "Classification Model Performance",
            classification,
            columns=["model", "accuracy", "weighted_f1"],
            max_rows=8,
        )
        add_table_page(
            pdf,
            "Top 10 Feature Importances",
            importance,
            columns=["feature", "importance"],
            max_rows=10,
        )
        add_insights_page(pdf, insights)
        add_table_page(
            pdf,
            "Ingredient Longevity Ranking",
            ingredients,
            columns=["ingredient", "avg_scent_duration", "std_scent_duration", "appearances", "ingredient_longevity_score"],
            max_rows=15,
        )

    print(pdf_path)


if __name__ == "__main__":
    main()
