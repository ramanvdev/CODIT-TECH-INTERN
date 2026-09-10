"""
Task 2 - Automated Report Generation
--------------------------------------
Reads tabular data from a CSV file, performs a short statistical analysis,
and generates a formatted PDF report (title page, summary table, key
metrics and an embedded chart) using ReportLab.

Usage:
    python generate_report.py sales_data.csv sales_report.pdf
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from datetime import date


def create_sample_csv(path: str):
    """Generates a small sample sales dataset if none is supplied."""
    data = {
        "Region": ["North", "South", "East", "West", "North", "South", "East", "West"],
        "Product": ["Widget-A", "Widget-A", "Widget-B", "Widget-B",
                    "Widget-B", "Widget-A", "Widget-A", "Widget-B"],
        "Units_Sold": [120, 98, 150, 87, 143, 110, 76, 132],
        "Revenue_INR": [60000, 49000, 90000, 52200, 85800, 55000, 45600, 79200],
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def analyze(df: pd.DataFrame) -> dict:
    summary = {
        "total_rows": len(df),
        "total_units": int(df["Units_Sold"].sum()),
        "total_revenue": int(df["Revenue_INR"].sum()),
        "avg_revenue_per_row": round(df["Revenue_INR"].mean(), 2),
        "top_region": df.groupby("Region")["Revenue_INR"].sum().idxmax(),
        "by_region": df.groupby("Region")[["Units_Sold", "Revenue_INR"]].sum(),
    }
    return summary


def make_chart(summary: dict, out_png: str):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    summary["by_region"]["Revenue_INR"].sort_values().plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_xlabel("Revenue (INR)")
    ax.set_title("Revenue by Region")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def build_pdf(df, summary, chart_png, out_pdf):
    doc = SimpleDocTemplate(out_pdf, pagesize=A4,
                             topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=22, spaceAfter=6)
    story = []

    story.append(Paragraph("Automated Sales Analysis Report", title_style))
    story.append(Paragraph(f"Generated on {date.today().strftime('%d %B %Y')}", styles["Normal"]))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("1. Executive Summary", styles["Heading2"]))
    story.append(Paragraph(
        f"This report was generated automatically from the source dataset "
        f"({summary['total_rows']} records). A total of {summary['total_units']} units were sold, "
        f"generating INR {summary['total_revenue']:,} in revenue. The average revenue per "
        f"transaction row was INR {summary['avg_revenue_per_row']:,}. The best performing region "
        f"was <b>{summary['top_region']}</b>.", styles["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("2. Regional Breakdown", styles["Heading2"]))
    table_data = [["Region", "Units Sold", "Revenue (INR)"]]
    for region, row in summary["by_region"].iterrows():
        table_data.append([region, int(row["Units_Sold"]), f"{int(row['Revenue_INR']):,}"])
    tbl = Table(table_data, colWidths=[5 * cm, 5 * cm, 5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86AB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("3. Revenue Chart", styles["Heading2"]))
    story.append(Image(chart_png, width=14 * cm, height=7.5 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("4. Conclusion", styles["Heading2"]))
    story.append(Paragraph(
        "The analysis indicates a healthy sales spread across all regions, with scope to "
        "improve performance in the lowest-revenue region through targeted promotions.",
        styles["BodyText"]))

    doc.build(story)
    print(f"Report saved to {out_pdf}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sales_data.csv"
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else "sales_report.pdf"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        create_sample_csv(csv_path)
        df = pd.read_csv(csv_path)

    summary = analyze(df)
    make_chart(summary, "revenue_chart.png")
    build_pdf(df, summary, "revenue_chart.png", pdf_path)
