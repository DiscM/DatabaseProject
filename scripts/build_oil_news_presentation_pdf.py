from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


OUT = Path("output") / "presentations" / "oil_news_project_presentation.pdf"
PAGE = landscape((13.333 * inch, 7.5 * inch))
W, H = PAGE

INK = colors.HexColor("#10202E")
MUTED = colors.HexColor("#5C6C7A")
FAINT = colors.HexColor("#E9EEF3")
LINE = colors.HexColor("#B8C5D0")
BLUE = colors.HexColor("#1E5C8A")
TEAL = colors.HexColor("#1F8A7B")
GOLD = colors.HexColor("#C58A1A")
RED = colors.HexColor("#B65B4B")


def text(c, value, x, y, size=18, color=INK, bold=False, max_width=None, leading=None):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    leading = leading or size * 1.22
    if max_width is None:
        c.drawString(x, y, value)
        return y - leading
    words = value.split()
    line = ""
    current_y = y
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, "Helvetica-Bold" if bold else "Helvetica", size) <= max_width:
            line = trial
        else:
            c.drawString(x, current_y, line)
            current_y -= leading
            line = word
    if line:
        c.drawString(x, current_y, line)
        current_y -= leading
    return current_y


def footer(c, page):
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(0.55 * inch, 0.28 * inch, "Semantic News and Oil Price Signals")
    c.drawRightString(W - 0.55 * inch, 0.28 * inch, f"Slide {page}")


def title(c, heading, sub=None):
    y = H - 0.72 * inch
    y = text(c, heading, 0.72 * inch, y, 28, INK, True, W - 1.44 * inch)
    if sub:
        y = text(c, sub, 0.72 * inch, y - 0.05 * inch, 13, MUTED, False, W - 1.6 * inch)
    c.setStrokeColor(TEAL)
    c.setLineWidth(3)
    c.line(0.72 * inch, y - 0.03 * inch, 2.0 * inch, y - 0.03 * inch)
    return y - 0.3 * inch


def metric(c, value, label, x, y, color=BLUE):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 30)
    c.drawString(x, y, value)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 10)
    c.drawString(x, y - 0.22 * inch, label)


def bullets(c, items, x, y, size=14, width=4.6 * inch):
    for item in items:
        y = text(c, f"- {item}", x, y, size, INK, False, width)
        y -= 0.05 * inch
    return y


def bar_chart(c, title_text, labels, values, x, y, w, h, color=BLUE):
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y + h + 0.2 * inch, title_text)
    max_value = max(values)
    bar_h = h / len(values) * 0.55
    gap = h / len(values) * 0.45
    current = y + h - bar_h
    for label, value in zip(labels, values):
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(x, current + 2, label)
        bar_x = x + 1.75 * inch
        bar_w = (w - 2.2 * inch) * value / max_value
        c.setFillColor(color)
        c.rect(bar_x, current, bar_w, bar_h, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(bar_x + bar_w + 4, current + 2, f"{value:g}")
        current -= bar_h + gap


def add_page(c, page):
    footer(c, page)
    c.showPage()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=PAGE)

    # 1
    text(c, "Semantic News to Oil Price Signals", 0.8 * inch, H - 1.1 * inch, 38, INK, True, 7.2 * inch)
    c.setStrokeColor(GOLD)
    c.setLineWidth(5)
    c.line(0.8 * inch, H - 2.05 * inch, 2.8 * inch, H - 2.05 * inch)
    text(
        c,
        "A MySQL-backed analytics project joining geopolitical risk, oil prices, events, and country exposure into a reproducible scikit-learn workflow.",
        0.8 * inch,
        H - 2.45 * inch,
        15,
        MUTED,
        False,
        6.8 * inch,
    )
    metric(c, "4,047", "daily market rows", 8.5 * inch, H - 1.55 * inch, BLUE)
    metric(c, "66,660", "country-month GPR rows", 8.5 * inch, H - 2.55 * inch, TEAL)
    metric(c, "1.516", "test RMSE, USD", 8.5 * inch, H - 3.55 * inch, GOLD)
    footer(c, 1)
    c.showPage()

    # 2
    y = title(c, "One workflow, three outputs", "Source CSVs become a queryable database, analysis surfaces, and predictive model artifacts.")
    labels = ["Datasets", "MySQL", "Views", "Model"]
    details = [
        "Semantic news, prices, events, country impact",
        "Operational tables plus dim/fact reporting tables",
        "Daily oil/news features and event/country analysis",
        "StandardScaler -> Ridge Regression",
    ]
    x = 0.9 * inch
    for i, (label, detail) in enumerate(zip(labels, details), 1):
        metric(c, str(i), label, x, y, [BLUE, TEAL, GOLD, RED][i - 1])
        text(c, detail, x, y - 0.55 * inch, 12, MUTED, False, 2.7 * inch)
        x += 3.05 * inch
    text(c, "Final handoff includes scripts, schema docs, trained model, report, PDF, and deck.", 0.9 * inch, 1.3 * inch, 16, INK, True, 11.4 * inch)
    add_page(c, 2)

    # 3
    y = title(c, "The data mixes market prices with semantic risk", "Coverage spans oil prices, GPR/news indices, events, and country-level exposure.")
    bar_chart(c, "Key dataset row counts", ["Market daily", "GPR daily", "GPR country monthly", "Events", "Countries"], [4047, 15078, 66660, 55, 18], 0.9 * inch, 1.15 * inch, 6.4 * inch, 4.2 * inch, TEAL)
    metric(c, "2010-2026", "market table date span", 8.0 * inch, y - 0.2 * inch, BLUE)
    metric(c, "1985-2026", "daily GPR date span", 8.0 * inch, y - 1.25 * inch, TEAL)
    text(c, "The modeling table aligns Brent, WTI, volatility, GPR, event flags, and lag features for supervised learning.", 8.0 * inch, y - 2.3 * inch, 14, INK, False, 4.2 * inch)
    add_page(c, 3)

    # 4
    y = title(c, "Two database layers keep analysis flexible", "Operational tables preserve source shape; dimensional tables support BI-style reporting.")
    text(c, "Operational layer", 0.9 * inch, y, 19, BLUE, True)
    bullets(c, ["ops_market_daily", "ops_gpr_daily and monthly", "ops_events", "ops_countries and petrol snapshots"], 0.9 * inch, y - 0.45 * inch)
    text(c, "Dimensional layer", 7.0 * inch, y, 19, TEAL, True)
    bullets(c, ["dim_date, dim_country, dim_event", "fact_market_daily", "fact_gpr_daily and monthly", "fact_country_impact and petrol prices"], 7.0 * inch, y - 0.45 * inch)
    text(c, "Primary joins: market_date = gpr_date, market_date = event_date, country_id, and iso3.", 0.9 * inch, 1.2 * inch, 16, INK, True, 11.4 * inch)
    add_page(c, 4)

    # 5
    y = title(c, "Views turn raw tables into analysis surfaces", "Three MySQL views package the joins needed for modeling, event study, and country impact analysis.")
    items = [
        ("vw_daily_oil_news_features", "Daily Brent/WTI, GPR, event, and volatility features for analysis and model inputs.", BLUE),
        ("vw_event_price_reaction", "Event-date oil price and risk indicators to inspect geopolitical market reactions.", TEAL),
        ("vw_country_petrol_impact", "Country exposure, petrol price snapshots, and vulnerability indicators in one reporting surface.", GOLD),
    ]
    x = 0.9 * inch
    for name, detail, color in items:
        text(c, name, x, y, 17, color, True, 3.4 * inch)
        text(c, detail, x, y - 0.55 * inch, 13, INK, False, 3.4 * inch)
        x += 4.05 * inch
    add_page(c, 5)

    # 6
    y = title(c, "A transparent scikit-learn model predicts next-day Brent", "Chronological split, standardized features, regularized linear model, exported metrics.")
    metric(c, "3,236", "training rows", 0.9 * inch, y - 0.35 * inch, BLUE)
    metric(c, "810", "test rows", 0.9 * inch, y - 1.4 * inch, TEAL)
    metric(c, "0.1", "Ridge alpha", 0.9 * inch, y - 2.45 * inch, GOLD)
    bullets(c, ["Train through 2022-12-21; test through 2026-03-12", "Scale market, risk, volatility, lag, and event features", "Fit Ridge regression and save as a joblib pipeline", "Export JSON metadata and CSV test predictions"], 5.0 * inch, y, 14, 6.8 * inch)
    add_page(c, 6)

    # 7
    y = title(c, "Features combine prices, risk, volatility, and events", "The model uses 20 fields from ops_market_daily.csv.")
    x_positions = [0.9, 3.95, 7.0, 10.05]
    rows = [(y, ["2 current oil prices", "2 macro indicators", "1 GPR risk index", "2 daily returns"]), (y - 1.65 * inch, ["6 lagged prices", "4 volatility fields", "1 Brent-WTI spread", "2 event features"])]
    for row_y, labels_row in rows:
        for idx, label in enumerate(labels_row):
            c.setFillColor([BLUE, TEAL, GOLD, RED][idx])
            c.rect(x_positions[idx] * inch, row_y - 0.55 * inch, 2.2 * inch, 0.14 * inch, fill=1, stroke=0)
            text(c, label, x_positions[idx] * inch, row_y, 17, INK, True, 2.2 * inch)
    add_page(c, 7)

    # 8
    y = title(c, "The model narrowly beats a strong baseline", "Next-day oil pricing is hard because yesterday's price is already an excellent predictor.")
    bar_chart(c, "Test error, USD", ["Model MAE", "Baseline MAE", "Model RMSE", "Baseline RMSE"], [1.118, 1.113, 1.516, 1.536], 0.9 * inch, 1.15 * inch, 6.6 * inch, 4.0 * inch, BLUE)
    metric(c, "1.516", "test RMSE, USD", 8.2 * inch, y, BLUE)
    metric(c, "1.118", "test MAE, USD", 8.2 * inch, y - 1.05 * inch, TEAL)
    metric(c, "0.967", "test R-squared", 8.2 * inch, y - 2.1 * inch, GOLD)
    text(c, "Evaluated on a future holdout period, not a shuffled split.", 8.2 * inch, y - 3.05 * inch, 14, INK, False, 4.0 * inch)
    add_page(c, 8)

    # 9
    y = title(c, "The handoff is reproducible", "Everything needed to load, train, predict, and present is in the workspace.")
    text(c, "Run path", 0.9 * inch, y, 19, BLUE, True)
    bullets(c, ["python -m pip install -r requirements.txt", "python src\\load_mysql.py --replace", "python src\\train_oil_model.py", "python src\\predict_oil_price.py"], 0.9 * inch, y - 0.45 * inch, 13, 5.0 * inch)
    text(c, "Deliverables", 7.0 * inch, y, 19, TEAL, True)
    bullets(c, ["MySQL loader and views", "Database schema guide", "Trained sklearn model", "Markdown/PDF report", "Editable deck source, PPTX, PDF, and HTML"], 7.0 * inch, y - 0.45 * inch, 13, 5.0 * inch)
    text(c, "Next improvements: foreign keys and indexes, dashboarding, rolling-window validation, and model prediction tables in MySQL.", 0.9 * inch, 1.1 * inch, 15, INK, True, 11.5 * inch)
    footer(c, 9)

    c.save()
    print(OUT)


if __name__ == "__main__":
    build()
