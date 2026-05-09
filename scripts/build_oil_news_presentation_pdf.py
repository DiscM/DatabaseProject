from __future__ import annotations

import csv
from datetime import datetime
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
    c.drawString(0.55 * inch, 0.28 * inch, "Semantic News and Oil Price Database Project")
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


def _parse_date(value: str):
    return datetime.fromisoformat(value).date()


def _load_predictions_chart_data(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    actual_col = "actual_brent_future" if "actual_brent_future" in rows[0] else "actual_brent_next"
    predicted_col = "predicted_brent_future" if "predicted_brent_future" in rows[0] else "predicted_brent_next"
    baseline_col = "baseline_previous_brent"
    return [
        {
            "date": _parse_date(row["market_date"]),
            "actual": float(row[actual_col]),
            "predicted": float(row[predicted_col]),
            "baseline": float(row[baseline_col]),
        }
        for row in rows
    ]


def _load_forecast_chart_data(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    rows.sort(key=lambda row: row["forecast_date"])
    return [
        {
            "date": _parse_date(row["forecast_date"]),
            "median": float(row["predicted_brent_usd"]),
            "p10": float(row["p10"]),
            "p25": float(row["p25"]),
            "p75": float(row["p75"]),
            "p90": float(row["p90"]),
        }
        for row in rows
    ]


def _nice_ticks(min_value, max_value, count=5):
    if max_value == min_value:
        return [min_value]
    step = (max_value - min_value) / max(1, count - 1)
    return [min_value + step * i for i in range(count)]


def _draw_chart_frame(c, x, y, w, h, title_text, subtitle_text=None):
    c.setFillColor(colors.white)
    c.rect(x, y, w, h, fill=1, stroke=0)
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.rect(x, y, w, h, fill=0, stroke=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y + h + 0.18 * inch, title_text)
    if subtitle_text:
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9.5)
        c.drawString(x, y + h + 0.02 * inch, subtitle_text)


def _draw_legend(c, items, x, y, step=95, font_size=8.5):
    current_x = x
    for label, color, style in items:
        if style == "line":
            c.setStrokeColor(color)
            c.setLineWidth(2)
            c.line(current_x, y + 4, current_x + 16, y + 4)
        elif style == "dashed":
            c.setStrokeColor(color)
            c.setLineWidth(1.5)
            c.setDash(4, 2)
            c.line(current_x, y + 4, current_x + 16, y + 4)
            c.setDash()
        else:
            c.setFillColor(color)
            c.circle(current_x + 8, y + 4, 3, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", font_size)
        c.drawString(current_x + 22, y, label)
        current_x += step


def _draw_axes(
    c,
    x,
    y,
    w,
    h,
    x_min,
    x_max,
    y_min,
    y_max,
    x_labels,
    y_label="Price (USD)",
    x_tick_values=None,
    x_tick_labels=None,
):
    plot_left = x + 0.55 * inch
    plot_bottom = y + 0.55 * inch
    plot_w = w - 0.8 * inch
    plot_h = h - 0.95 * inch

    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.line(plot_left, plot_bottom, plot_left, plot_bottom + plot_h)
    c.line(plot_left, plot_bottom, plot_left + plot_w, plot_bottom)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.5)

    for tick in _nice_ticks(y_min, y_max, 5):
        py = plot_bottom + (tick - y_min) / (y_max - y_min) * plot_h if y_max != y_min else plot_bottom
        c.setStrokeColor(colors.HexColor("#E4E9EF"))
        c.setLineWidth(0.7)
        c.line(plot_left, py, plot_left + plot_w, py)
        c.setFillColor(MUTED)
        c.drawRightString(plot_left - 6, py - 3, f"{tick:,.0f}")

    if x_tick_values is None:
        tick_count = min(6, len(x_labels))
        if tick_count >= 2:
            x_tick_values = [round(i * (len(x_labels) - 1) / (tick_count - 1)) for i in range(tick_count)]
        else:
            x_tick_values = [0]
        x_tick_labels = x_labels

    seen = set()
    for idx, tick in enumerate(x_tick_values):
        if tick in seen:
            continue
        seen.add(tick)
        px = plot_left + (tick - x_min) / (x_max - x_min) * plot_w if x_max != x_min else plot_left
        c.setStrokeColor(colors.HexColor("#E4E9EF"))
        c.setLineWidth(0.7)
        c.line(px, plot_bottom, px, plot_bottom + plot_h)
        c.setFillColor(MUTED)
        if x_tick_labels is None:
            label = f"{tick:g}"
        else:
            label = x_tick_labels[idx]
        c.saveState()
        c.translate(px - 4, plot_bottom - 18)
        c.rotate(45)
        c.drawString(0, 0, label)
        c.restoreState()

    c.setFillColor(MUTED)
    c.saveState()
    c.translate(x + 0.02 * inch, y + h / 2)
    c.rotate(90)
    c.setFont("Helvetica", 9)
    c.drawString(0, 0, y_label)
    c.restoreState()
    return plot_left, plot_bottom, plot_w, plot_h


def _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, x_value, y_value, x_min, x_max, y_min, y_max):
    px = plot_left + (x_value - x_min) / (x_max - x_min) * plot_w if x_max != x_min else plot_left
    py = plot_bottom + (y_value - y_min) / (y_max - y_min) * plot_h if y_max != y_min else plot_bottom
    return px, py


def scatter_chart(c, title_text, data, x, y, w, h):
    _draw_chart_frame(
        c,
        x,
        y,
        w,
        h,
        title_text,
        "Actual vs predicted test-set Brent prices",
    )
    xs = [row["actual"] for row in data]
    ys = [row["predicted"] for row in data]
    min_value = min(min(xs), min(ys))
    max_value = max(max(xs), max(ys))
    pad = (max_value - min_value) * 0.05 or 1
    min_value -= pad
    max_value += pad
    x_ticks = _nice_ticks(min_value, max_value, 5)
    x_tick_labels = [f"{tick:,.0f}" for tick in x_ticks]
    plot_left, plot_bottom, plot_w, plot_h = _draw_axes(
        c,
        x,
        y,
        w,
        h,
        min_value,
        max_value,
        min_value,
        max_value,
        [],
        y_label="Predicted Price (USD)",
        x_tick_values=x_ticks,
        x_tick_labels=x_tick_labels,
    )
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawCentredString(plot_left + plot_w / 2, y + 0.16 * inch, "Actual Price (USD)")
    _draw_legend(
        c,
        [("Test samples", TEAL, "point"), ("Perfect prediction", RED, "dashed")],
        x + 0.75 * inch,
        y + h - 0.35 * inch,
        step=115,
        font_size=8,
    )
    c.setStrokeColor(RED)
    c.setLineWidth(1.2)
    c.setDash(4, 3)
    x1, y1 = _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, min_value, min_value, min_value, max_value, min_value, max_value)
    x2, y2 = _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, max_value, max_value, min_value, max_value, min_value, max_value)
    c.line(x1, y1, x2, y2)
    c.setDash()
    for row in data:
        px, py = _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, row["actual"], row["predicted"], min_value, max_value, min_value, max_value)
        c.setFillColor(colors.Color(0.12, 0.54, 0.48))
        c.circle(px, py, 2.8, fill=1, stroke=0)


def _draw_line_series(c, points, color, plot_left, plot_bottom, plot_w, plot_h, x_min, x_max, y_min, y_max, width=2.0, dashed=False):
    if len(points) < 2:
        return
    c.setStrokeColor(color)
    c.setLineWidth(width)
    if dashed:
        c.setDash(5, 3)
    else:
        c.setDash()
    path = c.beginPath()
    first_x, first_y = points[0]
    px, py = _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, first_x, first_y, x_min, x_max, y_min, y_max)
    path.moveTo(px, py)
    for x_value, y_value in points[1:]:
        px, py = _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, x_value, y_value, x_min, x_max, y_min, y_max)
        path.lineTo(px, py)
    c.drawPath(path, stroke=1, fill=0)
    c.setDash()


def _draw_band(c, lower_points, upper_points, fill_color, plot_left, plot_bottom, plot_w, plot_h, x_min, x_max, y_min, y_max):
    if len(lower_points) < 2 or len(upper_points) < 2:
        return
    path = c.beginPath()
    first_x, first_y = lower_points[0]
    px, py = _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, first_x, first_y, x_min, x_max, y_min, y_max)
    path.moveTo(px, py)
    for x_value, y_value in lower_points[1:]:
        px, py = _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, x_value, y_value, x_min, x_max, y_min, y_max)
        path.lineTo(px, py)
    for x_value, y_value in reversed(upper_points):
        px, py = _plot_to_xy(plot_left, plot_bottom, plot_w, plot_h, x_value, y_value, x_min, x_max, y_min, y_max)
        path.lineTo(px, py)
    path.close()
    c.setFillColor(fill_color)
    c.setStrokeColor(fill_color)
    c.drawPath(path, stroke=0, fill=1)


def time_series_chart(c, title_text, predictions, forecast, x, y, w, h):
    _draw_chart_frame(
        c,
        x,
        y,
        w,
        h,
        title_text,
        "Historical test set, baseline, and Monte Carlo forecast window",
    )

    window_start = max(0, len(predictions) - 70)
    visible_predictions = predictions[window_start:]
    actual_points = [(i, row["actual"]) for i, row in enumerate(visible_predictions)]
    model_points = [(i, row["predicted"]) for i, row in enumerate(visible_predictions)]
    baseline_points = [(i, row["baseline"]) for i, row in enumerate(visible_predictions)]
    x_labels = [row["date"].isoformat() for row in visible_predictions]

    if forecast:
        bridge_index = len(visible_predictions) - 1
        x_labels.extend([row["date"].isoformat() for row in forecast])
        forecast_indexes = [bridge_index + i for i in range(len(forecast))]
        forecast_median = [(bridge_index, visible_predictions[-1]["predicted"])] + [
            (forecast_indexes[i], row["median"]) for i, row in enumerate(forecast)
        ]
        forecast_p10 = [(bridge_index, visible_predictions[-1]["predicted"])] + [
            (forecast_indexes[i], row["p10"]) for i, row in enumerate(forecast)
        ]
        forecast_p25 = [(bridge_index, visible_predictions[-1]["predicted"])] + [
            (forecast_indexes[i], row["p25"]) for i, row in enumerate(forecast)
        ]
        forecast_p75 = [(bridge_index, visible_predictions[-1]["predicted"])] + [
            (forecast_indexes[i], row["p75"]) for i, row in enumerate(forecast)
        ]
        forecast_p90 = [(bridge_index, visible_predictions[-1]["predicted"])] + [
            (forecast_indexes[i], row["p90"]) for i, row in enumerate(forecast)
        ]
    else:
        bridge_index = len(visible_predictions) - 1
        forecast_median = forecast_p10 = forecast_p25 = forecast_p75 = forecast_p90 = []

    all_y = [row["actual"] for row in visible_predictions] + [row["predicted"] for row in visible_predictions] + [row["baseline"] for row in visible_predictions]
    if forecast:
        all_y += [row["median"] for row in forecast] + [row["p10"] for row in forecast] + [row["p25"] for row in forecast] + [row["p75"] for row in forecast] + [row["p90"] for row in forecast]
        all_y.append(visible_predictions[-1]["predicted"])
    y_min = min(all_y)
    y_max = max(all_y)
    pad = (y_max - y_min) * 0.08 or 1
    y_min -= pad
    y_max += pad
    x_min = 0
    x_max = len(x_labels) - 1 if len(x_labels) > 1 else 1

    tick_count = min(6, len(x_labels))
    if tick_count >= 2:
        tick_indexes = [round(i * (len(x_labels) - 1) / (tick_count - 1)) for i in range(tick_count)]
    else:
        tick_indexes = [0]
    tick_labels = [x_labels[idx] for idx in tick_indexes]
    plot_left, plot_bottom, plot_w, plot_h = _draw_axes(
        c,
        x,
        y,
        w,
        h,
        x_min,
        x_max,
        y_min,
        y_max,
        x_labels,
        x_tick_values=tick_indexes,
        x_tick_labels=tick_labels,
    )

    if forecast:
        _draw_band(
            c,
            forecast_p10,
            forecast_p90,
            colors.Color(0.97, 0.82, 0.62),
            plot_left,
            plot_bottom,
            plot_w,
            plot_h,
            x_min,
            x_max,
            y_min,
            y_max,
        )
        _draw_band(
            c,
            forecast_p25,
            forecast_p75,
            colors.Color(0.98, 0.72, 0.42),
            plot_left,
            plot_bottom,
            plot_w,
            plot_h,
            x_min,
            x_max,
            y_min,
            y_max,
        )

    _draw_line_series(c, actual_points, colors.HexColor("#8CA7BE"), plot_left, plot_bottom, plot_w, plot_h, x_min, x_max, y_min, y_max, width=1.1)
    _draw_line_series(c, model_points, GOLD, plot_left, plot_bottom, plot_w, plot_h, x_min, x_max, y_min, y_max, width=2.6)
    _draw_line_series(c, baseline_points, colors.HexColor("#A5B0BA"), plot_left, plot_bottom, plot_w, plot_h, x_min, x_max, y_min, y_max, width=1.0, dashed=True)
    if forecast:
        _draw_line_series(c, forecast_median, RED, plot_left, plot_bottom, plot_w, plot_h, x_min, x_max, y_min, y_max, width=3.1, dashed=False)

    _draw_legend(
        c,
        [
            ("Actual", colors.HexColor("#8CA7BE"), "line"),
            ("Model", GOLD, "line"),
            ("Baseline", colors.HexColor("#A5B0BA"), "dashed"),
            ("Forecast median", RED, "line"),
        ],
        x + 0.75 * inch,
        y + h - 0.35 * inch,
        step=118,
        font_size=7.8,
    )


def add_page(c, page):
    footer(c, page)
    c.showPage()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=PAGE)

    # 1
    text(c, "Semantic News and Oil Price Database Project", 0.8 * inch, H - 1.1 * inch, 38, INK, True, 7.2 * inch)
    c.setStrokeColor(GOLD)
    c.setLineWidth(5)
    c.line(0.8 * inch, H - 2.05 * inch, 2.8 * inch, H - 2.05 * inch)
    text(
        c,
        "This project turns oil-market, geopolitical risk, semantic news, and country exposure CSV files into a reproducible analytics workflow with MySQL tables, SQL views, and a transparent pricing model.",
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
        "StandardScaler -> Ridge",
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
    predictions_path = Path("model_artifacts") / "test_predictions.csv"
    forecast_path = Path("model_artifacts") / "forward_forecast.csv"
    predictions = _load_predictions_chart_data(predictions_path)
    forecast = _load_forecast_chart_data(forecast_path)

    y = title(c, "The model narrowly beats a strong baseline", "Next-day oil pricing is hard because yesterday's price is already an excellent predictor.")
    scatter_chart(c, "Actual vs predicted Brent", predictions, 0.9 * inch, 1.35 * inch, 7.1 * inch, 4.4 * inch)
    metric(c, "1.516", "test RMSE, USD", 8.35 * inch, y, BLUE)
    metric(c, "1.118", "test MAE, USD", 8.35 * inch, y - 1.05 * inch, TEAL)
    metric(c, "0.967", "test R-squared", 8.35 * inch, y - 2.1 * inch, GOLD)
    text(c, "The diagonal shows perfect prediction. The cloud stays tightly clustered, which is what makes the result useful despite the small baseline gap.", 8.35 * inch, y - 3.05 * inch, 14, INK, False, 3.9 * inch)
    add_page(c, 8)

    # 9
    y = title(c, "The forecast extends from the test set", "The median forecast and probability band bridge from the last observed trading day.")
    time_series_chart(c, "Test set plus 10-day forecast", predictions, forecast, 0.9 * inch, 1.35 * inch, 8.2 * inch, 4.4 * inch)
    metric(c, "500", "Monte Carlo paths", 9.55 * inch, y, BLUE)
    metric(c, "10", "forecast trading days", 9.55 * inch, y - 1.05 * inch, TEAL)
    metric(c, "P10-P90", "outer forecast band", 9.55 * inch, y - 2.1 * inch, GOLD)
    text(c, "The shaded fan is the useful part for planning: it shows where the model is confident and where the path gets noisier.", 9.55 * inch, y - 3.05 * inch, 14, INK, False, 3.0 * inch)
    add_page(c, 9)

    # 10
    y = title(c, "The handoff is reproducible", "Everything needed to load, train, predict, and present is in the workspace.")
    text(c, "Run path", 0.9 * inch, y, 19, BLUE, True)
    bullets(c, ["python -m pip install -r requirements.txt", "python src\\load_mysql.py --replace", "python src\\train_oil_model.py", "python src\\predict_oil_price.py"], 0.9 * inch, y - 0.45 * inch, 13, 5.0 * inch)
    text(c, "Deliverables", 7.0 * inch, y, 19, TEAL, True)
    bullets(c, ["MySQL loader and views", "Database schema guide", "Trained sklearn model", "Markdown/PDF report", "Editable deck source, PPTX, PDF, and HTML"], 7.0 * inch, y - 0.45 * inch, 13, 5.0 * inch)
    text(c, "Next improvements: foreign keys and indexes, dashboarding, rolling-window validation, and model prediction tables in MySQL.", 0.9 * inch, 1.1 * inch, 15, INK, True, 11.5 * inch)
    footer(c, 10)

    c.save()
    print(OUT)


if __name__ == "__main__":
    build()
