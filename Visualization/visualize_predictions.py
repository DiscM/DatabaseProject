"""visualize_predictions.py
Interactive Plotly visualizations for the Brent oil price model.

Produces two charts -- all fully zoomable and pannable with week-by-week
range-selector buttons:

  1. Prediction accuracy scatter (actual vs predicted)
  2. Model vs Baseline vs Actual (test set) + forward forecast extension
     with a shaded +/-1 sigma confidence band

The forward forecast line bridges from the last test-set date and extends
at least 5 trading days (one week) beyond the data cut-off, with a
confidence band derived from the per-horizon test-set RMSE.

Run standalone:
    python Visualization/visualize_predictions.py

Or import and call create_visualizations() from a notebook.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _range_buttons() -> dict:
    """Week-by-week through All-time range selector buttons."""
    return {
        "buttons": [
            {"count": 7,  "label": "1W",  "step": "day",   "stepmode": "backward"},
            {"count": 14, "label": "2W",  "step": "day",   "stepmode": "backward"},
            {"count": 1,  "label": "1M",  "step": "month", "stepmode": "backward"},
            {"count": 3,  "label": "3M",  "step": "month", "stepmode": "backward"},
            {"count": 6,  "label": "6M",  "step": "month", "stepmode": "backward"},
            {"count": 1,  "label": "1Y",  "step": "year",  "stepmode": "backward"},
            {"step": "all", "label": "All"},
        ],
        "bgcolor": "#f0f2f6",
        "activecolor": "#4a90d9",
    }


def _time_xaxis(title: str = "Date", x_range: list | None = None) -> dict:
    """Standard date x-axis with range selector and slider."""
    axis = {
        "title": title,
        "type": "date",
        "rangeselector": _range_buttons(),
        "rangeslider": {"visible": True, "thickness": 0.05},
        "showgrid": True,
        "gridcolor": "#e5e5e5",
    }
    if x_range is not None:
        axis["range"] = x_range
    return axis


def _base_layout(**kwargs) -> dict:
    base = {
        "template": "plotly_white",
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "margin": {"t": 80, "b": 60},
        "plot_bgcolor": "#fafafa",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Forecast loading
# ---------------------------------------------------------------------------

def _load_forecast(
    forecast_csv_path: str | None,
    last_test_date: pd.Timestamp,
    last_test_price: float,
) -> tuple:
    """
    Load forward_forecast.csv and build bridged series starting from the
    last actual test-set point (so forecast and test lines connect visually).

    Supports two CSV formats:
      - Monte Carlo format: columns p10 / p25 / p75 / p90
      - Legacy format: columns lower_1sigma / upper_1sigma

    Returns:
        bridge_dates, bridge_median,
        bridge_p10, bridge_p25, bridge_p75, bridge_p90,
        forecast_end
    All band series are None when not present in the CSV.
    """
    _empty = (None, None, None, None, None, None, None)
    if forecast_csv_path is None:
        return _empty
    try:
        fc = pd.read_csv(forecast_csv_path)
    except FileNotFoundError:
        return _empty
    if fc.empty:
        return _empty

    fc["forecast_date"] = pd.to_datetime(fc["forecast_date"])
    fc_sorted = fc.sort_values("forecast_date")

    def _bridge(series: pd.Series) -> pd.Series:
        """Prepend the anchor (last test-set) value so lines connect."""
        return pd.concat([pd.Series([last_test_price]), series.astype(float)],
                         ignore_index=True)

    bridge_dates  = pd.concat(
        [pd.Series([last_test_date]), fc_sorted["forecast_date"]],
        ignore_index=True,
    )
    bridge_median = _bridge(fc_sorted["predicted_brent_usd"])

    # Monte Carlo percentile bands
    if all(c in fc_sorted.columns for c in ("p10", "p25", "p75", "p90")):
        bridge_p10 = _bridge(fc_sorted["p10"])
        bridge_p25 = _bridge(fc_sorted["p25"])
        bridge_p75 = _bridge(fc_sorted["p75"])
        bridge_p90 = _bridge(fc_sorted["p90"])
    # Legacy +/-1sigma fallback
    elif "lower_1sigma" in fc_sorted.columns:
        bridge_p10 = bridge_p25 = _bridge(fc_sorted["lower_1sigma"])
        bridge_p75 = bridge_p90 = _bridge(fc_sorted["upper_1sigma"])
    else:
        bridge_p10 = bridge_p25 = bridge_p75 = bridge_p90 = None

    forecast_end = fc_sorted["forecast_date"].max()
    return bridge_dates, bridge_median, bridge_p10, bridge_p25, bridge_p75, bridge_p90, forecast_end


# ---------------------------------------------------------------------------
# Forecast traces
# ---------------------------------------------------------------------------

def _add_forecast_traces(
    fig,
    bridge_dates,
    bridge_median,
    bridge_p10,
    bridge_p25,
    bridge_p75,
    bridge_p90,
    last_test_date,
    forecast_end,
):
    """Draw the Monte Carlo probability fan: outer/inner bands + median line."""
    if bridge_p10 is not None:
        fig.add_trace(go.Scatter(
            x=bridge_dates, y=bridge_p90, mode="lines",
            line={"width": 0}, showlegend=False, hoverinfo="skip", name="_p90",
        ))
        fig.add_trace(go.Scatter(
            x=bridge_dates, y=bridge_p10, mode="lines", line={"width": 0},
            fill="tonexty", fillcolor="rgba(255,140,0,0.10)",
            showlegend=True, name="P10-P90 Range",
            hovertemplate="%{x|%Y-%m-%d}<br>P10: $%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=bridge_dates, y=bridge_p75, mode="lines",
            line={"width": 0}, showlegend=False, hoverinfo="skip", name="_p75",
        ))
        fig.add_trace(go.Scatter(
            x=bridge_dates, y=bridge_p25, mode="lines", line={"width": 0},
            fill="tonexty", fillcolor="rgba(255,140,0,0.22)",
            showlegend=True, name="P25-P75 Range",
            hovertemplate="%{x|%Y-%m-%d}<br>P25: $%{y:.2f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=bridge_dates, y=bridge_median,
        name="Forecast Median",
        line={"color": "darkorange", "width": 2.5, "dash": "dash"},
        mode="lines+markers",
        marker={"size": 6, "color": "darkorange", "symbol": "circle"},
        hovertemplate="%{x|%Y-%m-%d}<br>Median: $%{y:.2f}<extra></extra>",
    ))
    fig.add_vrect(
        x0=last_test_date, x1=forecast_end,
        fillcolor="darkorange", opacity=0.04, layer="below", line_width=0,
        annotation={"text": "Forecast Window",
                        "font": {"size": 11, "color": "darkorange"}, "align": "left"},
        annotation_position="top left",
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _chart_scatter(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    horizon_label: str,
) -> go.Figure:
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[actual_col],
        y=df[predicted_col],
        mode="markers",
        marker={"color": "seagreen", "opacity": 0.45, "size": 6},
        name="Test Samples",
        hovertemplate="Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line={"color": "crimson", "dash": "dash", "width": 1.5},
        name="Perfect Prediction (y = x)",
    ))
    fig.update_layout(**_base_layout(
        title=f"Prediction Accuracy: Actual vs Predicted ({horizon_label})",
        xaxis={"title": "Actual Price (USD)", "showgrid": True, "gridcolor": "#e5e5e5"},
        yaxis={"title": "Predicted Price (USD)", "showgrid": True, "gridcolor": "#e5e5e5"},
        hovermode="closest",
    ))
    return fig


def _chart_model_vs_baseline(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    horizon_label: str,
    bridge_dates: pd.Series | None,
    bridge_median: pd.Series | None,
    bridge_p10: pd.Series | None,
    bridge_p25: pd.Series | None,
    bridge_p75: pd.Series | None,
    bridge_p90: pd.Series | None,
    last_test_date: pd.Timestamp | None,
    forecast_end: pd.Timestamp | None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["market_date"],
        y=df[actual_col],
        name="Actual Brent Price",
        line={"color": "royalblue", "width": 2},
        hovertemplate="%{x|%Y-%m-%d}<br>Actual: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["market_date"],
        y=df[predicted_col],
        name=f"Model Prediction ({horizon_label})",
        line={"color": "darkorange", "width": 2},
        hovertemplate="%{x|%Y-%m-%d}<br>Model: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["market_date"],
        y=df["baseline_previous_brent"],
        name="Baseline (Current Day Price)",
        line={"color": "gray", "width": 1.5, "dash": "dash"},
        opacity=0.65,
        hovertemplate="%{x|%Y-%m-%d}<br>Baseline: $%{y:.2f}<extra></extra>",
    ))

    # Append forecast if available
    if bridge_dates is not None:
        _add_forecast_traces(
            fig, bridge_dates, bridge_median,
            bridge_p10, bridge_p25, bridge_p75, bridge_p90,
            last_test_date, forecast_end,
        )
        x_end = forecast_end + pd.Timedelta(days=3)
    else:
        x_end = df["market_date"].max() + pd.Timedelta(days=3)

    # Default zoom: show ~6 weeks of history + full forecast window
    # This guarantees the forecast (at least 1 week) is clearly visible
    six_weeks_before_end = x_end - pd.DateOffset(weeks=6)
    x_start = min(six_weeks_before_end, last_test_date - pd.DateOffset(weeks=4))

    fig.update_layout(**_base_layout(
        title=f"Model vs Baseline vs Actual -- {horizon_label} (Test Set + Forecast)",
        xaxis=_time_xaxis(x_range=[x_start, x_end]),
        yaxis={"title": "Price (USD)", "showgrid": True, "gridcolor": "#e5e5e5"},
    ))
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_visualizations(
    predictions_csv_path: str,
    output_dir: str | None = None,   # kept for API compatibility; no files written
    forecast_csv_path: str | None = None,
    market_csv_path: str | None = None,  # kept for API compatibility; unused
) -> None:
    """
    Generate two interactive Plotly charts for the Brent oil price model.

    Charts display inline in Jupyter or open in the browser when run
    standalone. No PNG files are written to disk.

    Parameters
    ----------
    predictions_csv_path : str
        Path to model_artifacts/test_predictions.csv (from train_oil_model.py).
    forecast_csv_path : str | None
        Path to model_artifacts/forward_forecast.csv (from predict_oil_price.py).
        If present the forecast line + confidence band are added to chart 2.
    output_dir, market_csv_path : ignored (backward compat).
    """
    print(f"Loading predictions from {predictions_csv_path} ...")
    try:
        df = pd.read_csv(predictions_csv_path)
    except FileNotFoundError:
        print(f"Error: {predictions_csv_path} not found. Train the model first.")
        return

    df["market_date"] = pd.to_datetime(df["market_date"])
    df = df.sort_values("market_date")

    # Horizon label
    horizon_days   = int(df["horizon_days"].iloc[0]) if "horizon_days" in df.columns else 1
    calendar_weeks = round(horizon_days / 5)
    week_str       = f"~{calendar_weeks} Week{'s' if calendar_weeks != 1 else ''}"
    horizon_label  = f"{horizon_days} Trading Day{'s' if horizon_days != 1 else ''} ({week_str}) Ahead"

    # Column name compatibility (old vs new schema)
    actual_col    = "actual_brent_future"    if "actual_brent_future"    in df.columns else "actual_brent_next"
    predicted_col = "predicted_brent_future" if "predicted_brent_future" in df.columns else "predicted_brent_next"

    # Bridge point: connect forecast from the last predicted price
    last_test_date            = df["market_date"].iloc[-1]
    last_test_predicted_price = float(df[predicted_col].iloc[-1])

    # Load forecast (Monte Carlo format with percentile bands)
    (
        bridge_dates, bridge_median,
        bridge_p10, bridge_p25, bridge_p75, bridge_p90,
        forecast_end,
    ) = _load_forecast(forecast_csv_path, last_test_date, last_test_predicted_price)
    if forecast_csv_path and bridge_dates is None:
        print("Warning: forecast CSV not loaded -- chart will show test set only.")

    # 1. Scatter: accuracy correlation
    fig1 = _chart_scatter(df, actual_col, predicted_col, horizon_label)
    fig1.show()

    # 2. Time-series: model vs baseline + Monte Carlo fan + median
    fig2 = _chart_model_vs_baseline(
        df, actual_col, predicted_col, horizon_label,
        bridge_dates, bridge_median,
        bridge_p10, bridge_p25, bridge_p75, bridge_p90,
        last_test_date, forecast_end,
    )
    fig2.show()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _script_dir   = Path(os.path.abspath(__file__)).parent
    _project_root = _script_dir.parent

    _predictions = str(_project_root / "model_artifacts" / "test_predictions.csv")
    _forecast    = str(_project_root / "model_artifacts" / "forward_forecast.csv")

    create_visualizations(_predictions, forecast_csv_path=_forecast)
