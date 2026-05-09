"""visualize_predictions.py
Interactive Plotly visualizations for the Brent oil price model.

Produces four charts — all fully zoomable and pannable with week-by-week
range-selector buttons:

  1. Actual vs Predicted (test set line chart)
  2. Prediction accuracy scatter (actual vs predicted)
  3. Model vs Baseline vs Actual (test set)
  4. Forward forecast — recent actual prices bridged to future predictions

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
    return dict(
        buttons=[
            dict(count=7,  label="1W",  step="day",   stepmode="backward"),
            dict(count=14, label="2W",  step="day",   stepmode="backward"),
            dict(count=1,  label="1M",  step="month", stepmode="backward"),
            dict(count=3,  label="3M",  step="month", stepmode="backward"),
            dict(count=6,  label="6M",  step="month", stepmode="backward"),
            dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor="#f0f2f6",
        activecolor="#4a90d9",
    )


def _time_xaxis(title: str = "Date") -> dict:
    """Standard date x-axis with range selector and slider."""
    return dict(
        title=title,
        type="date",
        rangeselector=_range_buttons(),
        rangeslider=dict(visible=True, thickness=0.05),
        showgrid=True,
        gridcolor="#e5e5e5",
    )


def _base_layout(**kwargs) -> dict:
    base = dict(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60),
        plot_bgcolor="#fafafa",
    )
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _chart_actual_vs_predicted(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    horizon_label: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["market_date"],
        y=df[actual_col],
        name="Actual Brent Price",
        line=dict(color="royalblue", width=2),
        opacity=0.85,
        hovertemplate="%{x|%Y-%m-%d}<br>Actual: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["market_date"],
        y=df[predicted_col],
        name=f"Predicted ({horizon_label})",
        line=dict(color="darkorange", width=2),
        opacity=0.85,
        hovertemplate="%{x|%Y-%m-%d}<br>Predicted: $%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title=f"Brent Crude Oil: Actual vs Predicted — {horizon_label} (Test Set)",
        xaxis=_time_xaxis(),
        yaxis=dict(title="Price (USD)", showgrid=True, gridcolor="#e5e5e5"),
    ))
    return fig


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
        marker=dict(color="seagreen", opacity=0.45, size=6),
        name="Test Samples",
        hovertemplate="Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="crimson", dash="dash", width=1.5),
        name="Perfect Prediction (y = x)",
    ))

    fig.update_layout(**_base_layout(
        title=f"Prediction Accuracy: Actual vs Predicted ({horizon_label})",
        xaxis=dict(title="Actual Price (USD)", showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="Predicted Price (USD)", showgrid=True, gridcolor="#e5e5e5"),
        hovermode="closest",
    ))
    return fig


def _chart_model_vs_baseline(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    horizon_label: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["market_date"],
        y=df[actual_col],
        name="Actual Brent Price",
        line=dict(color="royalblue", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Actual: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["market_date"],
        y=df[predicted_col],
        name=f"Model Prediction ({horizon_label})",
        line=dict(color="darkorange", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Model: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["market_date"],
        y=df["baseline_previous_brent"],
        name="Baseline (Current Day Price)",
        line=dict(color="gray", width=1.5, dash="dash"),
        opacity=0.65,
        hovertemplate="%{x|%Y-%m-%d}<br>Baseline: $%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title=f"Model vs Baseline vs Actual — {horizon_label}",
        xaxis=_time_xaxis(),
        yaxis=dict(title="Price (USD)", showgrid=True, gridcolor="#e5e5e5"),
    ))
    return fig


def _chart_forward_forecast(
    forecast_csv_path: str | None,
    market_csv_path: str | None,
    horizon_label: str,
) -> go.Figure | None:
    if forecast_csv_path is None or market_csv_path is None:
        print("Skipping forward forecast plot (no forecast CSV or market CSV provided).")
        return None

    try:
        fc = pd.read_csv(forecast_csv_path)
        market = pd.read_csv(market_csv_path)
    except FileNotFoundError as exc:
        print(f"Skipping forward forecast plot: {exc}")
        return None

    fc["forecast_date"] = pd.to_datetime(fc["forecast_date"])
    market["market_date"] = pd.to_datetime(market["market_date"])
    market = market.sort_values("market_date")

    recent = market.tail(60).copy()
    last_actual_date = recent["market_date"].iloc[-1]
    last_actual_price = float(recent["brent_price_usd"].iloc[-1])

    fc_sorted = fc.sort_values("forecast_date")

    # Bridge: prepend the last actual point so the forecast line connects cleanly
    bridge_dates = pd.concat(
        [pd.Series([last_actual_date]), fc_sorted["forecast_date"]],
        ignore_index=True,
    )
    bridge_prices = pd.concat(
        [pd.Series([last_actual_price]), fc_sorted["predicted_brent_usd"].astype(float)],
        ignore_index=True,
    )

    fig = go.Figure()

    # Recent actual prices
    fig.add_trace(go.Scatter(
        x=recent["market_date"],
        y=recent["brent_price_usd"].astype(float),
        name="Actual Brent Price (Recent)",
        line=dict(color="royalblue", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Actual: $%{y:.2f}<extra></extra>",
    ))

    # Bridged forecast line
    fig.add_trace(go.Scatter(
        x=bridge_dates,
        y=bridge_prices,
        name="Forecasted Brent Price",
        line=dict(color="crimson", width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=7, color="crimson", symbol="circle"),
        hovertemplate="%{x|%Y-%m-%d}<br>Forecast: $%{y:.2f}<extra></extra>",
    ))

    # Shaded forecast window
    fig.add_vrect(
        x0=last_actual_date,
        x1=fc_sorted["forecast_date"].max(),
        fillcolor="crimson",
        opacity=0.06,
        layer="below",
        line_width=0,
        annotation=dict(
            text="Forecast Window",
            font=dict(size=11, color="crimson"),
            align="left",
        ),
        annotation_position="top left",
    )

    fig.update_layout(**_base_layout(
        title=f"Brent Crude Oil — Forward Price Forecast ({horizon_label})",
        xaxis=_time_xaxis(),
        yaxis=dict(title="Price (USD)", showgrid=True, gridcolor="#e5e5e5"),
    ))
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_visualizations(
    predictions_csv_path: str,
    output_dir: str | None = None,   # kept for API compatibility; no files are written
    forecast_csv_path: str | None = None,
    market_csv_path: str | None = None,
) -> None:
    """
    Generate four interactive Plotly charts for the Brent oil price model.

    Charts display inline in Jupyter or open in the browser when run as a
    standalone script. No PNG files are written to disk.

    Parameters
    ----------
    predictions_csv_path : str
        Path to model_artifacts/test_predictions.csv produced by train_oil_model.py.
    output_dir : str | None
        Ignored — kept for backwards compatibility.
    forecast_csv_path : str | None
        Path to model_artifacts/forward_forecast.csv produced by predict_oil_price.py.
    market_csv_path : str | None
        Path to datasets/ops_market_daily.csv (used for the forward forecast chart).
    """
    print(f"Loading predictions from {predictions_csv_path} ...")
    try:
        df = pd.read_csv(predictions_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {predictions_csv_path}. "
              "Ensure the model has been trained first.")
        return

    df["market_date"] = pd.to_datetime(df["market_date"])
    df = df.sort_values("market_date")

    # Horizon label
    horizon_days = int(df["horizon_days"].iloc[0]) if "horizon_days" in df.columns else 5
    calendar_weeks = round(horizon_days / 5)
    week_str = f"~{calendar_weeks} Week{'s' if calendar_weeks != 1 else ''}"
    horizon_label = f"{horizon_days} Trading Days ({week_str}) Ahead"

    # Column name compatibility (old vs new schema)
    actual_col    = "actual_brent_future"    if "actual_brent_future"    in df.columns else "actual_brent_next"
    predicted_col = "predicted_brent_future" if "predicted_brent_future" in df.columns else "predicted_brent_next"

    # 1. Actual vs Predicted line chart
    fig1 = _chart_actual_vs_predicted(df, actual_col, predicted_col, horizon_label)
    fig1.show()

    # 2. Scatter: correlation / accuracy
    fig2 = _chart_scatter(df, actual_col, predicted_col, horizon_label)
    fig2.show()

    # 3. Model vs Baseline vs Actual
    fig3 = _chart_model_vs_baseline(df, actual_col, predicted_col, horizon_label)
    fig3.show()

    # 4. Forward forecast
    fig4 = _chart_forward_forecast(forecast_csv_path, market_csv_path, horizon_label)
    if fig4 is not None:
        fig4.show()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _script_dir   = Path(os.path.abspath(__file__)).parent
    _project_root = _script_dir.parent

    _predictions = str(_project_root / "model_artifacts" / "test_predictions.csv")
    _forecast    = str(_project_root / "model_artifacts" / "forward_forecast.csv")
    _market      = str(_project_root / "datasets"        / "ops_market_daily.csv")

    create_visualizations(_predictions, forecast_csv_path=_forecast, market_csv_path=_market)
