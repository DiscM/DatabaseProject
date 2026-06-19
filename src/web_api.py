from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from fetch_market_data import get_live_market_data
from predict_oil_price import apply_momentum_blend, monte_carlo_forecast

_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_PATH = _ROOT / "model_artifacts" / "oil_price_model.json"

# ── Cached state ────────────────────────────────────────────────────────────

_forecast_cache: dict = {"data": None, "fetched_at": 0.0}
_FORECAST_TTL = 300  # seconds (5 min)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


def _load_model() -> None:
    if not _ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {_ARTIFACT_PATH}. "
            "Run `python main.py` first to train the model."
        )
    artifact = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
    model_path = Path(artifact["model_file"])
    if not model_path.is_absolute():
        model_path = _ROOT / model_path

    app.state.artifact = artifact
    app.state.model = joblib.load(model_path)
    app.state.momentum_blend = 0.4
    app.state.momentum_window = 10
    app.state.n_sims = 500
    app.state.seed = 42
    app.state.forecast_days = 10

    sigma = float(
        artifact.get("test_metrics", {}).get("rmse")
        or artifact.get("per_horizon", {}).get("1", {}).get("rmse")
        or 1.5
    )
    app.state.sigma = sigma
    app.state.model_rmse = sigma
    app.state.feature_columns = artifact["feature_columns"]
    app.state.trained_at = artifact.get("trained_at", "unknown")


app = FastAPI(title="Oil Price Forecaster", lifespan=lifespan)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_forecast() -> dict:
    now = time.time()

    if _forecast_cache["data"] is not None and (now - _forecast_cache["fetched_at"]) < _FORECAST_TTL:
        return _forecast_cache["data"]

    try:
        live_rows = get_live_market_data()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    if len(live_rows) < 35:
        raise HTTPException(503, detail=f"Insufficient live data ({len(live_rows)} rows, need 35+)")

    sorted_rows = sorted(live_rows, key=lambda r: r["market_date"])
    last_row = sorted_rows[-1]
    last_date = last_row["market_date"]

    forecasts = monte_carlo_forecast(
        app.state.model, app.state.feature_columns, sorted_rows,
        app.state.forecast_days, app.state.n_sims, app.state.sigma, app.state.seed,
    )

    forecasts, trend_slope = apply_momentum_blend(
        forecasts, sorted_rows,
        momentum_window=app.state.momentum_window,
        blend_weight=app.state.momentum_blend,
    )

    result = {
        "current_brent": round(float(last_row.get("brent_price_usd", 0)), 2),
        "current_wti": round(float(last_row.get("wti_price_usd", 0)), 2),
        "last_data_date": last_date,
        "model_rmse": round(app.state.model_rmse, 3),
        "trend_slope": round(float(trend_slope), 4),
        "fetched_at": now,
        "trained_at": app.state.trained_at,
        "forecast": forecasts,
    }

    _forecast_cache["data"] = result
    _forecast_cache["fetched_at"] = now
    return result


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/forecast")
def get_forecast() -> dict:
    return _run_forecast()


@app.get("/", response_class=HTMLResponse)
def get_dashboard() -> str:
    try:
        data = _run_forecast()
    except HTTPException as exc:
        return f"<h2>Error</h2><pre>{exc.detail}</pre>"

    table_rows = ""
    for day in data["forecast"]:
        table_rows += f"""<tr>
            <td>{day['forecast_date']}</td>
            <td>${day['predicted_brent_usd']:.2f}</td>
            <td>${day['p10']:.2f}</td>
            <td>${day['p25']:.2f}</td>
            <td>${day['p75']:.2f}</td>
            <td>${day['p90']:.2f}</td>
        </tr>"""

    forecast_json = json.dumps(data["forecast"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oil Price Forecaster</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #222; padding: 24px; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.6em; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 24px; }}
  .prices {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .price-card {{ flex: 1; min-width: 160px; background: #fff; border-radius: 10px;
                padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }}
  .price-card .label {{ font-size: 0.82em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  .price-card .value {{ font-size: 2em; font-weight: 700; margin-top: 4px; }}
  .price-card .value.brent {{ color: #d97706; }}
  .price-card .value.wti {{ color: #2563eb; }}
  .price-card .rmse {{ font-size: 0.78em; color: #999; margin-top: 6px; }}
  #chart {{ background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px;
          overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  th {{ background: #f0f2f6; text-align: left; padding: 12px 14px; font-size: 0.82em;
        text-transform: uppercase; letter-spacing: 0.3px; color: #555; }}
  td {{ padding: 10px 14px; border-top: 1px solid #eee; font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: #fafbfc; }}
  .refresh {{ margin-top: 16px; text-align: right; color: #999; font-size: 0.82em; }}
  .refresh a {{ color: #4a90d9; text-decoration: none; }}
  .error {{ background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 16px; color: #991b1b; }}
</style>
</head>
<body>
<div class="container">
  <h1>Brent Crude Oil — Forecast Dashboard</h1>
  <p class="subtitle">10-day Monte Carlo forecast using trained Ridge model</p>

  <div class="prices">
    <div class="price-card">
      <div class="label">Brent Crude</div>
      <div class="value brent">${data['current_brent']:.2f}</div>
      <div class="rmse">RMSE &plusmn;${data['model_rmse']:.2f}</div>
    </div>
    <div class="price-card">
      <div class="label">WTI Crude</div>
      <div class="value wti">${data['current_wti']:.2f}</div>
      <div class="rmse">Spread ${data['current_brent'] - data['current_wti']:.2f}</div>
    </div>
    <div class="price-card" style="min-width:200px">
      <div class="label">As of</div>
      <div style="font-size:1.1em; font-weight:600; margin-top:8px">{data['last_data_date']}</div>
      <div class="rmse">Model trained {data['trained_at']}</div>
    </div>
  </div>

  <div id="chart"></div>

  <table>
    <thead>
      <tr>
        <th>Date</th>
        <th>Median</th>
        <th>P10</th>
        <th>P25</th>
        <th>P75</th>
        <th>P90</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>

  <div class="refresh">
    <a href="/">Refresh now</a> &mdash; auto-refreshes every 5 min
  </div>
</div>

<script>
const fc = {forecast_json};
const dates = fc.map(d => d.forecast_date);
const median = fc.map(d => d.predicted_brent_usd);
const p10 = fc.map(d => d.p10);
const p25 = fc.map(d => d.p25);
const p75 = fc.map(d => d.p75);
const p90 = fc.map(d => d.p90);

const traces = [
  {{ x: dates, y: p90, mode: 'lines', line: {{width: 0}}, showlegend: false, name: 'p90' }},
  {{ x: dates, y: p10, mode: 'lines', line: {{width: 0}},
     fill: 'tonexty', fillcolor: 'rgba(255,140,0,0.10)', name: 'P10-P90 Range' }},
  {{ x: dates, y: p75, mode: 'lines', line: {{width: 0}}, showlegend: false, name: 'p75' }},
  {{ x: dates, y: p25, mode: 'lines', line: {{width: 0}},
     fill: 'tonexty', fillcolor: 'rgba(255,140,0,0.22)', name: 'P25-P75 Range' }},
  {{ x: dates, y: median, mode: 'lines+markers',
     line: {{color: 'darkorange', width: 2.5, dash: 'dash'}},
     marker: {{size: 6, color: 'darkorange'}}, name: 'Median' }}
];

Plotly.newPlot('chart', traces, {{
  title: '10-Day Brent Price Forecast',
  template: 'plotly_white',
  hovermode: 'x unified',
  xaxis: {{title: 'Date', type: 'date'}},
  yaxis: {{title: 'Price (USD)', tickprefix: '$'}},
  legend: {{orientation: 'h', y: 1.12, x: 0}},
  margin: {{t: 60, b: 40, l: 60, r: 20}}
}});
</script>
</body>
</html>"""
