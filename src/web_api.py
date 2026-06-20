from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from fetch_market_data import get_live_market_data
from predict_oil_price import apply_momentum_blend, monte_carlo_forecast

_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_PATH = _ROOT / "model_artifacts" / "oil_price_model.json"

_forecast_cache: dict = {"data": None, "fetched_at": 0.0, "key": ""}
_FORECAST_TTL = 300


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_models()
    yield


def _load_models() -> None:
    if not _ARTIFACT_PATH.exists():
        raise RuntimeError("Model artifact not found — run `python main.py --rf` first.")

    artifact = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))

    ridge_path = Path(artifact["model_file"])
    if not ridge_path.is_absolute():
        ridge_path = _ROOT / ridge_path
    app.state.model_ridge = joblib.load(ridge_path)

    app.state.model_rf = None
    if artifact.get("rf_available"):
        rf_path = Path(artifact["rf_model_file"])
        if not rf_path.is_absolute():
            rf_path = _ROOT / rf_path
        app.state.model_rf = joblib.load(rf_path)

    app.state.artifact = artifact
    app.state.feature_columns = artifact["feature_columns"]
    app.state.trained_at = artifact.get("trained_at", "unknown")
    app.state.momentum_blend = 0.4
    app.state.momentum_window = 10
    app.state.n_sims = 500
    app.state.seed = 42
    app.state.max_horizon = artifact.get("max_horizon_trading_days", 10)

    sigma = float(
        artifact.get("test_metrics", {}).get("rmse")
        or artifact.get("per_horizon", {}).get("1", {}).get("rmse")
        or 1.5
    )
    app.state.sigma_ridge = sigma
    app.state.sigma_rf = sigma

    if artifact.get("rf_available") and artifact.get("rf_test_metrics"):
        app.state.sigma_rf = float(artifact["rf_test_metrics"].get("rmse", sigma))

    app.state.model_ridge_rmse = sigma
    app.state.model_rf_rmse = app.state.sigma_rf


app = FastAPI(title="Oil Price Forecaster", lifespan=lifespan)


def _get_model(name: str):
    if name == "rf":
        if app.state.model_rf is None:
            raise HTTPException(400, "Random Forest model not trained. Run `python src/train_oil_model.py --rf`.")
        return app.state.model_rf, app.state.sigma_rf
    return app.state.model_ridge, app.state.sigma_ridge


def _run_forecast(forecast_days: int, model_name: str) -> dict:
    now = time.time()
    cache_key = f"{forecast_days}_{model_name}"

    cached = _forecast_cache["data"]
    if cached is not None and _forecast_cache["key"] == cache_key and (now - _forecast_cache["fetched_at"]) < _FORECAST_TTL:
        return cached

    try:
        live_rows = get_live_market_data()
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    if len(live_rows) < 35:
        raise HTTPException(503, detail=f"Insufficient live data ({len(live_rows)} rows, need 35+)")

    sorted_rows = sorted(live_rows, key=lambda r: r["market_date"])
    last_row = sorted_rows[-1]
    last_date = last_row["market_date"]

    model, sigma = _get_model(model_name)
    max_fc = min(forecast_days, app.state.max_horizon)

    forecasts = monte_carlo_forecast(
        model, app.state.feature_columns, sorted_rows,
        max_fc, app.state.n_sims, sigma, app.state.seed,
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
        "model_rmse": round(sigma, 3),
        "trend_slope": round(float(trend_slope), 4),
        "fetched_at": now,
        "trained_at": app.state.trained_at,
        "model_name": model_name,
        "forecast_days": max_fc,
        "forecast": forecasts,
        "rf_available": app.state.model_rf is not None,
    }

    _forecast_cache["data"] = result
    _forecast_cache["fetched_at"] = now
    _forecast_cache["key"] = cache_key
    return result


@app.get("/forecast")
def get_forecast(
    days: int = Query(10, ge=5, le=15, description="Forecast horizon in trading days"),
    model: str = Query("ridge", pattern="^(ridge|rf)$", description="Model algorithm"),
) -> dict:
    return _run_forecast(days, model)


@app.get("/", response_class=HTMLResponse)
def get_dashboard() -> str:
    try:
        data = _run_forecast(10, "ridge")
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

    rf_disabled = "" if data["rf_available"] else "disabled"
    rf_label = "Random Forest" if data["rf_available"] else "Random Forest (not trained)"

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

  .controls {{ display: flex; gap: 24px; align-items: center; margin-bottom: 16px;
               background: #fff; border-radius: 10px; padding: 16px 20px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.08); flex-wrap: wrap; }}
  .controls label {{ font-size: 0.85em; color: #555; display: flex; align-items: center; gap: 8px; }}
  .controls input[type=range] {{ width: 180px; }}
  .controls .range-value {{ font-weight: 600; color: #222; min-width: 3em; }}
  .controls select {{ padding: 6px 10px; border-radius: 6px; border: 1px solid #ccc;
                     font-size: 0.9em; background: #fafafa; }}
  .controls .model-meta {{ font-size: 0.78em; color: #999; }}
  .controls .status {{ margin-left: auto; font-size: 0.82em; color: #999; }}
  .controls .status.loading {{ color: #4a90d9; }}

  #chart {{ background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 16px; }}
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
  <p class="subtitle">Live 10-day Monte Carlo forecast via trained Ridge model</p>

  <div class="prices">
    <div class="price-card">
      <div class="label">Brent Crude</div>
      <div class="value brent" id="brent-price">${data['current_brent']:.2f}</div>
      <div class="rmse">RMSE &plusmn;<span id="model-rmse">{data['model_rmse']:.2f}</span></div>
    </div>
    <div class="price-card">
      <div class="label">WTI Crude</div>
      <div class="value wti" id="wti-price">${data['current_wti']:.2f}</div>
      <div class="rmse">Spread $<span id="spread">{data['current_brent'] - data['current_wti']:.2f}</span></div>
    </div>
    <div class="price-card" style="min-width:200px">
      <div class="label">As of</div>
      <div style="font-size:1.1em; font-weight:600; margin-top:8px" id="last-date">{data['last_data_date']}</div>
      <div class="rmse">Trained {data['trained_at']}</div>
    </div>
  </div>

  <div class="controls">
    <label>
      Forecast range:
      <input type="range" id="days-slider" min="5" max="15" value="10"
             oninput="updateForecast()">
      <span class="range-value" id="days-label">10 trading days</span>
    </label>
    <label>
      Algorithm:
      <select id="model-select" onchange="updateForecast()">
        <option value="ridge">Ridge (Linear)</option>
        <option value="rf" {rf_disabled}>{rf_label}</option>
      </select>
      <span class="model-meta">RMSE: Ridge $1.52 / RF $1.75</span>
    </label>
    <span class="status" id="status"></span>
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
    <tbody id="forecast-body">
      {table_rows}
    </tbody>
  </table>

  <div class="refresh">
    <a href="/">Refresh now</a> &mdash; auto-refreshes every 5 min
  </div>
</div>

<script>
let forecastData = {json.dumps(data["forecast"])};

function plotChart(fc) {{
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
    title: `Brent Price Forecast — ${{fc.length}} trading days`,
    template: 'plotly_white',
    hovermode: 'x unified',
    xaxis: {{title: 'Date', type: 'date'}},
    yaxis: {{title: 'Price (USD)', tickprefix: '$'}},
    legend: {{orientation: 'h', y: 1.12, x: 0}},
    margin: {{t: 60, b: 40, l: 60, r: 20}}
  }});
}}

function updateTable(fc) {{
  const tbody = document.getElementById('forecast-body');
  tbody.innerHTML = '';
  fc.forEach(d => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{d.forecast_date}}</td>
      <td>$${{d.predicted_brent_usd.toFixed(2)}}</td>
      <td>$${{d.p10.toFixed(2)}}</td>
      <td>$${{d.p25.toFixed(2)}}</td>
      <td>$${{d.p75.toFixed(2)}}</td>
      <td>$${{d.p90.toFixed(2)}}</td>`;
    tbody.appendChild(tr);
  }});
}}

function updateForecast() {{
  const days = document.getElementById('days-slider').value;
  const model = document.getElementById('model-select').value;
  document.getElementById('days-label').textContent = days + ' trading days';
  const status = document.getElementById('status');
  status.textContent = 'Loading\u2026';
  status.className = 'status loading';

  fetch(`/forecast?days=${{days}}&model=${{model}}`)
    .then(r => r.json())
    .then(d => {{
      forecastData = d.forecast;
      document.getElementById('brent-price').textContent = '$' + d.current_brent.toFixed(2);
      document.getElementById('wti-price').textContent = '$' + d.current_wti.toFixed(2);
      document.getElementById('spread').textContent = (d.current_brent - d.current_wti).toFixed(2);
      document.getElementById('last-date').textContent = d.last_data_date;
      document.getElementById('model-rmse').textContent = d.model_rmse.toFixed(2);
      document.getElementById('status').textContent = d.model_name === 'rf' ? 'Random Forest' : 'Ridge';
      document.getElementById('status').className = 'status';
      plotChart(d.forecast);
      updateTable(d.forecast);
    }})
    .catch(err => {{
      document.getElementById('status').textContent = 'Error: ' + err.message;
      document.getElementById('status').className = 'status';
    }});
}}

plotChart(forecastData);

/* Auto-refresh every 5 min */
setInterval(() => updateForecast(), 300000);
</script>
</body>
</html>"""
