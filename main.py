"""main.py
Run the full Brent oil price prediction pipeline (no SQL required).

Pipeline
--------
  1. Train  — fits the Ridge model on ops_market_daily.csv
  2. Predict — runs the Monte Carlo stochastic forecast
  3. Visualize — opens two interactive Plotly charts in the browser

Run from the project root:
    python main.py

All paths are resolved relative to this file so the script can be invoked
from any working directory.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Import path setup ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SRC  = str(_ROOT / "src")
_VIZ  = str(_ROOT / "Visualization")

for _p in (_SRC, _VIZ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local imports (after path setup) ─────────────────────────────────────────
import train_oil_model      as _train
import predict_oil_price    as _predict
import visualize_predictions as _viz


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(step: int, total: int, label: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  STEP {step}/{total} — {label}")
    print(f"{'=' * 62}")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def run_train(
    market_csv: Path,
    output_dir: Path,
    horizon: int = 1,
    alpha: float = 0.1,
    test_ratio: float = 0.2,
    max_horizon: int = 10,
) -> None:
    """Train one Ridge model per horizon (1 … max_horizon) and save artifacts."""
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import json
    from datetime import date

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _train.read_market_rows(market_csv)

    per_horizon: dict[int, dict] = {}
    h1_test_dates = h1_test_y = h1_test_preds = h1_baseline = None
    h1_train_metrics = h1_baseline_metrics = None

    print(f"  Training {max_horizon} Ridge models (alpha={alpha}) ...")

    for h in range(1, max_horizon + 1):
        x_rows, y_rows, dates = _train.build_examples(rows, horizon=h)
        if len(x_rows) < 50:
            continue

        train_x, train_y, _, test_x, test_y, test_dates = _train.split_chronological(
            x_rows, y_rows, dates, test_ratio
        )
        model = Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=alpha))])
        model.fit(train_x, train_y)

        test_preds  = model.predict(test_x).tolist()
        train_preds = model.predict(train_x).tolist()
        m_test  = _train.metrics(test_y, test_preds)
        m_train = _train.metrics(train_y, train_preds)

        mf = output_dir / f"oil_price_model_h{h}.joblib"
        joblib.dump(model, mf)
        mf_rel = mf.relative_to(Path.cwd()) if mf.is_absolute() else mf

        per_horizon[h] = {
            "model_file": str(mf_rel), "rmse": round(m_test["rmse"], 4),
            "mae": round(m_test["mae"], 4), "r2": round(m_test["r2"], 6),
            "mape_pct": round(m_test["mape_pct"], 4),
            "train_rows": len(train_x), "test_rows": len(test_x),
            "test_date_range": [test_dates[0], test_dates[-1]],
        }
        print(f"    h={h:02d}: RMSE={m_test['rmse']:.3f} USD  R²={m_test['r2']:.4f}")

        if h == 1:
            joblib.dump(model, output_dir / "oil_price_model.joblib")
            h1_test_dates, h1_test_y, h1_test_preds = test_dates, test_y, test_preds
            h1_baseline      = [row[0] for row in test_x]
            h1_train_metrics = m_train
            h1_baseline_metrics = _train.metrics(test_y, h1_baseline)

    _train.write_predictions(
        output_dir / "test_predictions.csv",
        h1_test_dates, h1_test_y, h1_test_preds, h1_baseline, horizon=1,
    )

    _rel = lambda p: str(p.relative_to(Path.cwd()) if p.is_absolute() else p)
    artifact = {
        "model_type": "sklearn.pipeline.Pipeline(StandardScaler, Ridge)",
        "strategy": "direct_multi_step", "max_horizon_trading_days": max_horizon,
        "horizon_trading_days": 1, "trained_at": date.today().isoformat(),
        "source_file": _rel(market_csv), "feature_columns": _train.FEATURE_COLUMNS,
        "computed_feature_names": _train.COMPUTED_FEATURE_NAMES,
        "model_file": _rel(output_dir / "oil_price_model.joblib"),
        "alpha": alpha, "per_horizon": per_horizon,
        "train_rows": per_horizon[1]["train_rows"],
        "test_rows": per_horizon[1]["test_rows"],
        "test_date_range": per_horizon[1]["test_date_range"],
        "train_metrics": h1_train_metrics,
        "test_metrics": {k: v for k, v in per_horizon[1].items()
                         if k in ("rmse", "mae", "r2", "mape_pct")},
        "baseline_previous_price_metrics": h1_baseline_metrics,
    }
    (output_dir / "oil_price_model.json").write_text(
        __import__("json").dumps(artifact, indent=2), encoding="utf-8"
    )
    print(f"  Saved {len(per_horizon)} models -> {output_dir}/")


def run_predict(
    artifact_path: Path,
    market_csv: Path,
    forecast_days: int = 10,
    n_sims: int = 500,
    momentum_blend: float = 0.4,
    momentum_window: int = 10,
    seed: int = 42,
) -> Path:
    """Run the Monte Carlo forecast and return the path to the output CSV."""
    import json, csv, joblib

    artifact        = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_path      = Path(artifact["model_file"])
    feature_columns = artifact["feature_columns"]
    sigma = float(
        artifact.get("test_metrics", {}).get("rmse")
        or artifact.get("per_horizon", {}).get("1", {}).get("rmse")
        or 1.5
    )
    model = joblib.load(model_path)

    with market_csv.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    sorted_rows = sorted(rows, key=lambda r: r["market_date"])
    last_row    = sorted_rows[-1]

    print(f"  Last data date : {last_row['market_date']}")
    print(f"  Current Brent  : ${float(last_row['brent_price_usd']):.2f}")
    print(f"  Paths={n_sims}  sigma=${sigma:.2f}  blend={momentum_blend:.0%}  "
          f"window={momentum_window}d")

    forecasts = _predict.monte_carlo_forecast(
        model, feature_columns, sorted_rows, forecast_days, n_sims, sigma, seed,
    )
    forecasts, slope = _predict.apply_momentum_blend(
        forecasts, sorted_rows, momentum_window, momentum_blend,
    )
    direction = "up" if slope >= 0 else "down"
    print(f"  Trend slope    : ${slope:+.3f}/day ({direction}ward)")

    output_dir    = model_path.parent
    forecast_path = output_dir / "forward_forecast.csv"
    with forecast_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "forecast_date", "predicted_brent_usd", "trading_days_ahead",
            "p10", "p25", "p75", "p90", "source_date",
        ])
        writer.writeheader()
        writer.writerows(forecasts)

    print(f"  Saved {len(forecasts)}-day forecast -> {forecast_path}")
    return forecast_path


def run_visualize(predictions_csv: Path, forecast_csv: Path) -> None:
    """Open the interactive Plotly charts in the default browser."""
    _viz.create_visualizations(
        str(predictions_csv),
        forecast_csv_path=str(forecast_csv),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    market_csv      = _ROOT / "datasets"        / "ops_market_daily.csv"
    output_dir      = _ROOT / "model_artifacts"
    artifact_path   = output_dir / "oil_price_model.json"
    predictions_csv = output_dir / "test_predictions.csv"
    forecast_csv    = output_dir / "forward_forecast.csv"

    t0 = time.time()

    _banner(1, 3, "Train Ridge model  (10 horizons, h=1..10)")
    run_train(market_csv, output_dir)

    _banner(2, 3, "Monte Carlo forecast  (500 paths, 40% momentum blend)")
    run_predict(artifact_path, market_csv)

    _banner(3, 3, "Interactive visualizations  (charts open in browser)")
    run_visualize(predictions_csv, forecast_csv)

    print(f"\nPipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
