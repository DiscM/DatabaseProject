"""predict_oil_price.py
Monte Carlo stochastic forecast of Brent crude oil prices.

Strategy: run N_SIMS independent simulation paths.  At each step the h=1
Ridge model predicts the next-day price; Gaussian noise (sigma = h=1 RMSE)
is added to represent prediction uncertainty.  Because each path diverges
independently the aggregate bands widen naturally with horizon.

Output columns in forward_forecast.csv:
    forecast_date, predicted_brent_usd (median), trading_days_ahead,
    p10, p25, p75, p90, source_date

Usage (from project root):
    python src/predict_oil_price.py
    python src/predict_oil_price.py --forecast-days 10 --n-sims 500
"""
from __future__ import annotations

import argparse
import csv
import json
import random as _random
from datetime import datetime, timedelta
from pathlib import Path

import joblib

from features import compute_derived, to_float_strict as to_float


def estimate_future_trading_date(base_date_str: str, trading_days_ahead: int) -> str:
    dt = datetime.strptime(base_date_str, "%Y-%m-%d")
    added = 0
    while added < trading_days_ahead:
        dt += timedelta(days=1)
        if dt.weekday() < 5:
            added += 1
    return dt.strftime("%Y-%m-%d")


def _rolling_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5


def _buf_get(buf: list[float], n: int) -> float:
    idx = -(n + 1)
    return buf[idx] if len(buf) > n else buf[0]


def _percentile(sorted_vals: list[float], pct: float) -> float:
    idx = max(0, min(len(sorted_vals) - 1, int(pct * len(sorted_vals))))
    return sorted_vals[idx]


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def monte_carlo_forecast(
    model,
    feature_columns: list[str],
    sorted_rows: list[dict[str, str]],
    forecast_days: int,
    n_sims: int,
    sigma: float,
    seed: int = 42,
) -> list[dict]:
    """
    Run `n_sims` independent stochastic forecast paths.

    Each path applies the h=1 Ridge model iteratively; at every step
    Gaussian noise N(0, sigma) is injected into the predicted price before
    it is fed back as the next input.  sigma = h=1 test-set RMSE, so the
    noise is calibrated to the model's actual 1-day prediction uncertainty.

    Returns one dict per forecast day with median + percentile statistics.
    """
    window   = 35
    recent   = sorted_rows[-window:]
    last_row = sorted_rows[-1]
    last_date = last_row["market_date"]

    brent_price_init  = [float(r["brent_price_usd"]) for r in recent if r.get("brent_price_usd")]
    wti_price_init    = [float(r["wti_price_usd"])   for r in recent if r.get("wti_price_usd")]
    brent_return_init = [float(r["brent_return"])     for r in recent if r.get("brent_return")]
    wti_return_init   = [float(r["wti_return"])       for r in recent if r.get("wti_return")]
    spread = float(last_row.get("brent_wti_spread", 0.0) or 0.0)

    anchor: dict[str, float] = {}
    for col in feature_columns:
        try:
            anchor[col] = to_float(last_row.get(col))
        except ValueError:
            anchor[col] = 0.0

    rng = _random.Random(seed)
    all_paths: list[list[float]] = []

    for _ in range(n_sims):
        brent_buf     = list(brent_price_init)
        wti_buf       = list(wti_price_init)
        brent_ret_buf = list(brent_return_init)
        wti_ret_buf   = list(wti_return_init)
        path: list[float] = []

        for step in range(1, forecast_days + 1):
            if step == 1:
                raw_vals    = {col: float(last_row.get(col, 0) or 0) for col in feature_columns}
                feature_vec = [raw_vals[col] for col in feature_columns] + compute_derived(raw_vals)
            else:
                brent     = brent_buf[-1]
                wti       = wti_buf[-1]
                brent_ret = brent_ret_buf[-1]
                wti_ret   = wti_ret_buf[-1]
                row: dict[str, float] = {}
                for col in feature_columns:
                    if   col == "brent_price_usd":     row[col] = brent
                    elif col == "wti_price_usd":       row[col] = wti
                    elif col == "brent_return":        row[col] = brent_ret
                    elif col == "wti_return":          row[col] = wti_ret
                    elif col == "brent_lag_1":         row[col] = _buf_get(brent_buf, 1)
                    elif col == "brent_lag_3":         row[col] = _buf_get(brent_buf, 3)
                    elif col == "brent_lag_7":         row[col] = _buf_get(brent_buf, 7)
                    elif col == "wti_lag_1":           row[col] = _buf_get(wti_buf, 1)
                    elif col == "wti_lag_3":           row[col] = _buf_get(wti_buf, 3)
                    elif col == "wti_lag_7":           row[col] = _buf_get(wti_buf, 7)
                    elif col == "brent_volatility_7d": row[col] = _rolling_std(brent_ret_buf[-7:])
                    elif col == "brent_volatility_30d":row[col] = _rolling_std(brent_ret_buf[-30:])
                    elif col == "wti_volatility_7d":   row[col] = _rolling_std(wti_ret_buf[-7:])
                    elif col == "wti_volatility_30d":  row[col] = _rolling_std(wti_ret_buf[-30:])
                    elif col == "brent_wti_spread":    row[col] = brent - wti
                    elif col in ("event_severity", "event_flag"): row[col] = 0.0
                    else:                              row[col] = anchor.get(col, 0.0)
                feature_vec = [row[col] for col in feature_columns] + compute_derived(row)

            pred_brent  = float(model.predict([feature_vec])[0])
            pred_brent += rng.gauss(0.0, sigma)   # stochastic noise
            path.append(pred_brent)

            prev_brent = brent_buf[-1]
            prev_wti   = wti_buf[-1]
            pred_wti   = pred_brent - spread
            brent_ret_new = (pred_brent - prev_brent) / prev_brent if prev_brent else 0.0
            wti_ret_new   = (pred_wti   - prev_wti)   / prev_wti   if prev_wti   else 0.0
            brent_buf.append(pred_brent)
            wti_buf.append(pred_wti)
            brent_ret_buf.append(brent_ret_new)
            wti_ret_buf.append(wti_ret_new)

        all_paths.append(path)

    # Aggregate across simulations
    forecasts: list[dict] = []
    for step_i in range(forecast_days):
        prices = sorted(p[step_i] for p in all_paths)
        forecasts.append({
            "forecast_date":       estimate_future_trading_date(last_date, step_i + 1),
            "predicted_brent_usd": round(_percentile(prices, 0.50), 4),
            "trading_days_ahead":  step_i + 1,
            "p10":                 round(_percentile(prices, 0.10), 4),
            "p25":                 round(_percentile(prices, 0.25), 4),
            "p75":                 round(_percentile(prices, 0.75), 4),
            "p90":                 round(_percentile(prices, 0.90), 4),
            "source_date":         last_date,
        })
    return forecasts


# ---------------------------------------------------------------------------
# Momentum blend  — makes the median less timid
# ---------------------------------------------------------------------------

def apply_momentum_blend(
    forecasts: list[dict],
    sorted_rows: list[dict[str, str]],
    momentum_window: int = 10,
    blend_weight: float = 0.4,
) -> list[dict]:
    """
    Correct for Ridge's mean-reversion bias by blending each step's MC median
    with a linear-trend extrapolation fitted to the last `momentum_window`
    actual trading days.

    blend_weight=0.0 -> pure MC median  (original behaviour)
    blend_weight=1.0 -> pure trend extrapolation
    blend_weight=0.4 -> 40 pct trend / 60 pct model (default)

    The entire distribution (p10/p25/p75/p90) is shifted by the same delta
    as the median, preserving the MC spread while moving the centre.
    """
    if blend_weight <= 0.0:
        return forecasts, 0.0

    recent = sorted_rows[-momentum_window:]
    prices = [float(r["brent_price_usd"]) for r in recent if r.get("brent_price_usd")]
    n = len(prices)
    if n < 2:
        return forecasts, 0.0  # not enough history

    # Ordinary least-squares slope over the window (price per trading day)
    xs = list(range(n))
    xm = sum(xs) / n
    ym = sum(prices) / n
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, prices)) / \
            sum((x - xm) ** 2 for x in xs)
    last_price = prices[-1]

    result: list[dict] = []
    for f in forecasts:
        step          = f["trading_days_ahead"]
        trend_price   = last_price + slope * step
        orig_median   = f["predicted_brent_usd"]
        blend_median  = (1.0 - blend_weight) * orig_median + blend_weight * trend_price
        shift         = blend_median - orig_median  # shift applied to all percentiles

        result.append({
            **f,
            "predicted_brent_usd": round(blend_median, 4),
            "p10": round(f["p10"] + shift, 4),
            "p25": round(f["p25"] + shift, 4),
            "p75": round(f["p75"] + shift, 4),
            "p90": round(f["p90"] + shift, 4),
        })
    return result, slope


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_forecast(
    artifact_path: Path,
    market_csv: Path,
    forecast_days: int = 10,
    n_sims: int = 500,
    momentum_blend: float = 0.4,
    momentum_window: int = 10,
    seed: int = 42,
) -> Path:
    artifact        = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_path      = Path(artifact.get("model_file", "model_artifacts/oil_price_model.joblib"))
    feature_columns = artifact["feature_columns"]

    sigma = float(
        artifact.get("test_metrics", {}).get("rmse")
        or artifact.get("per_horizon", {}).get("1", {}).get("rmse")
        or 1.5
    )

    model = joblib.load(model_path)

    with market_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    sorted_rows = sorted(rows, key=lambda r: r["market_date"])
    last_row    = sorted_rows[-1]

    print("=" * 60)
    print("  BRENT CRUDE OIL - MONTE CARLO FORECAST")
    print(f"  Paths: {n_sims}   Noise sigma: ${sigma:.2f}/day")
    print(f"  Momentum blend: {momentum_blend:.0%} trend  "
          f"({momentum_window}-day window)")
    print("=" * 60)
    print(f"  Last data date : {last_row['market_date']}")
    print(f"  Current Brent  : ${float(last_row['brent_price_usd']):.2f}")
    print(f"  Forecast window: {forecast_days} trading days")
    print("=" * 60)

    forecasts = monte_carlo_forecast(
        model, feature_columns, sorted_rows,
        forecast_days, n_sims, sigma, seed,
    )

    forecasts, trend_slope = apply_momentum_blend(
        forecasts, sorted_rows,
        momentum_window=momentum_window,
        blend_weight=momentum_blend,
    )
    direction = "up" if trend_slope >= 0 else "down"
    print(f"  Trend slope    : ${trend_slope:+.3f}/day ({direction}ward)"
          f"  blend={momentum_blend:.0%}")

    print(f"\n{'Date':<14} {'Med ($)':<12} {'P10':<10} {'P25':<10} {'P75':<10} {'P90'}")
    print("-" * 66)
    for f in forecasts:
        print(f"{f['forecast_date']:<14} "
              f"${f['predicted_brent_usd']:<10.2f} "
              f"${f['p10']:<8.2f} ${f['p25']:<8.2f} "
              f"${f['p75']:<8.2f} ${f['p90']:.2f}")

    output_dir    = Path(model_path).parent
    forecast_path = output_dir / "forward_forecast.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with forecast_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "forecast_date", "predicted_brent_usd", "trading_days_ahead",
            "p10", "p25", "p75", "p90", "source_date",
        ])
        writer.writeheader()
        writer.writerows(forecasts)
    print(f"\nSaved {len(forecasts)}-day Monte Carlo forecast -> {forecast_path}")
    return forecast_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo stochastic forecast for Brent crude oil."
    )
    parser.add_argument("--metadata",        type=Path,  default=Path("model_artifacts") / "oil_price_model.json")
    parser.add_argument("--model",           type=Path,  default=None)
    parser.add_argument("--market-csv",      type=Path,  default=Path("datasets") / "ops_market_daily.csv")
    parser.add_argument("--forecast-days",   type=int,   default=10)
    parser.add_argument("--n-sims",          type=int,   default=500,
                        help="Number of Monte Carlo simulation paths (default: 500).")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--momentum-blend",  type=float, default=0.4,
                        help="Weight given to the linear trend extrapolation "
                             "(0=pure MC, 1=pure trend, default: 0.4).")
    parser.add_argument("--momentum-window", type=int,   default=10,
                        help="Number of recent trading days used to estimate "
                             "the price trend (default: 10).")
    args = parser.parse_args()

    artifact_path = args.metadata
    model_path    = args.model
    market_csv    = args.market_csv

    if model_path is not None:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["model_file"] = str(model_path)
        artifact_path = Path("tmp_metadata.json")
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    run_forecast(
        artifact_path, market_csv,
        forecast_days=args.forecast_days,
        n_sims=args.n_sims,
        momentum_blend=args.momentum_blend,
        momentum_window=args.momentum_window,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
