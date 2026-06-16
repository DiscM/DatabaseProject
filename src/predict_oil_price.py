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
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np

from features import compute_derived
from features import to_float_strict as to_float


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
    Run `n_sims` independent stochastic forecast paths, vectorized across
    simulations.

    All `n_sims` paths share the same forecast step, so at each step a single
    batched `model.predict()` call scores the entire `(n_sims, n_features)`
    matrix at once instead of one row at a time.  Gaussian noise N(0, sigma)
    is injected per path so paths diverge; sigma = h=1 test-set RMSE.

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

    col_index = {col: i for i, col in enumerate(feature_columns)}
    n_feat = len(feature_columns)
    rng = np.random.default_rng(seed)

    # Per-simulation rolling buffers as 2-d arrays (n_sims, history_len).
    def _tile(init: list[float]) -> np.ndarray:
        return np.tile(np.asarray(init, dtype=np.float64), (n_sims, 1))

    brent_buf     = _tile(brent_price_init)
    wti_buf       = _tile(wti_price_init)
    brent_ret_buf = _tile(brent_return_init)
    wti_ret_buf   = _tile(wti_return_init)

    def _col(buf: np.ndarray, n: int) -> np.ndarray:
        """Vectorized _buf_get: column -(n+1) if long enough else column 0."""
        return buf[:, -(n + 1)] if buf.shape[1] > n else buf[:, 0]

    def _vol(buf: np.ndarray, k: int) -> np.ndarray:
        """Vectorized rolling sample std over the last k columns."""
        if buf.shape[1] < 2:
            return np.zeros(n_sims)
        return np.std(buf[:, -k:], axis=1, ddof=1)

    # Computed-feature indices (appended after raw features by the model).
    computed_cols = ["brent_momentum_7d", "brent_accel", "vol_regime"]

    all_paths = np.empty((n_sims, forecast_days), dtype=np.float64)

    for step in range(1, forecast_days + 1):
        feats = np.empty((n_sims, n_feat + len(computed_cols)), dtype=np.float64)

        if step == 1:
            base = np.array(
                [float(last_row.get(col, 0) or 0) for col in feature_columns],
                dtype=np.float64,
            )
            feats[:, :n_feat] = base
            derived = compute_derived(
                {col: float(last_row.get(col, 0) or 0) for col in feature_columns}
            )
            feats[:, n_feat:] = np.asarray(derived, dtype=np.float64)
        else:
            brent     = brent_buf[:, -1]
            wti       = wti_buf[:, -1]
            brent_ret = brent_ret_buf[:, -1]
            wti_ret   = wti_ret_buf[:, -1]
            for col in feature_columns:
                i = col_index[col]
                if col == "brent_price_usd":
                    feats[:, i] = brent
                elif col == "wti_price_usd":
                    feats[:, i] = wti
                elif col == "brent_return":
                    feats[:, i] = brent_ret
                elif col == "wti_return":
                    feats[:, i] = wti_ret
                elif col == "brent_lag_1":
                    feats[:, i] = _col(brent_buf, 1)
                elif col == "brent_lag_3":
                    feats[:, i] = _col(brent_buf, 3)
                elif col == "brent_lag_7":
                    feats[:, i] = _col(brent_buf, 7)
                elif col == "wti_lag_1":
                    feats[:, i] = _col(wti_buf, 1)
                elif col == "wti_lag_3":
                    feats[:, i] = _col(wti_buf, 3)
                elif col == "wti_lag_7":
                    feats[:, i] = _col(wti_buf, 7)
                elif col == "brent_volatility_7d":
                    feats[:, i] = _vol(brent_ret_buf, 7)
                elif col == "brent_volatility_30d":
                    feats[:, i] = _vol(brent_ret_buf, 30)
                elif col == "wti_volatility_7d":
                    feats[:, i] = _vol(wti_ret_buf, 7)
                elif col == "wti_volatility_30d":
                    feats[:, i] = _vol(wti_ret_buf, 30)
                elif col == "brent_wti_spread":
                    feats[:, i] = brent - wti
                elif col in ("event_severity", "event_flag"):
                    feats[:, i] = 0.0
                else:
                    feats[:, i] = anchor.get(col, 0.0)

            # Computed features (vectorized): momentum, accel, vol regime.
            b_price = feats[:, col_index["brent_price_usd"]]
            b_lag1  = feats[:, col_index["brent_lag_1"]]
            b_lag7  = feats[:, col_index["brent_lag_7"]]
            v7      = feats[:, col_index["brent_volatility_7d"]]
            v30     = feats[:, col_index["brent_volatility_30d"]]
            feats[:, n_feat + 0] = b_price - b_lag7
            feats[:, n_feat + 1] = b_lag1 - b_lag7
            with np.errstate(divide="ignore", invalid="ignore"):
                feats[:, n_feat + 2] = np.where(v30 != 0.0, v7 / v30, 1.0)

        pred = np.asarray(model.predict(feats), dtype=np.float64)
        pred = pred + rng.normal(0.0, sigma, size=n_sims)
        all_paths[:, step - 1] = pred

        prev_brent = brent_buf[:, -1]
        prev_wti   = wti_buf[:, -1]
        pred_wti   = pred - spread
        brent_ret_new = np.where(prev_brent != 0, (pred - prev_brent) / prev_brent, 0.0)
        wti_ret_new   = np.where(prev_wti != 0, (pred_wti - prev_wti) / prev_wti, 0.0)

        brent_buf     = np.column_stack([brent_buf, pred])
        wti_buf       = np.column_stack([wti_buf, pred_wti])
        brent_ret_buf = np.column_stack([brent_ret_buf, brent_ret_new])
        wti_ret_buf   = np.column_stack([wti_ret_buf, wti_ret_new])

    # Aggregate across simulations
    forecasts: list[dict] = []
    for step_i in range(forecast_days):
        col = all_paths[:, step_i]
        p10, p25, p50, p75, p90 = np.percentile(col, [10, 25, 50, 75, 90])
        forecasts.append({
            "forecast_date":       estimate_future_trading_date(last_date, step_i + 1),
            "predicted_brent_usd": round(float(p50), 4),
            "trading_days_ahead":  step_i + 1,
            "p10":                 round(float(p10), 4),
            "p25":                 round(float(p25), 4),
            "p75":                 round(float(p75), 4),
            "p90":                 round(float(p90), 4),
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
) -> tuple[list[dict], float]:
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
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, prices, strict=False)) / \
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
