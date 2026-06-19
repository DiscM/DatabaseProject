from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fetch_market_data import build_market_rows, download_market_history
from predict_oil_price import apply_momentum_blend, monte_carlo_forecast

ROOT = Path(__file__).resolve().parent

ARTIFACT_PATH = ROOT / "model_artifacts" / "oil_price_model.json"
MODEL_PATH = ROOT / "model_artifacts" / "oil_price_model.joblib"

CUTOFF_OFFSETS = [5, 10, 20, 30]  # trading days before the end


def _load_model():
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    sigma = float(
        artifact.get("test_metrics", {}).get("rmse")
        or artifact.get("per_horizon", {}).get("1", {}).get("rmse")
        or 1.5
    )
    return model, artifact["feature_columns"], sigma


def run_backtest() -> None:
    print("=" * 72)
    print("  BACKTEST — comparing model forecasts vs actual market prices")
    print("=" * 72)

    model, feature_cols, sigma = _load_model()

    all_rows = build_market_rows(download_market_history(120))
    n_total = len(all_rows)
    print(f"\n  Total history: {n_total} trading days")
    print(f"  Date range:    {all_rows[0]['market_date']} — {all_rows[-1]['market_date']}\n")

    cutoff_errors = {}

    for offset in CUTOFF_OFFSETS:
        cutoff = n_total - offset
        if cutoff < 35:
            print(f"  [skip] offset={offset}d — insufficient history")
            continue

        historical = all_rows[:cutoff]
        actual_future = all_rows[cutoff : cutoff + 10]
        actual_dates = [r["market_date"] for r in actual_future]
        actual_prices = np.array([float(r["brent_price_usd"]) for r in actual_future])

        if len(actual_future) < 1:
            continue

        sorted_rows = sorted(historical, key=lambda r: r["market_date"])

        forecasts = monte_carlo_forecast(
            model, feature_cols, sorted_rows,
            forecast_days=min(10, len(actual_future)),
            n_sims=500, sigma=sigma, seed=42,
        )
        forecasts, _slope = apply_momentum_blend(forecasts, sorted_rows)

        forecast_dates = [f["forecast_date"] for f in forecasts]
        forecast_medians = np.array([f["predicted_brent_usd"] for f in forecasts])
        forecast_p10 = np.array([f["p10"] for f in forecasts])
        forecast_p90 = np.array([f["p90"] for f in forecasts])

        overlap = min(len(actual_prices), len(forecast_medians))
        actual_p = actual_prices[:overlap]
        preds = forecast_medians[:overlap]

        errors = actual_p - preds
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mape = float(np.mean(np.abs(errors / actual_p)) * 100)
        within_p10_p90 = float(np.mean((actual_p >= forecast_p10[:overlap]) & (actual_p <= forecast_p90[:overlap]))) * 100

        cutoff_errors[offset] = {
            "mae": mae, "rmse": rmse, "mape_pct": mape,
            "within_p10_p90_pct": within_p10_p90,
            "n": overlap,
        }

        last_hist = historical[-1]
        print(f"\n  ── Cutoff {offset}d ago  ({last_hist['market_date']}) ──")
        print(f"  {'Date':<14} {'Actual':>8} {'Forecast':>9} {'Error':>7}  {'P10':>7} {'P90':>7}  {'In Band':>8}")
        print(f"  {'-'*14} {'-'*8} {'-'*9} {'-'*7}  {'-'*7} {'-'*7}  {'-'*8}")
        for i in range(overlap):
            in_band = "✓" if forecast_p10[i] <= actual_p[i] <= forecast_p90[i] else "✗"
            err = actual_p[i] - preds[i]
            print(f"  {forecast_dates[i]:<14} {actual_p[i]:>8.2f} {preds[i]:>9.2f} {err:>+7.2f}  "
                  f"{forecast_p10[i]:>7.2f} {forecast_p90[i]:>7.2f}  {in_band:>8}")

        print(f"  {'-'*63}")
        print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%  "
              f"within P10-P90={within_p10_p90:.0f}%")

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Offset':>6}  {'Cutoff Date':<14}  {'MAE':>8}  {'RMSE':>7}  "
          f"{'MAPE':>6}  {'InBand%':>7}  {'Days':>5}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*5}")
    for offset in CUTOFF_OFFSETS:
        if offset not in cutoff_errors:
            continue
        e = cutoff_errors[offset]
        cutoff_date = all_rows[n_total - offset - 1]["market_date"]
        print(f"  {offset:>6}d  {cutoff_date:<14}  {e['mae']:>8.3f}  {e['rmse']:>7.3f}  "
              f"{e['mape_pct']:>6.2f}%  {e['within_p10_p90_pct']:>6.0f}%  {e['n']:>5d}")


if __name__ == "__main__":
    run_backtest()
