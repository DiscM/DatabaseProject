from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib


def to_float(value: str | None) -> float:
    if value is None or value == "":
        raise ValueError("Missing numeric input.")
    return float(value)


def estimate_future_trading_date(base_date_str: str, trading_days_ahead: int) -> str:
    """Estimate a calendar date by adding N trading days (skips weekends)."""
    dt = datetime.strptime(base_date_str, "%Y-%m-%d")
    added = 0
    while added < trading_days_ahead:
        dt += timedelta(days=1)
        if dt.weekday() < 5:  # Mon-Fri
            added += 1
    return dt.strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a forward forecast of Brent crude oil prices."
    )
    parser.add_argument("--metadata", type=Path, default=Path("model_artifacts") / "oil_price_model.json")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--market-csv", type=Path, default=Path("datasets") / "ops_market_daily.csv")
    args = parser.parse_args()

    # --- Load model and config ---
    artifact = json.loads(args.metadata.read_text(encoding="utf-8"))
    model_path = args.model or Path(artifact["model_file"])
    horizon: int = artifact.get("horizon_trading_days", 5)
    model = joblib.load(model_path)
    feature_columns = artifact["feature_columns"]

    with args.market_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    sorted_rows = sorted(rows, key=lambda r: r["market_date"])
    last_date = sorted_rows[-1]["market_date"]

    # --- Single-point prediction from the latest row ---
    latest = sorted_rows[-1]
    raw_features = [to_float(latest[col]) for col in feature_columns]
    single_pred = model.predict([raw_features])[0]

    print("=" * 60)
    print("  BRENT CRUDE OIL - FORWARD PRICE FORECAST")
    print("=" * 60)
    print(f"  Last data date  : {last_date}")
    print(f"  Current Brent   : ${float(latest['brent_price_usd']):.2f}")
    print(f"  Model horizon   : {horizon} trading days")
    print(f"  Headline pred.  : ${single_pred:.2f}  ({horizon} days out)")
    print("=" * 60)

    # --- Multi-day forward forecast ---
    # The model predicts price at (source_date + horizon).
    # By feeding the last `horizon` rows of actual data, each row produces
    # a prediction for a different future date:
    #   row at D-horizon+1  →  predicts D+1
    #   row at D-horizon+2  →  predicts D+2
    #   ...
    #   row at D            →  predicts D+horizon
    # This gives us `horizon` predicted future prices.
    forecast_source_rows = sorted_rows[-horizon:]
    forecasts = []

    print(f"\n{'Date':<14} {'Days Ahead':<14} {'Predicted ($)':<16} {'Based On'}")
    print("-" * 60)

    for i, row in enumerate(forecast_source_rows):
        try:
            features = [to_float(row[col]) for col in feature_columns]
        except (ValueError, KeyError):
            continue
        pred = model.predict([features])[0]
        days_ahead = i + 1
        future_date = estimate_future_trading_date(last_date, days_ahead)
        source_date = row["market_date"]
        forecasts.append({
            "forecast_date": future_date,
            "predicted_brent_usd": round(pred, 4),
            "trading_days_ahead": days_ahead,
            "source_date": source_date,
        })
        print(f"{future_date:<14} +{days_ahead:<13} ${pred:<15.2f} {source_date}")

    # --- Save forecast CSV ---
    output_dir = Path(artifact["model_file"]).parent
    forecast_path = output_dir / "forward_forecast.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with forecast_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["forecast_date", "predicted_brent_usd", "trading_days_ahead", "source_date"],
        )
        writer.writeheader()
        writer.writerows(forecasts)

    print(f"\nSaved {len(forecasts)}-day forward forecast to {forecast_path}")


if __name__ == "__main__":
    main()
