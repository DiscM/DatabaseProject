from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib


def to_float(value: str | None) -> float:
    if value is None or value == "":
        raise ValueError("Missing numeric input.")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next trading-day Brent price from the latest market row.")
    parser.add_argument("--metadata", type=Path, default=Path("model_artifacts") / "oil_price_model.json")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--market-csv", type=Path, default=Path("datasets") / "ops_market_daily.csv")
    args = parser.parse_args()

    artifact = json.loads(args.metadata.read_text(encoding="utf-8"))
    model_path = args.model or Path(artifact["model_file"])
    model = joblib.load(model_path)
    feature_columns = artifact["feature_columns"]
    with args.market_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    latest = sorted(rows, key=lambda row: row["market_date"])[-1]
    raw_features = [to_float(latest[column]) for column in feature_columns]
    prediction = model.predict([raw_features])[0]

    print(f"Input market date: {latest['market_date']}")
    print(f"Current Brent: ${float(latest['brent_price_usd']):.2f}")
    print(f"Predicted next trading-day Brent: ${prediction:.2f}")


if __name__ == "__main__":
    main()
