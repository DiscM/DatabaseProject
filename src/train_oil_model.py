from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "brent_price_usd",
    "wti_price_usd",
    "dxy_index",
    "vix_index",
    "gpr_index",
    "brent_return",
    "wti_return",
    "brent_lag_1",
    "brent_lag_3",
    "brent_lag_7",
    "wti_lag_1",
    "wti_lag_3",
    "wti_lag_7",
    "brent_volatility_7d",
    "brent_volatility_30d",
    "wti_volatility_7d",
    "wti_volatility_30d",
    "brent_wti_spread",
    "event_severity",
    "event_flag",
]


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result


def read_market_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: row["market_date"])


def build_examples(
    rows: list[dict[str, str]], horizon: int = 10
) -> tuple[list[list[float]], list[float], list[str]]:
    """Build feature/label pairs where the target is `horizon` trading days ahead."""
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    dates: list[str] = []
    for idx, row in enumerate(rows[: len(rows) - horizon]):
        future_brent = to_float(rows[idx + horizon].get("brent_price_usd"))
        features = [to_float(row.get(column)) for column in FEATURE_COLUMNS]
        if future_brent is None or any(value is None for value in features):
            continue
        x_rows.append([float(value) for value in features])
        y_rows.append(future_brent)
        dates.append(rows[idx + horizon]["market_date"])
    return x_rows, y_rows, dates


def split_chronological(
    x_rows: list[list[float]], y_rows: list[float], dates: list[str], test_ratio: float
) -> tuple[list[list[float]], list[float], list[str], list[list[float]], list[float], list[str]]:
    split_index = max(1, int(len(x_rows) * (1.0 - test_ratio)))
    return (
        x_rows[:split_index],
        y_rows[:split_index],
        dates[:split_index],
        x_rows[split_index:],
        y_rows[split_index:],
        dates[split_index:],
    )


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "mape_pct": mean_absolute_percentage_error(actual, predicted) * 100,
        "r2": r2_score(actual, predicted),
    }


def write_predictions(
    path: Path,
    dates: list[str],
    actual: list[float],
    predicted: list[float],
    baseline: list[float],
    horizon: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["market_date", "actual_brent_future", "predicted_brent_future", "baseline_previous_brent", "horizon_days"])
        for d, a, p, b in zip(dates, actual, predicted, baseline):
            writer.writerow([d, a, p, b, horizon])



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Brent price model for a configurable horizon ahead.")
    parser.add_argument("--market-csv", type=Path, default=Path("datasets") / "ops_market_daily.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("model_artifacts"))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Number of trading days ahead to predict (default: 10 ≈ 2 calendar weeks).",
    )
    args = parser.parse_args()

    rows = read_market_rows(args.market_csv)
    x_rows, y_rows, dates = build_examples(rows, horizon=args.horizon)
    if len(x_rows) < 100:
        raise SystemExit("Not enough model-ready rows to train.")

    train_x, train_y, train_dates, test_x, test_y, test_dates = split_chronological(
        x_rows, y_rows, dates, args.test_ratio
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=args.alpha)),
        ]
    )
    model.fit(train_x, train_y)

    test_predictions = model.predict(test_x).tolist()
    train_predictions = model.predict(train_x).tolist()
    baseline = [row[0] for row in test_x]

    train_metrics = metrics(train_y, train_predictions)
    test_metrics = metrics(test_y, test_predictions)
    baseline_metrics = metrics(test_y, baseline)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "oil_price_model.joblib"
    joblib.dump(model, model_path)
    artifact = {
        "model_type": "sklearn.pipeline.Pipeline(StandardScaler, Ridge)",
        "target": f"{args.horizon}_trading_day_ahead_brent_price_usd",
        "horizon_trading_days": args.horizon,
        "trained_at": date.today().isoformat(),
        "source_file": str(args.market_csv),
        "feature_columns": FEATURE_COLUMNS,
        "model_file": str(model_path),
        "alpha": args.alpha,
        "train_rows": len(train_x),
        "test_rows": len(test_x),
        "train_date_range": [train_dates[0], train_dates[-1]],
        "test_date_range": [test_dates[0], test_dates[-1]],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_previous_price_metrics": baseline_metrics,
    }
    (args.output_dir / "oil_price_model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_predictions(args.output_dir / "test_predictions.csv", test_dates, test_y, test_predictions, baseline, horizon=args.horizon)

    print(f"scikit-learn model trained: {args.horizon}-trading-day-ahead Brent price (~2 calendar weeks)")
    print(f"Rows: train={len(train_x)} test={len(test_x)}")
    print(f"Test RMSE: {test_metrics['rmse']:.3f} USD")
    print(f"Test MAE:  {test_metrics['mae']:.3f} USD")
    print(f"Baseline RMSE: {baseline_metrics['rmse']:.3f} USD")
    print(f"Saved: {model_path}")
    print(f"Saved: {args.output_dir / 'oil_price_model.json'}")
    print(f"Saved: {args.output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
