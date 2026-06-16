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

from features import FEATURE_COLUMNS, COMPUTED_FEATURE_NAMES, compute_derived, to_float


def read_market_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: row["market_date"])


def build_examples(
    rows: list[dict[str, str]], horizon: int = 1
) -> tuple[list[list[float]], list[float], list[str]]:
    """Build feature/label pairs where the target is `horizon` trading days ahead."""
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    dates: list[str] = []
    for idx, row in enumerate(rows[: len(rows) - horizon]):
        future_brent = to_float(rows[idx + horizon].get("brent_price_usd"))
        raw_vals = {col: to_float(row.get(col)) for col in FEATURE_COLUMNS}
        raw_features = [raw_vals[col] for col in FEATURE_COLUMNS]
        computed = compute_derived(raw_vals)
        all_features = raw_features + computed
        if future_brent is None or any(v is None for v in all_features):
            continue
        x_rows.append([float(v) for v in all_features])
        y_rows.append(future_brent)
        dates.append(rows[idx + horizon]["market_date"])
    return x_rows, y_rows, dates


def split_chronological(
    x_rows, y_rows, dates, test_ratio
):
    split_index = max(1, int(len(x_rows) * (1.0 - test_ratio)))
    return (
        x_rows[:split_index], y_rows[:split_index], dates[:split_index],
        x_rows[split_index:], y_rows[split_index:], dates[split_index:],
    )


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    return {
        "mae":      mean_absolute_error(actual, predicted),
        "rmse":     mean_squared_error(actual, predicted) ** 0.5,
        "mape_pct": mean_absolute_percentage_error(actual, predicted) * 100,
        "r2":       r2_score(actual, predicted),
    }


def write_predictions(
    path: Path,
    dates: list[str],
    actual: list[float],
    predicted: list[float],
    baseline: list[float],
    horizon: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["market_date", "actual_brent_future", "predicted_brent_future",
                         "baseline_previous_brent", "horizon_days"])
        for d, a, p, b in zip(dates, actual, predicted, baseline):
            writer.writerow([d, a, p, b, horizon])


def train_models(
    market_csv: Path,
    output_dir: Path,
    alpha: float = 0.1,
    test_ratio: float = 0.2,
    max_horizon: int = 10,
) -> None:
    rows = read_market_rows(market_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_horizon: dict[int, dict] = {}
    h1_test_dates = h1_test_y = h1_test_preds = h1_baseline = None
    h1_train_dates = h1_train_metrics = h1_baseline_metrics = None

    print(f"Training {max_horizon} direct-horizon Ridge models "
          f"(alpha={alpha})  ...")

    for h in range(1, max_horizon + 1):
        x_rows, y_rows, dates = build_examples(rows, horizon=h)
        if len(x_rows) < 50:
            print(f"  h={h:02d}: not enough rows ({len(x_rows)}), skipping")
            continue

        train_x, train_y, train_dates, test_x, test_y, test_dates = split_chronological(
            x_rows, y_rows, dates, test_ratio
        )

        model = Pipeline([
            ("scaler",    StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ])
        model.fit(train_x, train_y)

        test_preds  = model.predict(test_x).tolist()
        train_preds = model.predict(train_x).tolist()
        m_test  = metrics(test_y,  test_preds)
        m_train = metrics(train_y, train_preds)

        model_file = output_dir / f"oil_price_model_h{h}.joblib"
        joblib.dump(model, model_file)
        model_file_rel = model_file.relative_to(Path.cwd()) if model_file.is_absolute() else model_file

        per_horizon[h] = {
            "model_file":      str(model_file_rel),
            "rmse":            round(m_test["rmse"],     4),
            "mae":             round(m_test["mae"],      4),
            "r2":              round(m_test["r2"],       6),
            "mape_pct":        round(m_test["mape_pct"], 4),
            "train_rows":      len(train_x),
            "test_rows":       len(test_x),
            "test_date_range": [test_dates[0], test_dates[-1]],
        }

        if h == 1:
            joblib.dump(model, output_dir / "oil_price_model.joblib")
            h1_test_dates    = test_dates
            h1_test_y        = test_y
            h1_test_preds    = test_preds
            h1_baseline      = [row[0] for row in test_x]
            h1_train_dates   = train_dates
            h1_train_metrics = m_train
            h1_baseline_metrics = metrics(test_y, h1_baseline)

        print(f"  h={h:02d}: RMSE={m_test['rmse']:.3f} USD  "
              f"MAE={m_test['mae']:.3f} USD  R²={m_test['r2']:.4f}")

    if not per_horizon:
        raise SystemExit("No horizon models trained — check dataset size.")

    write_predictions(
        output_dir / "test_predictions.csv",
        h1_test_dates, h1_test_y, h1_test_preds, h1_baseline, horizon=1,
    )

    _rel = lambda p: str(p.relative_to(Path.cwd()) if p.is_absolute() else p)
    artifact = {
        "model_type":              "sklearn.pipeline.Pipeline(StandardScaler, Ridge)",
        "strategy":                "direct_multi_step",
        "max_horizon_trading_days": max_horizon,
        "horizon_trading_days":    1,
        "trained_at":              date.today().isoformat(),
        "source_file":             _rel(market_csv),
        "feature_columns":         FEATURE_COLUMNS,
        "computed_feature_names":  COMPUTED_FEATURE_NAMES,
        "model_file":              _rel(output_dir / "oil_price_model.joblib"),
        "alpha":                   alpha,
        "per_horizon":             per_horizon,
        "train_rows":              per_horizon[1]["train_rows"],
        "test_rows":               per_horizon[1]["test_rows"],
        "train_date_range":        [h1_train_dates[0], h1_train_dates[-1]],
        "test_date_range":         per_horizon[1]["test_date_range"],
        "train_metrics":           h1_train_metrics,
        "test_metrics":            {k: v for k, v in per_horizon[1].items()
                                    if k in ("rmse", "mae", "r2", "mape_pct")},
        "baseline_previous_price_metrics": h1_baseline_metrics,
    }
    (output_dir / "oil_price_model.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )

    print(f"\nSaved {len(per_horizon)} models to {output_dir}/")
    print(f"h=1  test RMSE : {per_horizon[1]['rmse']:.3f} USD   R2: {per_horizon[1]['r2']:.4f}")
    if max_horizon in per_horizon:
        print(f"h={max_horizon:02d} test RMSE : "
              f"{per_horizon[max_horizon]['rmse']:.3f} USD   "
              f"R2: {per_horizon[max_horizon]['r2']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one Ridge model per forecast horizon (direct multi-step strategy)."
    )
    parser.add_argument("--market-csv",   type=Path,  default=Path("datasets") / "ops_market_daily.csv")
    parser.add_argument("--output-dir",   type=Path,  default=Path("model_artifacts"))
    parser.add_argument("--test-ratio",   type=float, default=0.2)
    parser.add_argument("--alpha",        type=float, default=0.1,
                        help="Ridge regularization strength (default: 0.1).")
    parser.add_argument("--max-horizon",  type=int,   default=10,
                        help="Number of horizon models to train, 1 through N (default: 10).")
    args = parser.parse_args()
    train_models(args.market_csv, args.output_dir, args.alpha, args.test_ratio, args.max_horizon)


if __name__ == "__main__":
    main()
