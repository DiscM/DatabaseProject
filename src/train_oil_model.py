from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import COMPUTED_FEATURE_NAMES, FEATURE_COLUMNS, compute_derived, to_float


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
        x_rows.append(all_features)  # type: ignore[arg-type]
        y_rows.append(future_brent)
        dates.append(rows[idx + horizon]["market_date"])
    return x_rows, y_rows, dates


def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Construct the full feature matrix X once (vectorized pandas operations).

    Returns (ndarray, date_list) where the array has shape (n_rows, 23):
        20 raw features + 3 computed.
    Rows with any missing feature value are dropped.
    """
    prices = df["brent_price_usd"].values.astype(np.float64)
    lag1   = df["brent_lag_1"].values.astype(np.float64)
    lag7   = df["brent_lag_7"].values.astype(np.float64)
    vol7   = df["brent_volatility_7d"].values.astype(np.float64)
    vol30  = df["brent_volatility_30d"].values.astype(np.float64)

    raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    momentum = prices - lag7
    accel    = lag1 - lag7
    with np.errstate(divide="ignore", invalid="ignore"):
        regime = np.where(vol30 != 0.0, vol7 / vol30, 1.0)

    x_mat = np.column_stack([raw, momentum, accel, regime])

    valid = ~np.isnan(x_mat).any(axis=1)
    return x_mat[valid], df["market_date"].values[valid].tolist()  # type: ignore


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
        "mae":      float(mean_absolute_error(actual, predicted)),
        "rmse":     float(mean_squared_error(actual, predicted) ** 0.5),
        "mape_pct": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "r2":       float(r2_score(actual, predicted)),
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
        for d, a, p, b in zip(dates, actual, predicted, baseline, strict=False):
            writer.writerow([d, a, p, b, horizon])


def _rel(path: Path) -> str:
    """Portable relative path (relative to CWD if possible)."""
    return str(path.relative_to(Path.cwd()) if path.is_absolute() else path)


def _valid_length(n: int, h: int) -> int:
    """Number of valid (X, y) pairs for horizon h with n source rows."""
    return n - h


def _load_feature_data(
    market_csv: Path, cache_dir: Path, use_cache: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (x_full, prices, date_list), reusing a cached .npz when the
    source CSV is unchanged so repeated runs (e.g. alpha sweeps) skip parsing.
    """
    cache_path = cache_dir / "feature_matrix.npz"
    src_mtime = market_csv.stat().st_mtime

    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if float(cached["src_mtime"]) == src_mtime:
            return cached["x_full"], cached["prices"], cached["dates"].tolist()

    df = pd.read_csv(market_csv, encoding="utf-8-sig").sort_values("market_date")
    prices = df["brent_price_usd"].to_numpy(dtype=np.float64)
    x_full, date_list = _build_feature_matrix(df)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            x_full=x_full,
            prices=prices,
            dates=np.array(date_list, dtype="U10"),
            src_mtime=np.array(src_mtime),
        )
    return x_full, prices, date_list


def train_models(
    market_csv: Path,
    output_dir: Path,
    alpha: float = 0.1,
    test_ratio: float = 0.2,
    max_horizon: int = 10,
    use_cache: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    x_full, prices, date_list = _load_feature_data(market_csv, output_dir, use_cache)
    n = len(x_full)

    if n < 50:
        raise SystemExit("Not enough valid rows — check dataset.")

    h1_valid = _valid_length(n, 1)
    h1_split = max(1, int(h1_valid * (1.0 - test_ratio)))
    h1_train_x = x_full[:h1_split]

    scaler = StandardScaler().fit(h1_train_x)
    x_scaled = scaler.transform(x_full)

    per_horizon: dict[int, dict] = {}
    h1_test_dates = h1_test_y = h1_test_preds = h1_baseline = None
    h1_train_metrics = h1_baseline_metrics = None
    h1_model: Ridge | None = None

    print(f"Training {max_horizon} direct-horizon Ridge models "
          f"(alpha={alpha}) on {n} feature rows ...")

    for h in range(1, max_horizon + 1):
        valid = _valid_length(n, h)
        if valid < 50:
            print(f"  h={h:02d}: not enough rows ({valid}), skipping")
            continue

        y = prices[h:n]
        split_idx = max(1, int(valid * (1.0 - test_ratio)))

        x_h = x_scaled[:valid]
        train_x = x_h[:split_idx]
        test_x  = x_h[split_idx:valid]
        train_y = y[:split_idx]
        test_y  = y[split_idx:valid]

        model = Ridge(alpha=alpha)
        model.fit(train_x, train_y)

        test_preds  = model.predict(test_x).tolist()
        train_preds = model.predict(train_x).tolist()
        m_test  = metrics(test_y.tolist(),  test_preds)
        m_train = metrics(train_y.tolist(), train_preds)

        model_file = output_dir / f"oil_price_model_h{h}.joblib"
        pipeline = Pipeline([("scaler", scaler), ("regressor", model)])
        joblib.dump(pipeline, model_file)
        model_file_rel = model_file.relative_to(Path.cwd()) if model_file.is_absolute() else model_file

        per_horizon[h] = {
            "model_file":      str(model_file_rel),
            "rmse":            round(m_test["rmse"],     4),
            "mae":             round(m_test["mae"],      4),
            "r2":              round(m_test["r2"],       6),
            "mape_pct":        round(m_test["mape_pct"], 4),
            "train_rmse":      round(m_train["rmse"],     4),
            "train_mae":       round(m_train["mae"],      4),
            "train_r2":        round(m_train["r2"],       6),
            "train_mape_pct":  round(m_train["mape_pct"], 4),
            "n_features":      train_x.shape[1],
            "train_rows":      len(train_x),
            "test_rows":       len(test_x),
            "test_date_range": [date_list[split_idx], date_list[valid - 1]],
        }

        if h == 1:
            joblib.dump(pipeline, output_dir / "oil_price_model.joblib")
            h1_test_dates    = list(date_list[split_idx:valid])
            h1_test_y        = test_y.tolist()
            h1_test_preds    = test_preds
            h1_baseline      = [float(row[0]) for row in test_x]
            h1_train_metrics = m_train
            h1_baseline_metrics = metrics(test_y.tolist(), h1_baseline)
            h1_model         = model

        print(f"  h={h:02d}: RMSE={m_test['rmse']:.3f} USD  "
              f"MAE={m_test['mae']:.3f} USD  R²={m_test['r2']:.4f}")

    if not per_horizon:
        raise SystemExit("No horizon models trained — check dataset size.")

    assert h1_test_dates is not None
    assert h1_test_y is not None
    assert h1_test_preds is not None
    assert h1_model is not None
    assert h1_baseline is not None
    write_predictions(
        output_dir / "test_predictions.csv",
        h1_test_dates, h1_test_y, h1_test_preds, h1_baseline, horizon=1,
    )

    feature_names = FEATURE_COLUMNS + COMPUTED_FEATURE_NAMES
    coefficients = dict(zip(feature_names, (round(float(c), 6) for c in h1_model.coef_), strict=False))
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
        "h1_coefficients":         coefficients,
        "data_rows":               len(x_full),
        "train_rows":              per_horizon[1]["train_rows"],
        "test_rows":               per_horizon[1]["test_rows"],
        "train_date_range":        [date_list[0], date_list[h1_split - 1]],
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
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the cached feature matrix and re-parse the CSV.")
    args = parser.parse_args()
    train_models(
        args.market_csv, args.output_dir, args.alpha, args.test_ratio,
        args.max_horizon, use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
