from __future__ import annotations

from pathlib import Path

from train_oil_model import train_models
from predict_oil_price import run_forecast
from visualize_predictions import create_visualizations


def run_train(
    market_csv: Path,
    output_dir: Path,
    horizon: int = 1,
    alpha: float = 0.1,
    test_ratio: float = 0.2,
    max_horizon: int = 10,
) -> None:
    train_models(market_csv, output_dir, alpha, test_ratio, max_horizon)


def run_predict(
    artifact_path: Path,
    market_csv: Path,
    forecast_days: int = 10,
    n_sims: int = 500,
    momentum_blend: float = 0.4,
    momentum_window: int = 10,
    seed: int = 42,
) -> Path:
    return run_forecast(
        artifact_path, market_csv,
        forecast_days=forecast_days,
        n_sims=n_sims,
        momentum_blend=momentum_blend,
        momentum_window=momentum_window,
        seed=seed,
    )


def run_visualize(predictions_csv: Path, forecast_csv: Path) -> None:
    create_visualizations(
        str(predictions_csv),
        forecast_csv_path=str(forecast_csv),
    )
