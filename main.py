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

_ROOT = Path(__file__).resolve().parent
_SRC  = str(_ROOT / "src")
_VIZ  = str(_ROOT / "Visualization")

for _p in (_SRC, _VIZ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import run_train, run_predict, run_visualize


def _banner(step: int, total: int, label: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  STEP {step}/{total} — {label}")
    print(f"{'=' * 62}")


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
