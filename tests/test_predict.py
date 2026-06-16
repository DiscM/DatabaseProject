from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict_oil_price import (
    _buf_get,
    _percentile,
    _rolling_std,
    estimate_future_trading_date,
)


class TestEstimateFutureTradingDate:
    def test_monday_to_tuesday(self):
        result = estimate_future_trading_date("2020-01-06", 1)  # Monday
        assert result == "2020-01-07"  # Tuesday

    def test_friday_to_monday(self):
        result = estimate_future_trading_date("2020-01-03", 1)  # Friday
        assert result == "2020-01-06"  # Monday

    def test_thursday_to_next_monday(self):
        result = estimate_future_trading_date("2020-01-02", 3)  # Thu -> Fri(1), Mon(2), Tue(3)
        assert result == "2020-01-07"  # Three trading days from Thu

    def test_five_trading_days(self):
        result = estimate_future_trading_date("2020-01-06", 5)  # Mon -> next Mon
        assert result == "2020-01-13"


class TestRollingStd:
    def test_single_value(self):
        assert _rolling_std([5.0]) == 0.0

    def test_two_values(self):
        assert _rolling_std([2.0, 4.0]) == pytest.approx(1.4142, abs=1e-4)

    def test_empty_list(self):
        assert _rolling_std([]) == 0.0


class TestBufGet:
    def test_in_range(self):
        buf = [10, 20, 30, 40, 50]
        assert _buf_get(buf, 1) == 40
        assert _buf_get(buf, 3) == 20

    def test_out_of_range_falls_back_to_first(self):
        buf = [10, 20, 30]
        assert _buf_get(buf, 5) == 10


class TestPercentile:
    def test_median(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 0.50) == 3.0

    def test_p10(self):
        vals = list(range(100))
        assert _percentile(vals, 0.10) == 10

    def test_p90(self):
        vals = list(range(100))
        assert _percentile(vals, 0.90) == 90

    def test_clamped_at_boundaries(self):
        vals = [1.0, 2.0, 3.0]
        assert _percentile(vals, 0.0) == 1.0
        assert _percentile(vals, 1.0) == 3.0


class TestMonteCarloIntegration:
    def test_basic_forecast_run(self, tmp_path):
        import csv
        import json

        from features import FEATURE_COLUMNS
        from predict_oil_price import monte_carlo_forecast

        data = _make_market_data(100)
        market_csv = tmp_path / "ops_market_daily.csv"
        _write_market_csv(market_csv, data)

        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=0.1))])

        from train_oil_model import build_examples, read_market_rows

        rows = read_market_rows(market_csv)
        x_rows, y_rows, _dates = build_examples(rows, horizon=1)
        model.fit(x_rows, y_rows)

        forecasts = monte_carlo_forecast(
            model, FEATURE_COLUMNS, rows, forecast_days=5, n_sims=50, sigma=1.5, seed=42,
        )
        assert len(forecasts) == 5
        for fc in forecasts:
            assert "forecast_date" in fc
            assert "predicted_brent_usd" in fc
            assert fc["p10"] <= fc["p25"] <= fc["predicted_brent_usd"] <= fc["p75"] <= fc["p90"]


def _make_row(i: int, date: str, brent: float) -> dict:
    return {
        "market_day_id": str(i),
        "market_date": date,
        "brent_price_usd": str(brent),
        "wti_price_usd": str(brent - 5),
        "dxy_index": "100",
        "vix_index": "20",
        "gpr_index": "100",
        "brent_return": "0.0",
        "wti_return": "0.0",
        "brent_lag_1": str(brent),
        "brent_lag_3": str(brent),
        "brent_lag_7": str(brent),
        "wti_lag_1": str(brent - 5),
        "wti_lag_3": str(brent - 5),
        "wti_lag_7": str(brent - 5),
        "brent_volatility_7d": "1.0",
        "brent_volatility_30d": "2.0",
        "wti_volatility_7d": "1.0",
        "wti_volatility_30d": "2.0",
        "brent_wti_spread": "5.0",
        "event_type": "",
        "event_description": "",
        "event_severity": "0",
        "event_flag": "0",
    }


def _make_market_data(n: int) -> list[dict]:
    rows = []
    from datetime import datetime, timedelta
    base_date = datetime(2020, 1, 1)
    for i in range(n):
        dt = base_date + timedelta(days=i)
        rows.append(_make_row(i, dt.strftime("%Y-%m-%d"), 60.0 + i * 0.1))
    return rows


def _write_market_csv(path: Path, rows: list[dict]) -> None:
    import csv as _csv
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
