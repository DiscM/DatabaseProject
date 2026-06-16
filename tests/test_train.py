from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_oil_model import build_examples, read_market_rows, split_chronological, metrics


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "market_day_id", "market_date", "brent_price_usd", "wti_price_usd",
        "dxy_index", "vix_index", "gpr_index",
        "brent_return", "wti_return",
        "brent_lag_1", "brent_lag_3", "brent_lag_7",
        "wti_lag_1", "wti_lag_3", "wti_lag_7",
        "brent_volatility_7d", "brent_volatility_30d",
        "wti_volatility_7d", "wti_volatility_30d",
        "brent_wti_spread", "event_type", "event_description",
        "event_severity", "event_flag",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_row(date: str, brent: float) -> dict:
    return {
        "market_day_id": date.replace("-", ""),
        "market_date": date,
        "brent_price_usd": str(brent),
        "wti_price_usd": str(brent - 5),
        "dxy_index": "100",
        "vix_index": "20",
        "gpr_index": "100",
        "brent_return": "0.01",
        "wti_return": "0.01",
        "brent_lag_1": str(brent - 0.5),
        "brent_lag_3": str(brent - 1.5),
        "brent_lag_7": str(brent - 3.5),
        "wti_lag_1": str(brent - 5.5),
        "wti_lag_3": str(brent - 6.5),
        "wti_lag_7": str(brent - 8.5),
        "brent_volatility_7d": "2.0",
        "brent_volatility_30d": "4.0",
        "wti_volatility_7d": "2.0",
        "wti_volatility_30d": "4.0",
        "brent_wti_spread": "5.0",
        "event_type": "",
        "event_description": "",
        "event_severity": "0",
        "event_flag": "0",
    }


class TestReadMarketRows:
    def test_sorts_by_date(self, tmp_path):
        path = tmp_path / "test.csv"
        _write_csv(path, [
            _make_row("2020-01-03", 65),
            _make_row("2020-01-01", 60),
            _make_row("2020-01-02", 62),
        ])
        rows = read_market_rows(path)
        assert len(rows) == 3
        assert rows[0]["market_date"] == "2020-01-01"
        assert rows[1]["market_date"] == "2020-01-02"
        assert rows[2]["market_date"] == "2020-01-03"


class TestBuildExamples:
    def test_basic(self, tmp_path):
        path = tmp_path / "test.csv"
        rows = [_make_row(f"2020-01-{d:02d}", 60.0 + d) for d in range(1, 31)]
        _write_csv(path, rows)
        data = read_market_rows(path)
        x, y, dates = build_examples(data, horizon=1)
        assert len(x) > 0
        assert len(x) == len(y) == len(dates)
        assert len(x[0]) == len(_get_feature_cols()) + 3  # 20 raw + 3 computed

    def test_horizon_5(self, tmp_path):
        path = tmp_path / "test.csv"
        rows = [_make_row(f"2020-01-{d:02d}", 60.0 + d) for d in range(1, 31)]
        _write_csv(path, rows)
        data = read_market_rows(path)
        x, y, dates = build_examples(data, horizon=5)
        assert len(x) > 0
        # y should be 5 days ahead; check a sample
        assert abs(y[0] - float(data[5]["brent_price_usd"])) < 1e-6


def _get_feature_cols():
    from features import FEATURE_COLUMNS
    return FEATURE_COLUMNS


class TestSplitChronological:
    def test_split_proportions(self):
        x = [[float(i)] for i in range(100)]
        y = [float(i) for i in range(100)]
        dates = [f"2020-01-{d:02d}" for d in range(1, 101)]
        train_x, train_y, train_dates, test_x, test_y, test_dates = split_chronological(
            x, y, dates, 0.2
        )
        assert len(train_x) == 80
        assert len(test_x) == 20
        assert train_dates[-1] < test_dates[0]  # chronological split


class TestMetrics:
    def test_perfect_prediction(self):
        m = metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["mape_pct"] == 0.0
        assert m["r2"] == 1.0

    def test_off_by_one(self):
        m = metrics([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert m["mae"] == 1.0
        assert m["rmse"] == 1.0
        assert m["r2"] < 1.0
