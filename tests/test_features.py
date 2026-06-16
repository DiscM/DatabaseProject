from __future__ import annotations

import pytest

from features import (
    COMPUTED_FEATURE_NAMES,
    FEATURE_COLUMNS,
    compute_derived,
    to_float,
    to_float_strict,
)


class TestToFloat:
    def test_valid_number(self):
        assert to_float("42.5") == 42.5

    def test_empty_string_returns_none(self):
        assert to_float("") is None

    def test_none_returns_none(self):
        assert to_float(None) is None

    def test_invalid_value_returns_none(self):
        assert to_float("abc") is None

    def test_negative_number(self):
        assert to_float("-10.0") == -10.0

    def test_zero(self):
        assert to_float("0") == 0.0


class TestToFloatStrict:
    def test_valid_number(self):
        assert to_float_strict("42.5") == 42.5

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Missing numeric input."):
            to_float_strict("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Missing numeric input."):
            to_float_strict(None)

    def test_zero(self):
        assert to_float_strict("0") == 0.0


class TestFeatureColumns:
    def test_length(self):
        assert len(FEATURE_COLUMNS) == 20

    def test_contains_key_fields(self):
        for col in ["brent_price_usd", "wti_price_usd", "gpr_index", "event_flag"]:
            assert col in FEATURE_COLUMNS

    def test_no_duplicates(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


class TestComputedFeatureNames:
    def test_length(self):
        assert len(COMPUTED_FEATURE_NAMES) == 3

    def test_contains_expected_names(self):
        for name in ["brent_momentum_7d", "brent_accel", "vol_regime"]:
            assert name in COMPUTED_FEATURE_NAMES


class TestComputeDerived:
    def test_all_valid_values(self):
        vals = {
            "brent_price_usd": 80.0,
            "brent_lag_1": 79.0,
            "brent_lag_7": 78.0,
            "brent_volatility_7d": 2.0,
            "brent_volatility_30d": 4.0,
        }
        result = compute_derived(vals, strict=True)
        assert len(result) == 3
        assert result[0] == pytest.approx(2.0)   # momentum_7d  = 80 - 78
        assert result[1] == pytest.approx(1.0)   # accel        = 79 - 78
        assert result[2] == pytest.approx(0.5)   # vol_regime   = 2.0 / 4.0

    def test_missing_values_strict(self):
        vals = {}
        result = compute_derived(vals, strict=True)
        assert result == [0.0, 0.0, 1.0]

    def test_missing_values_non_strict(self):
        vals = {}
        result = compute_derived(vals, strict=False)
        assert result == [0.0, 0.0, 1.0]

    def test_zero_volatility_30d_strict(self):
        vals = {
            "brent_price_usd": 80.0,
            "brent_lag_1": 79.0,
            "brent_lag_7": 78.0,
            "brent_volatility_7d": 2.0,
            "brent_volatility_30d": 0.0,
        }
        result = compute_derived(vals, strict=True)
        assert result[2] == 1.0  # fallback for div by zero
