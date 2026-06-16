from __future__ import annotations

from typing import Any

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

COMPUTED_FEATURE_NAMES = [
    "brent_momentum_7d",
    "brent_accel",
    "vol_regime",
]


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_float_strict(value: str | None) -> float:
    if value is None or value == "":
        raise ValueError("Missing numeric input.")
    return float(value)


def compute_derived(vals: dict[str, Any], strict: bool = False) -> list[float]:
    p = vals.get("brent_price_usd")
    l1 = vals.get("brent_lag_1")
    l7 = vals.get("brent_lag_7")
    v7 = vals.get("brent_volatility_7d")
    v30 = vals.get("brent_volatility_30d")

    momentum_7d: float | None = (p - l7) if p is not None and l7 is not None else None
    accel: float | None = (l1 - l7) if l1 is not None and l7 is not None else None

    vol_regime: float | None
    if v7 is not None and v30 is not None and v30 != 0.0:
        vol_regime = v7 / v30
    elif strict:
        vol_regime = 1.0
    else:
        vol_regime = None

    return [
        momentum_7d if momentum_7d is not None else 0.0,
        accel if accel is not None else 0.0,
        vol_regime if vol_regime is not None else 1.0,
    ]
