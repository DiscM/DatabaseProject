from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

YF_TICKERS = {
    "brent_price_usd": "BZ=F",
    "wti_price_usd": "CL=F",
    "dxy_index": "DX-Y.NYB",
    "vix_index": "^VIX",
}

LOOKBACK_DAYS = 90


def _period_str(days: int) -> str:
    """Map a number of days to the closest yfinance period string."""
    if days <= 5:
        return "5d"
    if days <= 31:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    return "1y"


def download_market_history(
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Download daily closing prices for Brent, WTI, DXY, VIX from yfinance.

    Returns a DataFrame indexed by date with columns:
        brent_price_usd, wti_price_usd, dxy_index, vix_index
    """
    tickers = list(YF_TICKERS.values())
    period = _period_str(lookback_days)
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False)
    if raw.empty:
        raise RuntimeError("No data returned from yfinance.")

    cols = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    col_map = {v: k for k, v in YF_TICKERS.items()}
    df = cols.rename(columns=col_map)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.dropna(how="any")
    return df


def build_market_rows(
    df: pd.DataFrame,
    gpr_value: float = 0.0,
) -> list[dict[str, str]]:
    """Convert a price DataFrame from download_market_history() into the list
    of row dicts expected by monte_carlo_forecast().

    All columns are stored as strings to match the CSV-based code path.
    Event fields are zeroed since they cannot be determined from market data.
    """
    dates = df.index
    brent = df["brent_price_usd"].values.astype(np.float64)
    wti   = df["wti_price_usd"].values.astype(np.float64)
    dxy   = df["dxy_index"].values.astype(np.float64)
    vix   = df["vix_index"].values.astype(np.float64)

    brent_ret = np.full_like(brent, np.nan)
    wti_ret   = np.full_like(wti, np.nan)
    brent_ret[1:] = np.diff(brent) / brent[:-1]
    wti_ret[1:]   = np.diff(wti)   / wti[:-1]

    n = len(df)

    def _lag(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[k:] = arr[: n - k]
        return out

    def _rolling_std(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        for i in range(k, n):
            out[i] = np.std(arr[i - k + 1 : i + 1], ddof=1)
        return out

    gpr_arr = np.full(n, gpr_value, dtype=np.float64)
    severity_arr = np.zeros(n, dtype=np.float64)
    flag_arr = np.zeros(n, dtype=np.float64)

    rows: list[dict[str, str]] = []
    for i in range(n):
        ds = dates[i].strftime("%Y-%m-%d")
        row: dict[str, str] = {}
        row["market_day_id"] = ds.replace("-", "")
        row["market_date"] = ds
        row["brent_price_usd"]     = _v(brent[i])
        row["wti_price_usd"]       = _v(wti[i])
        row["dxy_index"]           = _v(dxy[i])
        row["vix_index"]           = _v(vix[i])
        row["gpr_index"]           = _v(gpr_arr[i])
        row["brent_return"]        = _v(brent_ret[i])
        row["wti_return"]          = _v(wti_ret[i])
        row["brent_lag_1"]         = _v(_lag(brent, 1)[i])
        row["brent_lag_3"]         = _v(_lag(brent, 3)[i])
        row["brent_lag_7"]         = _v(_lag(brent, 7)[i])
        row["wti_lag_1"]           = _v(_lag(wti, 1)[i])
        row["wti_lag_3"]           = _v(_lag(wti, 3)[i])
        row["wti_lag_7"]           = _v(_lag(wti, 7)[i])
        row["brent_volatility_7d"] = _v(_rolling_std(brent_ret, 7)[i])
        row["brent_volatility_30d"]= _v(_rolling_std(brent_ret, 30)[i])
        row["wti_volatility_7d"]   = _v(_rolling_std(wti_ret, 7)[i])
        row["wti_volatility_30d"]  = _v(_rolling_std(wti_ret, 30)[i])
        row["brent_wti_spread"]    = _v(brent[i] - wti[i])
        row["event_type"]          = ""
        row["event_description"]   = ""
        row["event_severity"]      = _v(severity_arr[i])
        row["event_flag"]          = _v(flag_arr[i])
        rows.append(row)
    return rows


def _v(x: float | Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(round(float(x), 6))


def get_live_market_data(
    lookback_days: int = LOOKBACK_DAYS,
) -> list[dict[str, str]]:
    """High-level helper: download, build, and return rows ready for forecasting."""
    prices = download_market_history(lookback_days)
    return build_market_rows(prices)
