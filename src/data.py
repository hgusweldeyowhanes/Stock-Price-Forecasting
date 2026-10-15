"""Load prices, engineer OHLCV features, and build sliding windows for the LSTM."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "demo_prices.csv"
CACHE_DIR = ROOT / "data" / "cache"
CACHE_MAX_AGE_SECONDS = 12 * 3600

DEMO_TICKER = "DEMO"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FEATURES = ["ret", "hl", "oc", "rel_vol", "ma7_gap", "ma21_gap", "rsi14", "dow_sin", "dow_cos"]
# Rows needed before every feature is defined (MA21 is the longest lookback).
WARMUP = 21

_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=]{0,14}$")


class DataUnavailable(RuntimeError):
    """Raised when prices for a ticker cannot be fetched and no cache exists."""


def normalize_ticker(raw: str | None) -> str:
    ticker = (raw or DEMO_TICKER).strip().upper()
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker: {raw!r}")
    return ticker


class StandardScaler:
    """Per-column z-score scaler, persisted in meta.json (no sklearn dependency)."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, values: np.ndarray) -> "StandardScaler":
        arr = np.asarray(values, dtype=np.float64)
        self.mean = arr.mean(axis=0)
        self.std = np.maximum(arr.std(axis=0), 1e-8)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": np.atleast_1d(self.mean).tolist(), "std": np.atleast_1d(self.std).tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "StandardScaler":
        return cls(payload["mean"], payload["std"])


def generate_demo_prices(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily OHLCV with trend, weekly rhythm, and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    t = np.arange(n, dtype=np.float64)
    shocks = rng.normal(0.00035, 0.011, n)
    weekly = 0.0028 * np.sin(2 * np.pi * t / 5)
    yearly = 0.0009 * np.sin(2 * np.pi * t / 252)
    close = 92.0 * np.exp(np.cumsum(shocks + weekly + yearly))
    close = close * (1 + 0.035 * np.sin(2 * np.pi * t / 21))
    close = np.maximum(close, 5.0)
    prev = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(close, prev) * (1 + rng.uniform(0.001, 0.012, n))
    low = np.minimum(close, prev) * (1 - rng.uniform(0.001, 0.012, n))
    volume = rng.integers(18_000_000, 62_000_000, n)
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": np.round(prev, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume,
        }
    )


def ensure_demo_csv(path: Path | None = None) -> Path:
    dest = path or DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        generate_demo_prices().to_csv(dest, index=False)
    return dest


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in ["Date", "Close"] if col not in frame.columns]
    if missing:
        raise ValueError(f"CSV must include columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    for col in ("Open", "High", "Low"):
        if col not in frame.columns:
            frame[col] = frame["Close"]
    if "Volume" not in frame.columns:
        frame["Volume"] = 0
    frame = frame.dropna(subset=["Close"])
    frame = frame[frame["Close"] > 0]
    return frame.sort_values("Date").reset_index(drop=True)[["Date", *PRICE_COLUMNS]]


def load_frame(path: Path | None = None) -> pd.DataFrame:
    """Load the synthetic demo CSV (or any OHLCV CSV at ``path``)."""
    csv_path = ensure_demo_csv(path)
    return _clean_frame(pd.read_csv(csv_path))


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{re.sub(r'[^A-Z0-9.-]', '_', ticker)}.csv"


def _fetch_yfinance(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    history = yf.Ticker(ticker).history(period="6y", interval="1d", auto_adjust=True)
    if history is None or history.empty:
        raise DataUnavailable(f"No price history returned for {ticker}.")
    history = history.reset_index()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    return history[["Date", *PRICE_COLUMNS]]


def load_prices(ticker: str | None = None) -> tuple[pd.DataFrame, str]:
    """Return (frame, source). DEMO is synthetic and offline; other tickers use yfinance with a CSV cache."""
    symbol = normalize_ticker(ticker)
    if symbol == DEMO_TICKER:
        return load_frame(), "synthetic demo"

    cache = _cache_path(symbol)
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_MAX_AGE_SECONDS:
        return _clean_frame(pd.read_csv(cache)), "yfinance (cached)"
    try:
        fresh = _fetch_yfinance(symbol)
    except Exception as exc:
        if cache.exists():
            return _clean_frame(pd.read_csv(cache)), "yfinance (stale cache)"
        raise DataUnavailable(f"Could not load prices for {symbol}: {exc}") from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(cache, index=False)
    return _clean_frame(fresh), "yfinance"


def _shift(a: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[..., 1:] = a[..., :-1]
    return out


def _leading_nans(a: np.ndarray) -> int:
    row = a.reshape(-1, a.shape[-1])[0]
    valid = np.flatnonzero(~np.isnan(row))
    return int(valid[0]) if len(valid) else row.size


def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over the last axis; positions without a full window are NaN."""
    csum = np.cumsum(np.nan_to_num(a), axis=-1)
    out = np.empty_like(csum)
    out[..., window - 1 :] = csum[..., window - 1 :]
    out[..., window:] -= csum[..., :-window]
    out /= window
    out[..., : _leading_nans(a) + window - 1] = np.nan
    return out


def compute_features(open_, high, low, close, volume, weekday) -> np.ndarray:
    """Feature matrix with shape (..., T, len(FEATURES)). Works on 1-D series or batched 2-D paths."""
    open_, high, low, close, volume = (np.asarray(v, dtype=np.float64) for v in (open_, high, low, close, volume))
    weekday = np.broadcast_to(np.asarray(weekday, dtype=np.float64), close.shape)

    prev_close = _shift(close)
    ret = np.log(close / prev_close)
    hl = np.log(np.maximum(high, close) / np.maximum(np.minimum(low, close), 1e-8))
    oc = np.log(close / np.maximum(open_, 1e-8))
    log_vol = np.log1p(np.maximum(volume, 0))
    rel_vol = log_vol - _rolling_mean(log_vol, 20)
    ma7_gap = close / _rolling_mean(close, 7) - 1
    ma21_gap = close / _rolling_mean(close, 21) - 1

    gains = np.clip(ret, 0, None)
    losses = np.clip(-ret, 0, None)
    avg_gain = _rolling_mean(gains, 14)
    avg_loss = _rolling_mean(losses, 14)
    total = avg_gain + avg_loss
    rsi = np.where(total > 0, avg_gain / np.where(total > 0, total, 1), 0.5) - 0.5
    rsi = np.where(np.isnan(avg_gain), np.nan, rsi)

    dow_sin = np.sin(2 * np.pi * weekday / 5)
    dow_cos = np.cos(2 * np.pi * weekday / 5)
    return np.stack([ret, hl, oc, rel_vol, ma7_gap, ma21_gap, rsi, dow_sin, dow_cos], axis=-1)


def frame_features(frame: pd.DataFrame) -> np.ndarray:
    return compute_features(
        frame["Open"].to_numpy(),
        frame["High"].to_numpy(),
        frame["Low"].to_numpy(),
        frame["Close"].to_numpy(),
        frame["Volume"].to_numpy(),
        pd.to_datetime(frame["Date"]).dt.weekday.to_numpy(),
    )


def make_windows(
    features: np.ndarray, lookback: int, start: int = WARMUP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Windows of ``lookback`` feature rows ending at row ``i`` and the next-day log return at ``i + 1``.

    Returns (x, y, ends) where ``ends[k]`` is the last row index of window ``k``.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2.")
    first_end = start + lookback - 1
    ends = np.arange(first_end, len(features) - 1)
    if len(ends) == 0:
        raise ValueError(f"Need more than {first_end + 1} prices to build windows.")
    x = np.stack([features[end - lookback + 1 : end + 1] for end in ends])
    y = features[ends + 1, 0]
    return x, y, ends


def sliding_windows(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=np.float64).ravel()
    if lookback < 2:
        raise ValueError("lookback must be at least 2.")
    if len(values) <= lookback:
        raise ValueError(f"Need more than {lookback} prices to build windows.")
    x_rows = []
    y_rows = []
    for end in range(lookback, len(values)):
        x_rows.append(values[end - lookback : end])
        y_rows.append(values[end])
    x = np.array(x_rows, dtype=np.float64).reshape(-1, lookback, 1)
    y = np.array(y_rows, dtype=np.float64)
    return x, y


def chronological_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    split = int(len(x) * (1 - test_ratio))
    split = min(max(split, 1), len(x) - 1)
    return x[:split], x[split:], y[:split], y[split:]
