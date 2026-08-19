"""Load prices and build sliding-window sequences for the LSTM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "demo_prices.csv"


class MinMaxScaler:
    """Persistable 0–1 scaler (no sklearn dependency)."""

    def __init__(self, data_min: float = 0.0, data_max: float = 1.0):
        self.data_min = float(data_min)
        self.data_max = float(data_max)

    @property
    def span(self) -> float:
        return max(self.data_max - self.data_min, 1e-8)

    def fit(self, values: np.ndarray) -> "MinMaxScaler":
        arr = np.asarray(values, dtype=np.float64).ravel()
        self.data_min = float(arr.min())
        self.data_max = float(arr.max())
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return (arr - self.data_min) / self.span

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return arr * self.span + self.data_min

    def to_dict(self) -> dict:
        return {"data_min": self.data_min, "data_max": self.data_max}

    @classmethod
    def from_dict(cls, payload: dict) -> "MinMaxScaler":
        return cls(payload["data_min"], payload["data_max"])


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


def load_frame(path: Path | None = None) -> pd.DataFrame:
    csv_path = ensure_demo_csv(path)
    frame = pd.read_csv(csv_path, parse_dates=["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    if "Close" not in frame.columns:
        raise ValueError("CSV must include a Close column.")
    return frame


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
