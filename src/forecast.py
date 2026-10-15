"""Load a trained LSTM, evaluate it, and forecast future closes with uncertainty bands."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src import train as _train
from src.data import (
    WARMUP,
    StandardScaler,
    _clean_frame,
    compute_features,
    frame_features,
    load_prices,
    normalize_ticker,
)

_cache: dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _load(ticker: str | None, artifact_dir: Path | None):
    """(model, scaler, meta, dropout_sampler), cached until model.keras changes on disk."""
    dest = artifact_dir or _train.artifact_dir(ticker)
    model_path = dest / "model.keras"
    meta_path = dest / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No trained model in {dest}. Run: python -m src.train --ticker {normalize_ticker(ticker)}")
    stamp = model_path.stat().st_mtime_ns
    key = str(model_path.resolve())
    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1:]
        import tensorflow as tf
        from tensorflow import keras

        model = keras.models.load_model(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scaler = StandardScaler.from_dict(meta["feature_scaler"])
        # Compiled once per model; eager calls with training=True are ~8x slower.
        sampler = tf.function(lambda window: model(window, training=True), reduce_retracing=True)
        _cache[key] = (stamp, model, scaler, meta, sampler)
        return model, scaler, meta, sampler


def load_artifacts(ticker: str | None = None, artifact_dir: Path | None = None):
    """Return (model, scaler, meta) for a trained ticker."""
    return _load(ticker, artifact_dir)[:3]


def evaluate(ticker: str | None = None, artifact_dir: Path | None = None) -> dict:
    dest = artifact_dir or _train.artifact_dir(ticker)
    _, _, meta = load_artifacts(ticker, dest)
    saved = np.load(dest / "test_overlay.npz", allow_pickle=True)
    return {
        "ticker": meta.get("ticker"),
        "source": meta.get("source"),
        "model": "stacked_lstm_returns",
        "lookback": int(meta["lookback"]),
        "metrics": meta["metrics"],
        "comparison": meta["comparison"],
        "dates": saved["dates"].astype(str).tolist(),
        "actual": np.round(saved["actual"].astype(float), 2).tolist(),
        "predicted": np.round(saved["predicted"].astype(float), 2).tolist(),
        "naive": np.round(saved["naive"].astype(float), 2).tolist(),
    }


def frame_from_history(history: list[dict]) -> pd.DataFrame:
    rows = []
    for item in history:
        if not isinstance(item, dict) or "date" not in item or "close" not in item:
            raise ValueError("Each history row needs at least 'date' and 'close'.")
        rows.append(
            {
                "Date": item["date"],
                "Open": item.get("open", item["close"]),
                "High": item.get("high", item["close"]),
                "Low": item.get("low", item["close"]),
                "Close": item["close"],
                "Volume": item.get("volume", 0),
            }
        )
    return _clean_frame(pd.DataFrame(rows))


def predict_next_close(history: list[dict], ticker: str | None = None, artifact_dir: Path | None = None) -> dict:
    """Predict the next close from recent OHLCV rows (oldest first)."""
    model, scaler, meta = load_artifacts(ticker, artifact_dir)
    lookback = int(meta["lookback"])
    frame = frame_from_history(history)
    needed = WARMUP + lookback
    if len(frame) < needed:
        raise ValueError(f"Need at least {needed} daily rows ({WARMUP} warm-up + {lookback} lookback).")
    window = scaler.transform(frame_features(frame)[-lookback:])[np.newaxis]
    next_return = float(model.predict(window, verbose=0).ravel()[0]) * float(meta["target_std"])
    last_close = float(frame["Close"].iloc[-1])
    return {
        "ticker": meta.get("ticker"),
        "lookback": lookback,
        "last_close": round(last_close, 2),
        "predicted_return_pct": round(float(np.exp(next_return) - 1) * 100, 4),
        "predicted_next_close": round(last_close * float(np.exp(next_return)), 2),
    }


def _business_days_after(last_date, horizon: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_date) + pd.offsets.BDay(1)
    return pd.bdate_range(start, periods=horizon)


def forecast_horizon(
    horizon: int = 7,
    ticker: str | None = None,
    paths: int = 200,
    seed: int | None = 0,
    artifact_dir: Path | None = None,
) -> dict:
    """
    Simulate ``paths`` future price paths step by step.

    Each step runs the LSTM with dropout active (model uncertainty) and adds Gaussian noise sized
    from out-of-sample residuals (market noise), then feeds the simulated bar back in.
    """
    if horizon < 1 or horizon > 30:
        raise ValueError("horizon must be between 1 and 30.")
    if paths < 10 or paths > 1000:
        raise ValueError("paths must be between 10 and 1000.")
    _, scaler, meta, sampler = _load(ticker, artifact_dir)
    lookback = int(meta["lookback"])
    target_std = float(meta["target_std"])
    resid_std = float(meta["resid_std"])
    half_range = float(meta["hl_median"]) / 2

    frame, _ = load_prices(meta.get("ticker") or ticker)
    tail = frame.tail(lookback + WARMUP + 5)
    columns = {col: np.tile(tail[col].to_numpy(dtype=np.float64), (paths, 1)) for col in ["Open", "High", "Low", "Close", "Volume"]}
    weekdays = pd.to_datetime(tail["Date"]).dt.weekday.to_numpy()
    future_dates = _business_days_after(tail["Date"].iloc[-1], horizon)
    rng = np.random.default_rng(seed)
    simulated = np.empty((paths, horizon))

    for step in range(horizon):
        features = compute_features(
            columns["Open"], columns["High"], columns["Low"], columns["Close"], columns["Volume"], weekdays
        )
        window = scaler.transform(features[:, -lookback:, :]).astype("float32")
        drift = np.asarray(sampler(window)).ravel() * target_std
        returns = drift + rng.normal(0.0, resid_std, paths)

        prev_close = columns["Close"][:, -1]
        close = prev_close * np.exp(returns)
        bar = {
            "Open": prev_close,
            "High": np.maximum(prev_close, close) * np.exp(half_range),
            "Low": np.minimum(prev_close, close) * np.exp(-half_range),
            "Close": close,
            "Volume": np.median(columns["Volume"][:, -20:], axis=1),
        }
        for col, values in bar.items():
            columns[col] = np.column_stack([columns[col][:, 1:], values])
        weekdays = np.append(weekdays[1:], future_dates[step].weekday())
        simulated[:, step] = close

    p10, p50, p90 = np.percentile(simulated, [10, 50, 90], axis=0)
    return {
        "ticker": meta.get("ticker"),
        "last_close": round(float(frame["Close"].iloc[-1]), 2),
        "last_date": pd.Timestamp(frame["Date"].iloc[-1]).strftime("%Y-%m-%d"),
        "horizon": horizon,
        "paths": paths,
        "method": "MC dropout + residual noise",
        "dates": future_dates.strftime("%Y-%m-%d").tolist(),
        "forecast": np.round(p50, 2).tolist(),
        "lower": np.round(p10, 2).tolist(),
        "upper": np.round(p90, 2).tolist(),
    }
