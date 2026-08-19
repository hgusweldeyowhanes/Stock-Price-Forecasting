"""Load a trained LSTM and forecast the next closes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import MinMaxScaler, load_frame, sliding_windows
from src.train import ARTIFACTS, metrics


def load_artifacts(artifact_dir: Path | None = None):
    dest = artifact_dir or ARTIFACTS
    model_path = dest / "model.keras"
    meta_path = dest / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("No trained model. Run: python -m src.train")
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scaler = MinMaxScaler.from_dict(meta["scaler"])
    return model, scaler, meta


def evaluate(artifact_dir: Path | None = None, csv_path: Path | None = None) -> dict:
    dest = artifact_dir or ARTIFACTS
    model, scaler, meta = load_artifacts(dest)
    lookback = int(meta["lookback"])
    overlay_path = dest / "test_overlay.npz"
    if overlay_path.exists():
        saved = np.load(overlay_path, allow_pickle=True)
        actual = saved["actual"]
        predicted = saved["predicted"]
        dates = saved["dates"]
    else:
        frame = load_frame(csv_path)
        scaled = scaler.transform(frame["Close"].to_numpy(dtype=np.float64))
        x, y = sliding_windows(scaled, lookback)
        test_ratio = float(meta.get("test_ratio", 0.2))
        split = int(len(x) * (1 - test_ratio))
        predicted_scaled = model.predict(x[split:], verbose=0).ravel()
        actual = scaler.inverse_transform(y[split:])
        predicted = scaler.inverse_transform(predicted_scaled)
        dates = frame["Date"].astype(str).to_numpy()[-len(actual) :]
    return {
        "metrics": metrics(actual, predicted),
        "lookback": lookback,
        "model": "stacked_lstm_close",
        "dates": dates.tolist(),
        "actual": np.round(actual.astype(float), 2).tolist(),
        "predicted": np.round(predicted.astype(float), 2).tolist(),
    }


def predict_next_close(history: list[float] | np.ndarray, artifact_dir: Path | None = None) -> dict:
    """Predict the next close from a recent list of historical closes."""
    model, scaler, meta = load_artifacts(artifact_dir)
    lookback = int(meta["lookback"])
    series = np.asarray(history, dtype=np.float64).ravel()
    if len(series) < lookback:
        raise ValueError(f"Need at least {lookback} historical closes to predict the next close.")
    window = scaler.transform(series[-lookback:])
    x = np.asarray(window, dtype=np.float64).reshape(1, lookback, 1)
    next_scaled = float(model.predict(x, verbose=0).ravel()[0])
    next_close = float(scaler.inverse_transform(np.array([next_scaled], dtype=np.float64))[0])
    return {
        "lookback": lookback,
        "last_close": round(float(series[-1]), 2),
        "predicted_next_close": round(next_close, 2),
        "history_used": [round(float(v), 2) for v in series[-lookback:]],
    }


def forecast_horizon(horizon: int = 7, artifact_dir: Path | None = None, csv_path: Path | None = None) -> dict:
    if horizon < 1 or horizon > 30:
        raise ValueError("horizon must be between 1 and 30.")
    model, scaler, meta = load_artifacts(artifact_dir)
    lookback = int(meta["lookback"])
    frame = load_frame(csv_path)
    closes = frame["Close"].to_numpy(dtype=np.float64)
    window = scaler.transform(closes[-lookback:]).tolist()
    future = []
    for _ in range(horizon):
        x = np.array(window[-lookback:], dtype=np.float64).reshape(1, lookback, 1)
        next_scaled = float(model.predict(x, verbose=0).ravel()[0])
        window.append(next_scaled)
        future.append(float(scaler.inverse_transform(np.array([next_scaled]))[0]))
    last_date = frame["Date"].iloc[-1]
    dates = pd_bdate_range(last_date, horizon)
    return {
        "last_close": round(float(closes[-1]), 2),
        "horizon": horizon,
        "dates": dates,
        "forecast": [round(value, 2) for value in future],
    }


def pd_bdate_range(last_date, horizon: int) -> list[str]:
    import pandas as pd

    start = pd.Timestamp(last_date) + pd.offsets.BDay(1)
    return pd.bdate_range(start, periods=horizon).strftime("%Y-%m-%d").tolist()
