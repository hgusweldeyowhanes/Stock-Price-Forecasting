"""Train the stacked LSTM and persist artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data import (
    MinMaxScaler,
    chronological_split,
    load_frame,
    sliding_windows,
)
from src.model import build_lstm

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = pred - truth
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(truth), 1e-8)) * 100)
    naive = truth[1:] - truth[:-1]
    naive_rmse = float(np.sqrt(np.mean(naive**2))) if len(truth) > 1 else None
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "naive_rmse": round(naive_rmse, 4) if naive_rmse is not None else None,
        "n": int(len(truth)),
    }


def train(
    lookback: int = 60,
    epochs: int = 25,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    csv_path: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict:
    dest = artifact_dir or ARTIFACTS
    dest.mkdir(parents=True, exist_ok=True)

    frame = load_frame(csv_path)
    closes = frame["Close"].to_numpy(dtype=np.float64)
    split_price = int(len(closes) * (1 - test_ratio))
    scaler = MinMaxScaler().fit(closes[:split_price])
    scaled = scaler.transform(closes)

    x, y = sliding_windows(scaled, lookback)
    x_train, x_test, y_train, y_test = chronological_split(x, y, test_ratio)

    from tensorflow import keras

    model = build_lstm(lookback)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True,
            )
        ],
    )

    y_pred = model.predict(x_test, verbose=0).ravel()
    actual = scaler.inverse_transform(y_test)
    predicted = scaler.inverse_transform(y_pred)
    scores = metrics(actual, predicted)

    model_path = dest / "model.keras"
    model.save(model_path)
    meta = {
        "lookback": lookback,
        "epochs_ran": int(len(history.history["loss"])),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "metrics": scores,
        "scaler": scaler.to_dict(),
        "test_ratio": test_ratio,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez(
        dest / "test_overlay.npz",
        actual=actual,
        predicted=predicted,
        dates=frame["Date"].astype(str).to_numpy()[-len(actual) :],
    )
    return meta


if __name__ == "__main__":
    result = train()
    print(json.dumps(result["metrics"], indent=2))
