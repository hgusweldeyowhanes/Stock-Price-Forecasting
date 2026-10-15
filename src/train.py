"""Train the stacked LSTM on next-day returns, compare against baselines, and persist artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.baselines import baseline_prices
from src.data import (
    FEATURES,
    WARMUP,
    StandardScaler,
    frame_features,
    load_frame,
    load_prices,
    make_windows,
    normalize_ticker,
)
from src.model import build_lstm

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def artifact_dir(ticker: str | None = None) -> Path:
    return ARTIFACTS / normalize_ticker(ticker)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, last: np.ndarray | None = None) -> dict:
    """Price-space error metrics; with ``last`` (previous close) also the up/down direction hit rate."""
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = pred - truth
    scores = {
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "mape": round(float(np.mean(np.abs(err) / np.maximum(np.abs(truth), 1e-8)) * 100), 4),
        "n": int(len(truth)),
    }
    if last is not None:
        prev = np.asarray(last, dtype=np.float64).ravel()
        moved = truth != prev
        called = np.sign(pred - prev)
        if moved.any() and np.any(called[moved] != 0):
            hits = called[moved] == np.sign(truth - prev)[moved]
            scores["direction_acc"] = round(float(np.mean(hits)) * 100, 2)
        else:
            scores["direction_acc"] = None
    return scores


def train(
    ticker: str | None = None,
    lookback: int = 60,
    epochs: int = 25,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    csv_path: Path | None = None,
    artifact_dir_override: Path | None = None,
    seed: int = 42,
) -> dict:
    symbol = normalize_ticker(ticker)
    dest = artifact_dir_override or artifact_dir(symbol)
    dest.mkdir(parents=True, exist_ok=True)

    if csv_path is not None:
        frame, source = load_frame(csv_path), f"csv:{Path(csv_path).name}"
    else:
        frame, source = load_prices(symbol)
    closes = frame["Close"].to_numpy(dtype=np.float64)
    raw_features = frame_features(frame)
    x_raw, returns, ends = make_windows(raw_features, lookback)

    n_train = min(max(int(len(x_raw) * (1 - test_ratio)), 1), len(x_raw) - 1)
    scaler = StandardScaler().fit(raw_features[WARMUP : ends[n_train - 1] + 1])
    x = scaler.transform(x_raw)
    x_train, x_test = x[:n_train], x[n_train:]
    r_train, r_test = returns[:n_train], returns[n_train:]
    ends_test = ends[n_train:]
    target_std = float(max(np.std(r_train), 1e-8))

    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    model = build_lstm(lookback, n_features=len(FEATURES))
    history = model.fit(
        x_train,
        r_train / target_std,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ],
    )

    pred_returns = model.predict(x_test, verbose=0).ravel() * target_std
    last = closes[ends_test]
    actual = closes[ends_test + 1]
    predictions = {"lstm": last * np.exp(pred_returns)}
    predictions.update(baseline_prices(closes, ends_test, r_train, x_train, x_test))
    comparison = {name: metrics(actual, pred, last) for name, pred in predictions.items()}

    lstm_scores = dict(comparison["lstm"])
    naive_rmse = comparison["naive"]["rmse"]
    lstm_scores["naive_rmse"] = naive_rmse
    lstm_scores["skill_vs_naive"] = round((1 - lstm_scores["rmse"] / naive_rmse) * 100, 2) if naive_rmse else None

    model.save(dest / "model.keras")
    meta = {
        "ticker": symbol,
        "source": source,
        "lookback": lookback,
        "features": FEATURES,
        "target": "next_day_log_return",
        "epochs_ran": int(len(history.history["loss"])),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "test_ratio": test_ratio,
        "feature_scaler": scaler.to_dict(),
        "target_std": target_std,
        "resid_std": float(np.std(r_test - pred_returns)),
        "hl_median": float(np.median(raw_features[WARMUP : ends[n_train - 1] + 1, FEATURES.index("hl")])),
        "metrics": lstm_scores,
        "comparison": comparison,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez(
        dest / "test_overlay.npz",
        dates=frame["Date"].dt.strftime("%Y-%m-%d").to_numpy()[ends_test + 1],
        actual=actual,
        predicted=predictions["lstm"],
        naive=predictions["naive"],
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stacked LSTM for one ticker.")
    parser.add_argument("--ticker", default="DEMO", help="DEMO (synthetic, offline) or a Yahoo Finance symbol, e.g. AAPL")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    meta = train(ticker=args.ticker, lookback=args.lookback, epochs=args.epochs)
    print(f"{meta['ticker']} ({meta['source']}), {meta['epochs_ran']} epochs")
    print(f"{'model':<8}{'rmse':>9}{'mae':>9}{'mape%':>8}{'dir%':>8}")
    for name, row in meta["comparison"].items():
        direction = "-" if row.get("direction_acc") is None else row["direction_acc"]
        print(f"{name:<8}{row['rmse']:>9}{row['mae']:>9}{row['mape']:>8}{direction:>8}")
    print(f"LSTM skill vs naive RMSE: {meta['metrics']['skill_vs_naive']}%")


if __name__ == "__main__":
    main()
