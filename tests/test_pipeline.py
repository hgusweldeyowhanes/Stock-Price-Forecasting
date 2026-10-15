import time

import numpy as np
import pandas as pd
import pytest

from src import data as data_mod
from src.baselines import baseline_prices, lag_matrix, ridge_fit, ridge_predict
from src.charts import history_and_forecast_png, predicted_vs_actual_png
from src.data import (
    FEATURES,
    WARMUP,
    DataUnavailable,
    StandardScaler,
    compute_features,
    frame_features,
    generate_demo_prices,
    load_prices,
    make_windows,
    normalize_ticker,
    sliding_windows,
)
from src.jobs import TrainingJobs
from src.train import metrics

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_sliding_windows_shapes():
    series = np.arange(10, dtype=float)
    x, y = sliding_windows(series, lookback=3)
    assert x.shape == (7, 3, 1)
    assert y.shape == (7,)
    assert np.allclose(x[0].ravel(), [0, 1, 2])
    assert y[0] == 3
    assert y[-1] == 9


def test_standard_scaler_roundtrip():
    values = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    assert np.allclose(scaled.mean(axis=0), 0)
    assert np.allclose(scaler.inverse_transform(scaled), values)
    restored = StandardScaler.from_dict(scaler.to_dict())
    assert np.allclose(restored.transform(values), scaled)


def test_demo_prices_have_ohlc():
    frame = generate_demo_prices(n=40, seed=1)
    assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(frame) == 40
    assert (frame["High"] >= frame["Low"]).all()


def test_features_are_finite_after_warmup_and_batch_consistent():
    frame = data_mod._clean_frame(generate_demo_prices(n=120, seed=3))
    features = frame_features(frame)
    assert features.shape == (120, len(FEATURES))
    assert np.isfinite(features[WARMUP:]).all()

    cols = [np.stack([frame[c].to_numpy()] * 2) for c in ["Open", "High", "Low", "Close", "Volume"]]
    batched = compute_features(*cols, frame["Date"].dt.weekday.to_numpy())
    assert batched.shape == (2, 120, len(FEATURES))
    assert np.allclose(batched[1, WARMUP:], features[WARMUP:])


def test_make_windows_targets_next_day_return():
    frame = data_mod._clean_frame(generate_demo_prices(n=100, seed=4))
    features = frame_features(frame)
    x, y, ends = make_windows(features, lookback=10)
    assert x.shape == (len(ends), 10, len(FEATURES))
    assert ends[0] == WARMUP + 9
    closes = frame["Close"].to_numpy()
    assert np.allclose(y, np.log(closes[ends + 1] / closes[ends]))
    assert np.allclose(x[0, -1], features[ends[0]])


def test_metrics_with_direction():
    truth = np.array([10.0, 12.0, 11.0])
    pred = np.array([11.0, 12.0, 10.0])
    scores = metrics(truth, pred)
    assert scores["n"] == 3
    assert scores["mae"] == 0.6667
    assert scores["rmse"] > 0
    assert "direction_acc" not in scores

    last = np.array([9.0, 13.0, 12.0])
    # actual moves: up, down, down; predicted moves: up, down, down
    assert metrics(truth, pred, last)["direction_acc"] == 100.0
    # A model that never moves has no direction calls.
    assert metrics(truth, last, last)["direction_acc"] is None


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = 0.5 + x @ np.array([1.0, -2.0, 0.25])
    weights = ridge_fit(x, y, alpha=1e-6)
    assert np.allclose(ridge_predict(weights, x), y, atol=1e-4)


def test_baselines_shapes_and_naive():
    closes = np.linspace(100, 120, 60)
    ends = np.arange(40, 59)
    windows = np.random.default_rng(1).normal(size=(len(ends), 8, 2))
    preds = baseline_prices(closes, ends, np.full(30, 0.001), np.ones((30, 8, 2)), windows)
    assert set(preds) == {"naive", "drift", "ma5", "ridge"}
    assert np.allclose(preds["naive"], closes[ends])
    assert np.allclose(preds["ma5"][0], closes[36:41].mean())
    assert lag_matrix(windows).shape == (len(ends), 10)


def test_normalize_ticker():
    assert normalize_ticker(None) == "DEMO"
    assert normalize_ticker(" aapl ") == "AAPL"
    assert normalize_ticker("^gspc") == "^GSPC"
    for bad in ("../etc", "AAPL;rm", "A" * 20):
        with pytest.raises(ValueError):
            normalize_ticker(bad)


def _fake_history(n: int = 60) -> pd.DataFrame:
    frame = generate_demo_prices(n=n, seed=7)
    return frame[["Date", "Open", "High", "Low", "Close", "Volume"]]


def test_load_prices_fetches_then_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(data_mod, "_fetch_yfinance", lambda t: calls.append(t) or _fake_history())

    frame, source = load_prices("aapl")
    assert source == "yfinance" and len(frame) == 60
    assert (tmp_path / "AAPL.csv").exists()

    _, source = load_prices("AAPL")
    assert source == "yfinance (cached)"
    assert calls == ["AAPL"]


def test_load_prices_falls_back_to_stale_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path)
    _fake_history().to_csv(tmp_path / "MSFT.csv", index=False)
    monkeypatch.setattr(data_mod, "CACHE_MAX_AGE_SECONDS", 0)

    def offline(ticker):
        raise ConnectionError("offline")

    monkeypatch.setattr(data_mod, "_fetch_yfinance", offline)
    time.sleep(0.01)
    _, source = load_prices("MSFT")
    assert source == "yfinance (stale cache)"
    with pytest.raises(DataUnavailable):
        load_prices("NOPE")


def test_training_jobs_run_and_dedupe():
    jobs = TrainingJobs()
    seen = []

    def fake_train(ticker, **params):
        seen.append((ticker, params))
        time.sleep(0.05)
        return {"metrics": {"rmse": 1.0}}

    job, created = jobs.submit("AAPL", {"epochs": 1}, fake_train)
    again, created_again = jobs.submit("AAPL", {"epochs": 1}, fake_train)
    assert created and not created_again and again["id"] == job["id"]

    done = jobs.wait(job["id"], timeout=5)
    assert done["status"] == "succeeded"
    assert done["metrics"] == {"rmse": 1.0}
    assert seen == [("AAPL", {"epochs": 1})]

    def broken(ticker, **params):
        raise RuntimeError("boom")

    failed, _ = jobs.submit("AAPL", {}, broken)
    assert jobs.wait(failed["id"], timeout=5)["status"] == "failed"
    assert jobs.get(failed["id"])["error"] == "boom"


def test_chart_png_bytes():
    png = predicted_vs_actual_png(
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        [100, 101, 99],
        [100.2, 100.8, 99.4],
        naive=[99.5, 100, 101],
    )
    assert png[:8] == PNG_MAGIC
    band = history_and_forecast_png(["d1", "d2"], [100, 101], ["f1", "f2"], [102, 103], [100, 99], [104, 107])
    assert band[:8] == PNG_MAGIC


def test_tiny_lstm_fits_multifeature_windows():
    pytest.importorskip("tensorflow")
    from src.model import build_lstm

    x = np.random.default_rng(2).normal(size=(40, 5, 3))
    y = x[:, -1, 0] * 0.5
    model = build_lstm(5, n_features=3, learning_rate=0.01)
    model.fit(x, y, epochs=2, batch_size=8, verbose=0)
    pred = model.predict(x, verbose=0).ravel()
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_train_evaluate_forecast_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    from src.forecast import evaluate, forecast_horizon
    from src.train import train

    csv = tmp_path / "prices.csv"
    generate_demo_prices(n=260, seed=5).to_csv(csv, index=False)
    monkeypatch.setattr(data_mod, "DATA_PATH", csv)
    dest = tmp_path / "artifacts"

    meta = train(lookback=10, epochs=1, csv_path=csv, artifact_dir_override=dest)
    assert meta["ticker"] == "DEMO"
    assert set(meta["comparison"]) == {"lstm", "naive", "drift", "ma5", "ridge"}
    assert meta["metrics"]["naive_rmse"] == meta["comparison"]["naive"]["rmse"]
    assert meta["resid_std"] > 0

    result = evaluate(artifact_dir=dest)
    assert len(result["actual"]) == len(result["predicted"]) == len(result["naive"]) == result["metrics"]["n"]

    future = forecast_horizon(horizon=4, paths=30, artifact_dir=dest)
    assert len(future["dates"]) == len(future["forecast"]) == 4
    lower, mid, upper = (np.array(future[k]) for k in ("lower", "forecast", "upper"))
    assert np.all(lower <= mid) and np.all(mid <= upper)
    assert upper[-1] - lower[-1] > 0
