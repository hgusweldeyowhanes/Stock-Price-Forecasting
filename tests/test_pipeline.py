from pathlib import Path

import numpy as np
import pytest

from src.charts import predicted_vs_actual_png
from src.data import MinMaxScaler, generate_demo_prices, sliding_windows
from src.train import metrics


def test_sliding_windows_shapes():
    series = np.arange(10, dtype=float)
    x, y = sliding_windows(series, lookback=3)
    assert x.shape == (7, 3, 1)
    assert y.shape == (7,)
    assert np.allclose(x[0].ravel(), [0, 1, 2])
    assert y[0] == 3
    assert y[-1] == 9


def test_scaler_roundtrip():
    values = np.array([10.0, 20.0, 30.0])
    scaler = MinMaxScaler().fit(values)
    scaled = scaler.transform(values)
    assert np.allclose(scaled, [0, 0.5, 1])
    assert np.allclose(scaler.inverse_transform(scaled), values)


def test_demo_prices_have_ohlc():
    frame = generate_demo_prices(n=40, seed=1)
    assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(frame) == 40
    assert (frame["High"] >= frame["Low"]).all()


def test_metrics():
    truth = np.array([10.0, 12.0, 11.0])
    pred = np.array([11.0, 12.0, 10.0])
    scores = metrics(truth, pred)
    assert scores["n"] == 3
    assert scores["mae"] == 0.6667
    assert scores["rmse"] > 0


def test_chart_png_bytes():
    png = predicted_vs_actual_png(
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        [100, 101, 99],
        [100.2, 100.8, 99.4],
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_tiny_lstm_overfits_short_series(tmp_path):
    pytest.importorskip("tensorflow")
    from src.data import chronological_split
    from src.model import build_lstm

    series = np.linspace(0.1, 0.9, 80)
    x, y = sliding_windows(series, lookback=5)
    x_train, x_test, y_train, y_test = chronological_split(x, y, 0.2)
    model = build_lstm(5, learning_rate=0.01)
    model.fit(x_train, y_train, epochs=3, batch_size=8, verbose=0)
    pred = model.predict(x_test, verbose=0).ravel()
    assert pred.shape == y_test.shape
    assert np.isfinite(pred).all()
    model.save(tmp_path / "tiny.keras")
    assert (tmp_path / "tiny.keras").exists()
