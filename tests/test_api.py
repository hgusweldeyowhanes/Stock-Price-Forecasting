import json

import pytest

from src.data import ensure_demo_csv


@pytest.fixture()
def client(tmp_path, monkeypatch):
    csv = tmp_path / "demo_prices.csv"
    dest = tmp_path / "artifacts"
    dest.mkdir()
    ensure_demo_csv(csv)
    monkeypatch.setattr("src.data.DATA_PATH", csv)
    monkeypatch.setattr("src.train.ARTIFACTS", dest)
    monkeypatch.setattr("src.forecast.ARTIFACTS", dest)
    from api import app

    monkeypatch.setattr("api.ARTIFACTS", dest)
    monkeypatch.setattr("api.DATA_PATH", csv)
    app.config["TESTING"] = True
    return app.test_client()


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Stock Price Forecasting" in response.data


def test_health_without_model(client):
    response = client.get("/api/health")
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["model_ready"] is False


def test_evaluate_requires_model(client):
    response = client.get("/api/evaluate")
    assert response.status_code == 503


def test_series(client):
    response = client.get("/api/series")
    body = response.get_json()
    assert response.status_code == 200
    assert len(body["close"]) > 10
    assert len(body["dates"]) == len(body["close"])


def test_predict_next_close_from_recent_history(tmp_path):
    pytest.importorskip("tensorflow")
    from src.forecast import predict_next_close
    from src.model import build_lstm

    dest = tmp_path / "artifacts"
    dest.mkdir()
    model = build_lstm(5)
    model.save(dest / "model.keras")
    (dest / "meta.json").write_text(
        json.dumps({"lookback": 5, "scaler": {"data_min": 0.0, "data_max": 1.0}}),
        encoding="utf-8",
    )

    result = predict_next_close([0.1, 0.2, 0.3, 0.4, 0.5], artifact_dir=dest)
    assert result["lookback"] == 5
    assert "predicted_next_close" in result
    assert isinstance(result["predicted_next_close"], float)
