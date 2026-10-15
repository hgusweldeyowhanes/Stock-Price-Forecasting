import pytest

from src.data import generate_demo_prices

API_KEY = "test-key"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    csv = tmp_path / "demo_prices.csv"
    generate_demo_prices(n=260, seed=5).to_csv(csv, index=False)
    dest = tmp_path / "artifacts"
    dest.mkdir()
    monkeypatch.setattr("src.data.DATA_PATH", csv)
    monkeypatch.setattr("src.data.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("src.train.ARTIFACTS", dest)
    monkeypatch.delenv("TRAIN_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_TICKER", raising=False)
    from api import app

    app.config["TESTING"] = True
    return {"client": app.test_client(), "csv": csv, "artifacts": dest}


@pytest.fixture()
def client(env):
    return env["client"]


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Stock Price Forecasting" in response.data


def test_health_without_model(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert body["ticker"] == "DEMO"
    assert body["model_ready"] is False
    assert body["trained_tickers"] == []
    assert body["training_enabled"] is False


def test_evaluate_requires_model(client):
    response = client.get("/api/evaluate")
    assert response.status_code == 503
    assert "No trained model for DEMO" in response.get_json()["detail"]


def test_series(client):
    body = client.get("/api/series?ticker=demo").get_json()
    assert body["ticker"] == "DEMO"
    assert body["source"] == "synthetic demo"
    assert len(body["close"]) > 10
    assert len(body["dates"]) == len(body["close"])


def test_invalid_ticker_is_rejected(client):
    response = client.get("/api/series?ticker=../../etc")
    assert response.status_code == 400


def test_unavailable_ticker_returns_502(client, monkeypatch):
    def offline(ticker):
        raise ConnectionError("offline")

    monkeypatch.setattr("src.data._fetch_yfinance", offline)
    response = client.get("/api/series?ticker=ZZZZ")
    assert response.status_code == 502


def test_train_disabled_without_server_key(client):
    response = client.post("/api/train", json={"ticker": "DEMO"})
    assert response.status_code == 403


def test_train_rejects_wrong_key(client, monkeypatch):
    monkeypatch.setenv("TRAIN_API_KEY", API_KEY)
    response = client.post("/api/train", json={"ticker": "DEMO"}, headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_train_runs_in_background(client, monkeypatch):
    import api

    monkeypatch.setenv("TRAIN_API_KEY", API_KEY)
    calls = []

    def fake_train(ticker, epochs, lookback):
        calls.append((ticker, epochs, lookback))
        return {"metrics": {"rmse": 1.23}}

    monkeypatch.setattr("src.train.train", fake_train)
    response = client.post(
        "/api/train",
        json={"ticker": "aapl", "epochs": 500, "lookback": 5},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 202
    job = response.get_json()["job"]
    assert job["ticker"] == "AAPL"

    api.jobs.wait(job["id"], timeout=5)
    status = client.get(f"/api/train/{job['id']}").get_json()
    assert status["status"] == "succeeded"
    assert status["metrics"] == {"rmse": 1.23}
    # epochs and lookback are clamped to safe ranges
    assert calls == [("AAPL", 60, 10)]
    assert client.get("/api/train/missing").status_code == 404


def test_trained_model_endpoints(env):
    pytest.importorskip("tensorflow")
    from src.data import load_frame
    from src.train import train

    client = env["client"]
    train(lookback=10, epochs=1, csv_path=env["csv"], artifact_dir_override=env["artifacts"] / "DEMO")

    health = client.get("/api/health").get_json()
    assert health["model_ready"] is True
    assert health["trained_tickers"] == ["DEMO"]

    evaluated = client.get("/api/evaluate").get_json()
    assert set(evaluated["comparison"]) == {"lstm", "naive", "drift", "ma5", "ridge"}

    future = client.get("/api/forecast?horizon=3&paths=20").get_json()
    assert len(future["forecast"]) == len(future["lower"]) == len(future["upper"]) == 3
    assert client.get("/api/forecast?horizon=99").status_code == 400

    chart = client.get("/api/chart.png?kind=forecast")
    assert chart.status_code == 200 and chart.mimetype == "image/png"

    rows = [
        {"date": r.Date.strftime("%Y-%m-%d"), "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close, "volume": int(r.Volume)}
        for r in load_frame(env["csv"]).tail(40).itertuples()
    ]
    predicted = client.post("/api/predict", json={"history": rows}).get_json()
    assert predicted["last_close"] == round(rows[-1]["close"], 2)
    assert isinstance(predicted["predicted_next_close"], float)
    too_short = client.post("/api/predict", json={"history": rows[:5]})
    assert too_short.status_code == 400
