"""Flask API for stacked-LSTM stock close forecasting."""

from __future__ import annotations

import hmac
import io
import json
import os

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

from src import train as _train
from src.charts import history_and_forecast_png, predicted_vs_actual_png
from src.data import DataUnavailable, ensure_demo_csv, load_prices, normalize_ticker
from src.forecast import evaluate, forecast_horizon, predict_next_close
from src.jobs import TrainingJobs

ensure_demo_csv()

app = Flask(__name__, template_folder="templates")
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*"))
jobs = TrainingJobs()


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@app.errorhandler(ApiError)
def _api_error(exc: ApiError):
    return jsonify({"detail": exc.detail}), exc.status


@app.errorhandler(DataUnavailable)
def _data_unavailable(exc: DataUnavailable):
    return jsonify({"detail": str(exc)}), 502


def _ticker(value: str | None = None) -> str:
    raw = value if value is not None else request.args.get("ticker")
    try:
        return normalize_ticker(raw or os.environ.get("DEFAULT_TICKER"))
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc


def _model_ready(ticker: str) -> bool:
    dest = _train.artifact_dir(ticker)
    return (dest / "model.keras").exists() and (dest / "meta.json").exists()


def _require_model(ticker: str) -> None:
    if not _model_ready(ticker):
        raise ApiError(503, f"No trained model for {ticker}. Train it with POST /api/train or `python -m src.train --ticker {ticker}`.")


def _trained_tickers() -> list[str]:
    root = _train.ARTIFACTS
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "model.keras").exists())


def _int_arg(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except ValueError as exc:
        raise ApiError(400, f"{name} must be an integer.") from exc


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    ticker = _ticker()
    ready = _model_ready(ticker)
    meta = json.loads((_train.artifact_dir(ticker) / "meta.json").read_text(encoding="utf-8")) if ready else {}
    return jsonify(
        {
            "status": "ok",
            "ticker": ticker,
            "model_ready": ready,
            "lookback": meta.get("lookback"),
            "metrics": meta.get("metrics"),
            "trained_tickers": _trained_tickers(),
            "training_enabled": bool(os.environ.get("TRAIN_API_KEY")),
        }
    )


@app.get("/api/series")
def series():
    ticker = _ticker()
    frame, source = load_prices(ticker)
    tail = frame.tail(240)
    return jsonify(
        {
            "ticker": ticker,
            "source": source,
            "dates": tail["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "close": tail["Close"].round(2).tolist(),
        }
    )


@app.get("/api/evaluate")
def evaluate_view():
    ticker = _ticker()
    _require_model(ticker)
    return jsonify(evaluate(ticker))


@app.get("/api/forecast")
def forecast_view():
    ticker = _ticker()
    _require_model(ticker)
    try:
        return jsonify(forecast_horizon(_int_arg("horizon", 7), ticker, paths=_int_arg("paths", 200)))
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc


@app.post("/api/predict")
def predict_view():
    body = request.get_json(silent=True) or {}
    ticker = _ticker(body.get("ticker"))
    _require_model(ticker)
    history = body.get("history")
    if not isinstance(history, list) or not history:
        raise ApiError(400, "Provide 'history': a list of daily rows like {date, open, high, low, close, volume}, oldest first.")
    try:
        return jsonify(predict_next_close(history, ticker))
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc


@app.get("/api/chart.png")
def chart_png():
    ticker = _ticker()
    _require_model(ticker)
    if request.args.get("kind") == "forecast":
        future = forecast_horizon(_int_arg("horizon", 7), ticker)
        frame, _ = load_prices(ticker)
        tail = frame.tail(80)
        png = history_and_forecast_png(
            tail["Date"].dt.strftime("%Y-%m-%d").tolist(),
            tail["Close"].round(2).tolist(),
            future["dates"],
            future["forecast"],
            future["lower"],
            future["upper"],
        )
        filename = f"{ticker}_forecast.png"
    else:
        payload = evaluate(ticker)
        png = predicted_vs_actual_png(payload["dates"], payload["actual"], payload["predicted"], payload["naive"])
        filename = f"{ticker}_predicted_vs_actual.png"
    return send_file(io.BytesIO(png), mimetype="image/png", download_name=filename)


@app.post("/api/train")
def train_view():
    expected = os.environ.get("TRAIN_API_KEY", "")
    if not expected:
        raise ApiError(403, "Training is disabled. Set TRAIN_API_KEY on the server to enable it.")
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), expected):
        raise ApiError(401, "Missing or invalid X-API-Key header.")

    body = request.get_json(silent=True) or {}
    ticker = _ticker(body.get("ticker"))
    try:
        epochs = min(max(int(body.get("epochs", 25)), 1), 60)
        lookback = min(max(int(body.get("lookback", 60)), 10), 120)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "epochs and lookback must be integers.") from exc
    job, created = jobs.submit(ticker, {"epochs": epochs, "lookback": lookback}, _train.train)
    return jsonify({"created": created, "job": job}), 202


@app.get("/api/train/<job_id>")
def train_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise ApiError(404, "Unknown job id.")
    return jsonify(job)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "true").lower() == "true")
