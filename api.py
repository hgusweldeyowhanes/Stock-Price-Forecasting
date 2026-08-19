"""Flask API for stacked-LSTM stock close forecasting."""

from __future__ import annotations

import io
import os

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

from src.charts import history_and_forecast_png, predicted_vs_actual_png
from src.data import DATA_PATH, ensure_demo_csv, load_frame
from src.forecast import evaluate, forecast_horizon, load_artifacts, predict_next_close
from src.train import ARTIFACTS, train

ensure_demo_csv()

app = Flask(__name__, template_folder="templates")
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*"))


def _model_ready() -> bool:
    return (ARTIFACTS / "model.keras").exists() and (ARTIFACTS / "meta.json").exists()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    ready = _model_ready()
    meta = {}
    if ready:
        _, _, meta = load_artifacts()
    return jsonify(
        {
            "status": "ok",
            "model_ready": ready,
            "lookback": meta.get("lookback"),
            "metrics": meta.get("metrics"),
        }
    )


@app.get("/api/series")
def series():
    frame = load_frame()
    tail = frame.tail(240)
    return jsonify(
        {
            "ticker": "DEMO",
            "source": str(DATA_PATH.name),
            "dates": tail["Date"].astype(str).tolist(),
            "close": tail["Close"].round(2).tolist(),
        }
    )


@app.get("/api/evaluate")
def evaluate_view():
    if not _model_ready():
        return jsonify({"detail": "Model not trained. POST /api/train first."}), 503
    return jsonify(evaluate())


@app.get("/api/forecast")
def forecast_view():
    if not _model_ready():
        return jsonify({"detail": "Model not trained. POST /api/train first."}), 503
    try:
        horizon = int(request.args.get("horizon", 7))
        return jsonify(forecast_horizon(horizon=horizon))
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400


@app.post("/api/predict")
def predict_view():
    if not _model_ready():
        return jsonify({"detail": "Model not trained. POST /api/train first."}), 503
    body = request.get_json(silent=True) or {}
    history = body.get("history")
    if not isinstance(history, list) or not history:
        return jsonify({"detail": "Provide a non-empty 'history' list of recent close prices."}), 400
    try:
        return jsonify(predict_next_close(history))
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400


@app.get("/api/chart.png")
def chart_png():
    if not _model_ready():
        return jsonify({"detail": "Model not trained."}), 503
    payload = evaluate()
    kind = request.args.get("kind", "overlay")
    if kind == "forecast":
        future = forecast_horizon(horizon=int(request.args.get("horizon", 7)))
        frame = load_frame().tail(80)
        png = history_and_forecast_png(
            frame["Date"].astype(str).tolist(),
            frame["Close"].round(2).tolist(),
            future["dates"],
            future["forecast"],
        )
        filename = "forecast.png"
    else:
        png = predicted_vs_actual_png(payload["dates"], payload["actual"], payload["predicted"])
        filename = "predicted_vs_actual.png"
    return send_file(io.BytesIO(png), mimetype="image/png", download_name=filename)


@app.post("/api/train")
def train_view():
    body = request.get_json(silent=True) or {}
    epochs = int(body.get("epochs", 12))
    lookback = int(body.get("lookback", 60))
    epochs = min(max(epochs, 1), 40)
    lookback = min(max(lookback, 10), 90)
    meta = train(lookback=lookback, epochs=epochs)
    return jsonify({"ok": True, "metrics": meta["metrics"], "lookback": lookback, "epochs_ran": meta["epochs_ran"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "true").lower() == "true")
