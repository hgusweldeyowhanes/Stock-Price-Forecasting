# Stock Price Forecasting (LSTM)

Stacked LSTM that predicts the next **closing price** from a sliding window of past closes. TensorFlow + Pandas + Matplotlib, with a small Flask demo.

This is a portfolio demo on synthetic daily prices (learnable trend + weekly rhythm). It is not trading advice.

## Pipeline

1. Scale closes on the training period only (min–max)
2. Build windows of 60 days → next close
3. Chronological 80/20 split
4. Three stacked LSTMs (64 → 64 → 32) + dropout
5. Chart **predicted vs actual** on the held-out test set
6. Roll the last window forward for a 7-session forecast

## Run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m src.train             # ~1–3 min on CPU
python api.py
```

Open http://127.0.0.1:8000 (set `PORT=8001` if 8000 is taken).

Python 3.11/3.12 installs stable TensorFlow. Python 3.14 uses `tf-nightly` until a stable wheel exists.

| Method | Path | Notes |
|---|---|---|
| GET | `/` | Charts UI |
| GET | `/api/health` | Model status + test metrics |
| GET | `/api/evaluate` | Test dates, actual, predicted |
| GET | `/api/forecast?horizon=7` | Recursive future closes |
| GET | `/api/chart.png` | Matplotlib predicted vs actual |
| GET | `/api/chart.png?kind=forecast` | History + forecast PNG |
| POST | `/api/train` | Retrain `{ "epochs": 12, "lookback": 60 }` |

## Tests

```bash
pytest
```

## Deploy

See [DEPLOYMENT.md](DEPLOYMENT.md). TensorFlow needs a **Standard** Render instance (free 512 MB will run out of memory).
