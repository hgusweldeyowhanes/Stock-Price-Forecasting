# Stock Price Forecasting (LSTM)

A stacked LSTM predicts tomorrow's **return** from 60 days of OHLCV features, then turns that into a next-day close. It is scored against simple baselines on a held-out test period, and forecasts come with uncertainty bands. TensorFlow + Pandas + Matplotlib, with a Flask API and a small web UI.

Portfolio project, not investment advice.

## Pipeline

1. **Data**: `DEMO` is synthetic daily OHLCV (offline). Any Yahoo Finance symbol (`AAPL`, `MSFT`, `^GSPC`) is loaded with `yfinance` and cached for 12 hours. If Yahoo is unreachable, a stale cache is used.
2. **Features** (9 per day): log return, high–low range, open–close move, relative volume, gap to the 7- and 21-day moving averages, RSI(14), and day of week (sin/cos).
3. **Target**: next-day log return. Predicting returns instead of raw prices keeps the model's inputs and outputs on a stable scale even when prices trend out of the range seen in training.
4. **Split**: chronological 80/20. The feature scaler is fit on training rows only. Early stopping watches the last 10% of the training data, never the test set.
5. **Model**: three stacked LSTMs (64 → 64 → 32) with dropout.
6. **Evaluation**: RMSE, MAE, MAPE and **direction hit rate** (did it call up vs. down correctly), compared against baselines:
   - **Naive**: tomorrow closes where today closed
   - **Drift**: naive plus the average training return
   - **MA5**: 5-day average close
   - **Ridge**: linear regression on the last 5 days of features
7. **Forecast**: 200 simulated paths, 7 sessions ahead. Each step keeps dropout on (model uncertainty) and adds noise sized from test-set errors (market noise). The UI shows the median with a 10th–90th percentile band.

## Results

**DEMO** (synthetic, 284 test days):

| Model | RMSE | MAE | Direction |
|---|---|---|---|
| **LSTM** | **1.297** | **1.038** | **67.4%** |
| Ridge | 1.301 | 1.040 | 65.6% |
| Naive | 1.426 | 1.131 | – |
| Drift | 1.426 | 1.131 | 50.7% |
| MA5 | 2.662 | 2.209 | 36.2% |

The LSTM beats naive by **9% RMSE**. The synthetic series has weekly and monthly cycles that a model can learn. Ridge does almost as well, so most of the gain comes from the features rather than the LSTM itself.

**AAPL** (6 years from Yahoo Finance, 285 test days):

| Model | RMSE | MAE | Direction |
|---|---|---|---|
| Drift | 4.404 | 3.115 | 54.4% |
| Naive | 4.413 | 3.126 | – |
| LSTM | 4.429 | 3.142 | 46.7% |
| Ridge | 4.535 | 3.238 | 51.6% |

On real prices nothing meaningfully beats "tomorrow = today". Daily moves are close to a random walk, and that's what the baseline comparison is there to show. A model that only reports its own RMSE would look accurate here (a MAPE of about 1.1%) while adding nothing.

Your numbers will differ slightly between runs and as new market data arrives.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
python -m src.train                # DEMO, ~1–2 min on CPU
python -m src.train --ticker AAPL  # any Yahoo Finance symbol
python api.py
```

Open http://127.0.0.1:8000 (set `PORT=8001` if 8000 is taken) and switch tickers with `?ticker=AAPL` or the input box.

To train from the UI or API, set a key before starting the server:

```bash
set TRAIN_API_KEY=some-long-secret    # Windows (PowerShell: $env:TRAIN_API_KEY="...")
python api.py
```

Python 3.11/3.12 installs stable TensorFlow. Python 3.14 uses `tf-nightly` until a stable wheel exists.

## API

All GET endpoints take `?ticker=` (default `DEMO`, or `DEFAULT_TICKER`).

| Method | Path | Notes |
|---|---|---|
| GET | `/` | Charts UI |
| GET | `/api/health` | Model status, test metrics, trained tickers |
| GET | `/api/series` | Last 240 closes and data source |
| GET | `/api/evaluate` | Test dates, actual, LSTM, naive, baseline comparison |
| GET | `/api/forecast?horizon=7&paths=200` | Median forecast with `lower`/`upper` (10th/90th percentile) |
| POST | `/api/predict` | `{ "ticker": "DEMO", "history": [{date, open, high, low, close, volume}, …] }`, at least 81 rows |
| GET | `/api/chart.png` | Matplotlib predicted vs actual (`?kind=forecast` for the band chart) |
| POST | `/api/train` | Header `X-API-Key`; body `{ "ticker": "AAPL", "epochs": 25, "lookback": 60 }`. Returns `202` with a job |
| GET | `/api/train/<job_id>` | Job status: `queued`, `running`, `succeeded`, `failed` |

`POST /api/train` returns 403 when `TRAIN_API_KEY` is not set on the server and 401 for a wrong key. Training runs on one background thread, one job at a time. A second request for a ticker that's already training returns the existing job.

## Tests

```bash
pytest
```

GitHub Actions runs the same suite on every push (`.github/workflows/ci.yml`). The end-to-end tests train for a single epoch on a small synthetic series.

## Deploy

See [DEPLOYMENT.md](DEPLOYMENT.md). TensorFlow needs a **Standard** Render instance (free 512 MB will run out of memory).
