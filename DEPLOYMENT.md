# Deploy on Render

TensorFlow needs about **2 GB RAM**. Use a Standard web service.

## 1. Push

```bash
git add -A
git commit -m "Stock forecasting: returns LSTM, baselines, tickers, uncertainty bands."
git push origin main
```

## 2. Web service

| Field | Value |
|---|---|
| Language | Python |
| Build | `pip install -r requirements.txt` |
| Start | `gunicorn api:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180` |
| Health | `/api/health` |
| Instance | Standard |

Keep `--workers 1`. Training jobs and their status live in the web process, so a second worker would not see jobs started by the first.

The DEMO model is committed in `artifacts/DEMO/`, so the site works on first boot without training.

## 3. Environment

```
PYTHON_VERSION=3.12.8
ALLOWED_ORIGINS=*
TF_CPP_MIN_LOG_LEVEL=2
DEFAULT_TICKER=DEMO
TRAIN_API_KEY=<long random string>
```

- Without `TRAIN_API_KEY`, `POST /api/train` returns 403 and only committed models are served.
- With it, the UI shows a training panel for untrained tickers. Training runs in the background (1–3 minutes on CPU) and the browser polls `GET /api/train/<job_id>`.

## 4. Persistence

Render's filesystem is ephemeral. Models trained through the API, and the yfinance price cache, disappear on redeploy or restart. To keep a real-ticker model, train it locally and commit it:

```bash
python -m src.train --ticker AAPL
git add -f artifacts/AAPL
```

(`.gitignore` only includes `artifacts/DEMO/` by default.) Or attach a Render persistent disk mounted at `artifacts/`.
