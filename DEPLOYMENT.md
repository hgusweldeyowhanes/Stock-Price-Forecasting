# Deploy on Render

TensorFlow needs about **2 GB RAM**. Use a Standard web service.

## 1. Push

```bash
git add -A
git commit -m "Add stacked LSTM stock close forecasting."
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

Train locally and commit `artifacts/model.keras` plus `artifacts/meta.json` so Render does not train on boot.

## 3. Environment

```
PYTHON_VERSION=3.12.8
ALLOWED_ORIGINS=*
TF_CPP_MIN_LOG_LEVEL=2
```

If no artifacts are in the repo, the demo page will POST `/api/train` on first visit (slow, easy to time out).
