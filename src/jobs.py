"""In-process background training jobs (single worker, so run gunicorn with --workers 1)."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

ACTIVE = {"queued", "running"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrainingJobs:
    def __init__(self, max_history: int = 50):
        # One worker: TensorFlow training is CPU/RAM heavy, so jobs run one at a time.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._futures: dict[str, Future] = {}
        self._max_history = max_history

    def submit(self, ticker: str, params: dict, train_fn: Callable[..., dict]) -> tuple[dict, bool]:
        """Queue a job; returns (job, created). An active job for the same ticker is reused."""
        with self._lock:
            for job in self._jobs.values():
                if job["ticker"] == ticker and job["status"] in ACTIVE:
                    return dict(job), False
            job_id = uuid.uuid4().hex[:12]
            job = {
                "id": job_id,
                "ticker": ticker,
                "params": params,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "metrics": None,
            }
            self._jobs[job_id] = job
            self._prune()
            self._futures[job_id] = self._executor.submit(self._run, job_id, train_fn)
            return dict(job), True

    def _run(self, job_id: str, train_fn: Callable[..., dict]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _now()
            ticker, params = job["ticker"], dict(job["params"])
        try:
            meta = train_fn(ticker=ticker, **params)
            update = {"status": "succeeded", "metrics": meta.get("metrics")}
        except Exception as exc:
            update = {"status": "failed", "error": str(exc)}
        with self._lock:
            job.update(update, finished_at=_now())

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def wait(self, job_id: str, timeout: float | None = None) -> dict | None:
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.get(job_id)

    def _prune(self) -> None:
        finished = [jid for jid, job in self._jobs.items() if job["status"] not in ACTIVE]
        for jid in finished[: max(len(self._jobs) - self._max_history, 0)]:
            self._jobs.pop(jid, None)
            self._futures.pop(jid, None)
