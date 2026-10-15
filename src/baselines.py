"""Simple next-close baselines the LSTM has to beat."""

from __future__ import annotations

import numpy as np

RIDGE_LAGS = 5


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression with an unpenalised intercept; returns weights with intercept first."""
    design = np.column_stack([np.ones(len(x)), x])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def ridge_predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return weights[0] + x @ weights[1:]


def lag_matrix(windows: np.ndarray, lags: int = RIDGE_LAGS) -> np.ndarray:
    """Flatten the last ``lags`` feature rows of each window into one regression row."""
    return windows[:, -lags:, :].reshape(len(windows), -1)


def baseline_prices(
    closes: np.ndarray,
    ends: np.ndarray,
    train_returns: np.ndarray,
    train_windows: np.ndarray,
    test_windows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Predicted next close for each test window end, keyed by baseline name."""
    last = closes[ends]
    weights = ridge_fit(lag_matrix(train_windows), train_returns)
    ridge_returns = ridge_predict(weights, lag_matrix(test_windows))
    return {
        "naive": last.copy(),
        "drift": last * np.exp(float(np.mean(train_returns))),
        "ma5": np.array([closes[end - 4 : end + 1].mean() for end in ends]),
        "ridge": last * np.exp(ridge_returns),
    }
