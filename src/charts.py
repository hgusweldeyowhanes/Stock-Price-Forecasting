"""Predicted vs actual Matplotlib charts."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def predicted_vs_actual_png(dates, actual, predicted, title: str = "Predicted vs actual close") -> bytes:
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=140)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.plot(range(len(actual)), actual, color="#e5e7eb", linewidth=1.8, label="Actual")
    ax.plot(range(len(predicted)), predicted, color="#fbbf24", linewidth=1.8, label="LSTM predicted")
    ax.set_title(title, color="#f9fafb", loc="left", fontsize=13, pad=10)
    ax.set_ylabel("Close", color="#d1d5db")
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.grid(color="#1f2937", linestyle="--", linewidth=0.6)
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb")
    if dates:
        ticks = list(range(0, len(dates), max(len(dates) // 6, 1)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([dates[i] for i in ticks], rotation=0)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def history_and_forecast_png(history_dates, history_close, forecast_dates, forecast_close) -> bytes:
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=140)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.plot(range(len(history_close)), history_close, color="#e5e7eb", linewidth=1.6, label="History")
    offset = len(history_close) - 1
    xs = list(range(offset, offset + len(forecast_close) + 1))
    ys = [history_close[-1], *forecast_close]
    ax.plot(xs, ys, color="#34d399", linewidth=2, label="Forecast")
    ax.set_title("Close history and LSTM forecast", color="#f9fafb", loc="left", fontsize=13, pad=10)
    ax.set_ylabel("Close", color="#d1d5db")
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.grid(color="#1f2937", linestyle="--", linewidth=0.6)
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()
