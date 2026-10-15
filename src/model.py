"""Stacked LSTM that predicts the next-day log return from a window of OHLCV features."""

from __future__ import annotations


def build_lstm(lookback: int, n_features: int = 1, learning_rate: float = 0.001):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.Input(shape=(lookback, n_features), name="feature_window"),
            keras.layers.LSTM(64, return_sequences=True, name="lstm_1"),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, return_sequences=True, name="lstm_2"),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, name="lstm_3"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, name="next_return"),
        ],
        name="stacked_lstm_returns",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
