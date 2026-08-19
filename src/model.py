"""Stacked LSTM closing-price model."""

from __future__ import annotations


def build_lstm(lookback: int, learning_rate: float = 0.001):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.Input(shape=(lookback, 1), name="close_window"),
            keras.layers.LSTM(64, return_sequences=True, name="lstm_1"),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, return_sequences=True, name="lstm_2"),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, name="lstm_3"),
            keras.layers.Dense(1, name="next_close"),
        ],
        name="stacked_lstm_close",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
