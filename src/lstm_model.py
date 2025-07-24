# src/lstm_model.py
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model

from src.helper import quantile_loss  # Import the new loss function


def build_quantile_lstm_model(
    input_shape: tuple,
    quantile: float,
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-4,
) -> Model:
    """Builds a Bidirectional LSTM model for quantile regression."""
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=25, activation="relu")(x)

    # --- KEY CHANGE: Output layer for regression ---
    # Single neuron, linear activation (no activation function specified)
    outputs = Dense(units=1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    # --- KEY CHANGE: Compile with quantile loss and regression metrics ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=quantile_loss(quantile),
        metrics=["mean_absolute_error"],  # Use a regression metric
    )

    return model
