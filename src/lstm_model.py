# src/lstm_model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def build_lstm_model(input_shape: tuple, num_classes: int, lstm_units: int = 50, dropout_rate: float = 0.2, learning_rate: float = 1e-4) -> Model:
    """
    Builds a sequential LSTM model for time-series classification.

    Args:
        input_shape (tuple): The shape of the input data (lookback_period, num_features).
        num_classes (int): The number of output classes for classification.
        lstm_units (int): The number of units in each LSTM layer.
        dropout_rate (float): The dropout rate to apply after LSTM layers.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        tf.keras.models.Model: The compiled Keras model.
    """
    inputs = Input(shape=input_shape)

    # First LSTM layer with Dropout
    # return_sequences=True is necessary to pass the full sequence to the next LSTM layer
    x = LSTM(units=lstm_units, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)

    # Second LSTM layer with Dropout
    # return_sequences=False as we only need the output of the last time step
    x = LSTM(units=lstm_units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    
    # A dense layer to learn combinations of features from the LSTM output
    x = Dense(units=25, activation='relu')(x)

    # The final output layer with softmax for classification probabilities
    outputs = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with the same loss and metrics as the MPS for a fair comparison
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['sparse_categorical_accuracy']
    )
    
    return model