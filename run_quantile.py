# run_quantile.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data_ingestion import fetch_forex_data_yf
from src.data_preprocessing import preprocess_for_quantile_regression
from src.evaluate import comprehensive_evaluation
from src.lstm_model import build_quantile_lstm_model
from src.mps_model import build_quantile_mps_model


def run_quantile_pipeline():
    """Executes a full ML pipeline for quantile regression."""
    
    # --- STEP 1: Data Ingestion & Preprocessing ---
    print("--- 🔵 STEP 1: Starting Data Ingestion & Preprocessing for Quantiles ---")
    raw_data_dict = fetch_forex_data_yf()
    lookback_period = 12 
    
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = preprocess_for_quantile_regression(
        raw_data_dict,
        lookback=lookback_period
    )
    
    print("\n--- ✅ Data Ingestion & Preprocessing Complete ---")

    # --- Shared Parameters ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    quantiles = [0.1, 0.5, 0.9]
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            min_delta=1e-4,
            patience=7, 
            restore_best_weights=True, 
            verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    lstm_predictions = {}
    mps_predictions = {}
    lstm_histories = {}
    mps_histories = {}

    # --- STEP 2: Train a Model for Each Quantile ---
    for q in quantiles:
        print(f"\n" + "="*50)
        print(f"🧠 Training Models for Quantile: {q} �")
        print("="*50)

        # -- LSTM --
        print("\n--- Training LSTM ---")
        lstm_model = build_quantile_lstm_model(input_shape=input_shape, quantile=q)
        lstm_histories[q] = lstm_model.fit(
            X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
            callbacks=callbacks, verbose=1
        )
        pred_scaled = lstm_model.predict(X_test)
        lstm_predictions[q] = y_scaler.inverse_transform(pred_scaled)

        # -- MPS --
        print("\n--- Training MPS ---")
        mps_model = build_quantile_mps_model(input_shape=input_shape, quantile=q, bond_dim=4)
        mps_histories[q] = mps_model.fit(
            X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
            callbacks=callbacks, verbose=1
        )
        pred_scaled = mps_model.predict(X_test)
        mps_predictions[q] = y_scaler.inverse_transform(pred_scaled)

    # --- STEP 3: Comprehensive Evaluation ---
    y_test_unscaled = y_scaler.inverse_transform(y_test)
    
    comprehensive_evaluation(lstm_predictions, y_test_unscaled, lstm_histories, "LSTM Model")
    comprehensive_evaluation(mps_predictions, y_test_unscaled, mps_histories, "MPS Model")

    print("--- ✅ Pipeline Finished ---")

if __name__ == "__main__":
    run_quantile_pipeline()