# run_quantile.py

import time
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data_ingestion import fetch_forex_data_yf
from src.data_preprocessing import preprocess_for_quantile_regression
from src.evaluate import comprehensive_evaluation
from src.lstm_model import build_quantile_lstm_model
# Import the new callback generator from the mps_model script
from src.mps_model import build_quantile_mps_model, get_mps_callbacks


def run_quantile_pipeline():
    """Executes a full ML pipeline for quantile regression."""

    # --- STEP 1: Data Ingestion & Preprocessing ---
    print("--- 🔵 STEP 1: Starting Data Ingestion & Preprocessing for Quantiles ---")
    raw_data_dict = fetch_forex_data_yf()
    lookback_period = 40

    (
        X_train,
        y_train,
        X_test,
        y_test,
        x_scaler,
        y_scaler,
    ) = preprocess_for_quantile_regression(raw_data_dict, lookback=lookback_period)

    print("\n--- ✅ Data Ingestion & Preprocessing Complete ---")

    # --- Shared Parameters ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    quantiles = [0.1, 0.5, 0.9]

    # Shared callbacks for both models
    shared_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    lstm_predictions = {}
    mps_predictions = {}
    lstm_histories = {}
    mps_histories = {}

    lstm_training_times = {}
    mps_training_times = {}

    # --- STEP 2: Train a Model for Each Quantile ---
    for q in quantiles:
        print("\n" + "=" * 50)
        print(f"🧠 Training Models for Quantile: {q}")
        print("=" * 50)

        # -- LSTM --
        print("\n--- Training LSTM ---")
        lstm_model = build_quantile_lstm_model(input_shape=input_shape, quantile=q)
        start_time = time.time()
        lstm_histories[q] = lstm_model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=shared_callbacks, # Use only the shared callbacks for LSTM
            verbose=1,
        )
        lstm_training_times[q] = time.time() - start_time
        pred_scaled = lstm_model.predict(X_test)
        lstm_predictions[q] = y_scaler.inverse_transform(pred_scaled)

        # -- MPS --
        print("\n--- Training MPS ---")
        mps_model = build_quantile_mps_model(
            input_shape=input_shape, quantile=q, bond_dim=6
        )
        
        # --- CHANGE: Create specific callbacks for this MPS run ---
        # This will create a unique log directory for each quantile modelsss
        log_dir = os.path.join("logs", "fit", f"mps_q_{q}")
        mps_specific_callbacks = get_mps_callbacks(log_dir_base=log_dir)
        all_mps_callbacks = shared_callbacks + mps_specific_callbacks

        start_time = time.time()
        mps_histories[q] = mps_model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=all_mps_callbacks, # Pass the combined list of callbacks
            verbose=1,
        )
        mps_training_times[q] = time.time() - start_time
        pred_scaled = mps_model.predict(X_test)
        mps_predictions[q] = y_scaler.inverse_transform(pred_scaled)

    # --- STEP 3: Comprehensive Evaluation ---
    y_test_unscaled = y_scaler.inverse_transform(y_test)

    total_lstm_time = sum(lstm_training_times.values())
    total_mps_time = sum(mps_training_times.values())

    print(f"\nTotal LSTM Training Time: {total_lstm_time:.2f} seconds")
    print(f"Total MPS Training Time: {total_mps_time:.2f} seconds")

    comprehensive_evaluation(
        y_true=y_test_unscaled, 
        predictions=lstm_predictions, 
        histories=lstm_histories, 
        model_name="LSTM Model", 
        training_time=total_lstm_time
    )

    comprehensive_evaluation(
        y_true=y_test_unscaled, 
        predictions=mps_predictions, 
        histories=mps_histories, 
        model_name="MPS Model", 
        training_time=total_mps_time
    )

    print("--- ✅ Pipeline Finished ---")


if __name__ == "__main__":
    run_quantile_pipeline()
