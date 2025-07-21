# run_comparison.py

import numpy as np
import tensorflow as tf
import time  # <--- 1. Import the time module
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import your existing and new evaluate functions
from src.data_ingestion import fetch_data
from src.data_preprocessing import preprocess_data
from src.evaluate import (classification_metrics, plot_comparison_metrics,
                          plot_comparison_roc, plot_predictions, plot_training_times) # <--- 2. Import new plot function

# Import both model builders
from src.mps_model import build_model as build_mps_model
from src.lstm_model import build_lstm_model

def run_comparison_pipeline():
    """
    Executes a full ML pipeline to train and compare MPS and LSTM models.
    """
    # --- STEP 1: Data Ingestion & Preprocessing ---
    print("--- 🔵 STEP 1: Starting Data Ingestion & Preprocessing ---")
    raw_data_dict = fetch_data()
    lookback_period = 20
    X_train, y_train, X_test, y_test = preprocess_data(raw_data_dict, symbol='AAPL', lookback=lookback_period)
    print("--- ✅ Data Ingestion & Preprocessing Complete ---")

    # --- Shared Parameters and Callbacks ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nUsing Class Weights: {class_weight_dict}\n")
    results = {}

    # --- STEP 2.A: Train and Evaluate the MPS Model ---
    print("\n--- 🧠 STEP 2.A: Training MPS Model ---")
    mps_model = build_mps_model(
        input_shape=input_shape, num_classes=num_classes, bond_dim=10,
        learning_rate=1e-4, l2_lambda=1e-5
    )
    mps_model.summary()
    
    # 3. Time the MPS training
    mps_start_time = time.perf_counter()
    mps_model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
        callbacks=callbacks, class_weight=class_weight_dict, verbose=1
    )
    mps_end_time = time.perf_counter()
    mps_training_time = mps_end_time - mps_start_time
    
    print(f"--- ✅ MPS Model Training Complete in {mps_training_time:.2f} seconds ---")
    print("\n--- 📊 Evaluating MPS Model ---")
    mps_score, mps_report = classification_metrics(mps_model, X_test, y_test)
    results['MPS'] = {'roc_auc': mps_score, 'report': mps_report, 'training_time': mps_training_time}
    plot_predictions(mps_model, X_test, y_test)

    # --- STEP 2.B: Train and Evaluate the LSTM Model ---
    print("\n\n--- 🧠 STEP 2.B: Training LSTM Model ---")
    lstm_model = build_lstm_model(
        input_shape=input_shape, num_classes=num_classes, learning_rate=1e-4
    )
    lstm_model.summary()

    # 4. Time the LSTM training
    lstm_start_time = time.perf_counter()
    lstm_model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
        callbacks=callbacks, class_weight=class_weight_dict, verbose=1
    )
    lstm_end_time = time.perf_counter()
    lstm_training_time = lstm_end_time - lstm_start_time

    print(f"--- ✅ LSTM Model Training Complete in {lstm_training_time:.2f} seconds ---")
    print("\n--- 📊 Evaluating LSTM Model ---")
    lstm_score, lstm_report = classification_metrics(lstm_model, X_test, y_test)
    results['LSTM'] = {'roc_auc': lstm_score, 'report': lstm_report, 'training_time': lstm_training_time}
    plot_predictions(lstm_model, X_test, y_test)
    
    # --- STEP 3: Final Side-by-Side Comparison ---
    print("\n\n" + "="*50)
    print("🏁 FINAL MODEL COMPARISON 🏁")
    print("="*50 + "\n")

    # 5. Update the report to include training time
    for model_name, metrics in results.items():
        print(f"------ {model_name} Results ------")
        print(f"Training Time:   {metrics.get('training_time', 0):.2f} seconds")
        print(f"ROC-AUC Score:   {metrics['roc_auc']:.4f}")
        print("Classification Report:")
        print(metrics['report'])
        print("-" * (len(model_name) + 16) + "\n")

    # --- Show the visual comparison plots ---
    print("\n--- 📈 Generating Comparison Plots ---")
    trained_models = {'MPS': mps_model, 'LSTM': lstm_model}
    plot_comparison_roc(trained_models, X_test, y_test)
    plot_comparison_metrics(trained_models, X_test, y_test)
    plot_training_times(results) # <--- 6. Call the new plot function
        
    print("--- ✅ Pipeline Finished ---")

if __name__ == "__main__":
    run_comparison_pipeline()