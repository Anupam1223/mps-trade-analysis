# Assuming your source files are in a 'src' directory
# You may need to adjust the import paths based on your project structure
from src.data_ingestion import fetch_data
from src.data_preprocessing import preprocess_data
from src.model import build_model
from src.evaluate import print_evaluation_summary, plot_loss_history, plot_accuracy_history, plot_predictions, classification_metrics

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from sklearn.utils import class_weight
import numpy as np

def run_pipeline():
    """
    Executes the full machine learning pipeline from data ingestion to evaluation.
    This version includes key improvements for model stability and performance.
    """
    print("--- STEP 1: Starting Data Ingestion ---")
    raw_data_dict = fetch_data()
    print("--- Data Ingestion Complete ---")

    print("\n--- STEP 2: Starting Data Preprocessing ---")
    # --- IMPROVEMENT: Increased lookback window ---
    # A larger lookback (e.g., 20 days) provides more context for the model,
    # which is crucial for time-series data like stock prices.
    lookback_period = 20
    X_train, y_train, X_test, y_test = preprocess_data(raw_data_dict, symbol='AAPL', lookback=lookback_period)
    print("--- Data Preprocessing Complete ---")

    print("\n--- STEP 3: Starting Model Training ---")
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))

    # --- IMPROVEMENT: Increased bond dimension ---
    # A larger bond dimension (e.g., 10) increases the model's capacity,
    # allowing it to capture more complex correlations between time steps.
    bond_dimension = 10

    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        bond_dim=bond_dimension
    )
    model.summary() # Print model summary to verify the structure

    # Define log callback to print loss/accuracy per epoch for clarity
    log_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"Epoch {epoch+1:02d}: loss={logs['loss']:.4f}, acc={logs.get('sparse_categorical_accuracy', 0.0):.4f}, "
            f"val_loss={logs['val_loss']:.4f}, val_acc={logs.get('val_sparse_categorical_accuracy', 0.0):.4f}"
        )
    )

    # Callbacks for robust training
    callbacks = [
        # Stop training if validation loss doesn't improve for 7 epochs
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        # Reduce learning rate if validation loss plateaus
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        log_callback
    ]

    # --- Addressing Class Imbalance ---
    # Calculate class weights to give more importance to under-represented classes
    # during training. This is crucial for financial data.
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nUsing Class Weights: {class_weight_dict}\n")

    # --- Train the Model ---
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1 # Set to 1 to see the Keras progress bar
    )

    print("--- Model Training Complete ---")

    print("\n--- STEP 4: Starting Model Evaluation ---")
    # Evaluate the model's performance on the test set
    print_evaluation_summary(history)
    plot_loss_history(history)
    plot_accuracy_history(history)
    plot_predictions(model, X_test, y_test)
    classification_metrics(model, X_test, y_test)
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    run_pipeline()
