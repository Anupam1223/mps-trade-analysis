from src.data_ingestion import fetch_data
from src.data_preprocessing import preprocess_data
from src.model import build_model
from src.evaluate import print_evaluation_summary, plot_loss_history, plot_accuracy_history, plot_predictions
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def run_pipeline():
    """
    Executes the full machine learning pipeline from data ingestion to evaluation.
    """
    print("--- STEP 1: Starting Data Ingestion ---")
    raw_data_dict = fetch_data()
    print("--- Data Ingestion Complete ---")

    print("\n--- STEP 2: Starting Data Preprocessing ---")
    X_train, y_train, X_test, y_test = preprocess_data(raw_data_dict, symbol='AAPL', lookback=10)
    print("--- Data Preprocessing Complete ---")

    print("\n--- STEP 3: Starting Model Training ---")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape=input_shape, bond_dim=8)
    
    # Set verbose=0 to suppress the epoch-by-epoch output
    # --- Define Callbacks ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    # --- Train the Model with Callbacks ---
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0  # or verbose=1 if you want to see epoch logs
    )

    print("--- Model Training Complete ---")

    print("\n--- STEP 4: Starting Model Evaluation ---")
    
    # Print the final summary table
    print_evaluation_summary(history)

    # Display separate plots for loss and accuracy
    plot_loss_history(history)
    plot_accuracy_history(history)
    
    # Display the prediction plot
    plot_predictions(model, X_test, y_test)
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    run_pipeline()