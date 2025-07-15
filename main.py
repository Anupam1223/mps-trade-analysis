from src.data_ingestion import fetch_data
from src.data_preprocessing import preprocess_data # <-- Import
from src.model import build_model                   # <-- Import

def run_pipeline():
    """
    Executes the full machine learning pipeline from data ingestion to evaluation.
    """
    print("--- STEP 1: Starting Data Ingestion ---")
    raw_data_dict = fetch_data()
    print("--- Data Ingestion Complete ---")

    print("\n--- STEP 2: Starting Data Preprocessing ---")
    # We will model AAPL stock with a lookback of 10 days
    X_train, y_train, X_test, y_test = preprocess_data(raw_data_dict, symbol='AAPL', lookback=10)
    print("--- Data Preprocessing Complete ---")

    print("\n--- STEP 3: Starting Model Training ---")
    # The input shape is (lookback, features_per_step)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape=input_shape, bond_dim=8)
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    print("--- Model Training Complete ---")

    # print("\n--- STEP 4: Starting Model Evaluation ---")
    # evaluate_model(model, X_test, y_test)
    # print("--- Pipeline Finished ---")

if __name__ == "__main__":
    run_pipeline()