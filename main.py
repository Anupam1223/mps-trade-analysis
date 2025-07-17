from src.data_ingestion import fetch_data
from src.data_preprocessing import preprocess_data
from src.model import build_model
from src.dmrg_optimizer import train_with_dmrg # <-- MODIFICATION: Import the new trainer
from src.evaluate import print_evaluation_summary, plot_loss_history, plot_accuracy_history, plot_predictions, classification_metrics
import numpy as np

def run_pipeline():
    """
    Executes the full machine learning pipeline from data ingestion to evaluation,
    using a DMRG-style optimization for training.
    """
    print("--- STEP 1: Starting Data Ingestion ---")
    raw_data_dict = fetch_data()
    print("--- Data Ingestion Complete ---")

    print("\n--- STEP 2: Starting Data Preprocessing ---")
    # Using a lookback of 12 features for this example
    X_train, y_train, X_test, y_test = preprocess_data(raw_data_dict, symbol='AAPL', lookback=12)
    print("--- Data Preprocessing Complete ---")

    print("\n--- STEP 3: Starting Model Training with DMRG Optimizer ---")
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    
    # Build the model as before. The compilation step provides the optimizer and loss function
    # to the DMRG training function.
    model = build_model(
        input_shape=input_shape, 
        num_classes=num_classes, 
        bond_dim=8, 
        feature_map_type='angle' # Using non-linear feature map
    )
    
    model.summary()

    # --- MODIFICATION: Replace model.fit() with the DMRG training function ---
    # Callbacks and class weights are not used here as the training loop is custom.
    history = train_with_dmrg(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=10, # Number of full sweeps
        batch_size=64
    )
    # -------------------------------------------------------------------------

    print("--- Model Training Complete ---")

    print("\n--- STEP 4: Starting Model Evaluation ---")
    
    # The evaluation functions work with the history dictionary returned by the custom trainer.
    print_evaluation_summary(history)
    plot_loss_history(history)
    plot_accuracy_history(history)
    plot_predictions(model, X_test, y_test)
    classification_metrics(model, X_test, y_test)
    
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    run_pipeline()
