from src.data_ingestion import fetch_data
from src.data_preprocessing import preprocess_data
from src.model import build_model
from src.evaluate import print_evaluation_summary, plot_loss_history, plot_accuracy_history, plot_predictions
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.evaluate import classification_metrics
from sklearn.utils import class_weight
import numpy as np

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
    
    # --- THIS IS THE FIX ---
    # Determine the number of classes from the preprocessed labels (will be 2)
    num_classes = len(np.unique(y_train))
    
    # Pass the required 'num_classes' argument to the model builder
    model = build_model(input_shape=input_shape, num_classes=num_classes, bond_dim=8)
    
    # Define Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    # Compute class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # --- Train the Model with Callbacks ---
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1 # Changed to 1 to show training progress
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

    classification_metrics(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()
