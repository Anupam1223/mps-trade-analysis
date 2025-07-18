import pandas as pd
from src.data_ingestion import fetch_data
from src.data_preprocessing import preprocess_data_classical
from src.model import train_classical_models
from src.evaluate import evaluate_model

def run_pipeline():
    """
    Executes the full classical ML pipeline from data ingestion to evaluation.
    """
    print("--- STEP 1: Starting Data Ingestion ---")
    raw_data_dict = fetch_data()
    print("--- Data Ingestion Complete ---")

    print("\n--- STEP 2: Starting Data Preprocessing for Classical Models ---")
    X_train, y_train, X_test, y_test, scaler = preprocess_data_classical(
        raw_data_dict, symbol='GOOGL', lookback=12
    )
    print("--- Data Preprocessing Complete ---")

    print("\n--- STEP 3: Starting Model Training ---")
    trained_models = train_classical_models(X_train, y_train)
    print("--- Model Training Complete ---")

    print("\n--- STEP 4: Starting Model Evaluation ---")
    results = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, name, X_test, y_test)
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'ROC-AUC': metrics['roc_auc']
        })

    # --- STEP 5: Final Performance Summary ---
    results_df = pd.DataFrame(results)
    print("\n\n" + "="*50)
    print("      CLASSICAL MODEL PERFORMANCE SUMMARY (AAPL)")
    print("="*50)
    print(results_df.to_string(index=False))
    print("="*50)

    print("\n--- Classical Pipeline Finished ---")

if __name__ == "__main__":
    run_pipeline()