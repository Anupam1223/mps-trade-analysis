from src.data_ingestion import fetch_data
from src.data_preprocessing import preprocess_data
from src.model import build_model
from src.evaluate import classification_metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
import optuna # <-- Import Optuna

def objective(trial: optuna.Trial, raw_data_dict: dict, symbol: str) -> float:
    """
    The objective function for Optuna to optimize.
    A single trial represents one set of hyperparameters.
    """
    # --- 1. Define Hyperparameter Search Space using trial.suggest_* ---
    params = {
        'bond_dim': trial.suggest_categorical('bond_dim', [4, 8, 12, 16]),              
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'l2_lambda': trial.suggest_float('l2_lambda', 1e-6, 1e-2, log=True),
        'lookback': trial.suggest_categorical('lookback', [10, 15, 20])
    }
    
    print(f"\n--- [{symbol}] Starting Trial {trial.number} with params: {params} ---")

    # --- 2. Preprocess Data ---
    X_train, y_train, X_test, y_test = preprocess_data(
        raw_data_dict, symbol=symbol, lookback=params['lookback']
    )

    # --- 3. Build Model ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    
    model = build_model(
        input_shape=input_shape, 
        num_classes=num_classes, 
        bond_dim=params['bond_dim'],
        learning_rate=params['learning_rate'],
        l2_lambda=params['l2_lambda']
    )
    
    # --- 4. Train Model ---
    # --- FIX: Removed the TFKerasPruningCallback to resolve the ModuleNotFoundError ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        # The TFKerasPruningCallback was removed from here.
    ]

    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=0 
    )

    # --- 5. Evaluate and Return Score ---
    roc_auc, _ = classification_metrics(model, X_test, y_test)
    print(f"--- [{symbol}] Trial {trial.number} Result | ROC-AUC: {roc_auc:.4f} ---")
    
    return roc_auc


def run_hyperparameter_search():
    """
    Defines and runs an Optuna-based hyperparameter search across multiple stocks.
    """
    SYMBOLS_TO_TEST = ["AAPL", "MSFT", "NVDA"]
    N_TRIALS_PER_STOCK = 30 # Number of trials Optuna will run for each stock

    print("--- STARTING OPTUNA HYPERPARAMETER SEARCH ---")
    print(f"Testing {N_TRIALS_PER_STOCK} combinations for each of {SYMBOLS_TO_TEST}")

    raw_data_dict = fetch_data()
    
    all_best_trials = []
    for symbol in SYMBOLS_TO_TEST:
        print("\n\n" + "#"*80)
        print(f"# --- OPTIMIZING FOR SYMBOL: {symbol} ---")
        print("#"*80)

        # Create a study object and specify the direction is to maximize the score
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        
        # Start the optimization
        study.optimize(lambda trial: objective(trial, raw_data_dict, symbol), n_trials=N_TRIALS_PER_STOCK)

        # --- Store and print the best trial for the current symbol ---
        best_trial = study.best_trial
        print(f"\n--- Best Trial for {symbol} ---")
        print(f"  Value (ROC-AUC): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        best_trial_data = {**best_trial.params, 'symbol': symbol, 'roc_auc': best_trial.value}
        all_best_trials.append(best_trial_data)

    # --- Final Summary ---
    print("\n\n" + "="*80)
    print("--- OVERALL HYPERPARAMETER SEARCH COMPLETE ---")
    print("="*80)
    
    if all_best_trials:
        # Display a summary of the best parameters found for each stock
        summary_df = pd.DataFrame(all_best_trials).sort_values(by='roc_auc', ascending=False)
        print("--- Summary of Best Parameters Found ---")
        print(summary_df)
    else:
        print("No results were generated.")


if __name__ == "__main__":
    # To see less of Optuna's logging, you can uncomment the following lines:
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_hyperparameter_search()
