# src/evaluate.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_training_history(history_dict: dict, model_name: str):
    """
    Plots the training and validation loss for each quantile model.
    """
    quantiles = sorted(history_dict.keys())
    num_quantiles = len(quantiles)
    plt.figure(figsize=(15, 5 * num_quantiles))
    
    for i, q in enumerate(quantiles):
        history = history_dict[q]
        plt.subplot(num_quantiles, 1, i + 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Training History for Quantile {q}', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Pinball Loss)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.show()

def calculate_pinball_loss(y_true, y_pred, quantile):
    """Calculates the pinball loss for a single quantile."""
    err = y_true - y_pred
    return np.mean(np.maximum(quantile * err, (quantile - 1) * err))

def comprehensive_evaluation(predictions: dict, y_test_unscaled: np.ndarray, history_dict: dict, model_name: str):
    """
    Performs a full evaluation, including plots, metrics, and a summary table.
    """
    print("\n" + "="*60)
    print(f"📈 COMPREHENSIVE EVALUATION FOR: {model_name} 📈")
    print("="*60)
    
    # 1. Plot Training History
    print("\n--- 1. Training and Validation Loss ---")
    plot_training_history(history_dict, model_name)
    
    # 2. Plot Quantile Predictions
    print("\n--- 2. Visualizing Prediction Intervals ---")
    quantiles = sorted(predictions.keys())
    lower_q_pred = predictions[quantiles[0]].flatten()
    upper_q_pred = predictions[quantiles[-1]].flatten()
    
    plt.figure(figsize=(15, 7))
    plt.fill_between(
        range(len(y_test_unscaled)),
        lower_q_pred,
        upper_q_pred,
        alpha=0.3,
        color='orange',
        label=f'Prediction Interval ({quantiles[0]*100:.0f}th - {quantiles[-1]*100:.0f}th percentile)'
    )
    if 0.5 in predictions:
        plt.plot(predictions[0.5], 'r-', label='Median Prediction (50th percentile)')
    plt.plot(y_test_unscaled, 'b-', label='Actual Returns', alpha=0.7)
    plt.title(f'Quantile Regression Predictions: {model_name}', fontsize=16)
    plt.xlabel('Time Step (Test Set)')
    plt.ylabel('Future Return')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 3. Calculate and Report Metrics
    print("\n--- 3. Performance Metrics ---")
    results = []
    y_true_flat = y_test_unscaled.flatten()
    
    # Calculate metrics for the median prediction (if available)
    if 0.5 in predictions:
        y_pred_median = predictions[0.5].flatten()
        mae = mean_absolute_error(y_true_flat, y_pred_median)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_median))
    else:
        mae = np.nan
        rmse = np.nan
        
    # Calculate coverage
    coverage = np.mean((y_true_flat >= lower_q_pred) & (y_true_flat <= upper_q_pred)) * 100
    
    # Calculate pinball loss for all quantiles
    avg_pinball_loss = np.mean([
        calculate_pinball_loss(y_true_flat, predictions[q].flatten(), q) for q in quantiles
    ])
    
    results.append({
        "Model": model_name,
        "Median MAE": mae,
        "Median RMSE": rmse,
        "Avg. Pinball Loss": avg_pinball_loss,
        f"Coverage ({quantiles[0]}-{quantiles[-1]})": f"{coverage:.2f}%"
    })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("\n" + "="*60 + "\n")