# src/evaluate.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_prediction_intervals(y_true, y_preds, model_name, n=100):
    """Visualize prediction intervals and median predictions"""
    idx = np.arange(min(n, len(y_true)))
    plt.figure(figsize=(15, 6))
    plt.plot(idx, y_true[:n], "k.", markersize=5, label="True Values")

    # Conditionally plot the median prediction only if it exists
    if 0.5 in y_preds:
        plt.plot(idx, y_preds[0.5][:n], "b-", label="Predicted Median (q=0.5)")

    # Ensure keys exist before accessing
    lower_q = min(y_preds.keys())
    upper_q = max(y_preds.keys())
    plt.fill_between(
        idx,
        y_preds[lower_q][:n],
        y_preds[upper_q][:n],
        color="skyblue",
        alpha=0.4,
        label=f"{lower_q*100:.0f}-{upper_q*100:.0f}% Interval",
    )
    plt.title(f"{model_name} - Prediction Intervals vs. True Values (First 100 Points)")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def coverage_analysis(y_true, y_preds, model_name):
    """Plot empirical vs nominal coverage"""
    plt.figure(figsize=(7, 5))
    nominal_quantiles = sorted(y_preds.keys())
    empirical_quantiles = [np.mean(y_true < y_preds[q]) for q in nominal_quantiles]

    plt.plot(nominal_quantiles, empirical_quantiles, "o-", label="Model Calibration")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal Calibration")
    plt.xlabel("Nominal Quantile (Predicted)")
    plt.ylabel("Empirical Quantile (Actual)")
    plt.title(f"{model_name} - Quantile Coverage Calibration Plot")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_history(history_dict, model_name):
    """Plots the validation loss from the training history."""
    quantiles = sorted(history_dict.keys())
    plt.figure(figsize=(12, 4))
    for q in quantiles:
        history = history_dict[q]
        plt.plot(history.history["val_loss"], label=f"Q={q} Val Loss")
    plt.title(f"{model_name} - Validation Loss Across Quantiles")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def calculate_pinball_loss(y_true, y_pred, quantile):
    """Calculates the pinball loss for a given quantile."""
    err = y_true - y_pred
    return np.mean(np.maximum(quantile * err, (quantile - 1) * err))


def scatter_pred_vs_true(y_true, y_pred, quantile, model_name):
    """Creates a scatter plot of predicted vs. actual values."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.title(f"{model_name} - Predicted vs Actual (Quantile={quantile})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_prediction_interval_width(preds_dict, model_name):
    """Plots the width of the prediction interval over time."""
    lower = preds_dict[min(preds_dict)].flatten()
    upper = preds_dict[max(preds_dict)].flatten()
    width = upper - lower
    plt.figure(figsize=(12, 4))
    plt.plot(width, label="Interval Width")
    plt.title(f"{model_name} - Prediction Interval Width Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Width")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_error_distribution(y_true, y_pred, model_name):
    """Plots the distribution of prediction errors."""
    error = y_pred - y_true
    plt.figure(figsize=(8, 4))
    plt.hist(error, bins=50, alpha=0.7, color="purple")
    plt.title(f"{model_name} - Prediction Error Distribution (50th percentile)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_coverage_bar(y_true, preds_dict, model_name):
    """Plots a bar chart showing the prediction interval coverage."""
    quantiles = sorted(preds_dict)
    lower = preds_dict[quantiles[0]].flatten()
    upper = preds_dict[quantiles[-1]].flatten()
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    plt.figure(figsize=(6, 4))
    plt.bar(["Coverage"], [coverage], color="teal")
    plt.ylabel("Coverage (%)")
    plt.title(f"{model_name} - Prediction Interval Coverage")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- THIS IS THE ONLY FUNCTION YOU NEED TO CALL ---
def comprehensive_evaluation(
    predictions: dict,
    y_test_unscaled: np.ndarray,
    history_dict: dict,
    model_name: str,
    training_time: float = None,
):
    """
    Runs a comprehensive evaluation of a quantile regression model,
    generating plots and a summary metrics table.
    """
    print(f"\n{'='*80}\n📊 COMPREHENSIVE EVALUATION FOR MODEL: {model_name}\n{'='*80}")

    y_true_flat = y_test_unscaled.flatten()
    quantiles = sorted(predictions.keys())
    median_pred = predictions.get(0.5, None)

    # --- Section 1: Standard Performance Plots ---
    print("\n📌 1. Validation Loss Curve:")
    plot_training_history(history_dict, model_name)

    if median_pred is not None:
        print("\n📌 2. Predicted vs Actual (for Median):")
        scatter_pred_vs_true(y_true_flat, median_pred.flatten(), 0.5, model_name)

    print("\n📌 3. Prediction Interval Width Over Time:")
    plot_prediction_interval_width(predictions, model_name)

    # --- Section 2: Deeper Quantile Analysis ---
    print("\n📌 4. Prediction Intervals vs. True Values:")
    plot_prediction_intervals(
        y_true_flat, {q: p.flatten() for q, p in predictions.items()}, model_name
    )

    print("\n📌 5. Quantile Coverage Calibration:")
    coverage_analysis(
        y_true_flat, {q: p.flatten() for q, p in predictions.items()}, model_name
    )

    # --- Section 3: Final Metrics Table ---
    print("\n📌 6. Performance Metrics Table:")

    # Calculate metrics
    mae = (
        mean_absolute_error(y_true_flat, median_pred.flatten())
        if median_pred is not None
        else np.nan
    )
    rmse = (
        np.sqrt(mean_squared_error(y_true_flat, median_pred.flatten()))
        if median_pred is not None
        else np.nan
    )

    pinball_losses = {
        f"Pinball Loss (q={q})": calculate_pinball_loss(
            y_true_flat, predictions[q].flatten(), q
        )
        for q in quantiles
    }
    avg_pinball = np.mean(list(pinball_losses.values()))

    lower_q = min(quantiles)
    upper_q = max(quantiles)
    coverage = (
        np.mean(
            (y_true_flat >= predictions[lower_q].flatten())
            & (y_true_flat <= predictions[upper_q].flatten())
        )
        * 100
    )
    avg_width = np.mean(predictions[upper_q].flatten() - predictions[lower_q].flatten())

    # Create DataFrame
    results_data = {
        "Model": model_name,
        "Total Training Time (s)": (
            f"{training_time:.2f}" if training_time is not None else "N/A"
        ),
        "MAE (q=0.5)": mae,
        "RMSE (q=0.5)": rmse,
        "Avg. Pinball Loss": avg_pinball,
        f"Coverage ({lower_q*100:.0f}-{upper_q*100:.0f}%)": f"{coverage:.2f}%",
        f"Avg. Interval Width": f"{avg_width:.4f}",
    }
    # Add individual pinball losses to the results
    results_data.update(pinball_losses)

    results_df = pd.DataFrame([results_data])

    display(
        results_df.set_index("Model").style.background_gradient(cmap="YlGnBu", axis=1)
    )
    print("\n" + "=" * 80 + "\n")
