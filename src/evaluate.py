import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --- Helper Function ---
def pinball_loss(y_true, y_pred, quantile):
    """Calculates the pinball loss for quantile regression."""
    delta = y_true - y_pred
    return np.maximum(quantile * delta, (quantile - 1) * delta).mean()


# --- New Intuitive Plots for Coordinator ---

def plot_directional_accuracy(y_true, y_pred_median, model_name):
    """
    Visualizes the model's ability to predict the direction of change (up or down).
    - Green background: Correctly predicted the direction.
    - Red background: Predicted the wrong direction.
    """
    true_diff = np.diff(y_true, prepend=y_true[0])
    pred_diff = np.diff(y_pred_median, prepend=y_pred_median[0])
    correct_direction = (np.sign(true_diff) == np.sign(pred_diff))
    accuracy = np.mean(correct_direction) * 100
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Actual Value', color='black', zorder=5)
    
    for i in range(len(correct_direction)):
        color = 'green' if correct_direction[i] else 'red'
        plt.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.2, zorder=1)
        
    plt.title(f'{model_name}: Directional Accuracy ({accuracy:.2f}%)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    
    green_patch = patches.Patch(color='green', alpha=0.2, label='Correct Direction Predicted')
    red_patch = patches.Patch(color='red', alpha=0.2, label='Incorrect Direction Predicted')
    plt.legend(handles=[green_patch, red_patch, plt.Line2D([0], [0], color='black', label='Actual Value')])
    plt.tight_layout()
    plt.show()


def plot_confidence_gauge(quantile_preds, point_index, model_name):
    """
    Creates a "confidence gauge" for a single point in time.
    Visualizes the 50% and 90% prediction intervals like a speedometer.
    """
    required_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    if not all(q in quantile_preds for q in required_quantiles):
        print(f"Skipping Confidence Gauge: Missing one of the required quantiles {required_quantiles}")
        return

    p10 = quantile_preds[0.1][point_index]
    p25 = quantile_preds[0.25][point_index]
    p50 = quantile_preds[0.5][point_index]
    p75 = quantile_preds[0.75][point_index]
    p90 = quantile_preds[0.9][point_index]

    fig, ax = plt.subplots(figsize=(10, 6))
    center, radius = (0, 0), 1.0
    
    wedge_90 = patches.Wedge(center, radius, 0, 180, width=0.3, color='skyblue', label=f'90% Range [{p10:.2f}, {p90:.2f}]')
    ax.add_patch(wedge_90)
    wedge_50 = patches.Wedge(center, radius - 0.3, 0, 180, width=0.3, color='royalblue', label=f'50% Range [{p25:.2f}, {p75:.2f}]')
    ax.add_patch(wedge_50)

    gauge_min, gauge_max = p10, p90
    angle = 180 * (1 - (p50 - gauge_min) / (gauge_max - gauge_min)) if gauge_max > gauge_min else 90
    
    ax.arrow(0, 0, radius * np.cos(np.radians(angle)), radius * np.sin(np.radians(angle)),
             width=0.02, head_width=0.05, head_length=0.1, fc='black', ec='black', zorder=10)

    plt.text(0, -0.1, f'Best Guess: {p50:.2f}', ha='center', va='center', fontsize=16, weight='bold')
    plt.text(-radius, 0, f'{p10:.2f}\n(Low)', ha='center', va='center', fontsize=12)
    plt.text(radius, 0, f'{p90:.2f}\n(High)', ha='center', va='center', fontsize=12)

    ax.set_xlim(-radius * 1.2, radius * 1.2)
    ax.set_ylim(-0.2, radius * 1.2)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.title(f'{model_name}: Confidence Gauge for Timestep {point_index}', fontsize=16)
    plt.legend(handles=[wedge_90, wedge_50], loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    plt.show()


# --- Original Technical Plots ---

def plot_pred_vs_true(y_true, y_pred_median, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='True', color='black')
    plt.plot(y_pred_median, label='Predicted (0.5 quantile)', color='blue')
    plt.title(f"{model_name} - True vs Predicted (Median)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_quantile_fan(y_true, quantile_preds, model_name):
    plt.figure(figsize=(12, 6))
    sorted_quantiles = sorted(quantile_preds.keys())
    mid_q = 0.5 if 0.5 in sorted_quantiles else sorted_quantiles[len(sorted_quantiles) // 2]
    plt.plot(y_true, label='True', color='black', zorder=10)
    plt.plot(quantile_preds[mid_q], label=f'Predicted (q={mid_q})', color='blue', zorder=9)
    for i in range(len(sorted_quantiles) // 2):
        q_low = sorted_quantiles[i]
        q_high = sorted_quantiles[-(i + 1)]
        plt.fill_between(
            np.arange(len(y_true)), quantile_preds[q_low].flatten(),
            quantile_preds[q_high].flatten(), color='blue', alpha=0.15,
            label=f'Interval [{q_low}, {q_high}]' if i == 0 else None
        )
    plt.title(f"{model_name} - Quantile Fan Chart")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_intervals(y_true, quantile_preds, model_name):
    if 0.1 not in quantile_preds or 0.9 not in quantile_preds: return
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True', color='black')
    if 0.5 in quantile_preds: plt.plot(quantile_preds[0.5], label='Median (0.5)', color='blue')
    plt.fill_between(
        np.arange(len(y_true)), quantile_preds[0.1].flatten(),
        quantile_preds[0.9].flatten(), color='blue', alpha=0.2,
        label='Prediction Interval (0.1 - 0.9)'
    )
    plt.title(f"{model_name} - Prediction Interval (80%)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_quantile_residuals(y_true, quantile_preds, model_name):
    plt.figure(figsize=(10, 5))
    residuals_data = [{'Quantile': q, 'Residual': r} for q, pred in quantile_preds.items() for r in (y_true.flatten() - pred.flatten())]
    sns.boxplot(x='Quantile', y='Residual', data=pd.DataFrame(residuals_data), color='skyblue')
    plt.title(f"{model_name} - Quantile Residual Errors")
    plt.xlabel("Quantile")
    plt.ylabel("Residuals (True - Predicted)")
    plt.axhline(0, ls='--', color='red')
    plt.tight_layout()
    plt.show()


def plot_interval_width_over_time(quantile_preds, model_name):
    if 0.1 not in quantile_preds or 0.9 not in quantile_preds: return
    width = quantile_preds[0.9].flatten() - quantile_preds[0.1].flatten()
    plt.figure(figsize=(10, 4))
    plt.plot(width, label='Interval Width', color='orange')
    plt.title(f"{model_name} - Interval Width Over Time (0.1 to 0.9)")
    plt.xlabel("Time")
    plt.ylabel("Width")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pinball_loss_curve(y_true, quantile_preds, model_name):
    quantiles = sorted(quantile_preds.keys())
    losses = [pinball_loss(y_true, quantile_preds[q], q) for q in quantiles]
    plt.figure(figsize=(8, 4))
    plt.plot(quantiles, losses, marker='o', color='red')
    plt.title(f"{model_name} - Pinball Loss vs. Quantile")
    plt.xlabel("Quantile")
    plt.ylabel("Pinball Loss")
    plt.tight_layout()
    plt.show()


# --- Main Evaluation Function ---

def comprehensive_evaluation(y_true, predictions, histories, model_name, training_time):
    print(f"\n🔍 --- Comprehensive Evaluation for {model_name} ---")
    
    y_true_flat = y_true.flatten()
    quantiles = sorted(predictions.keys())
    metrics = [{'Quantile': q, 'MAE': mean_absolute_error(y_true_flat, predictions[q].flatten()),
                'RMSE': np.sqrt(mean_squared_error(y_true_flat, predictions[q].flatten())),
                'Pinball Loss': pinball_loss(y_true_flat, predictions[q].flatten(), q)} for q in quantiles]
    
    metrics_df = pd.DataFrame(metrics)
    print("--- Performance Metrics ---")
    print(metrics_df.round(4).to_string(index=False))

    if 0.1 in predictions and 0.9 in predictions:
        lower, upper = predictions[0.1].flatten(), predictions[0.9].flatten()
        coverage = np.mean((y_true_flat >= lower) & (y_true_flat <= upper)) * 100
        interval_width = np.mean(upper - lower)
        print(f"\n80% Interval Coverage: {coverage:.2f}%")
        print(f"Avg. Interval Width: {interval_width:.4f}")

    print(f"Total Training Time: {training_time:.2f} seconds")
    print("\n--- Generating Plots ---")

    # --- Plots for Coordinator (Intuitive First) ---
    print("Displaying plots for non-technical audience...")
    if 0.5 in predictions:
        plot_directional_accuracy(y_true_flat, predictions[0.5].flatten(), model_name)
    plot_confidence_gauge(predictions, point_index=-1, model_name=model_name)

    # --- Detailed Diagnostic Plots ---
    print("\nDisplaying detailed diagnostic plots for technical analysis...")
    if 0.5 in predictions:
        plot_pred_vs_true(y_true_flat, predictions[0.5].flatten(), model_name)
    plot_quantile_fan(y_true_flat, predictions, model_name)
    plot_prediction_intervals(y_true_flat, predictions, model_name)
    plot_quantile_residuals(y_true_flat, predictions, model_name)
    plot_interval_width_over_time(predictions, model_name)
    plot_pinball_loss_curve(y_true_flat, predictions, model_name)
    
    print(f"\n✅ Evaluation Complete for {model_name}.\n")

