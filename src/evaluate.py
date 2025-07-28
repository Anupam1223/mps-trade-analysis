import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report


# --- Helper Function ---
def pinball_loss(y_true, y_pred, quantile):
    """Calculates the pinball loss for quantile regression."""
    delta = y_true - y_pred
    return np.maximum(quantile * delta, (quantile - 1) * delta).mean()


# --- Plotting Functions ---

def plot_confusion_matrix(y_true, y_pred_median, model_name):
    """
    Derives directional predictions from regression output and plots a confusion matrix.
    """
    # Determine true direction (1 for Up, 0 for Down/Same)
    true_direction = (np.diff(y_true, prepend=y_true[0]) > 0).astype(int)
    
    # Determine predicted direction from the median prediction
    pred_direction = (np.diff(y_pred_median, prepend=y_pred_median[0]) > 0).astype(int)

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(true_direction, pred_direction)
    
    sns.heatmap(cm, annot=True, fmt='g', cmap='Greens', 
                xticklabels=['Predicted Down', 'Predicted Up'], 
                yticklabels=['Actual Down', 'Actual Up'])
    
    plt.title(f'{model_name} - Directional Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    print("\n--- Directional Classification Report ---")
    print(classification_report(true_direction, pred_direction, target_names=['Down', 'Up']))


def plot_directional_accuracy(y_true, y_pred_median, model_name):
    """
    Visualizes the model's ability to predict the direction of change (up or down).
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
        # --- ADDITION: Call the new confusion matrix plot ---
        plot_confusion_matrix(y_true_flat, predictions[0.5].flatten(), model_name)
        
    plot_confidence_gauge(predictions, point_index=-1, model_name=model_name)

    # --- Detailed Diagnostic Plots ---
    print("\nDisplaying detailed diagnostic plots for technical analysis...")
    if 0.5 in predictions:
        plot_pred_vs_true(y_true_flat, predictions[0.5].flatten(), model_name)
    plot_quantile_fan(y_true_flat, predictions, model_name)
    
    print(f"\n✅ Evaluation Complete for {model_name}.\n")
