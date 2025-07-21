# src/evaluate.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # <-- Added import for bar chart plotting
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report, roc_auc_score
import tensorflow as tf

def print_evaluation_summary(history):
    """
    Extracts and prints the final epoch's metrics from the history object.
    """
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_acc = history.history['sparse_categorical_accuracy'][-1]
    final_val_acc = history.history['val_sparse_categorical_accuracy'][-1]
    
    print("\n--- Final Model Evaluation Summary ---")
    print(f"Final Training Loss:       {final_loss:.6f}")
    print(f"Final Validation Loss:     {final_val_loss:.6f}")
    print(f"Final Training Accuracy:   {final_acc:.2%}")
    print(f"Final Validation Accuracy: {final_val_acc:.2%}")
    print("---------------------------------------")

def plot_loss_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Sparse Categorical Crossentropy)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(model, X_test, y_test):
    """
    For classification:
    - Shows predicted vs actual labels
    - Displays confusion matrix
    - Plots ROC curve
    """
    logits = model.predict(X_test)
    probabilities = tf.nn.softmax(logits).numpy()
    predicted_classes = np.argmax(probabilities, axis=1)
    positive_class_probs = probabilities[:, 1]

    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(y_test, label="Actual", marker='o', linestyle='--', alpha=0.7)
    plt.plot(predicted_classes, label="Predicted", marker='x', linestyle=':', alpha=0.7)
    plt.title("Predicted vs Actual Direction (Test Set)")
    plt.xlabel("Sample Index")
    plt.ylabel("Direction (0 = Down, 1 = Up)")
    plt.legend()
    plt.grid(True)
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, positive_class_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

def classification_metrics(model, X_test, y_test):
    """
    Calculates and returns classification metrics.
    
    Returns:
        float: The ROC-AUC score.
        str: The classification report string.
    """
    logits = model.predict(X_test, verbose=0)
    probabilities = tf.nn.softmax(logits).numpy()
    y_pred = np.argmax(probabilities, axis=1)
    positive_class_probs = probabilities[:, 1]

    score = roc_auc_score(y_test, positive_class_probs)
    report_str = classification_report(y_test, y_pred, digits=4)
    
    return score, report_str

# --- NEW COMPARISON FUNCTIONS ---

def plot_comparison_roc(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray):
    """
    Plots the ROC curves for multiple models on a single graph for comparison.

    Args:
        models_dict (dict): A dictionary where keys are model names (str) and
                            values are the trained model objects.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The true test labels.
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        # Get positive class probabilities
        logits = model.predict(X_test, verbose=0)
        probabilities = tf.nn.softmax(logits).numpy()
        positive_class_probs = probabilities[:, 1]
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, positive_class_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot the curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Formatting
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_comparison_metrics(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray):
    """
    Generates a bar chart comparing key performance metrics (ROC-AUC, Precision,
    Recall, F1-Score) across multiple models.

    Args:
        models_dict (dict): A dictionary where keys are model names (str) and
                            values are the trained model objects.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The true test labels.
    """
    metrics_data = []

    for name, model in models_dict.items():
        logits = model.predict(X_test, verbose=0)
        probabilities = tf.nn.softmax(logits).numpy()
        y_pred = np.argmax(probabilities, axis=1)
        positive_class_probs = probabilities[:, 1]
        
        # Get classification report as a dictionary for easier access
        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, positive_class_probs)
        
        # We will compare weighted averages to account for any class imbalance
        metrics_data.append({
            'Model': name,
            'ROC-AUC': roc_auc,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score']
        })
        
    # Create a pandas DataFrame for easy plotting
    df_metrics = pd.DataFrame(metrics_data).set_index('Model')
    
    # Plotting
    ax = df_metrics.plot(kind='bar', figsize=(12, 7), rot=0, width=0.8)
    plt.title('Comparison of Model Performance Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('')
    plt.ylim([0.0, max(df_metrics.max().max() * 1.1, 1.0)]) # Adjust ylim dynamically
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metrics')
    
    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                    
    plt.tight_layout()
    plt.show()


# Add this new function to the end of src/evaluate.py

def plot_training_times(results: dict):
    """
    Generates a bar chart comparing the training time of multiple models.

    Args:
        results (dict): The results dictionary containing a 'training_time' key
                        for each model.
    """
    model_names = list(results.keys())
    training_times = [res.get('training_time', 0) for res in results.values()]

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=model_names, y=training_times, palette="viridis")
    
    plt.title('Training Time Comparison', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xlabel('Model', fontsize=12)

    # Add data labels on top of each bar
    for index, value in enumerate(training_times):
        plt.text(index, value, f"{value:.2f} s", ha="center", va="bottom", fontsize=10)

    plt.show()