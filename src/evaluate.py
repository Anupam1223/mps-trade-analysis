import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def print_evaluation_summary(history):
    """
    Extracts and prints the final epoch's metrics from the history object.
    """
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
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
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
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
    predictions = model.predict(X_test).flatten()
    predicted_classes = (predictions > 0.5).astype(int)

    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    # --- 2. Prediction Comparison Plot ---
    plt.figure(figsize=(14, 4))
    plt.plot(y_test, label="Actual", marker='o', linestyle='--', alpha=0.7)
    plt.plot(predicted_classes, label="Predicted", marker='x', linestyle=':', alpha=0.7)
    plt.title("Predicted vs Actual Direction (Test Set)")
    plt.xlabel("Sample Index")
    plt.ylabel("Direction (0 = Down, 1 = Up)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 3. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, predictions)
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
