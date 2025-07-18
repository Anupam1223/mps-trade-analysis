import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    roc_auc_score
)

def evaluate_model(model, model_name: str, X_test, y_test):
    """
    Calculates and displays evaluation metrics for a single classical model.

    Args:
        model: The trained scikit-learn model object.
        model_name: The name of the model for titles and labels.
        X_test: The test feature data.
        y_test: The test target data.

    Returns:
        A dictionary with key performance metrics.
    """
    print(f"\n--- Evaluating: {model_name} ---")

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    # --- Print Classification Report ---
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # --- Plot Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.grid(False)
    plt.show()

    # --- Plot ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic: {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    # --- Return Metrics ---
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    return metrics