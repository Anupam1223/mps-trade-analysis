import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report, roc_auc_score
import tensorflow as tf

# --- FIX: Import the contraction function to use it in manual prediction loops ---
from src.dmrg_optimizer import contract_network_for_batch

def print_evaluation_summary(history):
    """
    Extracts and prints the final epoch's metrics from the history dictionary.
    """
    final_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_acc = history['sparse_categorical_accuracy'][-1]
    final_val_acc = history['val_sparse_categorical_accuracy'][-1]
    
    print("\n--- Final Model Evaluation Summary ---")
    print(f"Final Training Loss:       {final_loss:.6f}")
    print(f"Final Validation Loss:     {final_val_loss:.6f}")
    print(f"Final Training Accuracy:   {final_acc:.2%}")
    print(f"Final Validation Accuracy: {final_val_acc:.2%}")
    print("---------------------------------------")

def plot_loss_history(history):
    """
    Plots the training and validation loss from the history dictionary.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Sparse Categorical Crossentropy)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_history(history):
    """
    Plots the training and validation accuracy from the history dictionary.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def _manual_predict(model, X_data, batch_size=64):
    """
    Helper function to perform prediction manually, avoiding model.predict().
    """
    mps_layer = model.get_layer('mps_layer')
    all_logits = []
    
    dataset = tf.data.Dataset.from_tensor_slices(X_data).batch(batch_size)
    
    for X_batch in dataset:
        feature_vectors = mps_layer._feature_map(tf.squeeze(X_batch, axis=-1))
        logits = contract_network_for_batch(feature_vectors, mps_layer.mps_tensors, mps_layer.label_site)
        all_logits.append(logits)
        
    return tf.concat(all_logits, axis=0)

def plot_predictions(model, X_test, y_test):
    """
    For classification:
    - Shows predicted vs actual labels
    - Displays confusion matrix
    - Plots ROC curve
    """
    # --- FIX: Replace model.predict() with the manual prediction loop ---
    logits = _manual_predict(model, X_test)
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
    # --- FIX: Replace model.predict() with the manual prediction loop ---
    logits = _manual_predict(model, X_test)
    probabilities = tf.nn.softmax(logits).numpy()
    y_pred = np.argmax(probabilities, axis=1)
    positive_class_probs = probabilities[:, 1]

    score = roc_auc_score(y_test, positive_class_probs)
    report_str = classification_report(y_test, y_pred, digits=4)
    
    return score, report_str
