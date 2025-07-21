import tensorflow as tf
import numpy as np
import pandas as pd
import pytest
import io
import sys
from sklearn.metrics import roc_auc_score, classification_report

# Assuming modules are in src/
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluate import (
    print_evaluation_summary,
    plot_loss_history,
    plot_accuracy_history,
    plot_predictions,
    classification_metrics
)

# ------------------------
# Evaluation Tests
# ------------------------

def test_print_evaluation_summary(capsys):
    history = type('H', (), {'history': {
        'loss': [0.5, 0.4],
        'val_loss': [0.6, 0.45],
        'sparse_categorical_accuracy': [0.7, 0.75],
        'val_sparse_categorical_accuracy': [0.65, 0.7]
    }})()
    print_evaluation_summary(history)
    captured = capsys.readouterr().out
    assert 'Final Training Loss:       0.400000' in captured
    assert 'Final Validation loss:     0.450000' or 'Final Validation Loss:     0.450000' in captured
    assert '75.00%' in captured
    assert '70.00%' in captured

@pytest.mark.parametrize("y_true, probs, expected_auc", [
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 1.0),
    ([0, 1, 1, 0], [0.4, 0.6, 0.4, 0.6], 0.5)
])
def test_classification_metrics(y_true, probs, expected_auc):
    class DummyModel:
        def predict(self, X, verbose=0):
            logits = np.stack([1 - np.array(probs), np.array(probs)], axis=1)
            return logits
    X_test = np.zeros((len(y_true), 2))
    auc_score, report_str = classification_metrics(DummyModel(), X_test, np.array(y_true))
    assert np.isclose(auc_score, expected_auc)
    for cls in [0, 1]:
        assert f' {cls} ' in report_str
        assert 'precision' in report_str.lower()
        assert 'recall' in report_str.lower()

# Plot functions smoke tests

def test_plot_loss_history(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    history = type('H', (), {'history': {'loss': [0.1,0.2], 'val_loss': [0.15,0.25]}})()
    plot_loss_history(history)

def test_plot_accuracy_history(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    history = type('H', (), {'history': {'sparse_categorical_accuracy': [0.8,0.85], 'val_sparse_categorical_accuracy': [0.75,0.8]}})()
    plot_accuracy_history(history)

def test_plot_predictions(monkeypatch):
    class DummyModel:
        def predict(self, X):
            probs = [0.3,0.7]
            return np.stack([1-np.array(probs), np.array(probs)], axis=1)
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    monkeypatch.setattr(ConfusionMatrixDisplay, 'plot', lambda *args, **kwargs: None)
    X_test = np.zeros((2,2))
    plot_predictions(DummyModel(), X_test, np.array([0,1]))

if __name__ == "__main__":
    pytest.main()
