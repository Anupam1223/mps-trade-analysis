# tests/conftest.py
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_y_true():
    """Fixture for a sample ground truth array."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

@pytest.fixture
def sample_predictions():
    """Fixture for a sample predictions dictionary with multiple quantiles."""
    return {
        0.1: np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]),
        0.5: np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), # Median prediction
        0.9: np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    }
    
@pytest.fixture
def sample_predictions_no_median():
    """Fixture for sample predictions without a median (0.5) quantile."""
    return {
        0.2: np.array([0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8]),
        0.8: np.array([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2])
    }

@pytest.fixture
def sample_history_dict():
    """Fixture for a sample Keras history dictionary."""
    # A simplified mock of a Keras History object
    class MockHistory:
        def __init__(self, loss, val_loss):
            self.history = {'loss': loss, 'val_loss': val_loss}

    return {
        0.1: MockHistory(loss=[0.5, 0.4, 0.3], val_loss=[0.55, 0.45, 0.35]),
        0.5: MockHistory(loss=[0.2, 0.15, 0.1], val_loss=[0.25, 0.18, 0.12]),
        0.9: MockHistory(loss=[0.5, 0.4, 0.3], val_loss=[0.55, 0.45, 0.35]),
    }
