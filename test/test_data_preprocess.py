import tensorflow as tf
import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Assuming modules are in src/
from src.data_preprocessing import preprocess_data, create_sequences


def make_dummy_data(num_samples=100):
    """
    Create a dummy DataFrame mimicking Alpaca stock data structure for testing.
    """
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_samples, freq='H')
    df = pd.DataFrame({
        'open': np.random.rand(num_samples),
        'high': np.random.rand(num_samples),
        'low': np.random.rand(num_samples),
        'close': np.random.rand(num_samples),
        'volume': np.random.randint(1, 1000, size=num_samples)
    }, index=dates)
    return {'TEST': df}

@pytest.fixture
def dummy_data_dict():
    return make_dummy_data(200)

def test_feature_engineering_columns(dummy_data_dict):
    lookback = 10
    X_train, y_train, X_test, y_test = preprocess_data(dummy_data_dict, symbol='TEST', lookback=lookback)
    # After preprocessing, feature count should match create_sequences input
    num_features = X_train.shape[2]
    # We expect 9 features as defined in preprocess_data
    assert num_features == 9
    # Check that targets are binary
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})

def test_sequence_shapes_and_leakage(dummy_data_dict):
    lookback = 15
    X_train, y_train, X_test, y_test = preprocess_data(dummy_data_dict, symbol='TEST', lookback=lookback)
    # For training set
    train_len = int(len(make_dummy_data(200)['TEST'].dropna()) * 0.8) - lookback
    test_len = int(len(make_dummy_data(200)['TEST'].dropna()) * 0.2) - lookback
    assert X_train.shape[0] == train_len
    assert X_train.shape[1] == lookback
    # No leakage: windows should not mix train and test
    # Rough check: inputs in X_train should come from first 80% of data
    total = len(dummy_data_dict()['TEST'])
    cutoff_idx = int(total * 0.8)
    # Last window start index for train is cutoff_idx - lookback - 1
    assert not np.any(X_train.argmax(axis=(1,2)) >= cutoff_idx)

if __name__ == "__main__":
    pytest.main()
