# test/test_data_preprocess.py
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import preprocess_for_quantile_regression

@pytest.fixture
def dummy_data():
    """
    Creates a dummy data dictionary with enough data points to avoid errors
    from technical analysis indicators that require a certain lookback period.
    """
    def make_df(num_samples=200):
        # Ensure dates are far enough in the past to be realistic
        dates = pd.date_range(end='2023-01-01', periods=num_samples, freq='h')
        return pd.DataFrame({
            'open': np.random.rand(num_samples) * 100 + 50,
            'high': np.random.rand(num_samples) * 100 + 55,
            'low': np.random.rand(num_samples) * 100 + 45,
            'close': np.random.rand(num_samples) * 100 + 50,
            'volume': np.random.randint(1000, 10000, size=num_samples)
        }, index=dates)

    return {'TEST1': make_df(), 'TEST2': make_df()}

def test_output_shapes_and_types(dummy_data):
    """
    Tests the output shapes, data types, and number of returned items from the preprocessing function.
    """
    lookback = 15
    future_horizon = 1
    
    # --- Act ---
    # Call the function with the new signature (no 'symbol')
    # Unpack the 6 returned values
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = preprocess_for_quantile_regression(
        dummy_data, 
        lookback=lookback, 
        future_horizon=future_horizon
    )

    # --- Assert ---
    # 1. Check shapes
    assert X_train.ndim == 3, "X_train should be 3-dimensional (samples, lookback, features)"
    assert X_test.ndim == 3, "X_test should be 3-dimensional"
    assert y_train.ndim == 2, "y_train should be 2-dimensional (samples, 1)"
    assert y_test.ndim == 2, "y_test should be 2-dimensional"
    
    assert X_train.shape[1] == lookback, f"X_train lookback dimension should be {lookback}"
    assert X_train.shape[2] == 12, "There should be 12 features"
    assert X_test.shape[1] == lookback, f"X_test lookback dimension should be {lookback}"
    assert X_test.shape[2] == 12, "There should be 12 features"
    assert y_train.shape[1] == 1, "y_train should have 1 target column"
    assert y_test.shape[1] == 1, "y_test should have 1 target column"
    
    # 2. Check data types
    assert X_train.dtype == np.float64 or X_train.dtype == np.float32
    assert y_train.dtype == np.float64 or y_train.dtype == np.float32
    # The target is now continuous (returns), not binary classification
    assert not np.array_equal(np.unique(y_train), np.array([0, 1])), "Target should be continuous, not binary"

    # 3. Check scaler types
    assert isinstance(x_scaler, MinMaxScaler)
    assert isinstance(y_scaler, MinMaxScaler)

def test_no_nan_values(dummy_data):
    """
    Ensures that the final output arrays do not contain any NaN values,
    which would indicate an issue with feature engineering or data cleaning.
    """
    lookback = 20
    
    # --- Act ---
    X_train, y_train, X_test, y_test, _, _ = preprocess_for_quantile_regression(
        dummy_data, 
        lookback=lookback
    )

    # --- Assert ---
    assert not np.isnan(X_train).any(), "X_train should not contain NaN values"
    assert not np.isnan(y_train).any(), "y_train should not contain NaN values"
    assert not np.isnan(X_test).any(), "X_test should not contain NaN values"
    assert not np.isnan(y_test).any(), "y_test should not contain NaN values"

if __name__ == "__main__":
    pytest.main()

