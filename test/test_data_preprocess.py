import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import preprocess_data

def make_dummy_data(num_samples=200):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_samples, freq='h')
    df = pd.DataFrame({
        'open': np.random.rand(num_samples),
        'high': np.random.rand(num_samples),
        'low': np.random.rand(num_samples),
        'close': np.random.rand(num_samples),
        'volume': np.random.randint(1, 1000, size=num_samples)
    }, index=dates)
    return {'TEST': df}

@pytest.fixture
def dummy_data():
    return make_dummy_data()

def test_feature_engineering_columns(dummy_data):
    lookback = 10
    X_train, y_train, X_test, y_test = preprocess_data(dummy_data, symbol='TEST', lookback=lookback)
    assert X_train.shape[2] == 9  # nine engineered features
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})

def test_sequence_shapes_and_no_leakage(dummy_data):
    lookback = 15
    X_train, y_train, X_test, y_test = preprocess_data(dummy_data, symbol='TEST', lookback=lookback)

    # 1) counts match
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # 2) window size is correct
    assert X_train.shape[1] == lookback
    assert X_test.shape[1] == lookback

    # 3) no overlap between train/test indices
    # Reconstruct the dropped-nan DataFrame
    df = dummy_data['TEST'].copy()
    df['returns'] = df['close'].pct_change()
    df['rsi']     = df['close'].pct_change()  # simple proxy for shape
    df['target']  = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna().reset_index(drop=True)

    total = len(df)
    split = int(total * 0.8)

    max_train_idx = lookback + X_train.shape[0] - 1
    min_test_idx  = lookback + split

    assert max_train_idx < min_test_idx, "Train sequences overlap into test region"

if __name__ == "__main__":
    pytest.main()