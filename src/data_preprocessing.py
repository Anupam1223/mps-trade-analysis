import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta

def create_sequences(X_data, y_data, lookback):
    """
    Helper function to create overlapping sequences from time-series data.
    This is essential for models that learn from a sequence of past events.
    """
    Xs, ys = [], []
    # Iterate from the first possible start of a sequence to the last
    for i in range(len(X_data) - lookback):
        # The input sequence is the window of 'lookback' days of features
        Xs.append(X_data[i:(i + lookback)])
        # The target is the label for the day immediately following the sequence
        ys.append(y_data[i + lookback])
    return np.array(Xs), np.array(ys)

def preprocess_data(data_dict: dict, symbol: str, lookback: int = 20):
    """
    Preprocesses data by adding technical indicators and creating time-series sequences.
    This version correctly structures the data into (samples, lookback, features)
    for the MPS model.
    """
    df = data_dict[symbol].copy()

    # --- 1. Feature Engineering ---
    # Calculate a set of technical indicators to give the model more context.
    df['returns'] = df['close'].pct_change()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['bollinger_h'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bollinger_l'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['stoch'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()

    # --- 2. Define Target and Features ---
    # The target is the direction of the next day's price movement (binary classification).
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop any rows with NaN values that were created by the indicators/shifts.
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Define the list of features to be used in the model.
    # The MPS will learn the time dependency from the sequence structure,
    # so we don't need explicit lagged features here.
    feature_columns = ['close', 'volume', 'returns', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'atr', 'stoch']

    X_df = df[feature_columns]
    y_series = df['target']

    # --- 3. Data Splitting and Scaling ---
    train_size = int(len(X_df) * 0.8)
    X_train_df, X_test_df = X_df.iloc[:train_size], X_df.iloc[train_size:]
    y_train_series, y_test_series = y_series.iloc[:train_size], y_series.iloc[train_size:]

    # Scale features to the [0, 1] range. This is important for model stability.
    # Fit the scaler ONLY on the training data to prevent information leakage from the test set.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # --- 4. Sequence Creation ---
    # This is the crucial step: transform the flat 2D feature data into 3D sequences (windows).
    # The shape will be (samples, lookback_period, num_features).
    X_train, y_train = create_sequences(X_train_scaled, y_train_series.values, lookback)
    X_test, y_test = create_sequences(X_test_scaled, y_test_series.values, lookback)

    print(f"Data for {symbol} preprocessed.")
    print(f"Number of features: {X_train.shape[2]}")
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test
