import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta


def preprocess_data(data_dict: dict, symbol: str, lookback: int = 10):
    """
    Preprocesses the raw stock data for a given symbol.

    This involves:
    1.  Feature engineering using technical analysis indicators.
    2.  Creating lagged return features.
    3.  Defining a binary classification target.
    4.  Splitting data into training and testing sets.
    5.  Scaling features to the [0, 1] range.
    6.  Casting data to float32 to prevent type errors in TensorFlow.
    """
    df = data_dict[symbol].copy()

    # --- 1. Feature Engineering ---
    df['returns'] = df['close'].pct_change()
    df['log_volume'] = np.log1p(df['volume'])

    # Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['bollinger_h'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bollinger_l'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['stoch'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()

    # Add lagged returns as features
    for i in range(1, lookback + 1):
        df[f'return_lag_{i}'] = df['returns'].shift(i)

    # Define the target variable: 1 if next day's close is up/flat, 0 if down
    df['target'] = (df['returns'].shift(-1) >= 0).astype(int)

    # Drop rows with NaN values resulting from shifts and TA calculations
    df.dropna(inplace=True)

    # --- 2. Feature and Target Selection ---
    features = ['log_volume', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'atr', 'stoch'] + \
           [f'return_lag_{i}' for i in range(1, lookback + 1)]

    X = df[features]
    y = df['target']

    # --- 3. Data Splitting ---
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # --- 4. Scaling ---
    # Use MinMaxScaler to scale features to the [0, 1] range, suitable for the feature map
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- FIX: Explicitly cast data to float32 to match model weight types ---
    # This is the crucial fix for the TypeError.
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)

    # --- 5. Reshaping for the Model ---
    # Add a final dimension for the MPSLayer input shape (features, 1)
    X_train_tensor = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_tensor = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    print(f"Data for {symbol} preprocessed. Training shape: {X_train_tensor.shape}")
    return X_train_tensor, y_train.values, X_test_tensor, y_test.values
