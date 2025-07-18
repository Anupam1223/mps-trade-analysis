import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta

def preprocess_data_classical(data_dict: dict, symbol: str, lookback: int = 12):
    """
    Preprocesses the raw stock data for classical machine learning models.
    
    This version prepares data in the 2D format (samples, features) required
    by scikit-learn.
    """
    df = data_dict[symbol].copy()

    # --- 1. Feature Engineering (Identical to the TN version) ---
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

    # Define the target variable
    df['target'] = (df['returns'].shift(-1) >= 0).astype(int)

    # Drop rows with NaN values
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
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data for {symbol} preprocessed. Training shape: {X_train_scaled.shape}")
    
    # Return 2D arrays, scaler, and test sets
    return X_train_scaled, y_train.values, X_test_scaled, y_test.values, scaler