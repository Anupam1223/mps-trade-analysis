import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta


def preprocess_data(data_dict: dict, symbol: str, lookback: int = 10):
    df = data_dict[symbol].copy()

    # --- 1. Feature Engineering (based on EDA + TA indicators) ---
    df['returns'] = df['close'].pct_change()
    df['log_volume'] = np.log1p(df['volume'])

    # Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['bollinger_h'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bollinger_l'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['stoch'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()


    for i in range(1, lookback + 1):
        df[f'return_lag_{i}'] = df['returns'].shift(i)

    # Refined directional target: 1 if next day's return >= 0 else 0
    # This makes it a binary classification problem
    df['target'] = (df['returns'].shift(-1) >= 0).astype(int)


    df.dropna(inplace=True)

    features = ['log_volume', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'atr', 'stoch'] + \
           [f'return_lag_{i}' for i in range(1, lookback + 1)]

    X = df[features]
    y = df['target']

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_tensor = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    print(f"Data for {symbol} preprocessed. Training shape: {X_train_tensor.shape}")
    return X_train_tensor, y_train.values, X_test_tensor, y_test.values
