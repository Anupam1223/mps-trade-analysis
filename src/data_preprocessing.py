# src/data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import ta

# ... (create_sequences and preprocess_data functions are the same) ...
def create_sequences(X_data, y_data, lookback):
    Xs, ys = [], []
    for i in range(len(X_data) - lookback):
        Xs.append(X_data[i:(i + lookback)])
        ys.append(y_data[i + lookback])
    return np.array(Xs), np.array(ys)

def preprocess_data(data_dict: dict, symbol: str, lookback: int = 20):
    # This function is for single assets and can remain as is.
    df = data_dict[symbol].copy()
    # Feature Engineering
    df['returns'] = df['close'].pct_change().fillna(0) # Added fillna(0)
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['bollinger_h'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bollinger_l'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    # Target and Features
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    feature_columns = ['close', 'volume', 'returns', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'atr']
    X_df = df[feature_columns]
    y_series = df['target']
    # Splitting and Scaling
    train_size = int(len(X_df) * 0.8)
    X_train_df, X_test_df = X_df.iloc[:train_size], X_df.iloc[train_size:]
    y_train_series, y_test_series = y_series.iloc[:train_size], y_series.iloc[train_size:]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    # Sequence Creation
    X_train, y_train = create_sequences(X_train_scaled, y_train_series.values, lookback)
    X_test, y_test = create_sequences(X_test_scaled, y_test_series.values, lookback)
    print(f"Data for {symbol} preprocessed.")
    return X_train, y_train, X_test, y_test


def preprocess_data_universal(data_dict: dict, lookback: int = 20):
    all_X_train, all_y_train, all_X_test, all_y_test = [], [], [], []
    for symbol, df in data_dict.items():
        print(f"Processing {symbol}...")
        df = df.copy()

        # --- FIX FOR WARNING ---
        # Added .fillna(0) to handle the first NaN value and suppress the warning.
        df['returns'] = df['close'].pct_change().fillna(0)
        # --- END FIX ---
        
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
        df['macd'] = ta.trend.MACD(close=df['close']).macd()
        df['bollinger_h'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
        df['bollinger_l'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        # --- NEW: Time-Based Features ---
        df.index = pd.to_datetime(df.index) # Ensure index is datetime
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # --- NEW: Additional Technical Indicators ---
        df['stoch_osc'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
        df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()

        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        feature_columns = [
            'close', 'volume', 'returns', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'atr',
            'hour_of_day', 'day_of_week', 'stoch_osc', 'williams_r' # <-- Add new features
        ]
        X_df = df[feature_columns]
        y_series = df['target']
        
        train_size = int(len(X_df) * 0.8)
        X_train_df, X_test_df = X_df.iloc[:train_size], X_df.iloc[train_size:]
        y_train_series, y_test_series = y_series.iloc[:train_size], y_series.iloc[train_size:]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_df)
        X_test_scaled = scaler.transform(X_test_df)
        
        X_train_ind, y_train_ind = create_sequences(X_train_scaled, y_train_series.values, lookback)
        X_test_ind, y_test_ind = create_sequences(X_test_scaled, y_test_series.values, lookback)

        all_X_train.append(X_train_ind)
        all_y_train.append(y_train_ind)
        all_X_test.append(X_test_ind)
        all_y_test.append(y_test_ind)

    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_test = np.concatenate(all_X_test, axis=0)
    y_test = np.concatenate(all_y_test, axis=0)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    print("\nUniversal data preprocessing complete.")
    print(f"Total training samples: {X_train.shape[0]}")
    print(f"Total testing samples:  {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[2]}")

    return X_train, y_train, X_test, y_test