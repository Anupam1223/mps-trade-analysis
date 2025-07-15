import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data_dict: dict, symbol: str, lookback: int = 10):
    """
    Preprocesses the data for a single stock for the MPS model.
    
    Args:
        data_dict: Dictionary of raw dataframes from the ingestion step.
        symbol: The stock symbol to process (e.g., 'AAPL').
        lookback: The number of past time steps to use as features.
        
    Returns:
        X_train, y_train, X_test, y_test: Processed data splits.
    """
    df = data_dict[symbol].copy()

    # --- 1. Feature Engineering (based on EDA) ---
    df['returns'] = df['close'].pct_change()
    df['log_volume'] = np.log1p(df['volume'])
    
    # Create lagged return features
    for i in range(1, lookback + 1):
        df[f'return_lag_{i}'] = df['returns'].shift(i)

    # --- 2. Define Target ---
    # We want to predict the next day's return
    df['target'] = df['returns'].shift(-1)
    
    # Drop rows with NaN values created by shifts
    df.dropna(inplace=True)

    # --- 3. Feature Selection & Scaling ---
    features = ['log_volume'] + [f'return_lag_{i}' for i in range(1, lookback + 1)]
    X = df[features]
    y = df['target']
    
    # Split data chronologically before scaling
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- 4. Reshape data into Tensors [samples, timesteps, features] ---
    # Our MPS model will treat each feature at a different time step as a different site
    # So we reshape to (samples, lookback, num_base_features)
    # Note: This is one way to structure it. You might structure differently.
    # Here, we treat the 'lookback' dimension as the MPS chain.
    X_train_tensor = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_tensor = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    print(f"Data for {symbol} preprocessed. Training shape: {X_train_tensor.shape}")

    return X_train_tensor, y_train.values, X_test_tensor, y_test.values