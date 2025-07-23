# src/data_preprocessing.py

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils import shuffle


def create_sequences(X_data, y_data, lookback):
    """Helper function to create input sequences and corresponding labels."""
    Xs, ys = [], []
    for i in range(len(X_data) - lookback):
        Xs.append(X_data[i : (i + lookback)])
        ys.append(y_data[i + lookback])
    return np.array(Xs), np.array(ys)


def preprocess_for_quantile_regression(
    data_dict: dict, lookback: int = 20, future_horizon: int = 1
):
    """
    Prepares data for quantile regression from a dictionary of pandas DataFrames.
    The target is the future return over 'future_horizon' periods.

    This function performs the following steps:
    1. Iterates through all dataframes to calculate features and gather all training data.
    2. Fits scalers (MinMaxScaler) on the combined training data to prevent data leakage.
    3. Iterates through all dataframes again to apply the fitted scalers.
    4. Creates sequences for training and testing.
    5. Concatenates and shuffles the final training set.
    6. Concatenates the final test set (without shuffling to preserve time series order).

    Returns:
        - X_train, y_train, X_test, y_test (numpy arrays)
        - x_scaler, y_scaler (fitted sklearn scalers)
    """
    all_X_train, all_y_train, all_X_test, all_y_test = [], [], [], []

    # We fit scalers on the full training data from all symbols combined
    combined_X_train_df = pd.DataFrame()
    combined_y_train_series = pd.Series(dtype=np.float64)

    # --- First pass: Gather all training data to fit scalers ---
    for symbol, df in data_dict.items():
        df = df.copy()
        # Feature Engineering
        df["returns"] = df["close"].pct_change().fillna(0)
        df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
        df["macd"] = ta.trend.MACD(close=df["close"]).macd()
        df["bollinger_h"] = ta.volatility.BollingerBands(
            close=df["close"]
        ).bollinger_hband()
        df["bollinger_l"] = ta.volatility.BollingerBands(
            close=df["close"]
        ).bollinger_lband()
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"]
        ).average_true_range()
        df.index = pd.to_datetime(df.index)
        df["hour_of_day"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["stoch_osc"] = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"]
        ).stoch()
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            high=df["high"], low=df["low"], close=df["close"]
        ).williams_r()

        # Define target as future return
        df["target"] = df["returns"].shift(-future_horizon)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        feature_columns = [
            "close",
            "volume",
            "returns",
            "rsi",
            "macd",
            "bollinger_h",
            "bollinger_l",
            "atr",
            "hour_of_day",
            "day_of_week",
            "stoch_osc",
            "williams_r",
        ]

        train_size = int(len(df) * 0.8)
        X_train_df = df.loc[: train_size - 1, feature_columns]
        y_train_series = df.loc[: train_size - 1, "target"]

        combined_X_train_df = pd.concat(
            [combined_X_train_df, X_train_df], ignore_index=True
        )
        combined_y_train_series = pd.concat(
            [combined_y_train_series, y_train_series], ignore_index=True
        )

    # --- Fit scalers on the combined training data ---
    x_scaler = RobustScaler().fit(combined_X_train_df)
    # Reshape y for the scaler, which expects 2D input
    y_scaler = RobustScaler().fit(combined_y_train_series.values.reshape(-1, 1))

    # --- Second pass: Scale and create sequences for each symbol ---
    for symbol, df in data_dict.items():
        print(f"Processing {symbol} for quantile regression...")
        # Re-generate the full DataFrame with features and target
        df = df.copy()
        df["returns"] = df["close"].pct_change().fillna(0)
        df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
        df["macd"] = ta.trend.MACD(close=df["close"]).macd()
        df["bollinger_h"] = ta.volatility.BollingerBands(
            close=df["close"]
        ).bollinger_hband()
        df["bollinger_l"] = ta.volatility.BollingerBands(
            close=df["close"]
        ).bollinger_lband()
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"]
        ).average_true_range()
        df.index = pd.to_datetime(df.index)
        df["hour_of_day"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["stoch_osc"] = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"]
        ).stoch()
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            high=df["high"], low=df["low"], close=df["close"]
        ).williams_r()
        df["target"] = df["returns"].shift(-future_horizon)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        feature_columns = [
            "close",
            "volume",
            "returns",
            "rsi",
            "macd",
            "bollinger_h",
            "bollinger_l",
            "atr",
            "hour_of_day",
            "day_of_week",
            "stoch_osc",
            "williams_r",
        ]

        # Splitting
        train_size = int(len(df) * 0.8)
        X_train_df, X_test_df = (
            df.loc[: train_size - 1, feature_columns],
            df.loc[train_size:, feature_columns],
        )
        y_train_series, y_test_series = (
            df.loc[: train_size - 1, "target"],
            df.loc[train_size:, "target"],
        )

        # Scaling using the fitted scalers
        X_train_scaled = x_scaler.transform(X_train_df)
        X_test_scaled = x_scaler.transform(X_test_df)
        y_train_scaled = y_scaler.transform(y_train_series.values.reshape(-1, 1))
        y_test_scaled = y_scaler.transform(y_test_series.values.reshape(-1, 1))

        # Sequence Creation
        X_train_ind, y_train_ind = create_sequences(
            X_train_scaled, y_train_scaled, lookback
        )
        X_test_ind, y_test_ind = create_sequences(
            X_test_scaled, y_test_scaled, lookback
        )

        all_X_train.append(X_train_ind)
        all_y_train.append(y_train_ind)
        all_X_test.append(X_test_ind)
        all_y_test.append(y_test_ind)

    # Concatenate and shuffle the training data
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Concatenate the test data but DO NOT shuffle it to preserve time order for plotting
    X_test = np.concatenate(all_X_test, axis=0)
    y_test = np.concatenate(all_y_test, axis=0)

    print("\nUniversal data preprocessing for quantile regression complete.")
    return X_train, y_train, X_test, y_test, x_scaler, y_scaler
