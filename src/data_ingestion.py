# src/data_ingestion.py

import os
from datetime import datetime, timedelta
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf

def fetch_data(
    symbols: list,
    timeframe: TimeFrame = TimeFrame.Hour,
    days_back: int = 365
) -> dict[str, pd.DataFrame]:
    # This function for Alpaca data remains the same
    api_key = os.getenv('APCA_API_KEY_ID')
    secret_key = os.getenv('APCA_API_SECRET_KEY')
    if not api_key or not secret_key:
        raise ValueError("API keys not found. Please set env vars.")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = StockBarsRequest(
                        symbol_or_symbols=symbols,
                        timeframe=timeframe,
                        start=start_date,
                        end=end_date,
                        adjustment='raw'
                   )
    print(f"Fetching data for {symbols} from {start_date.date()} to {end_date.date()}...")
    stock_bars = client.get_stock_bars(request_params)
    print("Data fetched successfully.")
    data_by_symbol = {symbol: stock_bars.df.loc[symbol] for symbol in symbols}
    return data_by_symbol


def fetch_forex_data_yf(
    symbols: list = None,
    period: str = "1y",
    interval: str = "1h"
) -> dict[str, pd.DataFrame]:
    """
    Fetches historical forex data from Yahoo Finance with robust formatting.
    """
    if symbols is None:
        symbols = ["EURUSD=X", "JPY=X", "GBPUSD=X"]
        
    data_by_symbol = {}
    print(f"Fetching Forex data for {symbols} from Yahoo Finance...")

    for symbol in symbols:
        # Download with auto_adjust=False to get the original OHLCV columns
        df_raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
        
        # --- DEFINITIVE FIX: Rebuild the DataFrame to guarantee 1D columns ---
        df = pd.DataFrame(index=df_raw.index)
        # Use the original column names from yfinance
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_raw.columns:
                df[col.lower()] = df_raw[col].values.flatten()
        # --- END FIX ---
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        data_by_symbol[symbol] = df
        print(f"{symbol}: {len(df)} data points")

    print("Data fetched successfully.")
    return data_by_symbol