import os
from datetime import datetime, timedelta
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def fetch_data() -> dict[str, pd.DataFrame]:
    """
    Fetches historical stock data from Alpaca for a predefined list of symbols
    and returns it as a dictionary of pandas DataFrames.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are stock symbols
                                 and values are their corresponding dataframes.
    """
    # --- 1. Configuration ---
    api_key = os.getenv('APCA_API_KEY_ID')
    secret_key = os.getenv('APCA_API_SECRET_KEY')

    if not api_key or not secret_key:
        raise ValueError("API keys not found. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")

    SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=365)

    # --- 2. Data Fetching ---
    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = StockBarsRequest(
                        symbol_or_symbols=SYMBOLS,
                        timeframe=TimeFrame.Hour,
                        start=START_DATE,
                        end=END_DATE,
                        adjustment='raw'
                   )

    print(f"Fetching data for {SYMBOLS} from {START_DATE.date()} to {END_DATE.date()}...")
    stock_bars = client.get_stock_bars(request_params)
    print("Data fetched successfully.")

    # --- 3. Format and Return DataFrames ---
    # The SDK returns a MultiIndex DataFrame. We convert it into a dictionary
    # of single-index DataFrames for easier handling in the pipeline.
    data_by_symbol = {symbol: stock_bars.df.loc[symbol] for symbol in SYMBOLS}
    total_points = 0
    for symbol, df in data_by_symbol.items():
        count = len(df)
        total_points += count
        print(f"{symbol}: {count} data points")

    print(f"\nTotal data points fetched: {total_points}")
    
    return data_by_symbol