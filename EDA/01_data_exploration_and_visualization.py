import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ta

# Assuming 'data_ingestion.py' is in the same directory or a reachable path.
# This script reuses the same data fetching logic.
from src.data_ingestion import fetch_data

# --- 1. Feature Engineering (Adapted from your preprocessing script) ---
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators and the target variable, then adds them 
    to the DataFrame for visualization purposes.

    Args:
        df (pd.DataFrame): The raw stock data.

    Returns:
        pd.DataFrame: The DataFrame enriched with features.
    """
    df_copy = df.copy()
    
    # Calculate a set of technical indicators
    df_copy['returns'] = df_copy['close'].pct_change()
    df_copy['rsi'] = ta.momentum.RSIIndicator(close=df_copy['close']).rsi()
    df_copy['macd'] = ta.trend.MACD(close=df_copy['close']).macd()
    df_copy['bollinger_h'] = ta.volatility.BollingerBands(close=df_copy['close']).bollinger_hband()
    df_copy['bollinger_l'] = ta.volatility.BollingerBands(close=df_copy['close']).bollinger_lband()
    df_copy['atr'] = ta.volatility.AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close']).average_true_range()
    df_copy['stoch'] = ta.momentum.StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close']).stoch()

    # Define the target variable for classification
    df_copy['target'] = (df_copy['close'].shift(-1) > df_copy['close']).astype(int)

    # Drop rows with NaN values that were created by the indicators
    df_copy.dropna(inplace=True)
    # The index is the timestamp, which we want to keep for plotting
    df_copy.reset_index(inplace=True) 
    
    return df_copy

# --- 2. Visualization Functions ---
def plot_price_and_indicators(df: pd.DataFrame, symbol: str):
    """
    Plots the closing price, volume, and key technical indicators in subplots.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    fig.suptitle(f'📈 Exploratory Data Analysis for {symbol}', fontsize=18)

    # Plot 1: Price and Bollinger Bands
    axes[0].plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    axes[0].plot(df['timestamp'], df['bollinger_h'], label='Bollinger High', linestyle='--', color='red', alpha=0.7)
    axes[0].plot(df['timestamp'], df['bollinger_l'], label='Bollinger Low', linestyle='--', color='green', alpha=0.7)
    axes[0].fill_between(df['timestamp'], df['bollinger_l'], df['bollinger_h'], color='gray', alpha=0.1)
    axes[0].set_title('Stock Price and Bollinger Bands')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Trading Volume
    axes[1].bar(df['timestamp'], df['volume'], color='grey', alpha=0.6, width=0.01)
    axes[1].set_title('Trading Volume')
    axes[1].set_ylabel('Volume')
    axes[1].grid(True)

    # Plot 3: RSI and Stochastic Oscillator
    ax3_twin = axes[2].twinx()
    axes[2].plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
    ax3_twin.plot(df['timestamp'], df['stoch'], label='Stochastic Osc.', color='orange')
    axes[2].axhline(70, linestyle='--', color='red', alpha=0.5, label='Overbought (70)')
    axes[2].axhline(30, linestyle='--', color='green', alpha=0.5, label='Oversold (30)')
    axes[2].set_title('Momentum Indicators: RSI & Stochastic')
    axes[2].set_ylabel('RSI')
    ax3_twin.set_ylabel('Stochastic (%)')
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3_twin.legend(lines + lines2, labels + labels2, loc='upper left')
    axes[2].grid(True)

    # Plot 4: MACD
    axes[3].plot(df['timestamp'], df['macd'], label='MACD', color='darkcyan')
    axes[3].axhline(0, linestyle='--', color='black', alpha=0.5)
    axes[3].set_title('MACD Indicator')
    axes[3].set_ylabel('MACD Value')
    axes[3].set_xlabel('Date')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def plot_distributions_and_correlations(df: pd.DataFrame, symbol: str):
    """
    Plots the distribution of returns, target variable, and a feature correlation matrix.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'📊 Feature Distributions and Correlations for {symbol}', fontsize=18)

    # Plot 1: Distribution of Hourly Returns
    sns.histplot(df['returns'], kde=True, ax=axes[0], bins=50, color='skyblue')
    axes[0].set_title('Distribution of Hourly Returns')
    axes[0].set_xlabel('Return')
    axes[0].set_ylabel('Frequency')

    # Plot 2: Target Variable Distribution (Class Balance)
    sns.countplot(x='target', data=df, ax=axes[1], palette='viridis', hue='target', legend=False)
    axes[1].set_title('Target Variable Distribution')
    axes[1].set_xlabel('Target')
    axes[1].set_xticklabels(['Down/Same (0)', 'Up (1)'])
    
    # Plot 3: Correlation Matrix
    feature_cols = ['close', 'volume', 'returns', 'rsi', 'macd', 'bollinger_h', 'bollinger_l', 'atr', 'stoch', 'target']
    corr_matrix = df[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=axes[2])
    axes[2].set_title('Feature Correlation Matrix')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- 3. Main Execution Block ---
def explore_data_pipeline(symbol: str = 'AAPL'):
    """
    Runs the full data exploration and visualization pipeline for a given stock symbol.
    """
    print(f"--- Starting EDA for {symbol} ---")

    # Step 1: Fetch raw data from Alpaca
    print("Fetching raw data...")
    raw_data_dict = fetch_data()
    if symbol not in raw_data_dict:
        print(f"❌ Error: Symbol '{symbol}' not found in fetched data.")
        return
    
    # Step 2: Add technical indicators and target variable
    print("Adding technical indicators...")
    df_raw = raw_data_dict[symbol]
    df_features = add_technical_indicators(df_raw)

    # Step 3: Print descriptive statistics of the main features
    print("\n--- Descriptive Statistics of Features ---")
    print(df_features[['close', 'volume', 'returns', 'rsi', 'macd', 'target']].describe())
    print("------------------------------------------\n")

    # Step 4: Generate and display visualizations
    print("Generating visualizations...")
    plot_price_and_indicators(df_features, symbol)
    plot_distributions_and_correlations(df_features, symbol)

    print(f"--- ✅ EDA for {symbol} Complete ---")

if __name__ == "__main__":
    # You can easily change the symbol to analyze here (e.g., "GOOGL", "MSFT")
    explore_data_pipeline(symbol='AAPL')