# -*- coding: utf-8 -*-
"""
================================================================================
=== BTC/INR Trend Identification using Trend Scanning Models ===================
================================================================================

** DISCLAIMER: FOR EDUCATIONAL AND ILLUSTRATIVE PURPOSES ONLY. **
** THIS IS NOT FINANCIAL ADVICE. HIGHLY RISKY. **

This script analyzes historical price data to identify statistically significant
trends. It does not forecast future prices. Trading decisions should not be
based solely on this tool. Markets are volatile, and past performance is not
indicative of future results.

Description:
This script fetches hourly BTC/INR data and applies a "Trend Scanning"
methodology. It uses a rolling linear regression to calculate the t-statistic
for the trend's slope at each point in time, helping to identify when a trend
is statistically significant.

Requirements:
- pandas, numpy, scikit-learn, matplotlib, requests, statsmodels

Usage:
python your_script_name.py

================================================================================
"""
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from datetime import datetime

# --- Configuration ---
SYMBOL = 'BTC'
CURRENCY = 'INR'
DATA_LOOKBACK_DAYS = 15 # Days of hourly data to analyze
OBSERVATION_WINDOW = 24 # Hours in the rolling window for regression
MIN_SAMPLE_LENGTH = 20  # Min data points required in a window to calculate trend
T_VALUE_THRESHOLD = 2.0 # Threshold for t-statistic to be considered significant
CRYPTOCOMPARE_API_KEY = "YOUR_CRYPTOCOMPARE_API_KEY" # Optional, but recommended

def print_disclaimer():
    """Prints a prominent disclaimer to the console."""
    print("="*80)
    print("!!! CRITICAL RISK WARNING !!!")
    print("This script is an analysis tool, NOT a trading bot. It identifies past")
    print("trends and does NOT predict the future. Relying on this for live trading")
    print("without a robust risk management strategy is extremely dangerous.")
    print("="*80 + "\n")

def get_hourly_data(symbol, currency, days):
    """Fetches historical hourly cryptocurrency data."""
    limit = days * 24
    print(f"Fetching last {days} days ({limit} hours) of historical data for {symbol}-{currency}...")
    
    base_url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {'fsym': symbol.upper(), 'tsym': currency.upper(), 'limit': limit}
    if CRYPTOCOMPARE_API_KEY != "YOUR_CRYPTOCOMPARE_API_KEY":
        params['api_key'] = CRYPTOCOMPARE_API_KEY

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get('Response') == 'Error':
            print(f"API Error: {data.get('Message', 'Unknown error')}")
            return None

        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        print(f"✅ Data fetched successfully. ({len(df)} rows)")
        return df[['close']]
    except Exception as e:
        print(f"An unexpected error occurred during data fetching: {e}")
        return None

def trend_scanning_labels(closes, observation_window, min_sample_length=5):
    """
    Recreation of the trend_scanning_labels logic.

    This function scans through the closing prices using a rolling window,
    performs a linear regression for each window, and calculates the t-statistic
    of the slope to determine the trend's significance.

    Args:
        closes (pd.Series): Series of closing prices.
        observation_window (int): The size of the rolling window.
        min_sample_length (int): Minimum number of samples required in the window.

    Returns:
        pd.DataFrame: A DataFrame containing 't_value' and 'bin' (trend label).
    """
    print(f"Scanning trends with a {observation_window}-hour window...")
    results = []
    
    for i in range(len(closes)):
        window_end = i + 1
        window_start = max(0, window_end - observation_window)
        window = closes.iloc[window_start:window_end]

        if len(window) < min_sample_length:
            results.append({'t_value': np.nan, 'bin': np.nan})
            continue

        # Prepare data for regression
        X = np.arange(len(window)).reshape(-1, 1)
        y = window.values
        
        # Using statsmodels for detailed statistics including t-values
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        
        # The t-value for the slope is the second value in the tvalues series
        t_value = model.tvalues[1]
        
        # Determine trend bin based on t-value threshold
        bin_label = 0
        if t_value > T_VALUE_THRESHOLD:
            bin_label = 1  # Up trend
        elif t_value < -T_VALUE_THRESHOLD:
            bin_label = -1 # Down trend
            
        results.append({'t_value': t_value, 'bin': bin_label})

    print("✅ Trend scanning complete.")
    return pd.DataFrame(results, index=closes.index)

def analyze_and_plot():
    """Main function to perform trend analysis and plotting."""
    # 1. Fetch data
    closes = get_hourly_data(SYMBOL, CURRENCY, DATA_LOOKBACK_DAYS)['close']
    if closes is None or closes.empty:
        print("Could not proceed due to lack of data.")
        return

    # 2. Get trend labels using the trend scanning function
    trend_labels_df = trend_scanning_labels(closes, observation_window=OBSERVATION_WINDOW, min_sample_length=MIN_SAMPLE_LENGTH)
    
    # 3. Extract t-values and trend labels for plotting
    t_values = trend_labels_df['t_value'].dropna()
    trend_labels = trend_labels_df['bin'].dropna()

    # 4. Print summary of the latest trend
    latest_trend_label = trend_labels.iloc[-1]
    latest_t_value = t_values.iloc[-1]
    trend_map = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}
    print("\n--- Latest Trend Analysis ---")
    print(f"Time:           {trend_labels.index[-1]}")
    print(f"Identified Trend: {trend_map.get(latest_trend_label, 'UNKNOWN')}")
    print(f"T-Value:          {latest_t_value:.2f} (Threshold: +/- {T_VALUE_THRESHOLD})")
    print("-----------------------------\n")

    # 5. Plot the results
    print("Generating plot...")
    fig, ax1 = plt.subplots(figsize=(18, 10))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot closing prices on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date and Time', fontsize=12)
    ax1.set_ylabel('Closing Prices (INR)', color=color, fontsize=12)
    ax1.plot(closes.index, closes, color=color, label='Closing Prices')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the t-values
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('t-values', color=color, fontsize=12)
    ax2.plot(t_values.index, t_values, color=color, alpha=0.6, linestyle='--', label='t-values')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(T_VALUE_THRESHOLD, color=color, linestyle=':', alpha=0.5)
    ax2.axhline(-T_VALUE_THRESHOLD, color=color, linestyle=':', alpha=0.5)
    ax2.axhline(0, color='grey', linestyle=':', alpha=0.5)

    # Add trend labels as scatter points on the price chart
    colors = {1: 'green', 0: 'orange', -1: 'red'}
    for i in range(len(trend_labels)):
        label = trend_labels.iloc[i]
        date = trend_labels.index[i]
        if label != 0: # Only plot significant trends
            ax1.scatter(date, closes.loc[date], color=colors[label], s=50, zorder=5)

    # Add title and legend
    fig.suptitle('Trend Identification using Linear Regression Models', fontsize=16, weight='bold')
    
    # Create custom legend for trend labels
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Up Trend', markerfacecolor='green', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Down Trend', markerfacecolor='red', markersize=10)]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    
    file_name = f"{SYMBOL}_{CURRENCY}_trend_analysis.png"
    plt.savefig(file_name)
    print(f"✅ Analysis plot saved as '{file_name}'")
    # plt.show()

if __name__ == "__main__":
    print_disclaimer()
    analyze_and_plot()
