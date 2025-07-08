import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from datetime import datetime
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D

# --- Configuration ---
SYMBOL = 'BTC'
CURRENCY = 'INR'
DATA_LOOKBACK_DAYS = 15
OBSERVATION_WINDOW = 24
MIN_SAMPLE_LENGTH = 20
T_VALUE_THRESHOLD = 2.0

def print_disclaimer():
    print("="*80)
    print("!!! CRITICAL RISK WARNING !!!")
    print("This script is an analysis tool, NOT a trading bot.")
    print("It identifies past trends and does NOT predict the future.")
    print("="*80 + "\n")

def fetch_live_btc_inr():
    """Fetches live BTC to INR price using CoinGecko API"""
    resp = requests.get("https://api.coingecko.com/api/v3/simple/price",
                        params={"ids": "bitcoin", "vs_currencies": "inr"})
    return resp.json()["bitcoin"]["inr"]

def get_hourly_data(symbol, currency, days):
    limit = days * 24
    print(f"Fetching last {days} days ({limit} hours) of data for {symbol}-{currency}...")
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {'fsym': symbol.upper(), 'tsym': currency.upper(), 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    print(f"✅ Data fetched: {len(df)} points")
    return df[['close']]

def trend_scanning_labels(closes, observation_window, min_sample_length=5):
    print(f"Scanning trends with {observation_window}-hour window...")
    results = []
    for i in range(len(closes)):
        window_end = i + 1
        window_start = max(0, window_end - observation_window)
        window = closes.iloc[window_start:window_end]
        if len(window) < min_sample_length:
            results.append({'t_value': np.nan, 'bin': np.nan})
            continue
        X = np.arange(len(window)).reshape(-1, 1)
        y = window.values
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        t_value = model.tvalues[1]
        bin_label = 0
        if t_value > T_VALUE_THRESHOLD:
            bin_label = 1
        elif t_value < -T_VALUE_THRESHOLD:
            bin_label = -1
        results.append({'t_value': t_value, 'bin': bin_label})
    print("✅ Trend scanning complete.")
    return pd.DataFrame(results, index=closes.index)

def identify_dips(series, threshold=0.005):
    """Returns timestamps where price dipped more than threshold percent."""
    pct_changes = series.pct_change()
    return pct_changes[pct_changes < -threshold].index

def analyze_and_plot():
    closes = get_hourly_data(SYMBOL, CURRENCY, DATA_LOOKBACK_DAYS)['close']
    if closes is None or closes.empty:
        return
    live_price_inr = fetch_live_btc_inr()
    trend_df = trend_scanning_labels(closes, OBSERVATION_WINDOW, MIN_SAMPLE_LENGTH)
    t_values = trend_df['t_value'].dropna()
    trend_labels = trend_df['bin'].dropna()
    latest_trend_label = trend_labels.iloc[-1]
    latest_t_value = t_values.iloc[-1]
    trend_map = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}

    print("\n--- Latest Trend ---")
    print(f"Time: {trend_labels.index[-1]}")
    print(f"Trend: {trend_map.get(latest_trend_label)} | T-Value: {latest_t_value:.2f}")
    print("---------------------")

    dips = identify_dips(closes.iloc[-OBSERVATION_WINDOW:], threshold=0.005)

    fig, ax1 = plt.subplots(figsize=(18, 10))
    plt.style.use('seaborn-v0_8-darkgrid')

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('BTC/INR Price', color='blue', fontsize=12)
    ax1.plot(closes.index, closes, color='blue', label='Close Price')

    ax1.scatter(dips, closes.loc[dips], color='cyan', marker='v', s=70, label='Dip >0.5%')

    ax2 = ax1.twinx()
    ax2.set_ylabel('t-values', color='red', fontsize=12)
    ax2.plot(t_values.index, t_values, '--', color='red', alpha=0.7)
    ax2.axhline(T_VALUE_THRESHOLD, linestyle=':', color='red')
    ax2.axhline(-T_VALUE_THRESHOLD, linestyle=':', color='red')
    ax2.axhline(0, linestyle=':', color='grey')

    colors = {1: 'green', 0: 'orange', -1: 'red'}
    for i in range(len(trend_labels)):
        label = trend_labels.iloc[i]
        date = trend_labels.index[i]
        if label != 0:
            ax1.scatter(date, closes.loc[date], color=colors[label], s=50, zorder=5)

    price_box = AnchoredText(f"Live BTC/INR:\n₹{live_price_inr:,.0f}", loc='upper right', frameon=True, prop=dict(size=12))
    ax1.add_artist(price_box)

    fig.suptitle('BTC/INR Trend Analysis using Rolling Regression', fontsize=16, weight='bold')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Up Trend', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Down Trend', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='v', color='cyan', label='Dip >0.5%', markerfacecolor='cyan', markersize=10)
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    file_name = f"{SYMBOL}_{CURRENCY}_live_trend_analysis.png"
    plt.savefig(file_name)
    print(f"✅ Plot saved as '{file_name}'")
    plt.show()

if __name__ == "__main__":
    print_disclaimer()
    analyze_and_plot()
