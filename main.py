# main.py
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import click
from rich.console import Console
from rich.table import Table
import time

console = Console()

# --- 1. Data Fetching ---
def fetch_crypto_data(symbol='BTC', currency='USD', days=365):
    """
    Fetches historical cryptocurrency data from the CryptoCompare API.
    """
    console.print(f"Fetching last {days} days of historical data for {symbol}-{currency}...", style="cyan")
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            'fsym': symbol,
            'tsym': currency,
            'limit': days
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        if data['Response'] == 'Error':
            console.print(f"API Error: {data['Message']}", style="bold red")
            return None

        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        console.print("✅ Data fetched successfully.", style="green")
        return df
    except requests.exceptions.RequestException as e:
        console.print(f"Error fetching data: {e}", style="bold red")
        return None
    except Exception as e:
        console.print(f"An unexpected error occurred: {e}", style="bold red")
        return None


# --- 2. Feature Engineering ---
def create_features(df):
    """
    Creates features for the machine learning model from the raw data.
    """
    if df is None:
        return None, None

    console.print("Engineering features for the AI model...", style="cyan")
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['direction'] = np.where(df['returns'] > 0, 1, 0) # 1 for up, 0 for down

    # Create lagged returns as features
    lags = 5
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['returns'].shift(lag)

    df.dropna(inplace=True)

    features = [f'lag_{lag}' for lag in range(1, lags + 1)]
    target = 'direction'

    X = df[features]
    y = df[target]

    console.print("✅ Features created.", style="green")
    return X, y

# --- 3. AI Model Training ---
def train_model(X, y):
    """
    Trains a Random Forest Classifier to predict price direction.
    """
    if X is None or y is None:
        return None

    console.print("Training the AI model...", style="cyan")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    console.print(f"🤖 AI Model Accuracy on test data: {accuracy:.2%}", style="yellow")

    return model

# --- 4. Trading Simulation ---
def run_backtest(df, model):
    """
    Simulates trading based on the model's predictions.
    """
    if df is None or model is None:
        return

    console.print("\n🚀 Running trading simulation (backtest)...", style="bold magenta")
    time.sleep(1) # For dramatic effect

    # Prepare data for prediction
    lags = 5
    features = [f'lag_{lag}' for lag in range(1, lags + 1)]
    df_pred = df.copy()

    # Create lagged returns for the entire dataset
    df_pred['returns'] = np.log(df_pred['close'] / df_pred['close'].shift(1))
    for lag in range(1, lags + 1):
        df_pred[f'lag_{lag}'] = df_pred['returns'].shift(lag)
    df_pred.dropna(inplace=True)

    X_full = df_pred[features]
    df_pred['prediction'] = model.predict(X_full)

    # Simulation logic
    initial_capital = 10000.0
    capital = initial_capital
    position = 0  # Number of crypto units held
    trades = []
    
    for i in range(len(df_pred)):
        # Signal: 1 = Buy, 0 = Sell
        signal = df_pred['prediction'].iloc[i]
        price = df_pred['close'].iloc[i]
        date = df_pred.index[i].date()

        # Buy logic
        if signal == 1 and capital > 0:
            position = capital / price
            trades.append({'Date': date, 'Action': 'BUY', 'Price': f"${price:,.2f}", 'Capital': f"${capital:,.2f}", 'Position': f"{position:.6f}"})
            capital = 0
        # Sell logic
        elif signal == 0 and position > 0:
            capital = position * price
            trades.append({'Date': date, 'Action': 'SELL', 'Price': f"${price:,.2f}", 'Capital': f"${capital:,.2f}", 'Position': "0.000000"})
            position = 0

    # Final portfolio value
    final_capital = capital if position == 0 else position * df_pred['close'].iloc[-1]
    profit = final_capital - initial_capital
    profit_percent = (profit / initial_capital) * 100

    # Display results
    console.print("\n--- Backtest Results ---", style="bold green")

    # Trades Table
    if trades:
        table = Table(title="Simulated Trades")
        table.add_column("Date", justify="right", style="cyan", no_wrap=True)
        table.add_column("Action", style="magenta")
        table.add_column("Price", justify="right", style="green")
        table.add_column("Capital After Trade", justify="right", style="yellow")
        table.add_column("Position After Trade", justify="right", style="blue")

        for trade in trades[-10:]: # Show last 10 trades
            table.add_row(str(trade['Date']), trade['Action'], trade['Price'], trade['Capital'], trade['Position'])
        console.print(table)
    else:
        console.print("No trades were executed.", style="yellow")


    # Performance Summary
    summary_table = Table(title="Performance Summary")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Initial Capital", f"${initial_capital:,.2f}")
    summary_table.add_row("Final Capital", f"${final_capital:,.2f}")
    summary_table.add_row("Net Profit", f"${profit:,.2f}")
    summary_table.add_row("Profit Percentage", f"{profit_percent:.2f}%")
    console.print(summary_table)

    console.print("\n⚠️ [bold red]Disclaimer:[/bold red] This is a simplified simulation. Past performance is not indicative of future results. Always do your own research before investing.", style="italic")


@click.command()
@click.option('--symbol', default='BTC', help='The cryptocurrency symbol (e.g., BTC, ETH).')
@click.option('--currency', default='USD', help='The currency to compare against (e.g., USD, EUR).')
@click.option('--days', default=365, help='Number of past days to fetch data for.')
def main(symbol, currency, days):
    """
    An AI-powered CLI for simulating cryptocurrency trading algorithms.
    """
    console.print(f"📈 [bold green]AI Crypto Trading Simulator[/bold green] 🤖", justify="center")
    console.print(f"Simulating for: [bold]{symbol}-{currency}[/bold] over {days} days.", justify="center")
    console.print("-" * 50, justify="center")

    # 1. Get Data
    raw_data = fetch_crypto_data(symbol, currency, days)
    if raw_data is None:
        return

    # 2. Create Features
    X, y = create_features(raw_data)
    if X is None:
        return

    # 3. Train Model
    model = train_model(X, y)
    if model is None:
        return

    # 4. Run Simulation
    run_backtest(raw_data, model)


if __name__ == '__main__':
    main()