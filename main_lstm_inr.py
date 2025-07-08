# main_lstm_inr.py
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import click
from rich.console import Console
from rich.table import Table
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
console = Console()

# --- 1. Data Fetching ---
def fetch_crypto_data(symbol='BTC', currency='INR', days=365):
    """Fetches historical cryptocurrency data from the CryptoCompare API."""
    console.print(f"Fetching last {days} days of historical data for {symbol}-{currency}...", style="cyan")
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histoday"
        params = {'fsym': symbol, 'tsym': currency, 'limit': days}
        response = requests.get(url, params=params)
        response.raise_for_status()
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

# --- 2. AI Feature Engineering for LSTM ---
def create_sequences(data, sequence_length=60):
    """Scales data and creates sequences for the LSTM model."""
    console.print("Preparing data sequences for the LSTM model...", style="cyan")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    console.print("✅ Data sequences created.", style="green")
    return X, y, scaler

# --- 3. Build and Train LSTM Model ---
def build_and_train_lstm_model(X, y):
    """Builds and trains the LSTM model."""
    console.print("Building and training the LSTM model... (This may take a moment)", style="cyan")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=0)
    model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])
    console.print("🤖 AI Model trained successfully.", style="yellow")
    return model

# --- 4. Trading Simulation ---
def run_backtest(df, model, scaler, sequence_length=60):
    """Simulates trading based on the LSTM model's predictions."""
    console.print("\n🚀 Running trading simulation (backtest)...", style="bold magenta")
    all_close_prices = df['close'].values
    scaled_data = scaler.transform(all_close_prices.reshape(-1,1))
    
    inputs = [scaled_data[i-sequence_length:i, 0] for i in range(sequence_length, len(scaled_data))]
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
    
    predicted_prices_scaled = model.predict(inputs, verbose=0)
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled).flatten()
    
    prediction_df = df.iloc[sequence_length:].copy()
    prediction_df['predicted_close'] = predicted_prices

    initial_capital, capital, position, trades = 1000000.0, 1000000.0, 0, [] # Starting with 10 Lakh INR

    for i in range(len(prediction_df) - 1):
        current_price = prediction_df['close'].iloc[i]
        predicted_next_price = prediction_df['predicted_close'].iloc[i+1]
        date = prediction_df.index[i].date()

        if predicted_next_price > current_price and capital > 0:
            position = capital / current_price
            trades.append({'Date': date, 'Action': 'BUY', 'Price': f"₹{current_price:,.2f}", 'Capital': f"₹0.00", 'Position': f"{position:.6f}"})
            capital = 0
        elif predicted_next_price < current_price and position > 0:
            capital = position * current_price
            trades.append({'Date': date, 'Action': 'SELL', 'Price': f"₹{current_price:,.2f}", 'Capital': f"₹{capital:,.2f}", 'Position': "0.000000"})
            position = 0

    final_capital = capital if position == 0 else position * prediction_df['close'].iloc[-1]
    profit = final_capital - initial_capital
    profit_percent = (profit / initial_capital) * 100

    console.print("\n--- Backtest Results (LSTM Strategy) ---", style="bold green")
    if trades:
        table = Table(title="Simulated Trades (Last 10)")
        for col in ["Date", "Action", "Price", "Capital After Trade", "Position After Trade"]:
            table.add_column(col)
        for trade in trades[-10:]:
            table.add_row(str(trade['Date']), trade['Action'], trade['Price'], trade['Capital'], trade['Position'])
        console.print(table)
    
    summary_table = Table(title="Performance Summary")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Initial Capital", f"₹{initial_capital:,.2f}")
    summary_table.add_row("Final Capital", f"₹{final_capital:,.2f}")
    summary_table.add_row("Net Profit", f"[bold green]₹{profit:,.2f}[/bold green]" if profit > 0 else f"[bold red]₹{profit:,.2f}[/bold red]")
    summary_table.add_row("Profit Percentage", f"{profit_percent:.2f}%")
    console.print(summary_table)

# --- 5. Predict and Display Next Day Price ---
def predict_and_display_next_day(model, data, scaler, sequence_length):
    """Predicts the next day's price and displays it."""
    console.print("\n--- Next Day Prediction ---", style="bold green")
    
    last_sequence = data['close'].values[-sequence_length:]
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    
    X_pred = np.array([last_sequence_scaled.flatten()])
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
    
    predicted_price_scaled = model.predict(X_pred, verbose=0)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
    
    last_known_price = last_sequence[-1]
    last_known_date = data.index[-1].date()

    prediction_table = Table(title="🔮 AI Price Prediction")
    prediction_table.add_column("Item", style="bold")
    prediction_table.add_column("Value", justify="right")
    prediction_table.add_row(f"Last Known Price ({last_known_date})", f"₹{last_known_price:,.2f}")
    prediction_table.add_row("[bold yellow]Predicted Next Day Price[/bold yellow]", f"[bold yellow]₹{predicted_price:,.2f}[/bold yellow]")
    console.print(prediction_table)
    
    if predicted_price > last_known_price:
        console.print("Signal: [bold green]BULLISH[/bold green] - Model predicts the price will rise.", justify="center")
    else:
        console.print("Signal: [bold red]BEARISH[/bold red] - Model predicts the price will fall.", justify="center")

@click.command()
@click.option('--symbol', default='BTC', help='The cryptocurrency symbol (e.g., BTC, ETH).')
@click.option('--currency', default='INR', help='The currency to compare against (e.g., INR, USD).')
@click.option('--days', default=365, help='Number of past days to fetch data for.')
@click.option('--sequence-length', default=60, help='Number of past days to look at for each prediction.')
def main(symbol, currency, days, sequence_length):
    """An advanced AI-powered CLI for simulating cryptocurrency trading using an LSTM model."""
    console.print(f"📈 [bold green]Advanced AI Crypto Trading Simulator (LSTM)[/bold green] 🤖", justify="center")
    if days < sequence_length * 2:
        console.print(f"Error: Days must be at least double the sequence length. Recommended: {sequence_length*2+100}", style="bold red")
        return

    raw_data = fetch_crypto_data(symbol, currency, days)
    if raw_data is None: return

    X, y, scaler = create_sequences(raw_data, sequence_length)
    if X is None: return

    model = build_and_train_lstm_model(X, y)
    if model is None: return

    run_backtest(raw_data, model, scaler, sequence_length)
    predict_and_display_next_day(model, raw_data, scaler, sequence_length)
    console.print("\n⚠️ [bold red]Disclaimer:[/bold red] This is a simplified simulation for educational purposes. This is not financial advice. Past performance is not indicative of future results.", style="italic")

if __name__ == '__main__':
    main()