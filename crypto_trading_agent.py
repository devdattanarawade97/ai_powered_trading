# crypto_trading_agent.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import click
from rich.console import Console
from rich.table import Table
import warnings
from datetime import datetime, timedelta

# Suppress TensorFlow warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
console = Console()

# --- Helper Data ---
CURRENCY_MAP = {'INR': '₹', 'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥'}

# --- 1. Data Fetching (Using yfinance) ---
def fetch_crypto_data(symbol='BTC', currency='INR', days=730):
    """Fetches historical cryptocurrency data from Yahoo Finance."""
    ticker = f"{symbol.upper()}-{currency.upper()}"
    console.print(f"Fetching last {days} days of historical data for {ticker} from Yahoo Finance...", style="cyan")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)

        if df.empty:
            console.print(f"API Error for {ticker}: No data found. The ticker may be invalid.", style="bold red")
            return None

        # --- Data Cleaning ---
        df.dropna(inplace=True) # Remove rows with any missing values

        # Rename columns to match the rest of the script's expectations
        df.rename(columns={
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volumeto'
        }, inplace=True)

        # Ensure all required columns are present
        required_cols = ['close', 'high', 'low', 'volumeto']
        if not all(col in df.columns for col in required_cols):
            console.print(f"Error for {ticker}: Dataframe missing one or more required columns.", style="bold red")
            return None

        df = df[required_cols] # Keep only the necessary columns

        console.print(f"✅ Data for {ticker} fetched and cleaned successfully. ({len(df)} rows)", style="green")
        return df
    except Exception as e:
        console.print(f"An unexpected error occurred while fetching data for {ticker}: {e}", style="bold red")
        return None


# --- 2. AI Feature Engineering ---
def create_features(df):
    """Engineers features for the model: SMA, RSI, and Volume."""
    console.print("Engineering features (SMA, RSI, Volume)...", style="cyan")
    df['sma_20'] = df['close'].rolling(window=20).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['volume'] = df['volumeto']

    df.dropna(inplace=True)
    console.print("✅ Features engineered.", style="green")
    return df

def create_sequences(data, feature_cols, sequence_length=60):
    """Scales data and creates sequences for the LSTM model."""
    console.print(f"Creating sequences of length {sequence_length}...", style="cyan")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_cols])

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(data['close'].values.reshape(-1,1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    console.print("✅ Data sequences created.", style="green")
    return X, y, scaler, close_scaler

# --- 3. Build and Train LSTM Model ---
def build_and_train_lstm_model(X, y):
    """Builds and trains a more robust LSTM model."""
    console.print("Building and training the LSTM model... (This may take a moment)", style="cyan")

    input_shape = (X.shape[1], X.shape[2])
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(units=100, return_sequences=False),
        Dropout(0.3),
        Dense(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=0, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00001, verbose=0)

    model.fit(X, y, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop, reduce_lr])
    console.print("🤖 AI Model trained successfully.", style="yellow")
    return model

# --- 4. Trading Simulation (Backtest) - CORRECTED FUNCTION ---
def run_backtest(df, model, scaler, close_scaler, feature_cols, sequence_length, transaction_fee, buy_threshold, sell_threshold, stop_loss_pct, initial_capital):
    """Simulates trading and returns a dictionary of performance metrics."""
    console.print("\n🚀 Running advanced trading simulation (backtest)...", style="bold magenta")

    all_data = df[feature_cols].values
    scaled_all_data = scaler.transform(all_data)

    inputs = [scaled_all_data[i-sequence_length:i, :] for i in range(sequence_length, len(scaled_all_data))]
    inputs = np.array(inputs)

    predicted_prices_scaled = model.predict(inputs, verbose=0)
    predicted_prices = close_scaler.inverse_transform(predicted_prices_scaled).flatten()

    prediction_df = df.iloc[sequence_length:].copy()
    prediction_df['predicted_close'] = predicted_prices

    capital, position, stop_loss_price, buy_value = initial_capital, 0.0, 0.0, 0.0
    trades, wins, losses = [], 0, 0

    for i in range(len(prediction_df) - 1):
        # FIX: Use .iloc[i] and .iloc[i+1] to get single scalar values
        current_price = prediction_df['close'].iloc[i]
        predicted_next_price = prediction_df['predicted_close'].iloc[i+1]
        date = prediction_df.index[i].date()

        if position > 0 and current_price < stop_loss_price:
            action = 'STOP-LOSS'
        elif position > 0 and predicted_next_price < current_price * (1 - sell_threshold):
            action = 'SELL'
        elif position == 0 and predicted_next_price > current_price * (1 + buy_threshold):
            buy_value = capital
            fee = buy_value * transaction_fee
            position = (buy_value - fee) / current_price
            stop_loss_price = current_price * (1 - stop_loss_pct)
            trades.append({'Date': date, 'Action': 'BUY', 'Price': current_price, 'Size': position, 'Value': buy_value, 'PnL': 0})
            capital = 0
            continue
        else:
            continue

        trade_value = position * current_price
        fee = trade_value * transaction_fee
        capital = trade_value - fee
        pnl = capital - buy_value
        trades.append({'Date': date, 'Action': action, 'Price': current_price, 'Size': position, 'Value': capital, 'PnL': pnl})
        if pnl > 0: wins += 1
        else: losses += 1
        position = 0

    final_capital = capital if position == 0 else position * prediction_df['close'].iloc[-1]
    net_profit = final_capital - initial_capital
    profit_percent = (net_profit / initial_capital) * 100 if initial_capital > 0 else 0

    return {
        "trades": trades,
        "final_capital": final_capital,
        "net_profit": net_profit,
        "profit_percent": profit_percent,
        "wins": wins,
        "losses": losses
    }

# --- 5. Predict and Display Next Day Price ---
def predict_and_display_next_day(model, df, scaler, close_scaler, feature_cols, sequence_length, currency_symbol='₹'):
    """Predicts the next day's price and returns prediction details."""
    console.print("\n--- Next Day Prediction ---", style="bold green")

    last_sequence = df[feature_cols].values[-sequence_length:]
    last_sequence_scaled = scaler.transform(last_sequence)

    X_pred = np.array([last_sequence_scaled])
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]))

    predicted_price_scaled = model.predict(X_pred, verbose=0)
    predicted_price = close_scaler.inverse_transform(predicted_price_scaled)[0][0]

    last_known_price = df['close'].iloc[-1]
    last_known_date = df.index[-1].date()

    prediction_table = Table(title=f"🔮 AI Price Prediction for {df.name}")
    prediction_table.add_column("Item", style="bold")
    prediction_table.add_column("Value", justify="right")
    prediction_table.add_row(f"Last Known Price ({last_known_date})", f"{currency_symbol}{last_known_price:,.2f}")
    prediction_table.add_row("[bold yellow]Predicted Next Day Price[/bold yellow]", f"[bold yellow]{currency_symbol}{predicted_price:,.2f}[/bold yellow]")
    console.print(prediction_table)

    predicted_change_percent = ((predicted_price - last_known_price) / last_known_price) * 100
    if predicted_price > last_known_price:
        signal = "BULLISH"
        console.print(f"Signal: [bold green]{signal}[/bold green] - Model predicts a rise of {predicted_change_percent:.2f}%.", justify="center")
    else:
        signal = "BEARISH"
        console.print(f"Signal: [bold red]{signal}[/bold red] - Model predicts a fall of {predicted_change_percent:.2f}%.", justify="center")

    return {"signal": signal, "predicted_change_percent": predicted_change_percent}

def process_symbol(symbol, currency, days, sequence_length, fee, buy_thresh, sell_thresh, stop_loss, initial_capital, currency_symbol):
    """Runs the entire analysis workflow for a single symbol."""
    ticker_name = f"{symbol}-{currency}"
    console.rule(f"[bold yellow]Processing: {ticker_name}", style="yellow")

    raw_data = fetch_crypto_data(symbol, currency, days)
    if raw_data is None: return None
    raw_data.name = ticker_name

    featured_data = create_features(raw_data.copy())
    featured_data.name = ticker_name

    console.print(f"ℹ️ Data points remaining after feature engineering: {len(featured_data)}", style="blue")

    if len(featured_data) <= sequence_length:
        console.print(f"\n[bold red]Error for {ticker_name}: Not enough data to create sequences.[/bold red]")
        console.print(f"  - Sequence length required: {sequence_length}")
        console.print(f"  - Data points available: {len(featured_data)}")
        return None

    feature_cols = ['close', 'sma_20', 'rsi', 'volume']
    X, y, scaler, close_scaler = create_sequences(featured_data, feature_cols, sequence_length)

    model = build_and_train_lstm_model(X, y)
    if model is None: return None

    backtest_results = run_backtest(featured_data, model, scaler, close_scaler, feature_cols, sequence_length, fee, buy_thresh, sell_thresh, stop_loss, initial_capital)
    prediction_results = predict_and_display_next_day(model, featured_data, scaler, close_scaler, feature_cols, sequence_length, currency_symbol)

    return {
        "symbol": ticker_name,
        "final_capital": backtest_results["final_capital"],
        "net_profit": backtest_results["net_profit"],
        "profit_percent": backtest_results["profit_percent"],
        "signal": prediction_results["signal"],
        "predicted_change_percent": prediction_results["predicted_change_percent"]
    }

@click.command()
@click.option('--symbols', '-s', default='BTC,ETH', help='Comma-separated list of symbols (e.g., "BTC,ETH,SOL").')
@click.option('--currency', default='INR', help='Fiat currency (e.g., INR, USD).')
@click.option('--capital', default=100000.0, type=float, help='Initial capital for the simulation.')
@click.option('--days', default=730, help='Number of past days of data to use.')
@click.option('--sequence-length', default=60, help='Number of past days for each prediction.')
@click.option('--fee', default=0.001, help='Transaction fee percentage (0.1%).')
@click.option('--buy-thresh', default=0.01, help='Buy if predicted price is this % higher.')
@click.option('--sell-thresh', default=0.01, help='Sell if predicted price is this % lower.')
@click.option('--stop-loss', default=0.05, help='Sell if price drops by this % after buying.')
def main(symbols, currency, capital, days, sequence_length, fee, buy_thresh, sell_thresh, stop_loss):
    """An advanced AI-powered CLI for simulating cryptocurrency trading for multiple symbols."""
    console.print(f"📈 [bold green]Advanced AI Crypto Trading Simulator (using yfinance)[/bold green] 🤖", justify="center")

    cleaned_symbols = symbols.strip('[]')
    symbol_list = [s.strip().upper() for s in cleaned_symbols.split(',')]
    currency_symbol = CURRENCY_MAP.get(currency.upper(), '')

    all_results = []
    for symbol in symbol_list:
        result = process_symbol(symbol, currency, days, sequence_length, fee, buy_thresh, sell_thresh, stop_loss, capital, currency_symbol)
        if result:
            all_results.append(result)

    if not all_results:
        console.print("\n[bold red]No symbols were processed successfully.[/bold red]")
        return

    # --- Final Summary ---
    console.rule(f"[bold green]📊 Overall Market Analysis ({currency.upper()})", style="green")
    summary_table = Table(title="Consolidated Performance & Predictions")
    summary_table.add_column("Symbol", style="cyan", no_wrap=True)
    summary_table.add_column(f"Final Capital ({currency_symbol})", justify="right")
    summary_table.add_column(f"Net Profit ({currency_symbol})", justify="right")
    summary_table.add_column("Profit %", justify="right")
    summary_table.add_column("Next Day Signal", justify="center")
    summary_table.add_column("Predicted Change %", justify="right")

    total_net_profit = 0
    bullish_count = 0
    for res in all_results:
        total_net_profit += res['net_profit']
        if res['signal'] == 'BULLISH':
            bullish_count += 1

        profit_color = "green" if res['net_profit'] > 0 else "red"
        signal_color = "green" if res['signal'] == 'BULLISH' else "red"

        summary_table.add_row(
            res['symbol'],
            f"[{profit_color}]{res['final_capital']:,.2f}[/{profit_color}]",
            f"[{profit_color}]{res['net_profit']:,.2f}[/{profit_color}]",
            f"[{profit_color}]{res['profit_percent']:.2f}%[/{profit_color}]",
            f"[{signal_color}]{res['signal']}[/{signal_color}]",
            f"[{signal_color}]{res['predicted_change_percent']:.2f}%[/{signal_color}]"
        )

    console.print(summary_table)

    # --- Market Decision ---
    console.print("\n--- 🧠 AI Market Decision ---", style="bold yellow")
    overall_sentiment = "BULLISH" if bullish_count >= len(all_results) / 2 else "BEARISH"
    sentiment_color = "green" if overall_sentiment == "BULLISH" else "red"

    console.print(f"Overall Market Sentiment: [{sentiment_color}]{overall_sentiment}[/{sentiment_color}] ({bullish_count} of {len(all_results)} symbols are bullish)")

    total_profit_color = "green" if total_net_profit > 0 else "red"
    console.print(f"Total Combined Net Profit from AI Strategy: [{total_profit_color}]{currency_symbol}{total_net_profit:,.2f}[/{total_profit_color}]")

    console.print("\n⚠️ [bold red]Disclaimer:[/bold red] This is a simulation for educational purposes and not financial advice.", style="italic")

if __name__ == '__main__':
    main()