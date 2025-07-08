# main.py
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import argparse
import warnings

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore', category=FutureWarning)


def get_exchange_rate(currency='INR'):
    """
    Fetches the current exchange rate from USD to the target currency.

    Args:
        currency (str): The target currency code (e.g., 'INR', 'EUR').

    Returns:
        float: The exchange rate. Defaults to 1.0 (USD) if fetching fails.
    """
    if currency.upper() == 'USD':
        return 1.0
    try:
        ticker_symbol = f"USD{currency.upper()}=X"
        print(f"Fetching exchange rate for USD to {currency.upper()}...")
        rate_data = yf.Ticker(ticker_symbol)
        # Fetch the most recent closing price
        todays_data = rate_data.history(period='2d')
        if todays_data.empty:
            raise ValueError(f"No exchange rate data found for {ticker_symbol}")
        exchange_rate = todays_data['Close'].iloc[-1]
        print(f"Current USD to {currency.upper()} exchange rate: {exchange_rate:.2f}")
        return exchange_rate
    except Exception as e:
        print(f"Could not fetch exchange rate for USD to {currency.upper()}. Defaulting to 1.0 (USD). Error: {e}")
        return 1.0

def fetch_crypto_data(ticker='BTC-USD'):
    """
    Fetches all available historical cryptocurrency data from Yahoo Finance.

    Args:
        ticker (str): The ticker symbol for the cryptocurrency (e.g., 'BTC-USD').

    Returns:
        pandas.DataFrame: A DataFrame containing the historical data.
    """
    print(f"Fetching all available historical data for {ticker}...")
    # Using period="max" fetches all data since the listing date.
    data = yf.download(ticker, period="max", progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}. It might be delisted or an invalid symbol.")
    print(f"Data fetched successfully. Found data from {data.index[0].date()} to {data.index[-1].date()}.")
    return data

def clean_and_preprocess_data(data):
    """
    Cleans and preprocesses the financial data for model training.

    Args:
        data (pandas.DataFrame): The raw historical data.

    Returns:
        tuple: A tuple containing the scaled data, the scaler object, and the cleaned data.
    """
    print("Cleaning and preprocessing data...")
    # Use the 'Close' price for prediction
    df = data[['Close']].copy()

    # --- Data Cleaning ---
    # Check for missing values and fill them using forward fill
    if df.isnull().sum().values[0] > 0:
        print(f"Found {df.isnull().sum().values[0]} missing values. Filling them...")
        df.fillna(method='ffill', inplace=True)

    # --- Data Scaling ---
    # Scale the data to be between 0 and 1. This is crucial for LSTM performance.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    print("Data cleaned and preprocessed.")
    return scaled_data, scaler, df

def create_dataset(data, time_step=60):
    """
    Creates sequences of data for the LSTM model.

    Args:
        data (numpy.array): The preprocessed data.
        time_step (int): The number of past time steps to use for predicting the next time step.

    Returns:
        tuple: A tuple containing the input sequences (X) and the target values (y).
    """
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Builds the LSTM model architecture.

    Args:
        input_shape (tuple): The shape of the input data for the first LSTM layer.

    Returns:
        tensorflow.keras.models.Sequential: The compiled LSTM model.
    """
    print("Building LSTM model...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Model built successfully.")
    model.summary()
    return model

def plot_forecast(future_dates, future_forecast_usd, ticker, currency, exchange_rate):
    """
    Plots the future price forecast.

    Args:
        future_dates (pd.DatetimeIndex): The dates for the forecast period.
        future_forecast_usd (np.array): Future price forecast in USD.
        ticker (str): The cryptocurrency ticker symbol.
        currency (str): The target currency for display.
        exchange_rate (float): The USD to target currency exchange rate.
    """
    forecast_days = len(future_forecast_usd)
    # Convert forecast to the target currency
    future_forecast_target_currency = np.array(future_forecast_usd) * exchange_rate

    plt.figure(figsize=(12, 6))
    plt.title(f'{forecast_days}-Day Price Forecast for {ticker}', fontsize=20)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel(f'Predicted Close Price ({currency})', fontsize=18)

    # Plot future forecast with markers
    plt.plot(future_dates, future_forecast_target_currency, label=f'{forecast_days}-Day Forecast', color='green', marker='o', linestyle='--')

    # Formatting the plot for better readability
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout() # Adjust layout to make room for rotated date labels
    plt.show()

def main():
    """
    Main function to run the crypto prediction script.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction using LSTM.')
    parser.add_argument('--symbol', type=str, default='BTC-USD', help='Ticker symbol for the cryptocurrency (e.g., BTC, ETH, METIS).')
    parser.add_argument('--currency', type=str, default='INR', help='The currency for price display (e.g., INR, EUR, JPY).')
    args = parser.parse_args()
    
    # --- Configuration ---
    cli_symbol = args.symbol.upper()
    if '-' not in cli_symbol:
        TICKER = f"{cli_symbol}-USD"
    else:
        TICKER = cli_symbol

    CURRENCY = args.currency.upper()
    TIME_STEP = 60      # Number of past days to use for prediction
    EPOCHS = 50
    BATCH_SIZE = 32
    FORECAST_DAYS = 15  # Number of days to forecast into the future

    try:
        # 0. Get Exchange Rate
        exchange_rate = get_exchange_rate(CURRENCY)

        # 1. Fetch Data
        raw_data = fetch_crypto_data(TICKER)

        # Add a check for sufficient data length
        min_data_length = TIME_STEP + 60 # Require at least ~4 months of data
        if len(raw_data) < min_data_length:
             raise ValueError(f"Not enough historical data for {TICKER} to perform a reliable prediction. "
                              f"Found only {len(raw_data)} days of data, but require at least {min_data_length}.")

        # 2. Clean and Preprocess Data
        scaled_data, scaler, cleaned_df = clean_and_preprocess_data(raw_data)

        # 3. Create Training and Test sets
        training_size = int(len(scaled_data) * 0.80)
        train_data, test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:1]

        X_train, y_train = create_dataset(train_data, TIME_STEP)
        X_test, y_test = create_dataset(test_data, TIME_STEP)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # 4. Build and Train LSTM Model
        model = build_lstm_model((X_train.shape[1], 1))
        
        print("Training model...")
        history = model.fit(X_train, y_train, 
                              validation_data=(X_test, y_test), 
                              epochs=EPOCHS, 
                              batch_size=BATCH_SIZE, 
                              verbose=1)

        # 5. Predict the NEXT N DAYS' price
        print(f"Forecasting the next {FORECAST_DAYS} days...")
        future_predictions_scaled = []
        # Get the last `time_step` days from the original scaled data and flatten it
        last_sequence_scaled = list(scaled_data[-TIME_STEP:].flatten())

        for _ in range(FORECAST_DAYS):
            x_input = np.array(last_sequence_scaled).reshape(1, TIME_STEP, 1)
            prediction = model.predict(x_input, verbose=0)
            future_predictions_scaled.append(prediction[0][0])
            # Update the sequence: remove the first element and add the new prediction
            last_sequence_scaled.pop(0)
            last_sequence_scaled.append(prediction[0][0])
        
        # Inverse transform the future predictions to get USD values
        future_predictions_usd = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

        # 6. Generate dates for the forecast
        last_date = cleaned_df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

        # 7. Visualize the forecast results
        plot_forecast(future_dates, future_predictions_usd, TICKER, CURRENCY, exchange_rate)
        
        # 8. Plot training loss
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()

        # 9. Display the date-wise forecast table
        future_predictions_target_currency = future_predictions_usd * exchange_rate
        
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            f'Predicted Price ({CURRENCY})': [f"{price:,.2f}" for price in future_predictions_target_currency.flatten()]
        })
        print("\n" + "="*45)
        print(f"      Date-wise {FORECAST_DAYS}-Day Price Forecast")
        print("="*45)
        print(forecast_df.to_string(index=False))
        print("="*45)

        # 10. Get prices for the summary table
        current_price_usd = cleaned_df['Close'].iloc[-1]
        # Use the price predicted for N days from now for the summary
        future_price_usd = future_predictions_usd[-1][0]

        # 11. Convert prices to the target currency
        current_price_target_currency = current_price_usd * exchange_rate
        future_price_target_currency = future_price_usd * exchange_rate

        # 12. Calculate profit/loss and generate signal based on the forecast
        profit_percentage = ((future_price_target_currency - current_price_target_currency) / current_price_target_currency) * 100
        
        if profit_percentage > 1.0: # Using a threshold for the N-day signal
            signal = "Buy"
        elif profit_percentage < -1.0:
            signal = "Sell"
        else:
            signal = "Hold"

        # 13. Display the final summary table
        summary_data = {
            'Ticker': TICKER,
            f'Current Price ({CURRENCY})': f"{current_price_target_currency:,.2f}",
            f'Predicted Price in {FORECAST_DAYS} Days ({CURRENCY})': f"{future_price_target_currency:,.2f}",
            f'{FORECAST_DAYS}-Day Profit/Loss (%)': f"{profit_percentage:.2f}%",
            'Signal': signal
        }
        summary_df = pd.DataFrame([summary_data])
        
        print("\n" + "="*65)
        print(f"           CRYPTO PREDICTION & {FORECAST_DAYS}-DAY FORECAST SUMMARY")
        print("="*65)
        print(summary_df.to_string(index=False))
        print("="*65)
        print("\nDisclaimer: This is a model-based prediction and not financial advice.\n")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
