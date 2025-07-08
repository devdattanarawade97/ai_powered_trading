# main.py
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import argparse

def get_usd_to_inr_rate():
    """
    Fetches the latest USD to INR conversion rate from Yahoo Finance.
    Provides a fallback rate if the API call fails.
    """
    print("Fetching USD to INR conversion rate...")
    try:
        inr_rate_data = yf.Ticker("USDINR=X").history(period="1d")
        if inr_rate_data.empty or 'Close' not in inr_rate_data.columns:
            print("Could not fetch live INR rate, using a fallback value of 83.0")
            return 83.0  # Fallback value
        rate = inr_rate_data['Close'].iloc[-1]
        print(f"Current USD to INR rate: {rate:.2f}")
        return rate
    except Exception as e:
        print(f"Error fetching INR rate: {e}. Using a fallback value of 83.0")
        return 83.0  # Fallback value

def fetch_crypto_data(ticker='BTC-USD', start_date='2018-01-01', end_date=None):
    """
    Fetches historical cryptocurrency data from Yahoo Finance.

    Args:
        ticker (str): The ticker symbol for the cryptocurrency (e.g., 'BTC-USD').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        pandas.DataFrame: A DataFrame containing the historical data.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date or 'today'}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}. Please check the ticker symbol.")
    print("Data fetched successfully.")
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
    model = Sequential()
    
    # First LSTM layer with Dropout regularization
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Model built successfully.")
    model.summary()
    return model

def plot_predictions(results_df, ticker):
    """
    Plots the actual vs. predicted prices from the results DataFrame.

    Args:
        results_df (pandas.DataFrame): DataFrame with actual and predicted prices.
        ticker (str): The cryptocurrency ticker symbol.
    """
    plt.figure(figsize=(16, 8))
    plt.title(f'Future Price Prediction for {ticker} using LSTM')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price INR (₹)', fontsize=18)
    plt.plot(results_df.index, results_df['Actual Price (INR)'], label='Actual Price')
    plt.plot(results_df.index, results_df['Predicted Price (INR)'], label='Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_and_display_results(actual_prices, predicted_prices, dates, ticker):
    """
    Analyzes prediction results, displays them in a table, and plots them.

    Args:
        actual_prices (numpy.array): Array of actual prices in INR.
        predicted_prices (numpy.array): Array of predicted prices in INR.
        dates (pandas.Index): Dates corresponding to the prices.
        ticker (str): The cryptocurrency ticker symbol.
    """
    # Create a DataFrame for easy analysis
    results_df = pd.DataFrame({
        'Actual Price (INR)': actual_prices.flatten(),
        'Predicted Price (INR)': predicted_prices.flatten()
    }, index=dates)

    # Calculate Profit Percentage
    results_df['Profit Pct'] = ((results_df['Predicted Price (INR)'] - results_df['Actual Price (INR)']) / results_df['Actual Price (INR)']) * 100

    # Generate Buy/Sell Signal
    # Simple strategy: If predicted is higher, signal is Buy, else Sell.
    results_df['Signal'] = np.where(results_df['Predicted Price (INR)'] > results_df['Actual Price (INR)'], 'Buy', 'Sell')

    print("\n--- Prediction Analysis (in INR) ---")
    print(f"Showing last 10 days of predictions for {ticker}:")
    # Format the output for better readability
    results_df_display = results_df.copy()
    results_df_display['Actual Price (INR)'] = results_df_display['Actual Price (INR)'].map('₹{:,.2f}'.format)
    results_df_display['Predicted Price (INR)'] = results_df_display['Predicted Price (INR)'].map('₹{:,.2f}'.format)
    results_df_display['Profit Pct'] = results_df_display['Profit Pct'].map('{:.2f}%'.format)
    print(results_df_display.tail(10).to_string())

    # Call the plotting function with the original numeric data
    plot_predictions(results_df, ticker)


def main():
    """
    Main function to run the crypto prediction script.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction using LSTM.')
    parser.add_argument('--symbol', type=str, default='BTC-USD', help='Ticker symbol for the cryptocurrency (e.g., BTC, ETH, METIS). Prices will be shown in INR.')
    args = parser.parse_args()
    
    # --- Configuration ---
    cli_symbol = args.symbol.upper()
    # Append '-USD' if the ticker doesn't already contain a pair separator
    if '-' not in cli_symbol:
        TICKER = f"{cli_symbol}-USD"
    else:
        TICKER = cli_symbol

    START_DATE = '2019-01-01'
    TIME_STEP = 60      # Number of past days to use for prediction
    EPOCHS = 50
    BATCH_SIZE = 32

    try:
        # Get INR conversion rate
        inr_rate = get_usd_to_inr_rate()
        
        # 1. Fetch Data
        raw_data = fetch_crypto_data(TICKER, start_date=START_DATE)
        
        # --- Convert prices to INR ---
        print("Converting prices to INR...")
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_cols:
            if col in raw_data.columns:
                raw_data[col] = raw_data[col] * inr_rate

        # 2. Clean and Preprocess Data
        scaled_data, scaler, cleaned_df = clean_and_preprocess_data(raw_data)

        # 3. Create Training and Test sets
        training_size = int(len(scaled_data) * 0.80)
        train_data, test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:1]

        X_train, y_train = create_dataset(train_data, TIME_STEP)
        X_test, y_test = create_dataset(test_data, TIME_STEP)

        # Reshape input to be [samples, time steps, features] which is required for LSTM
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

        # 5. Make Predictions
        print("Making predictions...")
        test_predict = model.predict(X_test)

        # 6. Inverse Transform Predictions to Original Scale (which is now INR)
        test_predict = scaler.inverse_transform(test_predict)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # 7. Analyze, display, and visualize the results
        # We need the original dates for the results
        test_data_start_index = training_size + TIME_STEP + 1
        test_dates = cleaned_df.index[test_data_start_index:]
        
        analyze_and_display_results(y_test_actual, test_predict, test_dates, TICKER)
        
        # Plot training loss
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
