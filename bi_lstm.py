# BiLSTM-Based Crypto Price Forecasting Script (No News Sentiment)

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Download crypto historical data
def fetch_price_data(symbol="BTC-USD", start="2021-01-01", end="2025-01-01"):
    data = yf.download(symbol, start=start, end=end)
    return data[["Close"]]

# 2. Preprocess data (no sentiment input)
def preprocess_data(data, window_size=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        sequence = scaled[i - window_size:i, 0]
        X.append(sequence)
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# 3. Build BiLSTM model
def build_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 4. Train the model
def train_model(X, y):
    model = build_bilstm_model((X.shape[1], 1))
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, callbacks=[es])
    return model

# 5. Forecast next price
def forecast(model, X_last, scaler):
    pred_scaled = model.predict(X_last)
    pred = scaler.inverse_transform(pred_scaled)
    return pred

# 6. Run the workflow
if __name__ == "__main__":
    df = fetch_price_data("BTC-USD")
    X, y, scaler = preprocess_data(df)
    model = train_model(X, y)
    last_input = X[-1].reshape(1, X.shape[1], 1)
    prediction = forecast(model, last_input, scaler)
    print(f"Predicted next close price: ${prediction[0][0]:.2f}")
