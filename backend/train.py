import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os

# --- CONFIGURATION ---
TICKER = "AAPL"  # We train on Apple initially
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
PREDICTION_DAYS = 60  # Look back 60 days to predict tomorrow

def get_data():
    print(f"📉 Downloading {TICKER} data from Yahoo Finance...")
    data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    
    # We only care about the "Close" price
    return data[['Close']]

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    
    for i in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[i-PREDICTION_DAYS:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

def build_lstm_model(input_shape):
    """Deep LSTM Network for Time-Series"""
    model = Sequential([
        # Layer 1
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        # Layer 2
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        
        # Layer 3
        LSTM(units=50),
        Dropout(0.2),
        
        # Output Layer
        Dense(units=1)  # Prediction of the next closing price
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    # 1. Get Data
    df = get_data()
    print(f"📊 Loaded {len(df)} trading days.")
    
    # 2. Process
    x_train, y_train, scaler = prepare_data(df)
    
    # 3. Build Model
    model = build_lstm_model((x_train.shape[1], 1))
    
    # 4. Train
    print("🚀 Training Quantitative Model (LSTM)...")
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    
    # 5. Save
    model.save("stock_predictor.h5")

    print("✅ SUCCESS: Financial Model Saved as 'stock_predictor.h5'")