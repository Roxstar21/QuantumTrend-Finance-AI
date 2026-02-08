from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

# --- CONFIGURATION ---
MODEL_PATH = "stock_predictor.h5"
try:
    model = load_model(MODEL_PATH)
    print("✅ Quant Brain Loaded.")
except:
    print("❌ ERROR: Model not found. Run train.py first.")

app = FastAPI(title="QuantumTrend API v3.0")

class StockRequest(BaseModel):
    ticker: str

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index (Real Trading Math)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@app.post("/forecast")
def predict_stock(request: StockRequest):
    ticker = request.ticker.upper()
    
    # 1. Get 2 Years of Data (Better for Moving Averages)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=730)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if len(data) < 60:
        return {"error": "Not enough data points."}
    
    # 2. AI Prediction Logic (The Brain)
    closing_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)
    
    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    final_prediction = float(predicted_price[0][0])
    
    # 3. Calculate Technical Indicators (The "Working" Features)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data)
    
    # 4. Statistics
    current_price = float(closing_prices[-1][0])
    change_percent = ((final_prediction - current_price) / current_price) * 100
    
    # Get recent data for charts (last 100 days)
    chart_data = data.tail(100).reset_index()
    chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')
    
    # Handle NaN values in indicators
    chart_data = chart_data.fillna(0)

    return {
        "ticker": ticker,
        "current_price": current_price,
        "prediction": final_prediction,
        "change_percent": change_percent,
        "market_cap": 0, # yfinance often requires a separate call for this, strictly focusing on price action here
        "volume": int(chart_data['Volume'].iloc[-1]),
        "dates": chart_data['Date'].tolist(),
        "open": chart_data['Open'].values.flatten().tolist(),
        "high": chart_data['High'].values.flatten().tolist(),
        "low": chart_data['Low'].values.flatten().tolist(),
        "close": chart_data['Close'].values.flatten().tolist(),
        "rsi": chart_data['RSI'].values.flatten().tolist(),
        "sma": chart_data['SMA_50'].values.flatten().tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)