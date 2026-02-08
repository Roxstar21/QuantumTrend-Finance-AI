import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import datetime
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="QuantumTrend Pro", page_icon="💹", layout="wide")

# --- LOAD BRAIN (Cached for Speed) ---
@st.cache_resource
def load_brain():
    # We look for the model in the 'backend' folder
    model_path = os.path.join("backend", "stock_predictor.h5")
    if not os.path.exists(model_path):
        st.error(f"❌ MODEL NOT FOUND at {model_path}. Please check GitHub file structure.")
        return None
    return load_model(model_path)

model = load_brain()

# --- HELPER FUNCTIONS ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5305/5305052.png", width=70)
    st.title("QUANTUM DESK")
    st.markdown("`LIVE DEPLOYMENT`")
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    run_btn = st.button("RUN ANALYSIS", type="primary")

# --- MAIN LOGIC ---
st.title(f"💹 Market Intelligence // {ticker}")

if run_btn:
    if not model:
        st.warning("⚠️ AI Model is missing. Cannot predict.")
    else:
        with st.spinner(f"📡 DOWNLOADING LIVE DATA FOR {ticker}..."):
            try:
                # 1. Get Data
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=730)
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if len(data) < 60:
                    st.error("Not enough data points for this asset.")
                else:
                    # 2. AI Prediction
                    closing_prices = data[['Close']].values
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(closing_prices)
                    
                    last_60_days = scaled_data[-60:]
                    X_test = np.array([last_60_days])
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    
                    predicted_price = model.predict(X_test)
                    predicted_price = scaler.inverse_transform(predicted_price)
                    final_prediction = float(predicted_price[0][0])
                    
                    # 3. Technicals
                    data['SMA_50'] = data['Close'].rolling(window=50).mean()
                    data['RSI'] = calculate_rsi(data)
                    data = data.fillna(0)
                    
                    # 4. Display Metrics
                    current_price = float(closing_prices[-1][0])
                    change_percent = ((final_prediction - current_price) / current_price) * 100
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Price", f"${current_price:,.2f}")
                    c2.metric("AI Target", f"${final_prediction:,.2f}")
                    c3.metric("Forecast", f"{change_percent:+.2f}%")
                    
                    # 5. Charts
                    tab1, tab2 = st.tabs(["PRICE ACTION", "RSI"])
                    
                    with tab1:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Market'))
                        fig.add_trace(go.Scatter(x=[data.index[-1], "Forecast"], y=[current_price, final_prediction], mode='markers+lines', name='AI Vector', marker=dict(color='yellow', size=10)))
                        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with tab2:
                        st.line_chart(data['RSI'])

            except Exception as e:
                st.error(f"Error: {e}")