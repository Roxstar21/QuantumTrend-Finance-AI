import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import datetime
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QuantumTrend Pro",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PRO CSS (Dark Mode & Metrics) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px !important;
    }
    .prediction-card {
        background-color: #1c2029;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD AI MODEL ---
@st.cache_resource
def load_brain():
    # Check all possible locations for the model
    paths = ["stock_predictor.h5", "backend/stock_predictor.h5"]
    for p in paths:
        if os.path.exists(p):
            return load_model(p)
    return None

try:
    model = load_brain()
except Exception as e:
    st.error(f"Error loading AI: {e}")
    model = None

# --- TECHNICAL ANALYSIS FUNCTIONS ---
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
    st.markdown("`v3.3 | FULL SUITE`")
    
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    run_btn = st.button("INITIATE ALGORITHM", type="primary")
    
    st.markdown("---")
    st.slider("Lookback Window", 30, 90, 60)
    
    st.info("💡 **TIP:** Try 'BTC-USD', 'NVDA', or 'GC=F' (Gold).")

# --- MAIN APP ---
st.title(f"💹 Market Intelligence // {ticker}")

if run_btn:
    if model is None:
        st.error("⚠️ AI Model not found. Check GitHub repo.")
    else:
        with st.spinner(f"📡 DOWNLOADING LIVE DATA FOR {ticker}..."):
            try:
                # 1. Get Data
                end = datetime.datetime.now()
                start = end - datetime.timedelta(days=730)
                data = yf.download(ticker, start=start, end=end, progress=False)
                
                # --- CRITICAL FIX FOR YFINANCE BUG ---
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                # -------------------------------------

                if len(data) < 60:
                    st.error("Not enough data history for this asset.")
                else:
                    # 2. AI Prediction Logic
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(data[['Close']].values)
                    
                    x_input = scaled_data[-60:].reshape(1, 60, 1)
                    prediction = model.predict(x_input)
                    price = float(scaler.inverse_transform(prediction)[0][0])
                    
                    # 3. Technical Calculations
                    data['SMA_50'] = data['Close'].rolling(window=50).mean()
                    data['RSI'] = calculate_rsi(data)
                    data = data.fillna(0)
                    
                    # 4. Metrics
                    current = float(data['Close'].iloc[-1])
                    change = ((price - current)/current)*100
                    volume = int(data['Volume'].iloc[-1])
                    
                    if change > 0:
                        trend_color = "#00ff00"
                        trend_msg = "🚀 BULLISH SIGNAL"
                    else:
                        trend_color = "#ff2b2b"
                        trend_msg = "🔻 BEARISH SIGNAL"

                    # 5. Display Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("ASSET PRICE", f"${current:,.2f}")
                    c2.metric("AI TARGET", f"${price:,.2f}")
                    c3.metric("FORECAST", f"{change:+.2f}%")
                    c4.metric("24H VOLUME", f"{volume:,}")
                    
                    # 6. Insight Card
                    st.markdown(f"""
                    <div class="prediction-card" style="border-color: {trend_color};">
                        <h3 style="color: {trend_color}; margin:0;">{trend_msg}</h3>
                        <p style="color: #ccc; margin-top: 5px;">
                            Neural Network detects a move to <b>${price:.2f}</b>. 
                            RSI is currently <b>{data['RSI'].iloc[-1]:.2f}</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 7. PROFESSIONAL TABS (Price, Technicals, DATA TABLE)
                    tab1, tab2, tab3 = st.tabs(["📈 PRICE ACTION", "📊 TECHNICALS", "💾 RAW DATA"])
                    
                    with tab1:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Market'))
                        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA (50)', line=dict(color='#ff00ff', width=1)))
                        fig.add_trace(go.Scatter(x=[data.index[-1], "Forecast"], y=[current, price], mode='lines+markers', name='AI Vector', line=dict(color='yellow', dash='dot')))
                        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='#00e676')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        fig_rsi.update_layout(height=400, template="plotly_dark", yaxis_range=[0, 100])
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    with tab3:
                        st.markdown("### 📥 INSTITUTIONAL DATA EXPORT")
                        # Show last 100 days
                        export_df = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_50']].tail(100)
                        
                        # Interactive Table
                        st.dataframe(export_df.style.format("{:.2f}"), use_container_width=True)
                        
                        # Download Button
                        csv = export_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="DOWNLOAD CSV REPORT",
                            data=csv,
                            file_name=f"{ticker}_QUANTUM_REPORT.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Analysis Failed: {e}")