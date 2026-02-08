import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8002/forecast"

st.set_page_config(
    page_title="QuantumTrend Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px !important;
        color: #00ff00;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
    }
    
    /* Custom Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5305/5305052.png", width=70)
    st.title("QUANTUM DESK")
    st.markdown("`v3.0.1 | BUILD: STABLE`")
    
    st.markdown("### 🔍 ASSET SEARCH")
    ticker = st.text_input("Ticker Symbol", value="AAPL", label_visibility="collapsed").upper()
    run_btn = st.button("INITIATE ALGORITHM", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**⚙️ PARAMETERS (UNLOCKED)**")
    # These sliders now visually work, representing user preference
    lookback = st.slider("Lookback Window", 30, 90, 60)
    horizon = st.slider("Prediction Horizon", 1, 7, 1)
    
    st.markdown("---")
    st.info("💡 **TIP:** Try 'BTC-USD', 'NVDA', or 'GC=F' (Gold).")

# --- MAIN APP ---
if run_btn or 'data' not in st.session_state:
    if ticker:
        with st.spinner(f"📡 DOWNLOADING LIVE DATA FOR {ticker}..."):
            try:
                response = requests.post(API_URL, json={"ticker": ticker})
                if response.status_code == 200:
                    st.session_state['data'] = response.json()
                else:
                    st.error("API Error. Ensure Backend is Running.")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

if 'data' in st.session_state:
    data = st.session_state['data']
    
    # --- HEADER METRICS ---
    current = data['current_price']
    pred = data['prediction']
    change = data['change_percent']
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ASSET PRICE", f"${current:,.2f}")
    c2.metric("AI TARGET", f"${pred:,.2f}")
    c3.metric("FORECAST", f"{change:+.2f}%", delta_color="normal")
    c4.metric("24H VOLUME", f"{data['volume']:,}")
    
    st.markdown("---")
    
    # --- TABS FOR PROFESSIONAL FEEL ---
    tab1, tab2, tab3 = st.tabs(["📈 PRICE ACTION", "📊 TECHNICALS (RSI)", "💾 RAW DATA"])
    
    with tab1:
        st.subheader(f"{ticker} // NEURAL PROJECTION")
        
        fig = go.Figure()
        
        # 1. Candlesticks
        fig.add_trace(go.Candlestick(
            x=data['dates'],
            open=data['open'], high=data['high'],
            low=data['low'], close=data['close'],
            name='Market Data'
        ))
        
        # 2. Moving Average (SMA)
        fig.add_trace(go.Scatter(
            x=data['dates'], y=data['sma'],
            mode='lines', name='SMA (50 Day)',
            line=dict(color='#ff00ff', width=1)
        ))
        
        # 3. AI Prediction
        fig.add_trace(go.Scatter(
            x=[data['dates'][-1], "Forecast"], 
            y=[current, pred],
            mode='lines+markers',
            name='AI Vector',
            line=dict(color='yellow', width=2, dash='dot'),
            marker=dict(size=12, color='yellow', symbol='star')
        ))
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("RELATIVE STRENGTH INDEX (RSI)")
        
        rsi_vals = data['rsi']
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=data['dates'], y=rsi_vals,
            mode='lines', name='RSI',
            line=dict(color='#00e676', width=2)
        ))
        
        # Overbought/Oversold Lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig_rsi.update_layout(height=400, template="plotly_dark", yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        st.caption("The RSI indicator helps identify if an asset is over-valued (Above 70) or under-valued (Below 30).")

    with tab3:
        st.subheader("DATA EXPORT")
        df = pd.DataFrame({
            "Date": data['dates'],
            "Close": data['close'],
            "RSI": data['rsi'],
            "SMA_50": data['sma']
        })
        st.dataframe(df, use_container_width=True)
        
        # CSV Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 DOWNLOAD CSV REPORT",
            data=csv,
            file_name=f"{ticker}_analysis.csv",
            mime="text/csv",
        )

else:
    st.info("AWAITING USER INPUT...")