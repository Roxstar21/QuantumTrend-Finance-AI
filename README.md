# 📈 QuantumTrend Pro - AI & Technical Analysis Engine

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Finance](https://img.shields.io/badge/Strategy-Quant-green)

### 🔴 **Live Demo:** [Launch QuantumTrend Dashboard](https://quantamtrend.streamlit.app/OUR_LINK_HERE)

**QuantumTrend Pro** is an institutional-grade financial analytics dashboard. It combines Deep Learning (LSTM) for price forecasting with real-time Technical Analysis (RSI, SMA) to provide a 360-degree view of asset volatility.

## 🚀 v3.3 Capabilities
- **Neural Forecasting:** 60-day lookback LSTM model that predicts next-day price targets for any global asset.
- **Technical Indicators:** Real-time calculation of **RSI (14-day)** and **SMA (50-day)** to identify overbought/oversold conditions.
- **Institutional Charting:** Interactive Candlestick charts powered by Plotly for granular price action analysis.
- **Data Export:** Integrated CSV reporting engine for offline analysis.

## 🛠️ Tech Stack
- **AI Core:** TensorFlow/Keras (Sequential LSTM with Dropout Regularization)
- **Data Feed:** Yahoo Finance API (Real-time OHLCV data)
- **Visualization:** Streamlit, Plotly Graph Objects
- **Math:** NumPy, Pandas, Scikit-Learn

## 📸 Usage
1. **Launch:** Click the Live Demo link above.
2. **Search:** Enter any ticker (e.g., `BTC-USD`, `NVDA`, `GC=F`).
3. **Analyze:** View the "AI Insight Card" for immediate buy/sell signal context.
4. **Export:** Download raw market data via the "Raw Data" tab.

## 💿 Installation (Local)
```bash
git clone [https://github.com/Roxstar21/QuantumTrend-Finance-AI.git](https://github.com/Roxstar21/QuantumTrend-Finance-AI.git)
cd QuantumTrend-Finance-AI
pip install -r requirements.txt
streamlit run app_deploy.py