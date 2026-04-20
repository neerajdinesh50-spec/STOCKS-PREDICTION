# app.py — Streamlit Interactive Dashboard
# ---------------------------------------------------------------
# Run with:
#   streamlit run app.py
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import download_stock_data, load_stock_data
from preprocessing import preprocess, create_sequences, get_lr_features, LOOKBACK
from models.lstm_model import load_lstm, predict_lstm, build_lstm, train_lstm, save_lstm
from models.linear_model import (
    train_linear_regression, train_random_forest,
    evaluate_model, save_model
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

plt.style.use("seaborn-v0_8-darkgrid")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-card {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      border-radius: 12px; padding: 16px 20px;
      border-left: 4px solid #4E9AF1;
      margin-bottom: 8px;
  }
  .metric-card h3 { color: #8899aa; font-size: 12px; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
  .metric-card p  { color: #ffffff; font-size: 26px; font-weight: 700; margin: 4px 0 0 0; }
  .section-title  { color: #4E9AF1; font-weight: 600; font-size: 18px; margin-top: 12px; }
  .disclaimer     { background: #2d1b1b; border-left: 4px solid #FF6B6B;
                    border-radius: 8px; padding: 12px 16px; color: #ffaaaa;
                    font-size: 13px; margin-top: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

popular_tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "Custom (Write Ticker)"]
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", popular_tickers)

if selected_ticker == "Custom (Write Ticker)":
    ticker = st.sidebar.text_input("Write Ticker Symbol", value="AAPL").upper()
else:
    ticker = selected_ticker

period  = st.sidebar.selectbox("Historical Period", ["1y", "2y", "5y", "max"], index=2)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
run_btn = st.sidebar.button("🚀 Fetch & Train", use_container_width=True)

# ── Main Header ───────────────────────────────────────────────
st.markdown("""
<h1 style='color:#4E9AF1; margin-bottom:4px;'>📈 Stock Market Price Predictor</h1>
<p style='color:#8899aa; margin-top:0;'>ML + Deep Learning — Linear Regression · Random Forest · LSTM</p>
<hr style='border-color:#1e3a5f;'>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MDL_DIR  = os.path.join(os.path.dirname(__file__), "models")

# ── Session State ─────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False

@st.cache_data(show_spinner="Downloading stock data...")
def get_data(tkr, prd):
    return download_stock_data(tkr, period=prd, save_dir=DATA_DIR)

if run_btn:
    st.session_state.trained = False

    with st.spinner(f"Fetching {ticker} data..."):
        df = get_data(ticker, period)

    # ── EDA Section ───────────────────────────────────────────
    st.markdown('<p class="section-title">📊 Price History & Moving Averages</p>', unsafe_allow_html=True)

    df_plot = df.copy()
    df_plot["MA50"]  = df_plot["Close"].rolling(50).mean()
    df_plot["MA100"] = df_plot["Close"].rolling(100).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(df_plot.index, df_plot["Close"], color="#4E9AF1",  lw=1.5,  label="Close")
    axes[0].plot(df_plot.index, df_plot["MA50"],  color="#FF6B6B",  lw=2,    label="MA50")
    axes[0].plot(df_plot.index, df_plot["MA100"], color="#FFD93D",  lw=2,    label="MA100")
    axes[0].set_title(f"{ticker} — Closing Price & Moving Averages", fontweight="bold")
    axes[0].legend(); axes[0].set_ylabel("Price (USD)")
    axes[1].bar(df_plot.index, df_plot["Volume"], color="#6BCB77", alpha=0.7, width=1)
    axes[1].set_ylabel("Volume")
    plt.tight_layout()
    st.pyplot(fig)

    # ── Quick Stats ───────────────────────────────────────────
    last  = float(df["Close"].iloc[-1])
    high  = float(df["High"].max())
    low   = float(df["Low"].min())
    avg   = float(df["Close"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close",  f"${last:.2f}")
    c2.metric("5Y High",     f"${high:.2f}")
    c3.metric("5Y Low",      f"${low:.2f}")
    c4.metric("5Y Avg Close",f"${avg:.2f}")

    # ── Training ──────────────────────────────────────────────
    st.markdown('<p class="section-title">🤖 Training Models</p>', unsafe_allow_html=True)
    prog = st.progress(0, text="Preprocessing...")

    train_data, test_data, scaler, series = preprocess(df, "Close")
    total_scaled = np.concatenate([train_data, test_data])
    test_input   = total_scaled[len(train_data) - LOOKBACK:]
    X_train_lstm, y_train_lstm = create_sequences(train_data, LOOKBACK)
    X_test_lstm,  y_test_lstm  = create_sequences(test_input,  LOOKBACK)

    prog.progress(20, "Training LSTM...")
    lstm_model = build_lstm((LOOKBACK, 1))
    history    = train_lstm(lstm_model, X_train_lstm, y_train_lstm, epochs=30, batch_size=32)
    save_lstm(lstm_model, save_dir=MDL_DIR)
    lstm_preds  = predict_lstm(lstm_model, X_test_lstm, scaler)
    lstm_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).ravel()

    prog.progress(60, "Training Linear Regression & Random Forest...")
    X_tr, X_te, y_tr, y_te, lr_scaler, raw = get_lr_features(df, "Close")
    lr_model = train_linear_regression(X_tr, y_tr)
    rf_model = train_random_forest(X_tr, y_tr)
    save_model(lr_model, "linear_regression", MDL_DIR)
    save_model(rf_model, "random_forest",     MDL_DIR)

    lr_res = evaluate_model(lr_model, X_te, y_te, lr_scaler, "Linear Regression")
    rf_res = evaluate_model(rf_model, X_te, y_te, lr_scaler, "Random Forest")

    prog.progress(100, "Done!")

    # ── Loss Plot ─────────────────────────────────────────────
    st.markdown('<p class="section-title">📉 LSTM Training Loss</p>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(history.history["loss"],     label="Train Loss", color="#4E9AF1", lw=2)
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#FF6B6B", lw=2, ls="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE"); ax2.legend()
    ax2.set_title("LSTM — Training vs Validation Loss", fontweight="bold")
    st.pyplot(fig2)

    # ── Prediction Comparison Plot ────────────────────────────
    st.markdown('<p class="section-title">🔮 Actual vs Predicted (Test Set)</p>', unsafe_allow_html=True)
    min_len = min(len(lr_res["predictions"]), len(lstm_preds))
    x_ax = np.arange(min_len)

    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plot_data = [
        ("Linear Regression", lr_res["predictions"][-min_len:], lr_res["actual"][-min_len:], "#FF6B6B"),
        ("Random Forest",     rf_res["predictions"][-min_len:], rf_res["actual"][-min_len:], "#FFD93D"),
        ("LSTM",              lstm_preds[-min_len:],             lstm_actual[-min_len:],      "#6BCB77"),
    ]
    for ax, (name, pred, act, col) in zip(axes3, plot_data):
        ax.plot(x_ax, act,  color="#AAAAAA", lw=1.5, label="Actual")
        ax.plot(x_ax, pred, color=col,        lw=1.5, label=f"Predicted — {name}")
        rmse = np.sqrt(mean_squared_error(act, pred))
        mae  = mean_absolute_error(act, pred)
        ax.set_title(f"{name}  |  RMSE: ${rmse:.2f}   MAE: ${mae:.2f}", fontweight="bold")
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig3)

    # ── Metrics Table ─────────────────────────────────────────
    st.markdown('<p class="section-title">📋 Model Comparison</p>', unsafe_allow_html=True)
    lstm_rmse = np.sqrt(mean_squared_error(lstm_actual, lstm_preds))
    lstm_mae  = mean_absolute_error(lstm_actual, lstm_preds)
    results_df = pd.DataFrame({
        "Model":  ["Linear Regression", "Random Forest", "LSTM"],
        "RMSE ($)": [f"{lr_res['rmse']:.2f}", f"{rf_res['rmse']:.2f}", f"{lstm_rmse:.2f}"],
        "MAE ($)":  [f"{lr_res['mae']:.2f}",  f"{rf_res['mae']:.2f}",  f"{lstm_mae:.2f}"],
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # ── Next-Day Prediction ─────────────────────────────────
    st.markdown('<p class="section-title">🔭 Next-Day Prediction (LSTM)</p>', unsafe_allow_html=True)
    sc2 = MinMaxScaler(feature_range=(0, 1))
    sc2.fit(df["Close"].dropna().values.reshape(-1, 1))
    last_seq = sc2.transform(df["Close"].dropna().values[-LOOKBACK:].reshape(-1, 1))
    X_next   = last_seq.reshape(1, LOOKBACK, 1)
    pred_next = float(sc2.inverse_transform(lstm_model.predict(X_next))[0][0])
    change    = pred_next - last
    pct       = (change / last) * 100
    direction = "📈" if change > 0 else "📉"

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close",        f"${last:.2f}")
    col2.metric("Predicted Next Day", f"${pred_next:.2f}", delta=f"{change:+.2f} ({pct:+.2f}%)")
    col3.metric("Signal",            f"{direction} {'Bullish' if change > 0 else 'Bearish'}")

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This tool is for <strong>educational purposes only</strong>.
    Stock markets are influenced by countless unpredictable factors (news, geopolitics, earnings, etc.)
    that machine learning models cannot fully capture. <strong>Do NOT use this for real trading decisions.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.trained = True

else:
    if not st.session_state.trained:
        st.info("👈 Enter a stock ticker in the sidebar and click **Fetch & Train** to begin.")
        st.markdown("""
        ### How It Works
        | Step | Description |
        |------|-------------|
        | 1️⃣ Data | Download 5 years of OHLCV data via `yfinance` |
        | 2️⃣ EDA  | Visualize price trends, moving averages, volume |
        | 3️⃣ Train | Train Linear Regression, Random Forest, and LSTM |
        | 4️⃣ Evaluate | Compare RMSE & MAE across models |
        | 5️⃣ Predict | Forecast next trading day's closing price |

        ### Limitations
        - Models use only **price history** (no news, sentiment, fundamentals)
        - Past performance **does not** predict future markets
        - LSTM requires ~1 minute to train on CPU
        """)
