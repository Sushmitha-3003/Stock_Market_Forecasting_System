# app.py — Stock Dashboard (minimal, updated as per requirements)
import os
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed", "combined_features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SEQUENCE_LENGTH = 60

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "Bollinger_Upper", "Bollinger_Middle", "Bollinger_Lower",
    "sentiment_score",
]

# -----------------------------
# Cached Loaders
# -----------------------------
@st.cache_data
def load_features():
    df = pd.read_csv(PROCESSED_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    return df

@st.cache_resource
def load_model_and_scaler(ticker: str):
    model_path = os.path.join(MODELS_DIR, f"stock_price_lstm_model_{ticker}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"stock_price_scaler_{ticker}.joblib")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None
    model = load_model(model_path) if load_model else None
    scaler = joblib.load(scaler_path)
    return model, scaler

# -----------------------------
# Helpers
# -----------------------------
def get_last_sequence(df_all, ticker):
    sub = df_all[df_all["ticker"] == ticker].sort_values("date")
    if len(sub) < SEQUENCE_LENGTH:
        return None, None
    X = sub.tail(SEQUENCE_LENGTH)[FEATURE_COLS].values
    return sub.tail(SEQUENCE_LENGTH), X

def iterative_forecast(model, scaler, X_seq, days):
    close_idx = FEATURE_COLS.index("Close")
    preds = []
    current = X_seq.copy()
    for _ in range(days):
        scaled = scaler.transform(current)
        y_scaled = model.predict(scaled.reshape(1, *scaled.shape), verbose=0)[0][0]
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, close_idx] = y_scaled
        y = float(scaler.inverse_transform(dummy)[0, close_idx])
        preds.append(y)
        new_row = current[-1].copy()
        new_row[close_idx] = y
        current = np.vstack([current[1:], new_row])
    return preds

def historical_one_step_predictions(df_ticker, model, scaler):
    preds = []
    close_idx = FEATURE_COLS.index("Close")
    values = df_ticker[FEATURE_COLS].values
    for i in range(SEQUENCE_LENGTH, len(df_ticker)):
        X_hist = values[i-SEQUENCE_LENGTH:i]
        scaled_hist = scaler.transform(X_hist)
        pred_scaled = model.predict(scaled_hist.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLS)), verbose=0)[0][0]
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, close_idx] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0, close_idx]
        preds.append(pred_price)
    dates = df_ticker["date"].iloc[SEQUENCE_LENGTH:]
    return pd.Series(preds, index=dates, name="model_pred")

# -----------------------------
# Pages
# -----------------------------
def page_prediction(df_all):
    st.title("📊 Stock Prediction")

    ticker = st.session_state.get("selected_ticker", DEFAULT_TICKERS[0])
    st.subheader(f"Analysis for {ticker}")

    days = st.sidebar.slider("Prediction Days", 1, 30, 7)

    if st.sidebar.button("Run Prediction"):
        model, scaler = load_model_and_scaler(ticker)
        if not model:
            st.error(f"Model/scaler not found for {ticker}.")
            return

        seq_df, X_seq = get_last_sequence(df_all, ticker)
        if X_seq is None:
            st.error("Not enough data for this ticker.")
            return

        preds_future = iterative_forecast(model, scaler, X_seq, days)
        current_price = seq_df["Close"].iloc[-1]
        latest_sentiment = df_all[df_all["ticker"] == ticker].sort_values("date")["sentiment_score"].iloc[-1]
        hist_df = df_all[df_all["ticker"] == ticker].sort_values("date").copy()
        hist_pred_series = historical_one_step_predictions(hist_df, model, scaler)

        st.session_state["prediction_results"] = {
            "ticker": ticker, "days": days,
            "preds_future": preds_future,
            "current_price": current_price,
            "latest_sentiment": latest_sentiment,
            "hist_df": hist_df,
            "hist_pred_series": hist_pred_series,
            "seq_df": seq_df
        }

    if "prediction_results" in st.session_state:
        res = st.session_state["prediction_results"]
        if res["ticker"] == ticker:
            preds_future = res["preds_future"]
            current_price = res["current_price"]
            latest_sentiment = res["latest_sentiment"]
            hist_df = res["hist_df"]
            hist_pred_series = res["hist_pred_series"]
            seq_df = res["seq_df"]
            days = res["days"]

            # Sentiment arrow
            sentiment_arrow = "⬆️" if latest_sentiment > 0 else "⬇️" if latest_sentiment < 0 else "➡️"

            # Recommendation logic + color
            price_change = (preds_future[-1] - current_price) / current_price
            if price_change > 0.01:
                recommendation = "BUY"; rec_color = "green"
            elif price_change < -0.01:
                recommendation = "SELL"; rec_color = "red"
            else:
                recommendation = "HOLD"; rec_color = "blue"

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            prev_close = hist_df["Close"].iloc[-2] if len(hist_df) >= 2 else current_price
            col1.metric("Current Price", f"${current_price:.2f}", delta=f"{current_price - prev_close:.2f}")
            col2.metric(f"Predicted Price (+{days}d)", f"${preds_future[-1]:.2f}", delta=f"{preds_future[-1] - current_price:.2f}")
            col3.markdown(
                f"<div style='text-align:center;'><span style='background-color:{rec_color};"
                f"color:white;padding:5px 12px;border-radius:5px;font-weight:bold;'>{recommendation}</span></div>",
                unsafe_allow_html=True
            )
            col4.metric("Sentiment Score", f"{sentiment_arrow} {latest_sentiment:.2f}")

            # Forecast dates
            future_dates = pd.date_range(start=seq_df["date"].iloc[-1] + timedelta(days=1), periods=days)

            # Graph with month-level ticks
            # Graph with interactive zoom/pan
# Clean interactive prediction graph
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=hist_df["date"], y=hist_df["Close"], mode='lines+markers',
                name='Actual Price', line=dict(color='blue', width=1.5), marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=hist_pred_series.index, y=hist_pred_series.values, mode='lines+markers',
                name='Model Predicted', line=dict(color='red', width=1.5), marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=preds_future, mode='lines+markers',
                name='Forecast', line=dict(color='green', width=2, dash='dash'), marker=dict(size=6)
            ))

            fig.update_layout(
                title=f"Actual vs Predicted vs Forecast — {ticker}",
                xaxis_title="Date", yaxis_title="Price",
                template="plotly_white",
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date",
                    tickformat="%b %Y"
                ),
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)





def page_news(df_all):
    st.header("📰 Latest News & Sentiment")
    ticker = st.session_state.get("selected_ticker", DEFAULT_TICKERS[0])

    per_ticker_path = os.path.join(DATA_DIR, "raw", f"{ticker}_news.csv")
    if not os.path.exists(per_ticker_path):
        st.warning(f"No news file found for {ticker}.")
        return

    df_news = pd.read_csv(per_ticker_path)
    if "date" in df_news.columns:
        df_news["date"] = pd.to_datetime(df_news["date"]).dt.date
    elif "publishedAt" in df_news.columns:
        df_news["date"] = pd.to_datetime(df_news["publishedAt"]).dt.date
    else:
        df_news["date"] = pd.to_datetime("today").date()

    df_news = df_news.sort_values("date", ascending=False).head(5)

    if "headline" in df_news.columns:
        df_news["text"] = df_news["headline"].fillna("")
    else:
        df_news["text"] = df_news["title"].fillna("")

    # 🔧 Fix: Auto sentiment if label not present
    if "label" not in df_news.columns:
        def auto_label(txt):
            txt = str(txt).lower()
            if "gain" in txt or "rise" in txt or "up" in txt or "positive" in txt:
                return "positive"
            elif "fall" in txt or "down" in txt or "loss" in txt or "negative" in txt:
                return "negative"
            return "neutral"
        df_news["label"] = df_news["text"].apply(auto_label)

    for _, row in df_news.iterrows():
        sentiment = row['label'].lower()
        color = "green" if "pos" in sentiment else "red" if "neg" in sentiment else "blue"
        st.markdown(f"<div style='padding:10px;border-radius:6px;margin-bottom:10px;background-color:{color};color:white;font-size:18px;font-weight:600;'>{row['text']} ({row['label']})</div>", unsafe_allow_html=True)

def page_visualization(df_all):
    st.title("📈 Advanced Visualizations")
    ticker = st.session_state.get("selected_ticker", DEFAULT_TICKERS[0])
    df_ticker = df_all[df_all["ticker"] == ticker].sort_values("date")

    for window in [10, 20, 50]:
        df_ticker[f"SMA_{window}"] = df_ticker["Close"].rolling(window).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ticker["date"], y=df_ticker["Close"], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df_ticker["date"], y=df_ticker[f"SMA_{window}"], mode='lines', name=f"SMA {window}"))
        fig.update_layout(title=f"{ticker} {window}-Day Moving Average", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # RSI chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_ticker["date"], y=df_ticker["RSI"], mode='lines', line=dict(color="purple")))
    fig_rsi.add_hline(y=70, line_color="red", line_dash="dash", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_color="green", line_dash="dash", annotation_text="Oversold")
    fig_rsi.update_layout(title=f"RSI for {ticker}", template="plotly_white")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Price + Volume chart
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df_ticker["date"], y=df_ticker["Close"], mode='lines', name="Price"))
    fig_vol.add_trace(go.Bar(x=df_ticker["date"], y=df_ticker["Volume"], name="Volume", opacity=0.3, yaxis="y2"))
    fig_vol.update_layout(
        title=f"Price and Volume for {ticker}",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        template="plotly_white"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

def page_report(df_all):
    st.title("📑 Report")

    ticker = st.session_state.get("selected_ticker", DEFAULT_TICKERS[0])
    st.subheader(f"Last 30 Days Report for {ticker}")

    # Filter last 30 days
    df_ticker = df_all[df_all["ticker"] == ticker].sort_values("date")
    last_30 = df_ticker.tail(30).copy()

    # Add model predictions
    model, scaler = load_model_and_scaler(ticker)
    if not model or last_30.empty:
        st.warning("No data/model available for this ticker.")
        return

    hist_pred_series = historical_one_step_predictions(df_ticker, model, scaler)
    last_30 = last_30.merge(hist_pred_series.rename("Predicted"), left_on="date", right_index=True, how="left")

    # Display table
    st.dataframe(last_30[["date", "Close", "Predicted"]].rename(columns={
        "date": "Date", "Close": "Actual Price", "Predicted": "Predicted Price"
    }), use_container_width=True)

    # Download option
    csv = last_30[["date", "Close", "Predicted"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{ticker}_last30_report.csv",
        mime="text/csv"
    )

# -----------------------------
# App Layout
# -----------------------------
st.sidebar.title("Navigation")
try:
    df_all = load_features()
except Exception as e:
    st.error(f"Data error: {e}")
    df_all = pd.DataFrame()

TICKER_OPTIONS = sorted(df_all["ticker"].unique()) if not df_all.empty else DEFAULT_TICKERS
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = TICKER_OPTIONS[0]

st.sidebar.selectbox("Select Company", options=TICKER_OPTIONS,
                     index=TICKER_OPTIONS.index(st.session_state["selected_ticker"]),
                     key="selected_ticker")

menu = st.sidebar.radio("Go to", ["Prediction", "News & Analysis", "Visualization", 'Report'])

if menu == "Prediction": page_prediction(df_all)
elif menu == "News & Analysis": page_news(df_all)
elif menu == "Visualization": page_visualization(df_all)
elif menu == "Report": page_report(df_all)