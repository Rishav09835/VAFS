import streamlit as st
from streamlit_chat import message  # Assuming you are using streamlit-chat
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import plotly.graph_objs as go
import requests
import numpy as np
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_chat import message
import openai

# ✅ Streamlit Configuration
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

# --- Constants ---
PRED_DIR = "C:/Users/HP/Desktop/VAFS-main/predictions"
API_KEY = 'pub_846661939da17dc42bbdfb17f639afd2863ad'
INDICATOR_OPTIONS = ["SMA-20", "SMA-50", "RSI", "EMA-20", "EMA-50", "MACD"] # More indicator options

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main-title {
            font-size: 48px;
            font-weight: 700;
            color: #1E3A8A; /* A deeper, more professional blue */
            text-align: center;
            margin-bottom: 15px; /* More spacing */
        }

        .subtitle {
            font-size: 18px;
            font-weight: 500;
            text-align: center;
            color: #4A5568; /* A softer grey */
            margin-bottom: 25px; /* More spacing */
        }

        hr {
            border: none;
            border-top: 2px solid #CBD5E0; /* A lighter, more subtle line */
            margin: 30px 0; /* Increased margin */
        }

        .metric-value {
            font-size: 24px !important;
            font-weight: 600 !important;
            color: #2D3748 !important;
        }

        .metric-label {
            font-size: 14px !important;
            color: #718096 !important;
        }

        .metric-delta-positive {
            color: #38A169 !important;
        }

        .metric-delta-negative {
            color: #E53E3E !important;
        }

        .news-article {
            border: 1px solid #E2E8F0;
            background-color: #F7FAFC;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .news-title {
            font-size: 18px;
            color: #2C5282;
            margin-bottom: 5px;
        }

        .news-meta {
            font-size: 12px;
            color: #718096;
            margin-bottom: 8px;
        }

        .sentiment-positive {
            color: #38A169;
            font-weight: bold;
        }

        .sentiment-negative {
            color: #E53E3E;
            font-weight: bold;
        }

        .sentiment-neutral {
            color: #4A5568;
            font-weight: bold;
        }

        .sidebar .st-selectbox > div > div > div {
            color: #1A202C; /* Darker text for better readability */
        }

        .sidebar .st-multiselect > div > div > div {
            color: #1A202C; /* Darker text for better readability */
        }

    </style>
    <div class="main-title">Stock Insights Dashboard 📈</div>
    <div class="subtitle">Visualize stock forecasts, technical indicators, and news sentiment</div>
    <hr>
""", unsafe_allow_html=True)

# --- Functions ---
def get_news(stock_name):
    url = f"https://newsdata.io/api/1/latest?apikey={API_KEY}&q={stock_name}&language=en"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch news for {stock_name} (Status Code: {response.status_code})")
        return []
    return response.json().get("results", [])

def analyze_sentiment(news_list):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = []
    for article in news_list:
        headline = article.get("title", "")
        sentiment = analyzer.polarity_scores(headline)
        sentiment_label = "Neutral"
        if sentiment["compound"] > 0.05:
            sentiment_label = "Positive"
        elif sentiment["compound"] < -0.05:
            sentiment_label = "Negative"

        sentiment_data.append({
            "title": headline,
            "sentiment_score": sentiment["compound"],
            "sentiment_label": sentiment_label,
            "publishedAt": article.get("pubDate", "No Date Provided")
        })
    return sentiment_data

def get_stock_names(directory):
    return sorted(
        file.replace("predictions_", "").replace(".csv", "").upper() # Display in uppercase
        for file in os.listdir(directory)
        if file.endswith(".csv") and file.startswith("predictions_")
    )

def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    return macd_line, signal_line

# --- Sidebar Filters ---
with st.sidebar:
    st.header("⚙️ Filter & Configure")
    available_stocks = get_stock_names(PRED_DIR)
    selected_stock = st.selectbox("📌 Select a Stock", available_stocks)
    indicators = st.multiselect("📊 Technical Indicators", INDICATOR_OPTIONS)

# --- Main Area ---
if selected_stock:
    st.subheader(f"📈 Stock Performance for {selected_stock}")
    pred_file = os.path.join(PRED_DIR, f"predictions_{selected_stock.lower()}.csv") # Ensure lowercase filename
    try:
        df = pd.read_csv(pred_file, parse_dates=["Date"])
    except Exception as e:
        st.error(f"Error reading {pred_file}: {e}")
    else:
        with st.spinner("Generating chart..."):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Actual Price", line=dict(color='#636EFA')))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Predictions"], mode="lines", name="Predicted Price", line=dict(color='#FFA15A')))

            if "SMA-20" in indicators:
                df["SMA20"] = df["Close"].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], mode="lines", name="SMA-20", line=dict(color='#19D3F3')))

            if "SMA-50" in indicators:
                df["SMA50"] = df["Close"].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], mode="lines", name="SMA-50", line=dict(color='#FF6692')))

            if "RSI" in indicators:
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df["RSI"] = 100 - (100 / (1 + rs))
                fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI", line=dict(color='#B6E880'), yaxis="y2"))

                # Add a secondary y-axis for RSI
                fig.update_layout(
                    yaxis2=dict(
                        title="RSI",
                        overlaying="y",
                        side="right"
                    )
                )

            if "EMA-20" in indicators:
                df["EMA20"] = calculate_ema(df["Close"], 20)
                fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], mode="lines", name="EMA-20", line=dict(color='#F4D03F')))

            if "EMA-50" in indicators:
                df["EMA50"] = calculate_ema(df["Close"], 50)
                fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], mode="lines", name="EMA-50", line=dict(color='#377EB8')))

            if "MACD" in indicators:
                macd_line, signal_line = calculate_macd(df["Close"])
                fig.add_trace(go.Scatter(x=df["Date"], y=macd_line, mode="lines", name="MACD", line=dict(color='#9467BD'), yaxis="y3"))
                fig.add_trace(go.Scatter(x=df["Date"], y=signal_line, mode="lines", name="Signal Line (MACD)", line=dict(color='#E377C2'), yaxis="y3"))

                # Add a third y-axis for MACD
                fig.update_layout(
                    yaxis3=dict(
                        title="MACD",
                        overlaying="y",
                        side="right",
                        position=1.15 # Adjust position to avoid overlap
                    )
                )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                margin=dict(l=60, r=60, t=50, b=50),
                template="plotly_white", # A cleaner template
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
def calculate_average_accuracy(pred_dir):
    files = [f for f in os.listdir(pred_dir) if f.endswith(".csv") and f.startswith("predictions_")]
    accuracies = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(pred_dir, file))
            y_true = df["Close"]
            y_pred = df["Predictions"]
            r2 = r2_score(y_true, y_pred)
            accuracies.append(r2)
        except Exception as e:
            st.warning(f"Skipping {file} due to error: {e}")
    if accuracies:
        avg_r2 = sum(accuracies) / len(accuracies)
        return avg_r2 * 100  # convert to percentage
    else:
        return None

# --- Model Accuracy Section ---
st.subheader("🧮 Average Model Accuracy Across All Stocks")

average_accuracy = calculate_average_accuracy(PRED_DIR)

if average_accuracy is not None:
    st.write(f"Average Model Accuracy for LSTM across {len(os.listdir(PRED_DIR))} stocks: {average_accuracy:.2f}%")
else:
    st.write("No valid prediction files found to calculate accuracy.")

        # --- Metrics Section ---
st.subheader("📊 Key Metrics")
latest_date = df["Date"].max()
latest_close = df.loc[df["Date"] == latest_date, "Close"].values[0]

current_month = latest_date.month
current_year = latest_date.year
df_month = df[(df["Date"].dt.month == current_month) & (df["Date"].dt.year == current_year)]

if not df_month.empty:
            avg_month_close = df_month["Close"].mean()
            start_price = df_month.iloc[0]["Close"]
            end_price = df_month.iloc[-1]["Close"]
            pct_change = ((end_price - start_price) / start_price) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Price", f"₹{latest_close:.2f}", f"{pct_change:.2f}%", delta_color="normal")
            col2.metric("Monthly Average", f"₹{avg_month_close:.2f}")
            col3.metric("Month Change", f"{pct_change:.2f}%", delta_color="normal")
else:
            st.warning("Not enough data for monthly statistics.")

st.markdown("<hr>", unsafe_allow_html=True)

        # --- News Sentiment Section ---
st.subheader(f"📰 Latest News & Sentiment for {selected_stock}")
news = get_news(selected_stock.lower()) # Ensure lowercase for news API
sentiment_results = analyze_sentiment(news)

if sentiment_results:
            avg_sentiment_score = np.mean([res["sentiment_score"] for res in sentiment_results])

            sentiment_label_overall = "Neutral"
            if avg_sentiment_score > 0.05:
                sentiment_label_overall = "Positive"
            elif avg_sentiment_score < -0.05:
                sentiment_label_overall = "Negative"

            st.markdown(f"**Overall Sentiment:** <span class='sentiment-{sentiment_label_overall.lower()}'>{sentiment_label_overall}</span> (Score: `{avg_sentiment_score:.2f}`) ", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

            for article in sentiment_results:
                sentiment_class = f"sentiment-{article['sentiment_label'].lower()}"
                st.markdown(f"""
                    <div class="news-article">
                        <h4 class="news-title">📰 {article['title']}</h4>
                        <p class="news-meta">
                            🗓️ Published: {article['publishedAt']}<br>
                            📈 Sentiment: <span class="{sentiment_class}">{article['sentiment_label']}</span> (Score: `{article['sentiment_score']:.2f}`)
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"No recent news found for {selected_stock}.")
else:
        st.info("Select a stock from the sidebar to view its forecast and analysis.")


# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Callback to handle user input and generate response
def handle_input():
    user_input = st.session_state.user_input_field
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        user_text = user_input.lower()
        if "sma" in user_text:
            bot_reply = "SMA (Simple Moving Average) smooths out price data by averaging closing prices over a selected period. Example: SMA-20 shows the 20-day average."
        elif "ema" in user_text:
            bot_reply = "EMA (Exponential Moving Average) gives more weight to recent prices compared to SMA. EMA-20 reacts faster to price changes than SMA-20."
        elif "rsi" in user_text:
            bot_reply = "RSI (Relative Strength Index) measures the speed and change of price movements. Values >70 indicate overbought, <30 indicate oversold conditions."
        elif "macd" in user_text:
            bot_reply = "MACD (Moving Average Convergence Divergence) shows trend strength and momentum using the difference between two EMAs. Useful for spotting trend reversals."
        elif "indicator" in user_text or "help" in user_text:
            bot_reply = "You can select technical indicators from the sidebar (SMA, EMA, RSI, MACD) to visualize them on the stock chart. Just check the boxes!"
        elif "how to" in user_text or "usage" in user_text:
            bot_reply = "To use the dashboard: Select a stock from the sidebar, choose indicators, and view price trends, forecasts, and news sentiment analysis."
        elif "candlestick" in user_text or "chart" in user_text or "graph" in user_text:
            bot_reply = "Candlestick charts show the open, high, low, and close prices. A green (or hollow) candle shows price went up, a red (or filled) candle shows price went down."
        elif "support" in user_text:
            bot_reply = "Support is a price level where the stock tends to find buying interest, preventing it from falling further. It's visible as a floor on the chart."
        elif "resistance" in user_text:
            bot_reply = "Resistance is a price level where the stock tends to face selling pressure, preventing it from rising further. It acts like a ceiling."
        elif "trend" in user_text:
            bot_reply = "An uptrend is when prices make higher highs and higher lows. A downtrend is when prices make lower highs and lower lows. Trendlines help visualize them."
        elif "volume" in user_text:
            bot_reply = "Volume shows how many shares were traded during a time period. High volume during a price move confirms strength; low volume may signal weakness."
        else:
            bot_reply = "I'm a simple assistant. You can ask about SMA, EMA, RSI, MACD, or how to use this dashboard."

        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

        # Clear the input field (will only work because it's inside the callback)
        st.session_state.user_input_field = ""

# --- Chatbot UI ---
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("💬 Chatbot Assistant (Ask about indicators, usage help)", expanded=False):
    for i, chat in enumerate(st.session_state.chat_history):
        message(chat["content"], is_user=(chat["role"] == "user"), key=f"chat_{i}")

    # Text input with on_change callback (no need for button)
    st.text_input("Type your message and press Enter...", 
                  key="user_input_field", 
                  on_change=handle_input)
