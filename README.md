# 📈 ML-Based Stock Price Forecasting and Interactive Visualization Platform

This project is a Machine Learning–powered platform designed to forecast stock prices using historical data and visualize results through an interactive Streamlit dashboard. It combines advanced time-series models, sentiment analysis, and user-friendly design to assist retail investors, analysts, and learners in exploring stock trends.

---

## 🔍 Features

- ✅ Predict stock prices using advanced ML models (LSTM, GRU, BiLSTM, Attention-LSTM, Vanilla RNN)
- 🧠 Integrated sentiment analysis using news headlines (VADER NLP)
- 📊 Real-time interactive visualizations using Streamlit and Plotly
- 📈 Technical indicators (SMA, EMA, RSI, MACD)
- 💬 Built-in assistant bot to explain models and predictions
- 📥 Upload your own stock CSV or select from NIFTY-50 preloaded options (TO BE IMPLEMENTED SOON 🚧)
- 🧪 Model performance comparison with metrics (RMSE, MAE, R²)
- 📤 Export predictions and raw data as CSV (TO BE IMPLEMENTED SOON 🚧)


---

## 🧠 Models Implemented

| Model              | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| LSTM              | Capture long-term dependencies in price sequences        |
| BiLSTM            | Look at both past and future context (for training)      |
| GRU               | Efficient training, fewer parameters                     |
| Vanilla RNN       | Baseline performance model                               |
| Attention-LSTM    | Focus on important time steps for better prediction      |

---

## 💻 Tech Stack

- **Python 3.8+**
- **TensorFlow/Keras** – Deep learning models
- **Scikit-learn** – Preprocessing, metrics
- **NLTK + VADER** – Sentiment analysis
- **Pandas, NumPy** – Data manipulation
- **Plotly, Seaborn, Matplotlib** – Visualization
- **Streamlit** – Web interface
- **yfinance** – Fetching stock data
- **BeautifulSoup / Requests** – News scraping

---

## 🚀 How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. **Install dependencies**

3. **Run the app:**
---
# 🛠️ Future Enhancements
- 🔁 Real-time stock prediction using live APIs
- 📱 Mobile-friendly layout
- 🔔 Alert system for major price/sentiment shifts
- 🧠 Transformer-based forecasting models


