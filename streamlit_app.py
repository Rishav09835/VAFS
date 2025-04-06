import streamlit as st
import pandas as pd
import os
import plotly.graph_objs as go

# Set the folder path where your prediction CSV files are stored.
# Make sure this folder exists and is populated with CSV files from your training code.
PRED_DIR = "C:/Users/HP/Desktop/Majorproject2025/predictions"

# Helper function to get list of available stock names from prediction files
def get_stock_names(directory):
    stock_names = []
    for file in os.listdir(directory):
        if file.endswith(".csv") and file.startswith("predictions_"):
            name = file.replace("predictions_", "").replace(".csv", "")
            stock_names.append(name)
    return sorted(stock_names)

# Get the available stocks from the predictions folder.
available_stocks = get_stock_names(PRED_DIR)

# Build a minimalistic UI using Streamlit.
st.title("Stock Forecasting Dashboard")
st.markdown("### Select a stock to see Actual vs Predicted Prices:")

# Dropdown widget for stock selection.
selected_stock = st.selectbox("Select Stock", available_stocks)

# When a stock is selected, load its prediction CSV and create the Plotly graph.
if selected_stock:
    pred_file = os.path.join(PRED_DIR, f"predictions_{selected_stock}.csv")
    try:
        df = pd.read_csv(pred_file, parse_dates=["Date"])
    except Exception as e:
        st.error(f"Error reading {pred_file}: {e}")
    else:
        # Build the Plotly graph.
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                                 mode="lines", name="Actual Price"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Predictions"],
                                 mode="lines", name="Predicted Price"))
        fig.update_layout(
            title=f"{selected_stock}: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)